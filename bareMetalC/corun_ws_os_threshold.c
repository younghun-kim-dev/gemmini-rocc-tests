// corun_ws_os_threshold.c
// BOOM+Rocket 이기종에서: BOOM 하트는 ws_os_threshold 실행,
// Rocket 하트들은 메모리 스트레스 동시 실행.

// ─────────────  설정 매크로(컴파일 타임 -D로 덮어쓰기 가능)  ─────────────
#ifndef GEMMINI_HART_ID
#define GEMMINI_HART_ID 2   // 예: Rocket 2개(0,1) + BOOM 1개(2) 구성이면 2가 BOOM
#endif
#ifndef ROCKET_HART_MASK
#define ROCKET_HART_MASK 0x3 // 하트 0,1이 Rocket → (1<<0)|(1<<1)=0b011=0x3
#endif
#ifndef OFFSET_CYCLES
#define OFFSET_CYCLES 0     // 0이면 동시 시작, >0이면 hartid*OFFSET 만큼 지연
#endif
#ifndef STRESS_MODE
#define STRESS_MODE 1       // 1: STREAM-READ, 2: POINTER-CHASE, 3: STREAM-TRIAD
#endif
// ▶ “빨리 끝나는” 기본값(필요시 -DSTREAM_BYTES=(8<<20) 등으로 덮어쓰기)
#ifndef STREAM_BYTES
#define STREAM_BYTES (64<<10) // 64KB
#endif
#ifndef CHASE_ELEMS
#define CHASE_ELEMS (1<<14)   // 16K elems (~64KB for uint32)
#endif
#ifndef CHASE_ITERS
#define CHASE_ITERS (1<<16)   // 65536 steps
#endif

// ─────────────  기존 ws_os_threshold 본문  ─────────────
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_params.h"

/* 모든 하트를 main()으로 진입시키는 약한 별칭 (링커 옵션 없이 가능) */
extern int main(void);
int __main(void) __attribute__((weak, alias("main")));

// ======== 문제 크기 / 스윕 설정 ========
#define M_FIXED 8
#define N_FIXED 8
static const int KSET[] = {1,2,3,4,8};
#define KSET_LEN (sizeof(KSET)/sizeof(KSET[0]))

// 정적 버퍼 최대 크기 (여유)
#define M_MAX 32
#define N_MAX 32
#define K_MAX 128

// Row-major strides
#define STRIDE_A(k) (k)
#define STRIDE_B(n) (n)
#define STRIDE_D(n) (n)
#define STRIDE_C(n) (n)

// 정렬
#ifndef ALGN
#define ALGN __attribute__((aligned(64)))
#endif

// 정적 버퍼
static elem_t A_buf[M_MAX*K_MAX] ALGN;
static elem_t B_buf[K_MAX*N_MAX] ALGN;
static elem_t Ccpu_buf[M_MAX*N_MAX] ALGN;
static elem_t Cacc_buf[M_MAX*N_MAX] ALGN;
static acc_t  D_buf[M_MAX*N_MAX]   ALGN; // CPU acc32 임시 + Gemmini D/psum 겸용

// 심볼 가드
#ifndef OS
#define OS 0
#endif
#ifndef WS
#define WS 1
#endif
#ifndef NO_ACTIVATION
#define NO_ACTIVATION 0
#endif

// ======== 유틸 ========
static inline uint64_t rdcycle(){
  uint64_t c; asm volatile ("rdcycle %0" : "=r"(c)); return c;
}
static inline int8_t sat_i8(int32_t x){
  if (x > 127) return 127;
  if (x < -128) return -128;
  return (int8_t)x;
}
static void fill_i8(elem_t *a, int len, int seed){
  uint32_t s = (uint32_t)seed * 2654435761u;
  for (int i=0;i<len;i++){
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    a[i] = (elem_t)((int)(s & 0xFF) - 128);
  }
}
static void zero_i8(elem_t *a, int len){ memset(a, 0, len*sizeof(elem_t)); }
static void zero_i32(acc_t *a, int len){ memset(a, 0, len*sizeof(acc_t)); }

// ======== CPU 블로킹 GEMM ========
#define BI 8
#define BJ 8
#define BK 32
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif
static uint64_t cpu_matmul_i8_rowmajor_blocked(
    int M,int N,int K,
    const elem_t* restrict A,
    const elem_t* restrict B,
    elem_t* restrict C,
    acc_t* restrict ACC)
{
  for (int i=0;i<M*N;i++) ACC[i] = 0;
  uint64_t t0 = rdcycle();
  for (int ii=0; ii<M; ii+=BI){
    int i_end = MIN(ii+BI, M);
    for (int kk=0; kk<K; kk+=BK){
      int k_end = MIN(kk+BK, K);
      for (int jj=0; jj<N; jj+=BJ){
        int j_end = MIN(jj+BJ, N);
        for (int i=ii; i<i_end; i++){
          const int arow = i*K;
          const int crow = i*N;
          for (int k=kk; k<k_end; k++){
            const int8_t a_ik = A[arow + k];
            for (int j=jj; j<j_end; j++){
              ACC[crow + j] += (int32_t)a_ik * (int32_t)B[k*N + j];
            }
          }
        }
      }
    }
  }
  for (int i=0;i<M*N;i++){
    int32_t x = ACC[i];
    if (x > 127) x = 127;
    else if (x < -128) x = -128;
    C[i] = (elem_t)x;
  }
  uint64_t t1 = rdcycle();
  return t1 - t0;
}

// ======== Gemmini 경로 ========
static uint64_t gemmini_matmul(int dataflow, int M,int N,int K,
                               const elem_t* A,const elem_t* B, const acc_t* D, elem_t* C)
{
  gemmini_config_ex(dataflow, NO_ACTIVATION, 0);
  gemmini_flush(0);

  const size_t sA = STRIDE_A(K);
  const size_t sB = STRIDE_B(N);
  const size_t sD = STRIDE_D(N);
  const size_t sC = STRIDE_C(N);

  const scale_t      A_scale   = (scale_t)1.0f;
  const scale_t      B_scale   = (scale_t)1.0f;
  const scale_acc_t  D_scale   = (scale_acc_t)1;
  const int          act       = NO_ACTIVATION;
  const acc_scale_t  C_scale   = (acc_scale_t)1;
  const acc_scale_t  out_scale = (acc_scale_t)1;

  const bool repeating_bias=false, A_t=false, B_t=false, full_C=false, low_D=false;
  const uint8_t extra_flag=0;

  uint64_t t0 = rdcycle();

  tiled_matmul_auto(
    (size_t)M,(size_t)N,(size_t)K,
    A,B,(const void*)D,(void*)C,
    sA,sB,sD,sC,
    A_scale,B_scale,D_scale,
    act,C_scale,out_scale,
    repeating_bias,A_t,B_t, full_C,low_D,
    extra_flag,
    (enum tiled_matmul_type_t)dataflow
  );

  gemmini_fence();
  uint64_t t1 = rdcycle();
  return t1 - t0;
}

// ======== 스윕 & 임계점 ========
static void run_threshold_sweep(int M, int N){
  printf("== Offloading threshold sweep (M=%d, N=%d, row-major, blocked-CPU) ==\n", M,N);
  int kstar = -1;

  for (int idx=0; idx<KSET_LEN; idx++){
    int K = KSET[idx];
    if (K > K_MAX || M > M_MAX || N > N_MAX) break;

    fill_i8(A_buf, M*K, 1);
    fill_i8(B_buf, K*N, 2);
    zero_i8(Ccpu_buf, M*N);
    zero_i8(Cacc_buf, M*N);

    uint64_t cyc_cpu = cpu_matmul_i8_rowmajor_blocked(M,N,K, A_buf,B_buf, Ccpu_buf, D_buf);

    zero_i32(D_buf, M*N);

    uint64_t cyc_ws  = gemmini_matmul(WS, M,N,K, A_buf,B_buf, D_buf, Cacc_buf);
    uint64_t cyc_os  = gemmini_matmul(OS, M,N,K, A_buf,B_buf, D_buf, Cacc_buf);
    uint64_t cyc_acc = (cyc_ws < cyc_os) ? cyc_ws : cyc_os;
    int best = (cyc_ws < cyc_os) ? WS : OS;

    uint64_t spx100 = cyc_acc ? (cyc_cpu*100ULL)/cyc_acc : 0ULL;

    printf("[K=%2d] CPU:%10llu cyc | ACC(%s):%10llu cyc | speedup=%3llu.%02llux %s\n",
           K,
           (unsigned long long)cyc_cpu,
           (best==WS?"WS":"OS"),
           (unsigned long long)cyc_acc,
           (unsigned long long)(spx100/100), (unsigned long long)(spx100%100),
           (cyc_acc <= cyc_cpu ? "<- OFFLOAD" : "")
    );

    if (kstar < 0 && cyc_acc <= cyc_cpu) kstar = K;
  }

  if (kstar >= 0) printf("== Threshold K* = %d ==\n", kstar);
  else            printf("== No threshold found up to tested K ==\n");
}

// ─────────────  코런용 스트레스 루틴 & main  ─────────────
static volatile uint32_t go_flag = 0;

static inline void barrier_all() {
  size_t me; asm volatile ("csrr %0, mhartid" : "=r"(me));
  if (me == 0) __atomic_store_n(&go_flag, 1, __ATOMIC_RELEASE);
  while (!__atomic_load_n(&go_flag, __ATOMIC_ACQUIRE)) ;
}

static inline void delay_cycles(uint64_t c) {
  uint64_t t0 = rdcycle();
  while (rdcycle() - t0 < c) ;
}

// STREAM-READ (read-only) 스트레스
static uint64_t stream_read_bytes(size_t bytes) {
  static uint8_t ALGN buf[STREAM_BYTES];
  volatile uint64_t acc = 0;
  uint64_t t0 = rdcycle();
  for (size_t i = 0; i < bytes; i += 64) {
    acc += buf[i];
  }
  uint64_t t1 = rdcycle();
  if (acc == 0xdeadbeefULL) printf("acc=%llu\n", (unsigned long long)acc);
  return t1 - t0;
}

// STREAM-Triad: A = B + s*C (간단형)
static uint64_t stream_triad(size_t elems) {
  static int32_t ALGN A[STREAM_BYTES/4], B[STREAM_BYTES/4], C[STREAM_BYTES/4];
  const int32_t s = 3;
  uint64_t t0 = rdcycle();
  for (size_t i = 0; i < elems; i++) {
    A[i] = B[i] + s*C[i];
  }
  uint64_t t1 = rdcycle();
  return t1 - t0;
}

// 포인터 체이싱(랜덤 permutation)
static uint32_t ALGN chase_next[CHASE_ELEMS];
static void build_chase_list(void) {
  for (uint32_t i=0;i<CHASE_ELEMS;i++) chase_next[i] = i;
  uint32_t x=1;
  for (uint32_t i=0;i<CHASE_ELEMS;i++){
    x = x*1664525u + 1013904223u;
    uint32_t j = x % CHASE_ELEMS;
    uint32_t t = chase_next[i]; chase_next[i] = chase_next[j]; chase_next[j] = t;
  }
}
static uint64_t pointer_chase(size_t iters) {
  volatile uint32_t idx = 0;
  uint64_t t0 = rdcycle();
  for (size_t i=0;i<iters;i++) idx = chase_next[idx];
  uint64_t t1 = rdcycle();
  if (idx==0xFFFFFFFFu) printf("idx=%u\n", idx);
  return t1 - t0;
}

int main(){
  size_t me; asm volatile ("csrr %0, mhartid" : "=r"(me));
  barrier_all();
  if (OFFSET_CYCLES) delay_cycles(me * (uint64_t)OFFSET_CYCLES);

  if (me == GEMMINI_HART_ID) {
    printf("[ROLE] hart=%lu → GEMMINI_KSTAR\n", (unsigned long)me);
    run_threshold_sweep(M_FIXED, N_FIXED);
  } else if ( (ROCKET_HART_MASK >> me) & 0x1 ) {
    printf("[ROLE] hart=%lu → ROCKET_STRESS mode=%d\n", (unsigned long)me, (int)STRESS_MODE);
    uint64_t cyc=0;
    if (STRESS_MODE == 1) {
      cyc = stream_read_bytes(STREAM_BYTES);
    } else if (STRESS_MODE == 2) {
      build_chase_list();
      cyc = pointer_chase(CHASE_ITERS);
    } else /* STRESS_MODE == 3 */ {
      cyc = stream_triad(STREAM_BYTES/4);
    }
    printf("[STRESS] hart=%lu cycles=%llu\n", (unsigned long)me, (unsigned long long)cyc);
  } else {
    printf("[ROLE] hart=%lu → IDLE\n", (unsigned long)me);
  }

  printf("Done (hart=%lu).\n", (unsigned long)me);
  return 0;
}

