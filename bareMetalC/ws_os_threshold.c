// ws_os_threshold.c  (M=N=8, CPU 블로킹 버전만 사용)
// CPU(블로킹 GEMM) vs Gemmini(WS/OS) 사이클 비교로 오프로딩 임계점 찾기.
// Baremetal-safe: malloc/float 미사용. Row-major.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_params.h"

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

// ======== CPU 블로킹 GEMM (int8 x int8 -> acc32 -> clamp) ========
// 타일 크기: L1 캐시 친화적인 기본값 (필요시 조정)
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
    acc_t* restrict ACC)   // 크기 MxN 임시 acc32 버퍼
{
  // ACC 초기화
  for (int i=0;i<M*N;i++) ACC[i] = 0;

  uint64_t t0 = rdcycle();

  for (int ii=0; ii<M; ii+=BI){
    int i_end = MIN(ii+BI, M);
    for (int kk=0; kk<K; kk+=BK){
      int k_end = MIN(kk+BK, K);

      for (int jj=0; jj<N; jj+=BJ){
        int j_end = MIN(jj+BJ, N);

        // i - k - j 순서 (B는 연속 접근, C는 행 연속)
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

  // acc32 -> int8 clamp
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

    // 입력/버퍼 준비
    fill_i8(A_buf, M*K, 1);
    fill_i8(B_buf, K*N, 2);
    zero_i8(Ccpu_buf, M*N);
    zero_i8(Cacc_buf, M*N);

    // CPU 블로킹 GEMM (D_buf를 acc32 임시 버퍼로 사용)
    uint64_t cyc_cpu = cpu_matmul_i8_rowmajor_blocked(M,N,K, A_buf,B_buf, Ccpu_buf, D_buf);

    // Gemmini 호출 전 D_buf(acc/psum) 다시 0으로
    zero_i32(D_buf, M*N);

    // Gemmini WS/OS 중 빠른 쪽 선택
    uint64_t cyc_ws  = gemmini_matmul(WS, M,N,K, A_buf,B_buf, D_buf, Cacc_buf);
    uint64_t cyc_os  = gemmini_matmul(OS, M,N,K, A_buf,B_buf, D_buf, Cacc_buf);
    uint64_t cyc_acc = (cyc_ws < cyc_os) ? cyc_ws : cyc_os;
    int best = (cyc_ws < cyc_os) ? WS : OS;

    // 정수 speedup×100 (>=100 이면 가속기 승)
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

int main(){
  printf("Gemmini offloading threshold finder (M=N=8, K in {1,2,3,4,8})\n");
  run_threshold_sweep(M_FIXED, N_FIXED);
  printf("Done.\n");
  return 0;
}

