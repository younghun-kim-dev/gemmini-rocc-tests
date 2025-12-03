// ws_os_single.c — One-shot MM (CPU or Gemmini WS/OS) with optional warmup
// Baremetal-safe (no malloc/float)

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_params.h"

// === Run-mode (compile-time) ===============================================
// -DMODE=0  -> CPU (blocked int8 GEMM)
// -DMODE=1  -> GEMMINI (WS/OS)   [default]
#define MODE_CPU      0
#define MODE_GEMMINI  1
#ifndef MODE
#define MODE 1
#endif
// ==========================================================================

// ===== User-tunable compile-time knobs =====
// 예: -DONE_M=128 -DONE_N=256 -DONE_K=512  (미지정 시 아래 기본값)
#ifndef ONE_M
#define ONE_M 512
#endif
#ifndef ONE_N
#define ONE_N 512
#endif
#ifndef ONE_K
#define ONE_K 512
#endif

// 데이터플로우 선택: -DDATAFLOW=WS 또는 -DDATAFLOW=OS (GEMMINI 모드에서만 사용)
#ifndef DATAFLOW
#define DATAFLOW OS
#endif

// --- Warmup control ---
// 기본 1회 웜업(측정 제외). 필요 시 -DWARMUP_RUNS=0/2/… 로 조정
#ifndef WARMUP_RUNS
#define WARMUP_RUNS 1
#endif

// --- Cache eviction control (Cold-start 용) ---
// 기본 OFF. Cold 실험 시 -DDO_CACHE_EVICT=1 로 켜기
#ifndef DO_CACHE_EVICT
#define DO_CACHE_EVICT 0
#endif
#ifndef EVICT_BYTES
#define EVICT_BYTES (2*1024*1024) // L2보다 조금 크게
#endif

// ===== Strides / Align =====
#define STRIDE_A(k) (k)
#define STRIDE_B(n) (n)
#define STRIDE_D(n) (n)
#define STRIDE_C(n) (n)

#ifndef ALGN
#define ALGN __attribute__((aligned(64)))
#endif

// ===== Buffers =====
static elem_t A_buf[ONE_M * ONE_K] ALGN;
static elem_t B_buf[ONE_K * ONE_N] ALGN;
static elem_t C_out[ONE_M * ONE_N] ALGN;
static acc_t  D_bias[ONE_M * ONE_N] ALGN;  // GEMMINI: D/psum, CPU: ACC 임시 버퍼

#if DO_CACHE_EVICT
static volatile uint8_t cache_sweep[EVICT_BYTES] ALGN;
#endif

// ===== Utils =====
static inline uint64_t rdcycle(){
  uint64_t c; asm volatile ("rdcycle %0" : "=r"(c)); return c;
}
static inline void full_memory_fence() { asm volatile("fence" ::: "memory"); }

static void fill_i8(elem_t *a, int len, int seed){
  uint32_t s = (uint32_t)seed * 2654435761u;
  for (int i=0;i<len;i++){
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    int v = (int)(s & 0xFF) - 128;
    a[i] = (elem_t)v;
  }
}
static void zero_i8(elem_t *a, int len){ memset(a, 0, len*sizeof(elem_t)); }
static void zero_i32(acc_t *a, int len){ memset(a, 0, len*sizeof(acc_t)); }

static void evict_caches(void){
#if DO_CACHE_EVICT
  for (size_t i = 0; i < EVICT_BYTES; i += 64) cache_sweep[i] = (uint8_t)(i & 0xFF);
  for (size_t i = 0; i < EVICT_BYTES; i += 64) (void)cache_sweep[i];
  full_memory_fence();
#endif
}

// ===== CPU blocked GEMM (int8 x int8 -> acc32 -> clamp int8) ===============
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif
#ifndef BI
#define BI 8
#endif
#ifndef BJ
#define BJ 8
#endif
#ifndef BK
#define BK 32
#endif

static uint64_t cpu_mm_kernel(int M,int N,int K,
                              const elem_t* A,const elem_t* B,
                              acc_t* ACC, elem_t* C)
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

        for (int i=ii; i<i_end; i++){
          const int arow = i*K;
          const int crow = i*N;
          for (int k=kk; k<k_end; k++){
            const int8_t a_ik = (int8_t)A[arow + k];
            const int brow = k*N;
            for (int j=jj; j<j_end; j++){
              ACC[crow + j] += (int32_t)a_ik * (int32_t)((int8_t)B[brow + j]);
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
  return (t1 - t0);
}

// ===== Gemmini kernel (config + matmul; 측정은 여기서만) ===================
static uint64_t gemmini_mm_kernel(int dataflow, int M, int N, int K,
                                  elem_t *A, elem_t *B, acc_t *Dbias, elem_t *C)
{
#ifndef NO_ACTIVATION
#define NO_ACTIVATION 0
#endif
  const size_t stride_A = STRIDE_A(K);
  const size_t stride_B = STRIDE_B(N);
  const size_t stride_D = STRIDE_D(N);
  const size_t stride_C = STRIDE_C(N);

  const scale_t      A_scale   = (scale_t)1;
  const scale_t      B_scale   = (scale_t)1;
  const scale_acc_t  D_scale   = (scale_acc_t)1;
  const int          act       = NO_ACTIVATION;
  const acc_scale_t  C_scale   = (acc_scale_t)1;
  const acc_scale_t  out_scale = (acc_scale_t)1;

  const bool repeating_bias = false;
  const bool A_transpose    = false;
  const bool B_transpose    = false;
  const bool full_C         = false;
  const bool low_D          = false;
  const uint8_t extra_flag  = 0;

  gemmini_config_ex(dataflow, NO_ACTIVATION, 0);
  gemmini_fence();

  uint64_t t0 = rdcycle();

  tiled_matmul_auto(
    (size_t)M, (size_t)N, (size_t)K,
    (const elem_t*)A, (const elem_t*)B,
    (const void*)Dbias, (void*)C,
    stride_A, stride_B, stride_D, stride_C,
    A_scale, B_scale, D_scale,
    act, C_scale, out_scale,
    repeating_bias, A_transpose, B_transpose,
    full_C, low_D,
    extra_flag,
    (enum tiled_matmul_type_t)dataflow
  );

  gemmini_fence();
  uint64_t t1 = rdcycle();

  return t1 - t0;
}

// ===== 공통 실행 래퍼 (측정 대상 커널 1회) =================================
static uint64_t run_once(int M,int N,int K,
                         elem_t *A, elem_t *B,
                         acc_t *Dbias_or_ACC, elem_t *C)
{
#if (MODE == MODE_CPU)
  return cpu_mm_kernel(M,N,K, A,B, Dbias_or_ACC, C);
#else
  return gemmini_mm_kernel(DATAFLOW, M,N,K, A,B, Dbias_or_ACC, C);
#endif
}

int main(void){
  const int M = ONE_M, N = ONE_N, K = ONE_K;

  // 0) 입력/출력 초기화 (항상 같은 값 → 재현성)
  fill_i8(A_buf, M*K, 1);
  fill_i8(B_buf, K*N, 2);
  zero_i32(D_bias, M*N);   // CPU에서는 ACC 임시 버퍼, GEMMINI에서는 D/psum
  zero_i8(C_out,  M*N);

  // 1) 시작 시점 상태 정리 (외부 영향 최소화: 단 1회)
  full_memory_fence();
#if (MODE == MODE_GEMMINI)
  gemmini_fence();
  gemmini_flush(0);
#endif
  full_memory_fence();

  // 2) (옵션) Cold-start 원할 때만 캐시 축출
  evict_caches();

  // 3) 웜업 (측정 제외). 같은 커널로 자기 자신 예열
  for (int w = 0; w < WARMUP_RUNS; w++) {
    (void)run_once(M,N,K, A_buf, B_buf, D_bias, C_out);
  }

  // 4) 측정 1회
  uint64_t cyc = run_once(M,N,K, A_buf, B_buf, D_bias, C_out);

  // 5) 지표 계산/출력
  const uint64_t macs = (uint64_t)M * (uint64_t)N * (uint64_t)K;
  const uint64_t mac_per_100cyc = (cyc ? (macs*100ULL)/cyc : 0ULL);

  long sumC = 0;
  for (int i=0;i<M*N;i++) sumC += C_out[i];

#if (MODE == MODE_CPU)
  printf("CPU One-shot MM     | Backend=CPU-blocked  M=%d N=%d K=%d\n", M,N,K);
  printf("warmups=%d | cycles=%llu | MAC/100cyc=%llu | sum(C)=%ld\n",
         WARMUP_RUNS,
         (unsigned long long)cyc,
         (unsigned long long)mac_per_100cyc,
         sumC);
#else
  const char *df_str = (DATAFLOW == WS) ? "WS" : "OS";
  printf("Gemmini One-shot MM | DF=%s  M=%d N=%d K=%d\n", df_str, M, N, K);
  printf("warmups=%d | cycles=%llu | MAC/100cyc=%llu | sum(C)=%ld\n",
         WARMUP_RUNS,
         (unsigned long long)cyc,
         (unsigned long long)mac_per_100cyc,
         sumC);
#endif

  return 0;
}
