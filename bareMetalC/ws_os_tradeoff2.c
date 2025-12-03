// ws_os_single.c (baremetal-safe, no malloc; runtime WS/OS select; single-case run)
// Compare either WS(Weight-Stationary) or OS(Output-Stationary) for one M×N×K only.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_params.h"

// ======= user-selectable single case (compile-time) =======
// Dimensions for the *single* run (must be <= M_MAX/N_MAX/K_MAX)
#ifndef M_SEL
#define M_SEL 4
#endif
#ifndef N_SEL
#define N_SEL 4
#endif
#ifndef K_SEL
#define K_SEL 1
#endif

// Dataflow select: OS=0, WS=1 (matches gemmini tiled_matmul_type_t)
#ifndef DATAFLOW_SEL
#define DATAFLOW_SEL OS
#endif

// Warmups/repetitions (same dataflow only). Default: warmup OFF as requested.
#ifndef WARMUPS
#define WARMUPS 0
#endif
#ifndef REPS
#define REPS 1
#endif

// ======= static buffer sizing (upper bounds) =======
#define M_MAX 128
#define N_MAX 256
#define K_MAX 512

// Row-major strides
#define STRIDE_A(k) (k)
#define STRIDE_B(n) (n)
#define STRIDE_D(n) (n)
#define STRIDE_C(n) (n)

// 64B align helps DMA
#ifndef ALGN
#define ALGN __attribute__((aligned(64)))
#endif

// If not provided by includes:
#ifndef OS
#define OS 0
#endif
#ifndef WS
#define WS 1
#endif
#ifndef NO_ACTIVATION
#define NO_ACTIVATION 0
#endif

static elem_t A_buf[M_MAX * K_MAX] ALGN;
static elem_t B_buf[K_MAX * N_MAX] ALGN;
static elem_t Cws_buf[M_MAX * N_MAX] ALGN;
static elem_t Cos_buf[M_MAX * N_MAX] ALGN;
static acc_t  D_buf [M_MAX * N_MAX] ALGN;

// -------- util ----------
static inline uint64_t rdcycle(){
  uint64_t c; asm volatile ("rdcycle %0" : "=r"(c)); return c;
}

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

// -------- one matmul run (returns cycles) ----------
static uint64_t run_one(int dataflow, int M, int N, int K,
                        elem_t *A, elem_t *B, acc_t *Dacc, elem_t *C)
{
  // runtime dataflow select
  gemmini_config_ex(dataflow, NO_ACTIVATION, 0);
  gemmini_flush(0);

  const size_t stride_A = STRIDE_A(K);
  const size_t stride_B = STRIDE_B(N);
  const size_t stride_D = STRIDE_D(N);
  const size_t stride_C = STRIDE_C(N);

  const scale_t      A_scale   = (scale_t)1;   // avoid float const
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

  uint64_t t0 = rdcycle();

  tiled_matmul_auto(
    (size_t)M, (size_t)N, (size_t)K,
    (const elem_t*)A, (const elem_t*)B,
    (const void*)Dacc, (void*)C,
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

// -------- single-case driver ----------
static void run_single(int M, int N, int K, int dataflow) {
  // bounds check
  if (M > M_MAX || N > N_MAX || K > K_MAX) {
    printf("Error: dims exceed static buffers (M<=%d,N<=%d,K<=%d)\n",
           M_MAX, N_MAX, K_MAX);
    return;
  }

  elem_t *A = A_buf;
  elem_t *B = B_buf;
  acc_t  *D = D_buf;
  elem_t *C = (dataflow == WS) ? Cws_buf : Cos_buf;

  // init only the needed regions
  fill_i8(A, M*K, 1);
  fill_i8(B, K*N, 2);
  zero_i32(D, M*N);
  zero_i8(C, M*N);

  // warmups (same dataflow only)
  for (int w=0; w<WARMUPS; w++) (void)run_one(dataflow, M,N,K, A,B,D, C);

  // measure: REPS times, keep best (min) cycles
  uint64_t best = (uint64_t)-1;
  for (int r=0; r<REPS; r++) {
    uint64_t cyc = run_one(dataflow, M,N,K, A,B,D, C);
    if (cyc < best) best = cyc;
  }

  // integer-only throughput metric
  uint64_t macs = (uint64_t)M * (uint64_t)N * (uint64_t)K;
  uint64_t mac_per_100cyc = (best ? (macs*100ULL)/best : 0ULL);

  long long sum = 0;
  for (int i=0;i<M*N;i++) sum += C[i];

  printf("[single M=%d N=%d K=%d | %s | reps=%d warmups=%d] "
         "%llu cyc  (%llu MAC/100cyc) | sum=%lld\n",
         M,N,K, dataflow==WS ? "WS" : "OS", REPS, WARMUPS,
         (unsigned long long)best, (unsigned long long)mac_per_100cyc, sum);
}

int main(){
  printf("WS/OS single-run (static buffers, row-major)\n");
  run_single(M_SEL, N_SEL, K_SEL, DATAFLOW_SEL);
  printf("Done.\n");
  return 0;
}

