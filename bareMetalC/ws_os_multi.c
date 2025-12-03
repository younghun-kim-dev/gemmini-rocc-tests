// ws_os_multi.c — Multi-case Gemmini WS/OS harness with per-case isolation
// Baremetal-safe (no malloc/float), kernel-only timing, high reproducibility.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "include/gemmini.h"
#include "include/gemmini_params.h"

// ===================== User knobs =====================
// --- Sets to sweep (edit here) ---
static const int MSET[] = { 4, 8, 16 };
static const int NSET[] = { 4, 8 };
static const int KSET[] = { 4, 8 };
#define MSET_LEN (sizeof(MSET)/sizeof(MSET[0]))
#define NSET_LEN (sizeof(NSET)/sizeof(NSET[0]))
#define KSET_LEN (sizeof(KSET)/sizeof(KSET[0]))

// Max buffer sizes (must cover the largest in the sets)
#ifndef BUF_M_MAX
#define BUF_M_MAX 128
#endif
#ifndef BUF_N_MAX
#define BUF_N_MAX 256
#endif
#ifndef BUF_K_MAX
#define BUF_K_MAX 512
#endif

// Dataflows to run
#ifndef RUN_WS
#define RUN_WS 1
#endif
#ifndef RUN_OS
#define RUN_OS 1
#endif

// Warmup count (excluded from timing)
#ifndef WARMUP_RUNS
#define WARMUP_RUNS 1
#endif

// Cold-start for measurement (evict right before warmups+measure)
// Default OFF → steady-state friendly
#ifndef DO_CACHE_EVICT
#define DO_CACHE_EVICT 0
#endif

// Between-case reset: evict caches BEFORE initializing A/B/C/D
// This emulates "new process start → then init warms caches".
// Keeps multi-case results in-line with running the single binary per case.
#ifndef RESET_EVICT_BETWEEN_CASES
#define RESET_EVICT_BETWEEN_CASES 0
#endif

// Eviction sweep size (make it >= your L2)
#ifndef EVICT_BYTES
#define EVICT_BYTES (2*1024*1024)
#endif

// Optional repeat measurements (median); keep 1 for speed
#ifndef MEAS_REPEATS
#define MEAS_REPEATS 1
#endif

// ===================== Layout/Align =====================
#define STRIDE_A(k) (k)
#define STRIDE_B(n) (n)
#define STRIDE_D(n) (n)
#define STRIDE_C(n) (n)
#ifndef ALGN
#define ALGN __attribute__((aligned(64)))
#endif

// ===================== Buffers =====================
static elem_t A_buf[BUF_M_MAX * BUF_K_MAX] ALGN;
static elem_t B_buf[BUF_K_MAX * BUF_N_MAX] ALGN;
static elem_t C_buf[BUF_M_MAX * BUF_N_MAX] ALGN;
static acc_t  D_buf[BUF_M_MAX * BUF_N_MAX] ALGN;

#if (DO_CACHE_EVICT || RESET_EVICT_BETWEEN_CASES)
static volatile uint8_t cache_sweep[EVICT_BYTES] ALGN;
#endif

// ===================== Utils =====================
static inline uint64_t rdcycle(){ uint64_t c; asm volatile ("rdcycle %0" : "=r"(c)); return c; }
static inline void full_fence(){ asm volatile("fence" ::: "memory"); }

static void fill_i8(elem_t *a, int len, int seed){
  uint32_t s = (uint32_t)seed * 2654435761u;
  for (int i=0;i<len;i++){ s ^= s<<13; s ^= s>>17; s ^= s<<5; a[i] = (elem_t)((int)(s & 0xFF) - 128); }
}
static void zero_i8(elem_t *a, int len){ memset(a, 0, len*sizeof(elem_t)); }
static void zero_i32(acc_t *a, int len){ memset(a, 0, len*sizeof(acc_t)); }

static void evict_caches(void){
#if (DO_CACHE_EVICT || RESET_EVICT_BETWEEN_CASES)
  for (size_t i=0;i<EVICT_BYTES;i+=64) cache_sweep[i]=(uint8_t)(i & 0xFF);
  for (size_t i=0;i<EVICT_BYTES;i+=64) (void)cache_sweep[i];
  full_fence();
#endif
}

#ifndef NO_ACTIVATION
#define NO_ACTIVATION 0
#endif

static uint64_t gemmini_mm_kernel(int dataflow, int M, int N, int K,
                                  elem_t *A, elem_t *B, acc_t *Dbias, elem_t *C)
{
  const size_t sA = STRIDE_A(K), sB = STRIDE_B(N), sD = STRIDE_D(N), sC = STRIDE_C(N);
  const scale_t      A_scale   = (scale_t)1;
  const scale_t      B_scale   = (scale_t)1;
  const scale_acc_t  D_scale   = (scale_acc_t)1;
  const int          act       = NO_ACTIVATION;
  const acc_scale_t  C_scale   = (acc_scale_t)1;
  const acc_scale_t  out_scale = (acc_scale_t)1;
  const bool repeating_bias=false, A_T=false, B_T=false, full_C=false, low_D=false;
  const uint8_t extra=0;

  gemmini_config_ex(dataflow, NO_ACTIVATION, 0);
  gemmini_fence();

  uint64_t t0 = rdcycle();
  tiled_matmul_auto(M, N, K, A, B, (const void*)Dbias, (void*)C,
                    sA, sB, sD, sC,
                    A_scale, B_scale, D_scale,
                    act, C_scale, out_scale,
                    repeating_bias, A_T, B_T, full_C, low_D, extra,
                    (enum tiled_matmul_type_t)dataflow);
  gemmini_fence();
  return rdcycle() - t0;
}

static inline uint64_t u64_min(uint64_t a, uint64_t b){ return a<b?a:b; }
static uint64_t median3(uint64_t a, uint64_t b, uint64_t c){
  uint64_t x=a, y=b, z=c;
  if (x>y){uint64_t t=x;x=y;y=t;} // x<=y
  if (y>z){uint64_t t=y;y=z;z=t;} // y<=z
  if (x>y){uint64_t t=x;x=y;y=t;} // x<=y<=z
  return y;
}

static void run_one_isolated(int dataflow, int M, int N, int K){
  // 0) Between-case reset (optional): emulate "fresh process"
#if RESET_EVICT_BETWEEN_CASES
  evict_caches();
#endif

  // 1) Init only the needed regions (this also warms CPU caches on A/B/C/D)
  fill_i8(A_buf, M*K, 1);
  fill_i8(B_buf, K*N, 2);
  zero_i32(D_buf, M*N);
  zero_i8(C_buf, M*N);

  // 2) Clear Gemmini-internal state (not CPU/L2)
  full_fence(); gemmini_fence(); gemmini_flush(0); full_fence();

  // 3) (Optional) Cold-start for measurement: evict right before warmups/measure
#if DO_CACHE_EVICT
  evict_caches();
#endif

  // 4) Warmups (excluded)
  for (int w=0; w<WARMUP_RUNS; w++) (void)gemmini_mm_kernel(dataflow, M,N,K, A_buf,B_buf,D_buf,C_buf);

  // 5) Measure (repeat optional → median)
#if MEAS_REPEATS==1
  uint64_t cyc = gemmini_mm_kernel(dataflow, M,N,K, A_buf,B_buf,D_buf,C_buf);
#else
  uint64_t c1 = gemmini_mm_kernel(dataflow, M,N,K, A_buf,B_buf,D_buf,C_buf);
  uint64_t c2 = gemmini_mm_kernel(dataflow, M,N,K, A_buf,B_buf,D_buf,C_buf);
  uint64_t c3 = gemmini_mm_kernel(dataflow, M,N,K, A_buf,B_buf,D_buf,C_buf);
  uint64_t cyc = median3(c1,c2,c3);
#endif

  // 6) Metrics
  const uint64_t macs = (uint64_t)M*(uint64_t)N*(uint64_t)K;
  const uint64_t mac_per_100cyc = cyc ? (macs*100ULL)/cyc : 0ULL;
  long sumC=0; for (int i=0;i<M*N;i++) sumC += C_buf[i];
  const char *df = (dataflow==WS) ? "WS" : "OS";

  printf("DF=%s M=%d N=%d K=%d | warmups=%d | cycles=%llu | MAC/100cyc=%llu | sum(C)=%ld\n",
         df, M,N,K, WARMUP_RUNS,
         (unsigned long long)cyc,
         (unsigned long long)mac_per_100cyc,
         sumC);
}

int main(void){
  printf("Gemmini WS/OS Multi-case (kernel-only timing)\n");
  printf("Settings: WARMUP_RUNS=%d  DO_CACHE_EVICT=%d  RESET_EVICT_BETWEEN_CASES=%d  MEAS_REPEATS=%d\n",
         WARMUP_RUNS, DO_CACHE_EVICT, RESET_EVICT_BETWEEN_CASES, MEAS_REPEATS);

  // Safety guard: skip cases exceeding buffer maxima
  for (int im=0; im<MSET_LEN; im++){
    for (int in=0; in<NSET_LEN; in++){
      for (int ik=0; ik<KSET_LEN; ik++){
        const int M=MSET[im], N=NSET[in], K=KSET[ik];
        if (M>BUF_M_MAX || N>BUF_N_MAX || K>BUF_K_MAX){
          printf("SKIP (exceeds buffer): M=%d N=%d K=%d (max %d/%d/%d)\n",
                 M,N,K, BUF_M_MAX,BUF_N_MAX,BUF_K_MAX);
          continue;
        }
        if (RUN_WS) run_one_isolated(WS, M,N,K);
        if (RUN_OS) run_one_isolated(OS, M,N,K);
      }
    }
  }
  printf("Done.\n");
  return 0;
}

