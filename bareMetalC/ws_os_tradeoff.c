// ws_os_tradeoff.c (baremetal-safe, no malloc/float)
// Compare WS vs OS dataflows on Gemmini with static buffers.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_params.h"

// ------------ config for sweep ------------
#define M1 16
#define N1 16
#define K1 4

#define M2 16
#define N2 16
#define K2 4

#define M3 4
#define N3 4
#define K3 2

// pick max dims to size static buffers
#define M_MAX 128
#define N_MAX 256
#define K_MAX 512

// row-major strides
#define STRIDE_A(k) (k)
#define STRIDE_B(n) (n)
#define STRIDE_D(n) (n)
#define STRIDE_C(n) (n)

// 64B align helps DMA
#ifndef ALGN
#define ALGN __attribute__((aligned(64)))
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

#ifndef OS
#define OS 0
#endif
#ifndef WS
#define WS 1
#endif
#ifndef NO_ACTIVATION
#define NO_ACTIVATION 0
#endif

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

  const scale_t      A_scale   = (scale_t)1.0f;
  const scale_t      B_scale   = (scale_t)1.0f;
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

static void run_case(int M,int N,int K){
  elem_t *A   = A_buf;
  elem_t *B   = B_buf;
  elem_t *Cws = Cws_buf;
  elem_t *Cos = Cos_buf;
  acc_t  *D   = D_buf;

  // init only the needed regions
  fill_i8(A, M*K, 1);
  fill_i8(B, K*N, 2);
  zero_i32(D, M*N);
  zero_i8(Cws, M*N);
  zero_i8(Cos, M*N);

  // warmup
  (void)run_one(WS, M,N,K, A,B,D, Cws);
  (void)run_one(OS, M,N,K, A,B,D, Cos);

  // measure
  uint64_t cyc_ws = run_one(WS, M,N,K, A,B,D, Cws);
  uint64_t cyc_os = run_one(OS, M,N,K, A,B,D, Cos);

  // integer-only metrics (no float)
  uint64_t macs = (uint64_t)M * (uint64_t)N * (uint64_t)K;
  uint64_t mac_per_100cyc_ws = (cyc_ws ? (macs*100ULL)/cyc_ws : 0ULL);
  uint64_t mac_per_100cyc_os = (cyc_os ? (macs*100ULL)/cyc_os : 0ULL);

  long sum_ws=0, sum_os=0;
  for (int i=0;i<M*N;i++){ sum_ws += Cws[i]; sum_os += Cos[i]; }

  printf("[M=%d N=%d K=%d] WS: %llu cyc  (%llu MAC/100cyc) | OS: %llu cyc  (%llu MAC/100cyc) | sums: %ld/%ld\n",
         M,N,K,
         (unsigned long long)cyc_ws, (unsigned long long)mac_per_100cyc_ws,
         (unsigned long long)cyc_os, (unsigned long long)mac_per_100cyc_os,
         sum_ws, sum_os);
}

int main(){
  printf("WS vs OS tradeoff (static buffers, row-major)\n");

  run_case(M1,N1,K1);
  run_case(M2,N2,K2);
  run_case(M3,N3,K3);

  printf("Done.\n");
  return 0;
}
