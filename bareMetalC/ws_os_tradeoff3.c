// ws_os_single.c  — One-shot Gemmini MM (WS or OS) with cache/state hygiene
// Baremetal-safe (no malloc/float)

// ===== Includes =====
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_params.h"

// ===== User-tunable compile-time knobs =====
// 선택: -DONE_M=128 -DONE_N=256 -DONE_K=512  (미지정 시 아래 기본값)
#ifndef ONE_M
#define ONE_M 128
#endif
#ifndef ONE_N
#define ONE_N 128
#endif
#ifndef ONE_K
#define ONE_K 128
#endif

// 데이터플로우 선택: -DDATAFLOW=WS 또는 -DDATAFLOW=OS (미지정 시 WS)
#ifndef DATAFLOW
#define DATAFLOW WS
#endif

// 캐시 축출 옵션: 필요없으면 -DDO_CACHE_EVICT=0
#ifndef DO_CACHE_EVICT
#define DO_CACHE_EVICT 0
#endif

// 축출 버퍼 크기(바이트). L2보다 넉넉히(예: 2MB).
#ifndef EVICT_BYTES
#define EVICT_BYTES (2*1024*1024)
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
// ONE_* 최대치에 맞춰 정적 할당
static elem_t A_buf[ONE_M * ONE_K] ALGN;
static elem_t B_buf[ONE_K * ONE_N] ALGN;
static elem_t C_out[ONE_M * ONE_N] ALGN;
static acc_t  D_bias[ONE_M * ONE_N] ALGN;

// 캐시 축출용 (volatile로 최적화 억제)
#if DO_CACHE_EVICT
static volatile uint8_t cache_sweep[EVICT_BYTES] ALGN;
#endif

// ===== Utils =====
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

static inline void full_memory_fence() {
  asm volatile("fence" ::: "memory");
}

// L2/LLC 축출: 큰 버퍼를 stride로 왕복 접근해서 캐시를 더럽혀 버림
static void evict_caches(void){
#if DO_CACHE_EVICT
  // 전방/후방 모두 한 번씩 접근 (쓰기도 섞음)
  for (size_t i = 0; i < EVICT_BYTES; i += 64) {
    cache_sweep[i] = (uint8_t)(i & 0xFF);
  }
  for (size_t i = 0; i < EVICT_BYTES; i += 64) {
    (void)cache_sweep[i];
  }
  full_memory_fence();
#endif
}

// ===== One-shot run =====
static uint64_t run_oneshot(int dataflow, int M, int N, int K,
                            elem_t *A, elem_t *B, acc_t *Dbias, elem_t *C)
{
  // 가속기/메모리 상태를 최대한 깨끗하게
  full_memory_fence();
  gemmini_fence();
  gemmini_flush(0);        // Gemmini 내부 상태(fence류) 비움
  full_memory_fence();

  // 캐시 축출로 선행 실행 영향 최소화
  evict_caches();

  const size_t stride_A = STRIDE_A(K);
  const size_t stride_B = STRIDE_B(N);
  const size_t stride_D = STRIDE_D(N);
  const size_t stride_C = STRIDE_C(N);

  // 정수 경로: Gemmini 타입에 맞춰 스케일/활성화 설정
#ifndef NO_ACTIVATION
#define NO_ACTIVATION 0
#endif
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

  // 런타임 데이터플로우 설정 + 동기화
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
  full_memory_fence();

  return t1 - t0;
}

int main(void){
  const int M = ONE_M, N = ONE_N, K = ONE_K;

  // 입력/출력 초기화 (항상 같은 값 → 결과 재현성)
  fill_i8(A_buf, M*K, 1);
  fill_i8(B_buf, K*N, 2);
  zero_i32(D_bias, M*N);
  zero_i8(C_out,  M*N);

  // 한 번만 측정 (웜업 없음)
  uint64_t cyc = run_oneshot(DATAFLOW, M, N, K, A_buf, B_buf, D_bias, C_out);

  // 정수형 지표
  const uint64_t macs = (uint64_t)M * (uint64_t)N * (uint64_t)K;
  const uint64_t mac_per_100cyc = (cyc ? (macs*100ULL)/cyc : 0ULL);

  long sumC = 0;
  for (int i=0;i<M*N;i++) sumC += C_out[i];

  // 어떤 데이터플로우였는지 문자열
  const char *df_str = (DATAFLOW == WS) ? "WS" : "OS";

  printf("Gemmini One-shot MM  | DF=%s  M=%d N=%d K=%d\n", df_str, M, N, K);
  printf("cycles=%llu  | MAC/100cyc=%llu  | sum(C)=%ld\n",
         (unsigned long long)cyc,
         (unsigned long long)mac_per_100cyc,
         sumC);

  return 0;
}

