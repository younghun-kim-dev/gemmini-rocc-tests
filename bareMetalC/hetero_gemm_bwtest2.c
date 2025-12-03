// hetero_gemm_bwtest.c — Chipyard hetero test (BOOM=Gemmini GEMM, Rocket=Mem BW stress)
// - 모든 hart가 crt_threaded.S 를 통해 thread_entry(cid, nc)로 진입
// - hart2(BOOM): Gemmini GEMM 실행
// - hart0/1(Rocket): 메모리 대역폭 스트레스(안전한 선형 스윕, 낮은 강도 기본)
// - 베어메탈 printf는 %zu 를 지원하지 않으므로 %lu/%llu 로 캐스팅 출력

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"  // barrier(), rdcycle(), 등
#include "include/gemmini.h"
#include "include/gemmini_testutils.h"

// ====== 하트→역할 매핑(필요시 -DROLE_HART_* 로 덮어쓰기) ======
#ifndef ROLE_HART_R0
#define ROLE_HART_R0 0   // Rocket #1
#endif
#ifndef ROLE_HART_R1
#define ROLE_HART_R1 1   // Rocket #2
#endif
#ifndef ROLE_HART_BOOM
#define ROLE_HART_BOOM 2 // BOOM (Gemmini 붙은 코어)
#endif

// ====== 안전한 기본 파라미터(필요시 빌드시 -D로 조절) ======
#ifndef GEMM_M
#define GEMM_M 256
#endif
#ifndef GEMM_N
#define GEMM_N 256
#endif
#ifndef GEMM_K
#define GEMM_K 256
#endif

#ifndef STRESS_MB
#define STRESS_MB 8       // Rocket 메모리 스트레스 총 용량 (MiB)
#endif
#ifndef STRESS_PASSES
#define STRESS_PASSES 1   // 각 Rocket이 반복할 횟수
#endif
#ifndef STRESS_STRIDE
#define STRESS_STRIDE 64  // 64B cache-line stride
#endif

// ====== (추가) 프로파일 토글/헬퍼 ======
#ifndef PROF_KERNEL
#define PROF_KERNEL 1      // 커널 정규화 로그(cycles/MAC + tag)
#endif
#ifndef PROF_TILE
#define PROF_TILE  1       // 블록 타일 커널 로깅(외부 블록화). 켜려면 1
#endif
#ifndef CFG_TAG
#define CFG_TAG "BASE"     // 실험 라벨: -DCFG_TAG=\"OS_L2b4_2MB\" 등
#endif
#ifndef GEMM_DATAFLOW
#define GEMM_DATAFLOW OS   // OS/WS 토글: -DGEMM_DATAFLOW=WS
#endif
#ifndef TILE_I
#define TILE_I 64          // 외부 블록 크기(행)
#endif
#ifndef TILE_J
#define TILE_J 64          // 외부 블록 크기(열)
#endif
#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

static inline unsigned long long macs_total(void) {
  return (unsigned long long)GEMM_M * GEMM_N * GEMM_K;
}

// ====== 멀티코어 printf 잠금(옵션) ======
#ifdef PRINT_LOCK
static volatile int printf_lock = 0;
static inline void lock_print(void){ while(__sync_lock_test_and_set(&printf_lock,1)); }
static inline void unlock_print(void){ __sync_lock_release(&printf_lock); }
#define PLOCK()   lock_print()
#define PUNLOCK() unlock_print()
#else
#define PLOCK()
#define PUNLOCK()
#endif

// ====== Rocket 메모리 스트레스 버퍼(전역 정적: 베어메탈에서 malloc 회피) ======
#define STRESS_BYTES ( (size_t)STRESS_MB * 1024UL * 1024UL )
static volatile uint8_t stress_buf[STRESS_BYTES];

// ====== Rocket: 안전한 선형 메모리 스트레스(순방향 read-only 스윕) ======
static void memory_bw_stress_linear(const char *tag, int passes) {
  const size_t BYTES = STRESS_BYTES;
  volatile const uint8_t *buf = stress_buf;
  // 간단 초기화(쓰기 최소화)
  for (size_t i = 0; i < BYTES; i += 64) {
    ((volatile uint8_t*)buf)[i] = (uint8_t)(i & 0xFF);
  }

  uint64_t t0 = rdcycle();
  size_t sink = 0;

  for (int p = 0; p < passes; ++p) {
    // 순방향 선형 read 스윕 (가장 안전)
    for (size_t off = 0; off < BYTES; off += STRESS_STRIDE) {
      sink += buf[off];
    }
  }

  asm volatile("fence iorw, iorw");
  uint64_t t1 = rdcycle();

  PLOCK();
  printf("[Rocket %s linear] stress=%u MiB, stride=%lu, passes=%d -> cycles=%llu, sink=%lu\n",
         tag, (unsigned)STRESS_MB, (unsigned long)STRESS_STRIDE, passes,
         (unsigned long long)(t1 - t0), (unsigned long)sink);
  PUNLOCK();
}

// ====== BOOM: Gemmini GEMM 실행 + 시간 측정 (커널 한방) ======
static void run_gemmini_matmul_and_time(const char *tag) {
  // 정적 버퍼(스택 폭주 방지)
  static elem_t A[GEMM_M * GEMM_K];
  static elem_t B[GEMM_K * GEMM_N];
  static elem_t C[GEMM_M * GEMM_N];

  // 입력 초기화(작은 값)
  for (size_t i = 0; i < GEMM_M * GEMM_K; ++i) A[i] = (elem_t)(i % 7);
  for (size_t i = 0; i < GEMM_K * GEMM_N; ++i) B[i] = (elem_t)(i % 5);
  for (size_t i = 0; i < GEMM_M * GEMM_N; ++i) C[i] = 0;

  // Gemmini fence로 파이프 정리
  gemmini_flush(0);

  uint64_t t0 = rdcycle();

  // gemmini.h (Chipyard Feb’24 계열) 시그니처에 맞춰 모든 인자 전달
  tiled_matmul_auto(
    GEMM_M, GEMM_N, GEMM_K,
    A, B, /*D=*/NULL, C,
    /*stride_A=*/GEMM_K, /*stride_B=*/GEMM_N, /*stride_D=*/GEMM_N, /*stride_C=*/GEMM_N,
    /*A_scale=*/1, /*B_scale=*/1, /*D_scale=*/0,
    /*act=*/NO_ACTIVATION, /*scale=*/0, /*bert_scale=*/0,
    /*repeating_bias=*/false,
    /*transpose_A=*/false, /*transpose_B=*/false,
    /*full_C=*/true, /*low_D=*/false,
    /*weightA=*/0,
    /*type=*/GEMM_DATAFLOW   // OS/WS 선택
  );

  gemmini_fence();

  uint64_t t1 = rdcycle();

#if PROF_KERNEL
  unsigned long long cyc = (unsigned long long)(t1 - t0);
  unsigned long long mac = macs_total();
  unsigned long long cpm = cyc / (mac ? mac : 1);  // cycles per MAC
  PLOCK();
  printf("[BOOM %s] GEMM %lux%lu * %lux%lu -> cycles=%llu, c/MAC=%llu, tag=%s\n",
         tag,
         (unsigned long)GEMM_M, (unsigned long)GEMM_K,
         (unsigned long)GEMM_K, (unsigned long)GEMM_N,
         cyc, cpm, CFG_TAG);
  PUNLOCK();
#else
  PLOCK();
  printf("[BOOM %s] GEMM %lux%lu * %lux%lu -> cycles=%llu\n",
         tag,
         (unsigned long)GEMM_M, (unsigned long)GEMM_K,
         (unsigned long)GEMM_K, (unsigned long)GEMM_N,
         (unsigned long long)(t1 - t0));
  PUNLOCK();
#endif
}

// ====== (추가) BOOM: 블록 타일 커널 로깅(외부 블록화) ======
#if PROF_TILE
static void run_gemmini_matmul_tilelogged(const char *tag) {
  static elem_t A[GEMM_M * GEMM_K];
  static elem_t B[GEMM_K * GEMM_N];
  static elem_t C[GEMM_M * GEMM_N];

  // 입력 초기화
  for (size_t i = 0; i < GEMM_M * GEMM_K; ++i) A[i] = (elem_t)(i % 7);
  for (size_t i = 0; i < GEMM_K * GEMM_N; ++i) B[i] = (elem_t)(i % 5);
  for (size_t i = 0; i < GEMM_M * GEMM_N; ++i) C[i] = 0;

  gemmini_flush(0);

  // (선택) 전체 블록 실행 총합을 별도로 보고 싶으면 아래 주석 해제
  // uint64_t kernel0 = rdcycle();

  for (size_t i0 = 0; i0 < GEMM_M; i0 += TILE_I) {
    for (size_t j0 = 0; j0 < GEMM_N; j0 += TILE_J) {
      size_t im = MIN(TILE_I, GEMM_M - i0);
      size_t jn = MIN(TILE_J, GEMM_N - j0);

      const elem_t* Ablk = A + i0 * GEMM_K;        // im x K
      const elem_t* Bblk = B + j0;                 // K x jn (row-major, stride_B=GEMM_N)
      elem_t*       Cblk = C + i0 * GEMM_N + j0;   // im x jn

      uint64_t t0 = rdcycle();

      tiled_matmul_auto(
        /*I*/ im, /*J*/ jn, /*K*/ GEMM_K,
        Ablk, Bblk, /*D=*/NULL, Cblk,
        /*sA=*/GEMM_K, /*sB=*/GEMM_N, /*sD=*/GEMM_N, /*sC=*/GEMM_N,
        /*A_scale=*/1, /*B_scale=*/1, /*D_scale=*/0,
        /*act=*/NO_ACTIVATION, /*scale=*/0, /*bert_scale=*/0,
        /*repeating_bias=*/false,
        /*tA=*/false, /*tB=*/false,
        /*full_C=*/true, /*low_D=*/false,
        /*weightA=*/0,
        /*type=*/GEMM_DATAFLOW
      );

      gemmini_fence();
      uint64_t t1 = rdcycle();

      PLOCK();
      printf("[tile i=%lu j=%lu] im=%lu jn=%lu -> cycles=%llu\n",
             (unsigned long)i0, (unsigned long)j0,
             (unsigned long)im, (unsigned long)jn,
             (unsigned long long)(t1 - t0));
      PUNLOCK();
    }
  }

  // (선택) 블록 전체 총합/정규화 보고
  // uint64_t kernel1 = rdcycle();
  // unsigned long long total = (unsigned long long)(kernel1 - kernel0);
  // unsigned long long cpm   = total / (macs_total() ? macs_total() : 1);
  // PLOCK();
  // printf("[BOOM %s] TILE-KERNEL total=%llu cycles, c/MAC=%llu, tag=%s\n",
  //        tag, total, cpm, CFG_TAG);
  // PUNLOCK();
}
#endif  // PROF_TILE

// ====== 역할 함수들 ======
static void role_rocket0(void) {
  PLOCK(); printf("[Hart %d - Rocket#1] mem-stress 시작\n", ROLE_HART_R0); PUNLOCK();
  memory_bw_stress_linear("R0", STRESS_PASSES);
  PLOCK(); printf("[Hart %d - Rocket#1] 완료\n", ROLE_HART_R0); PUNLOCK();
}

static void role_rocket1(void) {
  PLOCK(); printf("[Hart %d - Rocket#2] mem-stress 시작\n", ROLE_HART_R1); PUNLOCK();
  memory_bw_stress_linear("R1", STRESS_PASSES);
  PLOCK(); printf("[Hart %d - Rocket#2] 완료\n", ROLE_HART_R1); PUNLOCK();
}

static void role_boom(void) {
  PLOCK(); printf("[Hart %d - BOOM] Gemmini GEMM 시작\n", ROLE_HART_BOOM); PUNLOCK();
#if PROF_TILE
  run_gemmini_matmul_tilelogged("GEMM");
#else
  run_gemmini_matmul_and_time("GEMM");
#endif
  PLOCK(); printf("[Hart %d - BOOM] 완료\n", ROLE_HART_BOOM); PUNLOCK();
}

// ====== 진입점 ======
void thread_entry(int cid, int nc) {
  if (cid == 0) {
    printf("--- Hetero GEMM + MemBW test 시작 (총 %d 코어) ---\n", nc);
  }

  barrier(nc);

  if (cid == ROLE_HART_BOOM) {
    role_boom();
  } else if (cid == ROLE_HART_R0) {
    role_rocket0();
  } else if (cid == ROLE_HART_R1) {
    role_rocket1();
  } else {
    PLOCK(); printf("[Hart %d] 역할 없음\n", cid); PUNLOCK();
  }

  PLOCK();
  printf(">> Hart %d finished at cycle %llu\n", cid, (unsigned long long)rdcycle());
  PUNLOCK();

  barrier(nc);

  if (cid == 0) {
    asm volatile("fence iorw, iorw");
    printf("--- 모든 코어 임무 완료, 테스트 종료 ---\n");
    exit(0);  // tohost 정상 종료
  } else {
    while (1) asm volatile("wfi");
  }
}

// pk/linux 링크 시 main 요구 대응(베어메탈에는 영향 없음)
__attribute__((weak)) int main(void) { return 0; }
