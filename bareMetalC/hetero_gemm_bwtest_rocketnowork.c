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

// >>>>>>> 추가: Rocket을 작업에서 배제 (다른 부분은 변경 없음)
#define SLEEP_ROCKETS 1
// <<<<<<< 추가 끝

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

// ====== BOOM: Gemmini GEMM 실행 + 시간 측정 ======
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
  // static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
  //   const elem_t* A, const elem_t* B, const void *D, void *C,
  //   size_t sA, size_t sB, size_t sD, size_t sC,
  //   scale_t As, scale_t Bs, scale_acc_t Ds,
  //   int act, acc_scale_t scale, acc_scale_t bert_scale,
  //   bool repeating_bias, bool tA, bool tB, bool full_C, bool low_D,
  //   uint8_t weightA, enum tiled_matmul_type_t t);
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
    /*type=*/OS
  );

  gemmini_fence();

  uint64_t t1 = rdcycle();

  PLOCK();
  printf("[BOOM %s] GEMM %lux%lu * %lux%lu -> cycles=%llu\n",
         tag,
         (unsigned long)GEMM_M, (unsigned long)GEMM_K,
         (unsigned long)GEMM_K, (unsigned long)GEMM_N,
         (unsigned long long)(t1 - t0));
  PUNLOCK();
}

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
  run_gemmini_matmul_and_time("GEMM");
  PLOCK(); printf("[Hart %d - BOOM] 완료\n", ROLE_HART_BOOM); PUNLOCK();
}

// ====== 진입점 ======
void thread_entry(int cid, int nc) {
  if (cid == 0) {
    printf("--- Hetero GEMM + MemBW test 시작 (총 %d 코어) ---\n", nc);
  }

  barrier(nc);

  // >>>>>>> 추가: Rocket 하트는 일을 전혀 수행하지 않고 에필로그로 이동
#if SLEEP_ROCKETS
  if (cid == ROLE_HART_R0 || cid == ROLE_HART_R1) {
    goto epilogue;
  }
#endif
  // <<<<<<< 추가 끝

  if (cid == ROLE_HART_BOOM) {
    role_boom();
  } else if (cid == ROLE_HART_R0) {
    role_rocket0();
  } else if (cid == ROLE_HART_R1) {
    role_rocket1();
  } else {
    PLOCK(); printf("[Hart %d] 역할 없음\n", cid); PUNLOCK();
  }

epilogue:
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
