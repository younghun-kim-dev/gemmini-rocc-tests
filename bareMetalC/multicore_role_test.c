// multicore_role_test.c — Chipyard + riscv-tests 베어메탈 멀티코어 예제
// - 모든 hart가 thread_entry(cid, nc)로 진입 (crt_threaded.S 필요)
// - hart0만 exit(0)로 정상 종료, 나머지는 wfi 대기
// - rdcycle()는 encoding.h 매크로를 직접 사용 (사용자 함수 만들지 말 것)

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"   // riscv-tests/benchmarks/common/util.h (barrier 등)

// ------- printf 직렬화 옵션 (빌드시 -DPRINT_LOCK 주면 활성화) -------
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
// -------------------------------------------------------------------

// ====== 하트→역할 매핑 (필요시 빌드 시 -DROLE_HART_BOOM=...로 덮어써라) ======
#ifndef ROLE_HART_R0
#define ROLE_HART_R0 0   // Rocket #1 (환경에 맞게 조정)
#endif
#ifndef ROLE_HART_R1
#define ROLE_HART_R1 1   // Rocket #2
#endif
#ifndef ROLE_HART_BOOM
#define ROLE_HART_BOOM 2 // BOOM
#endif

static void role_rocket0(void) {
  PLOCK(); printf("[Hart %d - Rocket#1] 카운팅 시작\n", ROLE_HART_R0); PUNLOCK();
  for (int i = 0; i < 3; ++i) {
    PLOCK(); printf("[Hart %d] %d\n", ROLE_HART_R0, i); PUNLOCK();
  }
  PLOCK(); printf("[Hart %d - Rocket#1] 완료\n", ROLE_HART_R0); PUNLOCK();
}

static void role_rocket1(void) {
  PLOCK(); printf("[Hart %d - Rocket#2] 100부터 카운팅\n", ROLE_HART_R1); PUNLOCK();
  for (int i = 100; i < 103; ++i) {
    PLOCK(); printf("[Hart %d] %d\n", ROLE_HART_R1, i); PUNLOCK();
  }
  PLOCK(); printf("[Hart %d - Rocket#2] 완료\n", ROLE_HART_R1); PUNLOCK();
}

static void role_boom(void) {
  int a = 123, b = 456, c = a + b;
  PLOCK(); printf("[Hart %d - BOOM] %d + %d = %d\n", ROLE_HART_BOOM, a, b, c); PUNLOCK();
  PLOCK(); printf("[Hart %d - BOOM] 완료\n", ROLE_HART_BOOM); PUNLOCK();
}

// ─────────────────────────────────────────────────────────────────────────────
// 멀티코어 진입: riscv-tests 런타임/커스텀 CRT가 모든 hart를 여기로 부름
void thread_entry(int cid, int nc) {
  if (cid == 0) {
    PLOCK(); printf("--- 멀티코어 역할 분담 테스트 시작 (총 %d 코어) ---\n", nc); PUNLOCK();
  }

  barrier(nc);  // 시작 동기화

  if (cid == ROLE_HART_R0) {
    role_rocket0();
  } else if (cid == ROLE_HART_R1) {
    role_rocket1();
  } else if (cid == ROLE_HART_BOOM) {
    role_boom();
  } else {
    PLOCK(); printf("[Hart %d] 역할 없음\n", cid); PUNLOCK();
  }

  PLOCK();
  printf(">> Hart %d finished at cycle %llu\n",
         cid, (unsigned long long) rdcycle());  // ← 매크로 호출
  PUNLOCK();

  barrier(nc);  // 종료 전 동기화

  if (cid == 0) {
    asm volatile("fence iorw, iorw");
    PLOCK(); printf("--- 모든 코어 임무 완료, 테스트 종료 ---\n"); PUNLOCK();
    exit(0);  // tohost 경유 정상 종료
  } else {
    while (1) asm volatile("wfi");
  }
}

// linux/pk 변형 링크러가 main을 요구할 때를 위한 안전장치 (베어메탈엔 영향 없음)
__attribute__((weak)) int main(void) { return 0; }

