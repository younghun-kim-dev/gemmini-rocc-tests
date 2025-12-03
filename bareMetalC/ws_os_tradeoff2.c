// ====== 추가: 단일 실행용 매크로 ======
#ifndef M_SEL
#define M_SEL 128
#endif
#ifndef N_SEL
#define N_SEL 256
#endif
#ifndef K_SEL
#define K_SEL 512
#endif

// ORDER=0 → WS 먼저, ORDER=1 → OS 먼저
#ifndef ORDER
#define ORDER 0
#endif

#ifndef WARMUP
#define WARMUP 1
#endif

static void run_single_in_order(int first_flow, int second_flow, int M, int N, int K) {
  elem_t *A   = A_buf;
  elem_t *B   = B_buf;
  elem_t *C1  = Cws_buf;   // 첫 번째 결과 버퍼
  elem_t *C2  = Cos_buf;   // 두 번째 결과 버퍼
  acc_t  *D   = D_buf;

  // 입력/출력 초기화 (필요 영역만)
  fill_i8(A, M*K, 1);
  fill_i8(B, K*N, 2);
  zero_i32(D, M*N);
  zero_i8(C1, M*N);
  zero_i8(C2, M*N);

  // 선택적 웜업
  if (WARMUP) {
    (void)run_one(first_flow,  M,N,K, A,B,D, C1);
    (void)run_one(second_flow, M,N,K, A,B,D, C2);
  }

  // 측정
  uint64_t cyc_first  = run_one(first_flow,  M,N,K, A,B,D, C1);
  uint64_t cyc_second = run_one(second_flow, M,N,K, A,B,D, C2);

  uint64_t macs = (uint64_t)M * (uint64_t)N * (uint64_t)K;
  uint64_t first_eff  = (cyc_first  ? (macs*100ULL)/cyc_first  : 0ULL);
  uint64_t second_eff = (cyc_second ? (macs*100ULL)/cyc_second : 0ULL);

  long sum1=0, sum2=0;
  for (int i=0;i<M*N;i++){ sum1 += C1[i]; sum2 += C2[i]; }

  const char* fstr = (first_flow==WS ? "WS" : "OS");
  const char* sstr = (second_flow==WS ? "WS" : "OS");

  printf("[M=%d N=%d K=%d] %s first: %llu cyc (%llu MAC/100cyc) | %s second: %llu cyc (%llu MAC/100cyc) | sums: %ld/%ld\n",
         M,N,K,
         fstr, (unsigned long long)cyc_first,  (unsigned long long)first_eff,
         sstr, (unsigned long long)cyc_second, (unsigned long long)second_eff,
         sum1, sum2);
}

int main(){
  printf("WS vs OS (single case, order-controlled)\n");

  if (ORDER == 0) {
    // WS → OS
    run_single_in_order(WS, OS, M_SEL, N_SEL, K_SEL);
  } else {
    // OS → WS
    run_single_in_order(OS, WS, M_SEL, N_SEL, K_SEL);
  }

  printf("Done.\n");
  return 0;
}

