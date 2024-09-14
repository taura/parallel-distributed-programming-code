//% file: mm_2.c
//% cmd: g++ -Wall -Wextra -O3 -march=native mm_2.c -I. -o mm_2 

#include "mm_main.h"

/* vectorize this code. hint: along j axis */
template<idx_t M,idx_t N,idx_t K,
  idx_t lda,idx_t ldb,idx_t ldc,
  idx_t bM,idx_t bN>
long gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  for (idx_t i = 0; i < M; i++) {
    for (idx_t j = 0; j < N; j++) {
      asm volatile("# loop begins (%0,%1)x(%1,%2)" :: "i" (1), "i" (K), "i" (1));
      for (idx_t k = 0; k < K; k++) {
	C(i,j) += A(i,k) * B(k,j);
      }
      asm volatile("# loop ends");
    }
  }
  return (long)M * (long)N * (long)K;
}

