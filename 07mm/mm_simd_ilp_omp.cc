//% file: mm_4.c
//% cmd: g++ -Wall -Wextra -O3 -march=native mm_4.c -I. -o mm_4 

#include "mm_main.h"

/* 
 * vectorize; and
 * concurrently update several rows and several columns of C to 
 * increase ILP
 * obtain >80% of the peak performance (50 flops/clock with AVX512F)
 */
template<idx_t M,idx_t N,idx_t K,
         idx_t lda,idx_t ldb,idx_t ldc,
         idx_t bM,idx_t bN>
long gemm1(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  assert(M % bM == 0);
  assert(N % bN == 0);
  for (idx_t j = 0; j < N; j += bN) {
    for (idx_t i = 0; i < M; i += bM) {
      realv c[bM][bN/L];
      for (idx_t ii = 0; ii < bM; ii++) {
        for (idx_t jj = 0; jj < bN; jj += L) {
          c[ii][jj/L] = C.v(i+ii,j+jj);
        }
      }
      asm volatile("# loop begins");
      for (idx_t k = 0; k < K; k++) {
        for (idx_t ii = 0; ii < bM; ii++) {
          for (idx_t jj = 0; jj < bN; jj += L) {
            c[ii][jj/L] += A(i+ii,k) * B.v(k,j+jj);
          }
        }
      }
      asm volatile("# loop ends");
      for (idx_t ii = 0; ii < bM; ii++) {
        for (idx_t jj = 0; jj < bN; jj += L) {
          C.v(i+ii,j+jj) = c[ii][jj/L];
        }
      }
    }
  }
  return (long)M * (long)N * (long)K;
}

template<idx_t M,idx_t N,idx_t K,
         idx_t lda,idx_t ldb,idx_t ldc,
         idx_t bM,idx_t bN>
long gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C) {
  const idx_t M1 = 512;
  const idx_t N1 = 320;
#pragma omp parallel for collapse(2) schedule(runtime)
  for (idx_t i = 0; i < M; i += M1) {
    for (idx_t j = 0; j < N; j += N1) {
      matrix_c<M1,K,lda> sA = A.template sub<M1,K>(i, 0);
      matrix_c<K,N1,ldb> sB = B.template sub<K,N1>(0, j);
      matrix_c<M1,N1,ldc> sC = C.template sub<M1,N1>(i, j);
      
      gemm1<M1,N1,K,lda,ldb,ldc,bM,bN>(sA, sB, sC);
    }
  }
  return (long)M * (long)N * (long)K;
}
