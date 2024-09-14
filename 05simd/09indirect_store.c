/**
   @file 09indirect_store.c
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>

#include <assert.h>

void loop_indirect_store(float a, float * restrict x, int * idx, float b,
                         float * restrict y, long n) {
  /* tell the compiler x and y are 64 bytes-aligned (a multiple of 64) */
  x = __builtin_assume_aligned(x, 64);
  y = __builtin_assume_aligned(y, 64);
  /* tell the compiler n is a multiple of 16 */
  n = (n / 16) * 16;
  asm volatile("# loop begins");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    y[idx[i]] += a * x[i] + b;
  }
  asm volatile("# loop ends");
}

#if __AVX512F__
typedef float floatv __attribute__((vector_size(64),aligned(sizeof(float))));
typedef int intv __attribute__((vector_size(64),aligned(sizeof(int))));
#else
#error "this code requires __AVX512F__"
#endif
const int L = sizeof(floatv) / sizeof(float);

#define V(p) *((floatv*)&(p))
#define IV(p) *((intv*)&(p))

void loop_indirect_store_v(float a, float * x, int * idx, float b,
                           float * y, long n) {
  asm volatile("# vloop begins");
  for (long i = 0; i < n; i += L) {
    intv iv = IV(idx[i]);
    floatv yiv = _mm512_i32gather_ps((__m512i)iv, y, sizeof(float));
    _mm512_i32scatter_ps(y, (__m512i)iv, yiv + a * V(x[i]) + b, sizeof(float));
  }
  asm volatile("# vloop ends");
}

int main(int argc, char ** argv) {
  long  n = (argc > 1 ? atol(argv[1]) : 32);
  float a = (argc > 2 ? atof(argv[2]) : 1.234);
  float b = (argc > 3 ? atof(argv[3]) : 4.567);
  long seed = (argc > 4 ? atol(argv[4]) : 8901234567);
  n = (n / 16) * 16;
  float *  x = _mm_malloc(n * sizeof(float), 64);
  float *  y = _mm_malloc(n * sizeof(float), 64);
  float * yv = _mm_malloc(n * sizeof(float), 64);
  int  * idx = _mm_malloc(n * sizeof(int), 64);
  unsigned short rg[3] = { (seed >> 32) & 65535,
                           (seed >> 16) & 65535,
                           (seed >>  0) & 65535 };
  for (long i = 0; i < n; i++) {
    idx[i] = i;
    x[i] = erand48(rg) - 0.5;
    y[i] = 1.0;
    yv[i] = 1.0;
  }
  for (long i = 0; i < n; i++) {
    long j = nrand48(rg) % n;
    int t = idx[i];
    idx[i] = idx[j];
    idx[j] = t;
  }
  loop_indirect_store(a, x, idx, b, y, n);
  loop_indirect_store_v(a, x, idx, b, yv, n);
  for (long i = 0; i < n; i++) {
    //printf("y[%ld] = %.9f, yv[%ld] = %.9f\n", i, y[i], i, yv[i]);
    assert(fabs(y[i] - yv[i]) < 1.0e-4);
  }
  printf("OK\n");
  return 0;
}
