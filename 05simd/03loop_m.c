/**
   @file 03loop_m.c
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

void loop_loop_m(float a, float * restrict x, float b,
                 float * restrict y, long n, long m) {
  /* tell the compiler x and y are 64 bytes-aligned (a multiple of 64) */
  x = __builtin_assume_aligned(x, 64);
  y = __builtin_assume_aligned(y, 64);
  /* tell the compiler n is a multiple of 16 */
  n = (n / 16) * 16;
  asm volatile("# loop begins");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    y[i] = x[i];
    for (long j = 0; j < m; j++) {
      y[i] = a * y[i] + b;
    }
  }
  asm volatile("# loop ends");
}
#if __AVX512F__
typedef float floatv __attribute__((vector_size(64),aligned(sizeof(float))));
#elif __AVX__
typedef float floatv __attribute__((vector_size(32),aligned(sizeof(float))));
#else
#error "this code requires __AVX512F__ or __AVX__"
#endif
const int L = sizeof(floatv) / sizeof(float);

#define V(p) *((floatv*)&(p))

void loop_loop_m_v(float a, float * x, float b,
                   float * y, long n, long m) {
  asm volatile("# vloop begins");
  for (long i = 0; i < n; i += L) {
    V(y[i]) = V(x[i]);
    for (long j = 0; j < m; j++) {
      V(y[i]) = a * V(y[i]) + b;
    }
  }
  asm volatile("# vloop ends");
}

int main(int argc, char ** argv) {
  long    n = (argc > 1 ? atol(argv[1]) : 1024);
  long    m = (argc > 2 ? atol(argv[2]) : 10);
  float   a = (argc > 3 ? atof(argv[3]) : 1.234);
  float   b = (argc > 4 ? atof(argv[4]) : 4.567);
  long seed = (argc > 5 ? atol(argv[5]) : 8901234567);
  n = (n / 16) * 16;
  float *  x = _mm_malloc(n * sizeof(float), 64);
  float *  y = _mm_malloc(n * sizeof(float), 64);
  float * yv = _mm_malloc(n * sizeof(float), 64);
  unsigned short rg[3] = { (seed >> 32) & 65535,
                           (seed >> 16) & 65535,
                           (seed >>  0) & 65535 };
  for (long i = 0; i < n; i++) {
    x[i] = erand48(rg) - 0.5;
    y[i] = 1.0;
    yv[i] = 2.0;
  }
  loop_loop_m(a, x, b, y, n, m);
  loop_loop_m_v(a, x, b, yv, n, m);
  for (long i = 0; i < n; i++) {
    assert(y[i] == yv[i]);
  }
  printf("OK\n");
  return 0;
}
