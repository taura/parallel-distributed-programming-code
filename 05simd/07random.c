/**
   @file 07random.c
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

void loop_random(float a, float * restrict x, float b,
                 float * restrict y, long n) {
  /* tell the compiler x and y are 64 bytes-aligned (a multiple of 64) */
  x = __builtin_assume_aligned(x, 64);
  y = __builtin_assume_aligned(y, 64);
  /* tell the compiler n is a multiple of 16 */
  n = (n / 16) * 16;
  asm volatile("# loop begins");
#pragma omp simd
  for (long i = 0; i < n; i++) {
    y[i] = a * x[i * i] + b;
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

intv ztoL() {
  int ztoL_[L];
  for (int i = 0; i < L; i++) ztoL_[i] = i;
  return *((intv*)ztoL_);
}

void loop_random_v(float a, float * x, float b, float * y, long n) {
  intv iv = (intv)ztoL();
  intv Lv = (intv)_mm512_set1_epi32(L);
  asm volatile("# vloop begins");
  for (long i = 0; i < n; i += L, iv += Lv) {
    floatv xiv = _mm512_i32gather_ps((__m512i)(iv * iv), x, sizeof(float));
    V(y[i]) = a * xiv + b;
  }
  asm volatile("# vloop ends");
}

int main(int argc, char ** argv) {
  long  n = (argc > 1 ? atol(argv[1]) : 1024);
  float a = (argc > 2 ? atof(argv[2]) : 1.234);
  float b = (argc > 3 ? atof(argv[3]) : 4.567);
  long seed = (argc > 4 ? atol(argv[4]) : 8901234567);
  n = (n / 16) * 16;
  float *  x = _mm_malloc(n * n * sizeof(float), 64);
  float *  y = _mm_malloc(n * sizeof(float), 64);
  float * yv = _mm_malloc(n * sizeof(float), 64);
  unsigned short rg[3] = { (seed >> 32) & 65535,
                           (seed >> 16) & 65535,
                           (seed >>  0) & 65535 };
  for (long i = 0; i < n * n; i++) {
    x[i] = erand48(rg) - 0.5;
  }
  for (long i = 0; i < n; i++) {
    y[i] = 1.0;
    yv[i] = 2.0;
  }
  loop_random(a, x, b, y, n);
  loop_random_v(a, x, b, yv, n);
  for (long i = 0; i < n; i++) {
    assert(y[i] == yv[i]);
  }
  printf("OK\n");
  return 0;
}
