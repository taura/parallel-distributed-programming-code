/**
   @file udr_varlen_vect.c
   @brief a simple demonstration of how to use user-defined reductions
   to reduce variable-length vectors
 */
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  long n;
  double * a;
} vec_t;

int vec_init(vec_t * v, long n) {
  double * a = (double *)malloc(sizeof(double) * n);
  for (long i = 0; i < n; i++) {
    a[i] = 0;
  }
  v->n = n;
  v->a = a;
  return 0;
}

/**
   @brief initialize v, taking the number of elements from orig
 */
int vec_init_from(vec_t * v, vec_t * orig) {
  long n = orig->n;
  double * a = (double *)malloc(sizeof(double) * n);
  for (long i = 0; i < n; i++) {
    a[i] = 0;
  }
  v->n = n;
  v->a = a;
  return 0;
}

void vec_add(vec_t * y, vec_t * x) {
  long n = y->n;
  for (long i = 0; i < n; i++) {
    y->a[i] += x->a[i];
  }
}

#pragma omp declare reduction (vplus : vec_t : vec_add(&omp_out,&omp_in)) \
  initializer(vec_init_from(&omp_priv, &omp_orig))

int main(int argc, char ** argv) {
  vec_t y;
  long n = (argc > 1 ? atol(argv[1]) : 5);
  vec_init(&y, n);
#pragma omp parallel
#pragma omp for reduction(vplus : y)
  for (long j = 0; j < 1000000; j++) {
    y.a[j % n] += 1;
  }
  for (long i = 0; i < n; i++) {
    printf("a[%ld] = %f\n", i, y.a[i]);
  }
  return 0;
}

