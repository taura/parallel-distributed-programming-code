/* 
 * udr_simple.c --- user-defined reduction on 3 elements vectors
 */
#include <stdio.h>
#include <stdlib.h>

/* three element vector */
typedef struct {
  double a[3];
} vec_t;

/* initialize v = 0,0,0 */
int vec_init(vec_t * v) {
  for (int i = 0; i < 3; i++) {
    v->a[i] = 0;
  }
  return 0;
}

/* y += x */
void vec_add(vec_t * y, vec_t * x) {
  for (long i = 0; i < 3; i++) {
    y->a[i] += x->a[i];
  }
}

/* user-defined reduction on vec_t */
#pragma omp declare reduction \
  (vp : vec_t : vec_add(&omp_out,&omp_in))      \
  initializer(vec_init(&omp_priv))

//  reduction(vp : y)
int main() {
  vec_t y;
  vec_init(&y);                 /* y={0,0,0} */
#pragma omp parallel
#pragma omp for reduction(vp:y)
  for (long i = 0; i < 10000; i++) {
    y.a[i % 3] += 1;
  }
  for (long i = 0; i < 3; i++) {
    printf("a[%ld] = %f\n", i, y.a[i]);
  }
  return 0;
}

