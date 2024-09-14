/**
   @file hello.c
  */
#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>
#include <omp.h>

void worker() {
  int rank = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  int cpu = sched_getcpu();
  printf("hello from thread %d of %d on %d\n", rank, nthreads, cpu);
}

int main() {
  printf("hello before omp parallel\n");
#pragma omp parallel
  worker();
  printf("hello after omp parallel\n");
  return 0;
}
