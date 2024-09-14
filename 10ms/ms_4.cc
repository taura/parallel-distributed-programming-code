//% file: ms_4.cc
// merge sort with TBB tasks
// TBB task_group (need -std=gnu++11 with old gcc versions)
//% cmd: g++ -Wall -Wextra -std=gnu++11 -O3 -march=native ms_4.cc -o ms_4 -ltbb -lpthread

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>
#include "clock.h"

/* return m a[m] = min {a[begin:end]} */ 
long choose_min_idx(float * a, long begin, long end) {
  long m = begin;
  for (long i = begin + 1; i < end; i++) {
    if (a[i] < a[m]) {
      m = i;
    }
  }
  return m;
}

/* sort a[p:q] by insertion sort */
void insertion_sort(float * a, long p, long q) {
  for (long i = p; i < q; i++) {
    long j = choose_min_idx(a, i, q);
    float t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
}

/* merge a[p:q] and a[r:s] --> b[d:] */
void merge(float * a, float * b, long p, long q, long r, long s, long d) {
  long i = p;
  long j = r;
  long k = d;
  while (i < q && j < s) {
    if (a[i] < a[j]) {
      b[k++] = a[i++];
    } else {
      b[k++] = a[j++];
    }
  }
  while (i < q) {
    b[k++] = a[i++];
  }
  while (j < s) {
    b[k++] = a[j++];
  }
}


/* sort a[p:q] into g[p:q], using b[p:q] as a temporary space */
void ms(float * a, float * b, float * g, long p, long q, long th0) {
  if (q - p < th0) {
    if (g != a) {
      for (long i = p; i < q; i++) {
        g[i] = a[i];
      }
    }
    insertion_sort(g, p, q);
  } else {
    long r = (p + q) / 2;
    /* get partial results into the other array != g */
    float * h = (g == a ? b : a);
    ms(a, b, h, p, r, th0);
    ms(a, b, h, r, q, th0);
    /* merge h[p:r] and h[r:q] -> g[p:]*/
    merge(h, g, p, r, r, q, p);
  }
}

/* make a random array of n elements */
float * random_array(long n, long seed) {
  float * a = (float *)malloc(sizeof(float) * n);
  unsigned short rg[3] = { (unsigned short)((seed >> 32) & 65535),
                           (unsigned short)((seed >> 16) & 65535),
                           (unsigned short)((seed >> 0 ) & 65535) };
  for (long i = 0; i < n; i++) {
    a[i] = erand48(rg);
  }
  return a;
}

/* make a random array of n elements */
float * const_array(long n, float c) {
  float * a = (float *)malloc(sizeof(float) * n);
  for (long i = 0; i < n; i++) {
    a[i] = c;
  }
  return a;
}

/* check if a[0:n] is sorted */
int count_unsorted(float * a, long n) {
  int err = 0;
  for (long i = 0; i < n - 1; i++) {
    if (a[i] > a[i+1]) {
      fprintf(stderr, "a[%ld] = %f > a[%ld] = %f\n",
              i, a[i], i + 1, a[i + 1]);
      err++;
    }
    assert(a[i] <= a[i+1]);
  }
  return err;
}

int main(int argc, char ** argv) {
  int i = 1;
  long n                = (argc > i ? atol(argv[i]) : 10000000); i++;
  /* n <= ms_threshold -> insertion sort */
  long ms_threshold     = (argc > i ? atol(argv[i]) : 50);       i++;
  long seed             = (argc > i ? atol(argv[i]) : 12345678); i++;
  float * a = random_array(n, seed);
  float * b = const_array(n, 0);
  printf("sort %ld elements\n", n);
  clock_counters_t cc = mk_clock_counters();
  clocks_t c0 = clock_counters_get(cc);

  /* real thing happens here */
  ms(a, b, a, 0, n, ms_threshold);

  clocks_t c1 = clock_counters_get(cc);
  long cpu  = c1.cpu_clock  - c0.cpu_clock;
  long ref  = c1.ref_clock  - c0.ref_clock;
  long wall = c1.wall_clock - c0.wall_clock;
  if (cpu == 0) {
    char * cpu_freq_s = getenv("CLOCK_ADJUST_CPU");
    char * ref_freq_s = getenv("CLOCK_ADJUST_REF");
    if (cpu_freq_s && ref_freq_s) {
      double cpu_freq = atof(cpu_freq_s);
      double ref_freq = atof(ref_freq_s);
      fprintf(stderr, "get cpu cycles by ref cycles x %f / %f\n", cpu_freq, ref_freq);
      fflush(stderr);
      cpu = ref * cpu_freq / ref_freq;
    }
  }
  printf("%ld nsec\n", wall);
  printf("%ld ref clocks\n", ref);
  if (cpu) {
    printf("%ld cpu clocks\n", cpu);
  } else {
    printf("-------- cpu clocks\n");
  }
  long us = count_unsorted(a, n);
  if (us == 0) {
    printf("OK\n");
  } else {
    printf("NG\n");
  }
  return 0;
}
