/* 
 * comm.c

 * usage:
     OMP_PROC_BIND=true ./comm N
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/mman.h>

#include <x86intrin.h>

#if _OPENMP
#include <omp.h>
#else
int omp_get_thread_num(void) {
  return 0;
}

int omp_get_max_threads() {
  return 1;
}
#endif

typedef struct {
  union {
    struct {
      union { volatile long p[1]; char pad[1088]; };
      union { volatile long q[1]; char pad_[1088]; };
      long src_load;
      unsigned long long src_clock;
      long dst_load;
      unsigned long long dst_clock;
    };
    char pad____[4096];
  };
} record;


void ping_pong(record * r, long n, int role) {
  volatile long * p = r->p;
  volatile long * q = r->q;
  long i;
  long load = 0;
  long long c0 = 0, c1;
  if (role == 0) {
    asm volatile ("# src loop begin");
    for (i = 0; i < n + 1; i++) {
      *p = i;
      while (*q < i) {
	asm volatile("pause");
	//__builtin_ia32_pause();
	load++;
      }
      if (i == 0) c0 = _rdtsc();
    }
    asm volatile ("# src loop end");
    c1 = _rdtsc();
    r->src_load = load;
    r->src_clock = c1 - c0;
  } else {
    asm volatile ("# dst loop begin");
    for (i = 0; i < n + 1; i++) {
      while (*p < i) {
	asm volatile("pause");
	//__builtin_ia32_pause();
	load++;
      }
      *q = i;
      if (i == 0) c0 = _rdtsc();
    }
    asm volatile ("# dst loop end");
    c1 = _rdtsc();
    r->dst_load = load;
    r->dst_clock = c1 - c0;
  }
}


void worker_main(int rank, int n_threads, 
		 record a[n_threads][n_threads], long n) {
  int s, d;
  
  /* init communication buffer */
  for (d = 0; d < n_threads; d++) {
    a[rank][d].p[0] = -1;
    a[rank][d].q[0] = -1;
  }
  /* measure */
  for (s = 0; s < n_threads; s++) {
    for (d = 0; d < n_threads; d++) {
      if (s == d) {
	if (rank == s) {
	  a[s][d].src_load = 0;
	  a[s][d].src_clock = 0;
	  a[s][d].dst_load = 0;
	  a[s][d].dst_clock = 0;
	}
      } else {
	/* s -> d */
	if (rank == s) {
	  ping_pong(&a[s][d], n, 0);
	} else if (rank == d) {
	  ping_pong(&a[s][d], n, 1);
	}
      }
#if _OPENMP
#pragma omp barrier
#endif
    }
  }
}

void * mk_array(int n) {
  void * a = mmap(0, n * n * sizeof(record), 
		  PROT_READ|PROT_WRITE,
		  MAP_PRIVATE|MAP_ANONYMOUS,
		  -1, 0);
  assert(a != MAP_FAILED);
  // memset(a, 0, n * n * sizeof(record));
  return a;
}

void show_record(int nt, record a[nt][nt], long n) {
  int s, d;
  for (s = 0; s < nt; s++) {
    for (d = 0; d < nt; d++) {
      printf("%d -> %d : src_loads=%6.2f dst_loads=%6.2f src_clocks=%6.2f dst_clocks=%6.2f\n", 
	     s, d,
	     a[s][d].src_load  / (double)n,
	     a[s][d].dst_load  / (double)n,
	     a[s][d].src_clock / (double)n,
	     a[s][d].dst_clock / (double)n);
    }
  }
}

int main(int argc, char ** argv) {
  long n  = (argc > 1 ? atol(argv[1]) : 1 << 14);
  long nt = omp_get_max_threads();
  record (*a)[nt] = mk_array(nt);

#if _OPENMP
#pragma omp parallel
#endif
  {
    int rank = omp_get_thread_num();
    worker_main(rank, nt, a, n);
  }
  show_record(nt, a, n);
  return 0;
}
