/** 
 * @file mem.cc
 */
#include <assert.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#if _OPENMP
#include <omp.h>
#endif

#include "event.h"
#include "get_pfn_info.h"

#ifndef max_chains_per_thread
#define max_chains_per_thread 30
#endif

#if __AVX512F__
enum { vwidth = 64 };
#define mk_longv(c) { c, c, c, c, c, c, c, c }
#elif __AVX__
enum { vwidth = 32 };
#define mk_longv(c) { c, c, c, c }
#else
#error "__AVX512F__ or __AVX__ must be defined"
#endif
enum {
  //valign = sizeof(real),
  valign = vwidth
};
typedef long longv __attribute__((vector_size(vwidth),aligned(valign)));

/* record is the maximum-sized record that can be
   fetched with a single instruction */
template<int rec_sz>
struct record {
  union {
    struct {
      struct record<rec_sz> * volatile next;
      struct record<rec_sz> * volatile prefetch;
    };      
    char payloadc[rec_sz];
    volatile longv payload[0];
  };
};

/* check if links starting from a form a cycle
   of n distinct elements in a[0] ... a[n-1] */
template<int rec_sz>
void check_links_cyclic(record<rec_sz> * a, long n) {
  char * c = (char *)calloc(sizeof(char), n);
  volatile record<rec_sz> * p = a;
  for (long i = 0; i < n; i++) {
    assert(p - a >= 0);
    assert(p - a < n);
    assert(c[p - a] == 0);
    c[p - a] = 1;
    p = p->next;
  }
  p = a;
  for (long i = 0; i < n; i++) {
    assert(c[p - a]);
    p = p->next;
  }
  free(c);
}

/* return 1 if x is a prime */
int is_prime(long x) {
  if (x == 1) return 0;
  long y = 2;
  while (y * y <= x) {
    if (x % y == 0) return 0;
    y++;
  }
  return 1; 
}

static inline long calc_random_next0(long t, long idx, long n) {
  long next_idx = idx + 2 * t + 1;
  if (next_idx - n >= 0) next_idx = next_idx - n;
  if (next_idx - n >= 0) next_idx = next_idx - n;
  return next_idx;
}

static inline long calc_random_next1(long t, long idx, long n) {
  (void)t;
  return n - idx;
}

static inline long calc_random_next2(long t, long idx, long n) {
  long next_idx = idx - (2 * t + 1);
  if (next_idx < 0) next_idx = next_idx + n;
  if (next_idx < 0) next_idx = next_idx + n;
  return next_idx;
}

static inline long calc_random_next(long t, long idx, long n) {
  if (t < (n - 1) / 2) {
    return calc_random_next0(t, idx, n);
  } else if (t == (n - 1) / 2) {
    return calc_random_next1(t, idx, n);
  } else {
    return calc_random_next2(t, idx, n);
  }
}

inline long calc_stride_next(long idx, long stride, long n) {
  idx += stride;
  if (idx >= n) idx -= n;
  return idx;
}

/* set next pointers, so that
   a[0] .. a[n-1] form a cycle */
template<int rec_sz>
void randomize_next_pointers(record<rec_sz> * a, long n) {
  long idx = 0;
  assert(n % 4 == 3);
  assert(is_prime(n));
  for (long t = 0; t < n; t++) {
    long next_idx = calc_random_next(t, idx, n);
    assert(next_idx >= 0);
    assert(next_idx < n);
    a[idx].next = &a[next_idx];
    idx = next_idx;
  }
}

template<int rec_sz>
void make_prefetch_pointers(record<rec_sz> * h, long n, long d) {
  record<rec_sz> * p = h;
  record<rec_sz> * q = h;
  /* q = d nodes ahead of p */
  for (long i = 0; i < d; i++) {
    q = q->next;
  }
  for (long i = 0; i < n; i++) {
    p->prefetch = q;
    p = p->next;
    q = q->next;
  }
}

/* make H[0] ... H[N * NC - 1] NC chains x N elements;
   if shuffle is 1, next pointers of each array are shuffled */
template<int rec_sz>
void mk_arrays(long n, int nc, record<rec_sz> * H,
               record<rec_sz> * a[max_chains_per_thread],
	       int shuffle, long prefetch_dist) {
  /* make NC arrays */
  for (int c = 0; c < nc; c++) {
    record<rec_sz> * h = H + n * c;
    /* default: next points to the immediately
       following element in the array */
    for (long i = 0; i < n; i++) {
      h[i].next = &h[(i + 1) % n];
    }
    if (shuffle) {
      randomize_next_pointers(h, n);
    }
    /* check if links form a cycle */
    check_links_cyclic(h, n);
    /* prefetch pointers */
    make_prefetch_pointers(h, n, prefetch_dist);
    /* install the head pointer */
    a[c] = h;
  }
}

int get_rank();

template<int rec_sz, int n_chains, int access_payload, int prefetch>
record<rec_sz> * scan_seq(record<rec_sz> * a[n_chains], long n, long n_scans, long prefetch_dist) {
  for (long s = 0; s < n_scans; s++) {
#if 0
    long t0 = rdtsc();
#endif
    asm volatile("# seq loop begin (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
    for (long t = 0, u = prefetch_dist; t < n; t++, u++) {
      for (int c = 0; c < n_chains; c++) {
	if (access_payload) {
          for (int i = 0; i < rec_sz / (int)sizeof(longv); i++) {
            a[c][t].payload[i];
          }
	} else {
	  a[c][t].next;
        }
	if (prefetch) {
	  __builtin_prefetch(&a[c][u]);
	}
      }
    }
    asm volatile("# seq loop end (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
#if 0
    long t1 = rdtsc();
    if (get_rank() == 0) {
      printf("%ld : %ld\n", s, t1 - t0);
    }
#endif
  }
  return &a[n_chains-1][n - 1];
}

template<int rec_sz, int n_chains, int access_payload, int prefetch>
record<rec_sz> * scan_store_seq(record<rec_sz> * a[n_chains], long n, long n_scans, long prefetch_dist) {
  longv Z = mk_longv(1);
  for (long s = 0; s < n_scans; s++) {
    asm volatile("# store seq loop begin (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
    for (long t = 0, u = prefetch_dist; t < n; t++, u++) {
      for (int c = 0; c < n_chains; c++) {
	if (access_payload) {
          for (int i = 0; i < rec_sz / (int)sizeof(longv); i++) {
            a[c][t].payload[i] = Z;
          }
	} else {
	  a[c][t].next;
	}
	if (prefetch) {
	  __builtin_prefetch(&a[c][u], 1);
	}
      }
    }
    asm volatile("# store seq loop end (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
  }
  return &a[n_chains-1][n - 1];
}

template<int rec_sz, int n_chains, int access_payload, int prefetch>
record<rec_sz> * scan_stride(record<rec_sz> * a[n_chains], long n, long n_scans,
		     long stride, long prefetch_dist) {
  long idx = 0;
  long p_idx = 0;
  if (prefetch) {
    for (long t = 0; t < prefetch_dist; t++) {
      p_idx = calc_stride_next(t, p_idx, n);
    }
  }
  for (long s = 0; s < n_scans; s++) {
    asm volatile("# stride loop begin (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
    for (long t = 0; t < n; t++) {
      for (int c = 0; c < n_chains; c++) {
	if (access_payload) {
          for (int i = 0; i < rec_sz / (int)sizeof(longv); i++) {
            a[c][idx].payload[i];
          }
	} else {
	  a[c][idx].next;
	}
	if (prefetch) {
	  __builtin_prefetch(&a[c][p_idx]);
	}
      }
      idx = calc_stride_next(idx, stride, n);
      if (prefetch) {
	p_idx = calc_stride_next(p_idx, stride, n);
      }
    }
    asm volatile("# stride loop end (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
  }
  return &a[n_chains-1][idx];
}

/* access a[k,0..n] for each k with stride s, m times */
template<int rec_sz, int n_chains, int access_payload, int prefetch>
record<rec_sz> * scan_random(record<rec_sz> * a[n_chains], long n, long n_scans, long prefetch_dist) {
  long idx = 0;
  long p_idx = 0;
  for (long t = 0; t < prefetch_dist; t++) {
    p_idx = calc_random_next(t, p_idx, n);
  }
  for (long s = 0; s < n_scans; s++) {
    asm volatile("# random loop begin (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
    for (long t = 0; t < n; t++) {
      for (int c = 0; c < n_chains; c++) {
	if (access_payload) {
          for (int i = 0; i < rec_sz / (int)sizeof(longv); i++) {
            a[c][idx].payload[i];
          }
	} else {
	  a[c][idx].next;
	}
	if (prefetch) {
	  __builtin_prefetch(&a[c][p_idx]);
	}
      }
      idx = calc_random_next(t, idx, n);
      if (prefetch) {
	p_idx = calc_random_next(t, p_idx, n);
      }
    }
    asm volatile("# random loop end (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
  }
  return &a[n_chains-1][idx];
}

/* access a[k,0..n] for each k with stride s, m times */
template<int rec_sz, int n_chains, int access_payload, int prefetch>
record<rec_sz> * scan_store_random(record<rec_sz> * a[n_chains], long n, long n_scans, long prefetch_dist) {
  long idx = 0;
  long p_idx = 0;
  for (long t = 0; t < prefetch_dist; t++) {
    p_idx = calc_random_next(t, p_idx, n);
  }
  longv Z = mk_longv(1);
  for (long s = 0; s < n_scans; s++) {
    asm volatile("# store random loop begin (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
    for (long t = 0; t < n; t++) {
      for (int c = 0; c < n_chains; c++) {
	if (access_payload) {
          for (int i = 0; i < rec_sz / (int)sizeof(longv); i++) {
            a[c][idx].payload[i] = Z;
          }
	} else {
	  a[c][idx].next;
	}
	if (prefetch) {
	  __builtin_prefetch(&a[c][p_idx]);
	}
      }
      idx = calc_random_next(t, idx, n);
      if (prefetch) {
	p_idx = calc_random_next(t, p_idx, n);
      }
    }
    asm volatile("# store random loop end (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
  }
  return &a[n_chains-1][idx];
}

template<int rec_sz, int n_chains, int access_payload, int prefetch>
/* traverse n_chains pointers in parallel */
record<rec_sz> * scan_ptr_chase(record<rec_sz> * a[n_chains], long n, long n_scans) {
  /* init pointers */
  record<rec_sz> * p[n_chains];
  for (int c = 0; c < n_chains; c++) {
    p[c] = a[c];
  }
  for (long s = 0; s < n_scans; s++) {
    asm volatile("# pointer chase loop begin (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
    for (long t = 0; t < n; t++) {
      for (int c = 0; c < n_chains; c++) {
	if (access_payload) {
          for (int i = 0; i < rec_sz / (int)sizeof(longv); i++) {
            p[c]->payload[i];
          }
	}
	record<rec_sz> * next = p[c]->next;
	if (prefetch) {
	  __builtin_prefetch(p[c]->prefetch);
	}
	p[c] = next;
      }
    }
    asm volatile("# pointer chase loop end (n_chains = %0, payload = %1, prefetch = %2, rec_sz = %3)" 
		 : : "i" (n_chains), "i" (access_payload), "i" (prefetch), "i" (rec_sz));
  }
  for (int c = 0; c < n_chains; c++) {
    asm volatile("" : : "q" (p[c]));
  }
  return p[0];
}

template<int rec_sz, int n_chains, int access_payload, int prefetch>
record<rec_sz> * scan(record<rec_sz> * a[n_chains], long n, long n_scans,
                      const char * method, long stride, long prefetch_dist) {
  switch (method[0]) {
  case 's' :
    return scan_seq<rec_sz,n_chains,access_payload,prefetch>(a, n, n_scans, prefetch_dist);
  case 'S' :
    return scan_store_seq<rec_sz,n_chains,access_payload,prefetch>(a, n, n_scans, prefetch_dist);
  case 't' :
    return scan_stride<rec_sz,n_chains,access_payload,prefetch>(a, n, n_scans, stride, prefetch_dist);
  case 'r' :
    return scan_random<rec_sz,n_chains,access_payload,prefetch>(a, n, n_scans, prefetch_dist);
  case 'R' :
    return scan_store_random<rec_sz,n_chains,access_payload,prefetch>(a, n, n_scans, prefetch_dist);
  default : 
  case 'p': {
    return scan_ptr_chase<rec_sz,n_chains,access_payload,prefetch>(a, n, n_scans);
  }
  }
}

template<int rec_sz, int n_chains, int access_payload>
record<rec_sz> * scan(record<rec_sz> * a[n_chains], long n, long n_scans,
	      const char * method, long stride, int prefetch) {
  if (prefetch) {
    return scan<rec_sz,n_chains,access_payload,1>(a, n, n_scans, method, stride, prefetch);
  } else {
    return scan<rec_sz,n_chains,access_payload,0>(a, n, n_scans, method, stride, prefetch);
  }
}

template<int rec_sz, int n_chains>
record<rec_sz> * scan(record<rec_sz> * a[n_chains], long n, long n_scans,
                      const char * method, int access_payload, long stride,
                      long prefetch) {
  if (access_payload) {
    return scan<rec_sz,n_chains,1>(a, n, n_scans, method, stride, prefetch);
  } else {
    return scan<rec_sz,n_chains,0>(a, n, n_scans, method, stride, prefetch);
  }
}


template<int rec_sz>
record<rec_sz> * scan(record<rec_sz> * a[max_chains_per_thread], long n, long n_scans,
                      const char * method, int nc, int access_payload,
                      long stride, long prefetch) {
  if (nc >= max_chains_per_thread) {
    fprintf(stderr, "number of chains = %d >= %d\n", nc, max_chains_per_thread);
    fprintf(stderr, "either give a smaller nc or change max_chains in the source; abort\n");
    exit(1);
  }

  switch (nc) {
  case 1:
    return scan<rec_sz,1>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 2:
    return scan<rec_sz,2>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 3:
    return scan<rec_sz,3>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 4:
    return scan<rec_sz,4>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 5:
    return scan<rec_sz,5>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 6:
    return scan<rec_sz,6>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 7:
    return scan<rec_sz,7>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 8:
    return scan<rec_sz,8>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 9:
    return scan<rec_sz,9>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 10:
    return scan<rec_sz,10>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 11:
    return scan<rec_sz,11>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 12:
    return scan<rec_sz,12>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 13:
    return scan<rec_sz,13>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 14:
    return scan<rec_sz,14>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 15:
    return scan<rec_sz,15>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 16:
    return scan<rec_sz,16>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 17:
    return scan<rec_sz,17>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 18:
    return scan<rec_sz,18>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 19:
    return scan<rec_sz,19>(a, n, n_scans, method, access_payload, stride, prefetch);
  case 20:
    return scan<rec_sz,20>(a, n, n_scans, method, access_payload, stride, prefetch);
  default:
    fprintf(stderr, "number of chains = %d, must be >= 0 and <= 20\n", nc);
    exit(1);
    break;
  }
  return 0;
}


/* find a prime of 4m+3 no greater than x */
long good_prime(long x) {
  if (x < 3) return 3;
  else {
    long y = x - (x % 4) + 3;
    long z = (y > x ? y - 4 : y);
    assert(z % 4 == 3);
    assert(z <= x);
    while (z > 0) {
      if (is_prime(z)) return z;
      z -= 4;
    }
    assert(0);
  }
}

int get_n_threads() {
#if _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int get_rank() {
#if _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

void barrier() {
#if _OPENMP
#pragma omp barrier
#endif
}

typedef struct {
  long long c;                  /**< cpu clock */
  long long r;                  /**< ref clock */
  long long t;                  /**< wall clock */
  perf_event_values_t v;        /**< counters */
} timestamps_t;

typedef struct {
  timestamps_t ts[2];           /**< timestamps before and after */
  long n;                       /**< number of clements */
  long n_chains;                /**< number of chains per thread */
  long n_threads;               /**< number of threads */
  long n_scans;                 /**< number of scans */
} scan_record_t;

timestamps_t get_all_stamps_before(perf_event_counters_t cc,
                                   perf_event_counters_t mc) {
  timestamps_t ts;
  ts.t = cur_time_ns();
  ts.v = perf_event_counters_get(mc);
  ts.c = perf_event_counters_get_i(cc, 0);
  ts.r = rdtsc();
  return ts;
}

timestamps_t get_all_stamps_after(perf_event_counters_t cc,
                                  perf_event_counters_t mc) {
  timestamps_t ts;
  ts.r = rdtsc();
  ts.c = perf_event_counters_get_i(cc, 0);
  ts.v = perf_event_counters_get(mc);
  ts.t = cur_time_ns();
  return ts;
}

template<int rec_sz>
void worker(int rank, int n_threads, record<rec_sz> * H,
	    long n, long n_scans, long repeat, int shuffle,
	    const char * method, long nc, int access_payload,
            long stride, long prefetch, const char * events) {
  record<rec_sz> * a[max_chains_per_thread];
  mk_arrays(n, nc, &H[n * nc * rank], a, shuffle, prefetch);
  perf_event_counters_t cc, mc;
  scan_record_t * scan_records = 0;
  if (rank == 0) {
    cc = mk_perf_event_counters("cycles");
    mc = mk_perf_event_counters(events);
    scan_records = (scan_record_t *)calloc(repeat, sizeof(scan_record_t));
  }
  for (long r = 0; r < repeat; r++) {
    scan_record_t * R = &scan_records[r];
    barrier();
    if (rank == 0) {
      R->ts[0] = get_all_stamps_before(cc, mc);
    }
    scan(a, n, n_scans, method, nc, access_payload, stride, prefetch);
    barrier();
    if (rank == 0) {
      R->ts[1] = get_all_stamps_after(cc, mc);
    }
  }
  if (rank == 0) {
#if 0    
    long n_loads = n * nc * n_scans
      * (access_payload ? (sizeof(record<rec_sz>) / sizeof(longv)) : 1);
    double ovf = show_cache_set_info(&H[n * nc * rank],
                                     &H[n * nc * (rank + 1)]);
    printf("overflow percentage : %f\n", ovf);
    printf("n_loads : %ld\n", n_loads);
    printf("expected misses : %f\n", ovf * n_loads);
#endif
    for (long r = 0; r < repeat; r++) {
      printf("--------- %ld ---------\n", r);
      scan_record_t * R = &scan_records[r];
      long long dr = R->ts[1].r - R->ts[0].r;
      long long dc = R->ts[1].c - R->ts[0].c;
      long long dt = R->ts[1].t - R->ts[0].t;
      long n_elements = n * nc * n_threads;
      long n_records  = n_elements * n_scans;
      long access_sz  = n_records * (access_payload ? sizeof(record<rec_sz>) : sizeof(record<rec_sz>*));
      long n_iters_per_thread = n * n_scans;
      for (int i = 0; i < mc.n; i++) {
        long long m0 = R->ts[0].v.values[i];
        long long m1 = R->ts[1].v.values[i];
        long long dm = m1 - m0;
        printf("metric:%s = %lld -> %lld = %lld\n", mc.events[i], m0, m1, dm);
      }
      printf("%lld CPU clocks\n", dc);
      printf("%lld REF clocks\n", dr);
      printf("%lld nano sec\n", dt);
      printf("throughput %.3f bytes/REF clock\n", access_sz / (double)dr);
      printf("throughput %.3f GiB/sec\n", access_sz * pow(2.0, -30) * 1.0e9 / dt);
      printf("latency %.3f CPU clocks\n", dc / (double)n_iters_per_thread);
      printf("latency %.3f REF clocks\n", dr / (double)n_iters_per_thread);
      printf("latency %.3f nano sec\n", dt / (double)n_iters_per_thread);
    }
  }
  perf_event_counters_destroy(mc);
  perf_event_counters_destroy(cc);
}

const char * canonical_method_string(const char * method) {
  switch (method[0]) {
  case 's': 
    return "sequential";
  case 'S': 
    return "store-sequential";
  case 't': 
    return "stride";
  case 'r':
    return "random";
  case 'R':
    return "store-random";
  default :
  case 'p':
    return "ptrchase";
  }
}

struct opts {
  const char * method;
  long n_elements;
  long n_chains;
  long n_scans;
  int repeat;
  int shuffle;
  int payload;
  long stride;
  long prefetch;
  const char * events;
  int rec_sz;
  opts() {
    method = "ptrchase";
    n_elements = 1 << 9;
    n_chains = 1;
    n_scans = -1;
    repeat = 3;
    shuffle = 1;
    payload = 1;
    prefetch = 0;
    stride = 1;
    events = 0;
    rec_sz = 64;
  }
};

void usage(char * prog) {
  opts o;
  fprintf(stderr, "usage:\n");
  fprintf(stderr, "  %s [options]\n", prog);
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -m,--method ptrchase/sequential/stride/random (%s)\n", o.method);
  fprintf(stderr, "  -n,--n_elements N (%ld)\n", o.n_elements);
  fprintf(stderr, "  -c,--n_chains N (%ld)\n", o.n_chains);
  fprintf(stderr, "  -S,--n_scans N (%ld)\n", o.n_scans);
  fprintf(stderr, "  -r,--repeat N (%d)\n", o.repeat);
  fprintf(stderr, "  -x,--shuffle 0/1 (%d)\n", o.shuffle);
  fprintf(stderr, "  -l,--payload 0/1 (%d)\n", o.payload);
  fprintf(stderr, "  -s,--stride N (%ld)\n", o.stride);
  fprintf(stderr, "  -p,--prefetch 0/1 (%ld)\n", o.prefetch);
  fprintf(stderr, "  -e,--events ev,ev,ev,.. (%s)\n", o.events);
}

opts * parse_cmdline(int argc, char * const * argv, opts * o) {
  static struct option long_options[] = {
    {"method",     required_argument, 0, 'm' },
    {"n_elements", required_argument, 0, 'n' },
    {"n_scans",    required_argument, 0, 'S' },
    {"n_chains",   required_argument, 0, 'c' },
    {"repeat",     required_argument, 0, 'r' },
    {"shuffle",    required_argument, 0, 'x' },
    {"payload",    required_argument, 0, 'l' },
    {"stride",     required_argument, 0, 's' },
    {"prefetch",   required_argument, 0, 'p' },
    {"events",     required_argument, 0, 'e' },
    {"rec_sz",     required_argument, 0, 'z' },
    {0,         0,                 0,  0 }
  };

  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "m:n:c:s:r:x:l:p:S:e:z:",
			long_options, &option_index);
    if (c == -1) break;

    switch (c) {
    case 'm':
      o->method = strdup(optarg);
      break;
    case 'n':
      o->n_elements = atol(optarg);
      break;
    case 'S':
      o->n_scans = atol(optarg);
      break;
    case 'c':
      o->n_chains = atoi(optarg);
      break;
    case 'r':
      o->repeat = atoi(optarg);
      break;
    case 'x':
      o->shuffle = atoi(optarg);
      break;
    case 'l':
      o->payload = atoi(optarg);
      break;
    case 'p':
      o->prefetch = atol(optarg);
      break;
    case 's':
      o->stride = atol(optarg);
      break;
    case 'e':
      o->events = strdup(optarg);
      break;
    case 'z':
      o->rec_sz = atoi(optarg);
      break;
    default:
      usage(argv[0]);
      exit(1);
    }
  }
  return o;
}

template<int rec_sz>
int real_main(opts o) {
  int n_threads = get_n_threads();

  const char * method = o.method;
  const char * events = o.events;
  /* nc : number of chains per thread */
  int nc        = o.n_chains;
  /* n : number of elements per chain */
  long n        = good_prime(o.n_elements / nc / n_threads);
  long shuffle  = o.shuffle;
  /* number of times an array is scanned */
  long n_scans  = (o.n_scans >= 0 ?
                   o.n_scans :
                   ((1 << 25) / (n * nc * n_threads) + 1));
  int repeat    = o.repeat;
  int access_payload = o.payload;
  long prefetch = o.prefetch;
  long stride = o.stride % n;
  long n_elements = n * nc * n_threads;
  long n_records  = n_elements * n_scans;
  long data_sz    = sizeof(record<rec_sz>) * n_elements;
  long n_loads = n_records * (access_payload ? (sizeof(record<rec_sz>) / sizeof(longv)) : 1);

  record<rec_sz> * H = (record<rec_sz> *)mmap(0, data_sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (H == MAP_FAILED) { 
    perror("mmap"); exit(1);
  }
  //memset(H, -1, data_sz);
  assert(sizeof(record<rec_sz>) == rec_sz);
  printf("%ld elements"
	 " x %d chains"
	 " x %ld scans"
	 " x %d threads"
	 " = %ld record accesses"
	 " = %ld loads.\n", 
	 n, nc, n_scans, n_threads, n_records, n_loads);
  printf("record_size: %ld bytes\n", sizeof(record<rec_sz>));
  printf("data: %ld bytes\n", data_sz);
  printf("shuffle: %ld\n", shuffle);
  printf("payload: %d\n", access_payload);
  printf("stride: %ld\n", stride);
  printf("prefetch: %ld\n", prefetch);
  printf("method: %s\n", canonical_method_string(method));
  printf("events: %s\n", events);
  fflush(stdout);

#if _OPENMP
#pragma omp parallel
#endif
  {
    int rank = get_rank();
    worker(rank, n_threads, H,
	   n, n_scans, repeat, shuffle, 
	   method, nc, access_payload, stride, prefetch, events);
  }
  return 0;
}

int main(int argc, char * const * argv) {
  opts o;
  parse_cmdline(argc, argv, &o);
  switch (o.rec_sz) {
  case 64:
    return real_main<64>(o);
  case 128:
    return real_main<128>(o);
  case 256:
    return real_main<256>(o);
  case 4096:
    return real_main<4096>(o);
  case 65536:
    return real_main<65536>(o);
  default:
    fprintf(stderr, "invalid record size %d\n", o.rec_sz);
    return EXIT_FAILURE;
  }
}

