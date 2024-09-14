/**
   @file axpb.cc
 */

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <err.h>
#include <x86intrin.h>

#include "clock.h"

#if _OPENMP
#include <sched.h>
#include <omp.h>
#endif

#if __NVCC__
/* cuda_util.h incudes various utilities to make CUDA 
   programming less error-prone. check it before you
   proceed with rewriting it for CUDA */
#include "cuda_util.h"
#endif

/* GCC vector extension to define a vector of floats */
#if __AVX512F__
const int vwidth = 64;
#elif __AVX__
const int vwidth = 32;
#else
#error "you'd better have a better machine"
#endif

//const int valign = vwidth;
const int valign = sizeof(float);
typedef float floatv __attribute__((vector_size(vwidth),aligned(valign)));
/* SIMD lanes */
const int L = sizeof(floatv) / sizeof(float);

/**
   @brief type of axpb functions
  */

typedef enum {
  algo_scalar, 
  algo_simd, 
  algo_simd_c, 
  algo_simd_m, 
  algo_simd_m_nmn, 
  algo_simd_m_mnm, 
  algo_simd_parallel_m_mnm, 
  algo_cuda,
  algo_cuda_c,
  algo_invalid,
} algo_t;

/**
   @brief command line options
 */

typedef struct {
  const char * algo_str;
  algo_t algo;
  long bs;                      /**< cuda block size */
  long w;                       /**< active threads per warp */
  long c;                       /**< the number of floats concurrently updated */
  long m;                       /**< the number of floats */
  long n;                       /**< the number of times each variable is updated */
  long seed;                    /**< random seed */
  long n_elems_to_show;         /**< the number of variables to show results */
  int help;
  int error;
} axpb_options_t;

/** 
    @brief repeat x = a x + b for a scalar type (float) variable x
    @param (n) the number of times you do ax+b for x
    @param (a) a of a x + b
    @param (X) array of float elements (only use X[0])
    @param (b) b of a x + b

    @details it should run at 4 clocks/iter (the latency of fma
    instruction), or 0.5 flops/clock
 */
long axpb_scalar(axpb_options_t opt, float a, float* X, float b) {
  assert(opt.m == 1);
  assert(opt.bs == 1);
  long n = opt.n;
  float x = X[0];
  asm volatile ("# axpb_scalar: ax+b loop begin");
  for (long i = 0; i < n; i++) {
    x = a * x + b;
  }
  asm volatile ("# axpb_scalar: ax+b loop end");
  X[0] = x;
  return 0;
}

/** 
    @brief repeat x = a x + b with SIMD instructions
    @param (n) the number of times you do ax+b
    @param (a) a of a x + b
    @param (X) array of float elements (use only L elements)
    @param (b) b of a x + b

    @details it should run at 4 clocks/iter (the latency of fma
    instruction) = 4 flops/clock with AVX and 8 flops/clock with AVX512F 
 */
//#pragma GCC optimize("unroll-loops", 4)
long axpb_simd(axpb_options_t opt, float a, float* X_, float b) {
  assert(opt.m == L);
  assert(opt.bs == 1);
  long n = opt.n;
  floatv * X = (floatv*)X_;
  floatv x = X[0];
  asm volatile ("# axpb_simd: ax+b loop begin");
  for (long i = 0; i < n; i++) {
    x = a * x + b;
  }
  asm volatile ("# axpb_simd: ax+b loop end");
  X[0] = x;
  return 0;
}

/** 
    @brief repeat x = a x + b for a constant number of 
    vector variables
    @param (m) size of X. ignored. it always updates c vector elements
    @param (n) the number of times you do ax+b for each variable
    @param (a) a of a x + b
    @param (X) array of float elements (use only c * L elements)
    @param (b) b of a x + b

    @details when you increase nv, it should remain running at 4 
    clocks/iter until it reaches the limit of 2 FMAs/cycle,
    where it achieves the peak performance. nv=8 should achieve
    64 flops/clock with AVX512F.
    
    $ srun -p big bash -c "./axpb simd_c 8"

    4.001386 CPU clocks/iter, 3.966710 REF clocks/iter, 1.893479 ns/iter
    63.977836 flops/CPU clock, 64.537118 flops/REF clock, 135.200880 GFLOPS
    
 */
template<int c>
long axpb_simd_c(axpb_options_t opt, float a, float* X_, float b) {
  assert(opt.m == c * L);
  assert(opt.bs == 1);
  long n = opt.n;
  floatv * X = (floatv*)X_;
  asm volatile ("# axpb_simd_c<%0>: ax+c loop begin" :: "i"(c));
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < c; j++) {
      X[j] = a * X[j] + b;
    }
  }
  asm volatile ("# axpb_simd_c<%0>: ax+c loop end" :: "i"(c));
  return 0;
}

/** 
    @brief repeat x = a x + b for m (variable) vector type (floatv) variables
    @param (m) the number of variables updated
    @param (n) the number of times you do ax+b for each variable
    @param (a) a of a x + b
    @param (X) array of m float elements
    @param (b) b of a x + b

    @details this is similar to axpb_simc_c, but works on a variable
    number of vectors (m), which makes it impossible to completely
    unroll the j loop and therefore register-promote X.
    each innermost iteration therefore needs a load, an fma and a store
    instruction, which makes the latency longer and the throughput
    limited by the throughput of store instructions.
    
    $ srun -p big bash -c "./axpb simd_m 8"
    algo = simd_m
    m = 8
    n = 100000000
    flops = 25600000000
    1802238053 CPU clocks, 1397861786 REF clocks, 667250569 ns
    18.022381 CPU clocks/iter, 13.978618 REF clocks/iter, 6.672506 ns/iter
    14.204561 flops/CPU clock, 18.313685 flops/REF clock, 38.366397 GFLOPS

 */
long axpb_simd_m(axpb_options_t opt, float a, float * X_, float b) {
  long m = opt.m;
  assert(m % L == 0);
  assert(opt.bs == 1);
  long n = opt.n;
  floatv * X = (floatv*)X_;
  asm volatile ("# axpb_simd_m: ax+c loop begin");
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < m / L; j++) {
      X[j] = a * X[j] + b;
    }
  }
  asm volatile ("# axpb_simd_m: ax+c loop end");
  return 0;
}

/** 
    @brief repeat x = a x + b for m (variable) vector type (floatv) variables
    by updating a single variable a few times
    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + b
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (b) b of a x + b

 */
long axpb_simd_m_nmn(axpb_options_t opt, float a, float* X_, float b) {
  const int steps_inner = 4;
  long m = opt.m;
  long n = opt.n;
  assert(m % L == 0);
  assert(n % steps_inner == 0);
  assert(opt.bs == 1);
  floatv * X = (floatv*)X_;
  asm volatile ("# axpb_simd_m_nmn: ax+b loop begin");
  for (long i = 0; i < n; i += steps_inner) {
    for (long j = 0; j < m / L; j++) {
      for (long ii = 0; ii < steps_inner; ii++) {
        X[j] = a * X[j] + b;
      }
    }
  }
  asm volatile ("# axpb_simd_m_nmn: ax+b loop end");
  return 0;
}

/** 
    @brief repeat x = a x + c for m (variable) vector type (floatv) variables,
    nv variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+c for each variable
    @param (a) a of a x + c
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (c) c of a x + c

    @details the innsermost two loops look similar to axpb_simd_c

 */
template<int c>
long axpb_simd_m_mnm(axpb_options_t opt, float a, float * X_, float b) {
  long m = opt.m;
  long n = opt.n;
  assert(m % (c * L) == 0);
  assert(opt.bs == 1);
  floatv * X = (floatv*)X_;
  for (long j = 0; j < m / L; j += c) {
    asm volatile ("# axpb_simd_m_mnm<%0>: ax+c inner loop begin" :: "i"(c));
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < c; jj++) {
        X[j+jj] = a * X[j+jj] + b;
      }
    }
    asm volatile ("# axpb_simd_m_mnm<%0>: ax+c inner loop end" :: "i"(c));
  }
  return 0;
}

/** 
    @brief repeat x = a x + b for m (variable) vector type (floatv) variables in parallel,
    nv variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+b for each variable
    @param (a) a of a x + b
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (b) b of a x + b

    @details
    $ srun -p big -n 1 --exclusive bash -c "OMP_PROC_BIND=true OMP_NUM_THREADS=64 ./axpb simd_parallel_m_mnm 8 512 100000000"
    should achieve something like this on the big partition
    4.125885 CPU clocks/iter, 4.708529 REF clocks/iter, 2.247610 ns/iter
    3971.026909 flops/CPU clock, 3479.643183 flops/REF clock, 7289.520058 GFLOPS

 */
template<int c>
long axpb_simd_parallel_m_mnm(axpb_options_t opt, float a, float * X__, float b) {
  long m = opt.m;
  long n = opt.n;
  assert(c % L == 0);
  assert(m % c == 0);
  assert(opt.bs == 1);
  floatv * X_ = (floatv*)X__;
#pragma omp parallel for schedule(static)
  for (long j = 0; j < m / L; j += c) {
    floatv X[c];
    for (long jj = 0; jj < c; jj++) {
      X[jj] = X_[j+jj];
    }
    asm volatile ("# axpb_simd_parallel_m_mnm<%0>: ax+c inner loop begin" :: "i"(c));
    for (long i = 0; i < n; i++) {
      for (long jj = 0; jj < c; jj++) {
        X[jj] = a * X[jj] + b;
      }
    }
    asm volatile ("# axpb_simd_parallel_m_mnm<%0>: ax+c inner loop end" :: "i"(c));
    for (long jj = 0; jj < c; jj++) {
      X_[j+jj] = X[jj];
    }
  }
  return 0;
}

/** 
    @brief repeat x = a x + b for m (variable) vector type (floatv) variables in parallel,
    nv variables at a time

    @param (m) the number of variables updated
    @param (n) the number of times you do ax+b for each variable
    @param (a) a of a x + b
    @param (X) array of m floatv elements (i.e., m * L floats)
    @param (b) b of a x + b

    @details
    $ srun -p big -n 1 --exclusive bash -c "OMP_PROC_BIND=true OMP_NUM_THREADS=64 ./axpb simd_parallel_m_mnm 8 512 100000000"
    should achieve something like this on the big partition
    4.125885 CPU clocks/iter, 4.708529 REF clocks/iter, 2.247610 ns/iter
    3971.026909 flops/CPU clock, 3479.643183 flops/REF clock, 7289.520058 GFLOPS

 */

#if __NVCC__

typedef struct {
  long c0;
  long c1;
} thread_rec_t;

long thread_rec_get_span(thread_rec_t * R, long nthreads) {
  long min_c = -1;
  long max_c = -1;
  for (long i = 0; i < nthreads; i++) {
    if (i == 0 || R[i].c0 < min_c) {
      min_c = R[i].c0;
    }
    if (i == 0 || R[i].c1 > max_c) {
      max_c = R[i].c1;
    }
  }
  return max_c - min_c;
}

__global__ void axpb_dev(axpb_options_t opt, float a, float * X, float b,
                         thread_rec_t * dR) {
  int j = get_thread_id_x();
  if (j < opt.m) {
    long n = opt.n;
    thread_rec_t dr;
    dr.c0 = clock64();
    asm("// axpb_dev loop begins");
    for (long i = 0; i < n; i++) {
      X[j] = a * X[j] + b;
    }
    asm("// axpb_dev loop ends");
    dr.c1 = clock64();
    dR[j] = dr;
  }
}

long axpb_cuda(axpb_options_t opt, float a, float * X, float b) {
  long m = opt.m;
  long bs = opt.bs;
  size_t sz  = sizeof(float) * m;
  size_t rsz = sizeof(thread_rec_t) * m;
  float * X_dev = (float *)dev_malloc(sz);
  thread_rec_t * R     = (thread_rec_t *)malloc(rsz);
  thread_rec_t * R_dev = (thread_rec_t *)dev_malloc(rsz);
  to_dev(X_dev, X, sz);
  long nb = (m + bs - 1) / bs;
  check_launch_error((axpb_dev<<<nb,bs>>>(opt, a, X_dev, b, R_dev)));
  check_api_error(cudaDeviceSynchronize());
  to_host(X, X_dev, sz);
  to_host(R, R_dev, rsz);
  dev_free(X_dev);
  dev_free(R_dev);
  long clocks = thread_rec_get_span(R, m);
  return clocks;
}

#define make_string(c) #c
#define expand(c) make_string(c)

template<int c>
__global__ void axpb_c_dev(axpb_options_t opt,
                           long nthreads,
                           float a, float * X_, float b,
                           thread_rec_t * dR) {
  int tid = get_thread_id_x();
  if (tid < nthreads) {
    assert(c * (tid + 1) <= opt.m);
    long j0 = c * tid;
    thread_rec_t dr;
    long n = opt.n;
    float X[c];
    for (long j = 0; j < c; j++) {
      X[j] = X_[j0 + j];
    }
    dr.c0 = clock64();
    asm("// axpb_c_dev<%0> loop begins" :: "r"(c));
    for (long i = 0; i < n; i++) {
      for (long j = 0; j < c; j++) {
        X[j] = a * X[j] + b;
      }
    }
    asm("// axpb_c_dev<%0> loop ends" :: "r"(c));
    dr.c1 = clock64();
    for (long j = 0; j < c; j++) {
      X_[j0 + j] = X[j];
    }
    dR[tid] = dr;
  }
}

template<int c>
long axpb_cuda_c(axpb_options_t opt, float a, float * X, float b) {
  long m = opt.m;
  long bs = opt.bs;
  assert(m % c == 0);
  long nthreads = m / c;
  size_t sz  = sizeof(float) * nthreads;
  size_t rsz = sizeof(thread_rec_t) * nthreads;
  float * X_dev = (float *)dev_malloc(sz);
  thread_rec_t * R     = (thread_rec_t *)malloc(rsz);
  thread_rec_t * R_dev = (thread_rec_t *)dev_malloc(rsz);
  to_dev(X_dev, X, sz);
  long nb = (nthreads + bs - 1) / bs;
  check_launch_error((axpb_c_dev<c><<<nb,bs>>>(opt, nthreads, a, X_dev, b, R_dev)));
  to_host(X, X_dev, sz);
  to_host(R, R_dev, rsz);
  dev_free(X_dev);
  dev_free(R_dev);
  long clocks = thread_rec_get_span(R, nthreads);
  return clocks;
}

#endif  /* __NVCC__ */

typedef long (*axpb_fun_t)(axpb_options_t, float a, float* X, float b);

typedef struct {
  axpb_fun_t t[algo_invalid];
} axpb_funs_t;

template<int c>
axpb_funs_t make_axpb_funs_c() {
  axpb_funs_t funs;
  funs.t[algo_scalar]     = axpb_scalar;
  funs.t[algo_simd]       = axpb_simd;
  funs.t[algo_simd_c]     = axpb_simd_c<c>;
  funs.t[algo_simd_m]     = axpb_simd_m;
  funs.t[algo_simd_m_nmn] = axpb_simd_m_nmn;
  funs.t[algo_simd_m_mnm] = axpb_simd_m_mnm<c>;
  funs.t[algo_simd_parallel_m_mnm] = axpb_simd_parallel_m_mnm<c>;
#if __NVCC__
  funs.t[algo_cuda]       = axpb_cuda;
  funs.t[algo_cuda_c]     = axpb_cuda_c<c>;
#endif
  return funs;
};

#define aac(c) make_axpb_funs_c<c>()
axpb_funs_t axpb_funs_table[] = {
  // avoid aac(0) to avoid compiler warning
  aac(1), aac(1), aac(2), aac(3), aac(4),
  aac(5), aac(6), aac(7), aac(8), aac(9), 
  aac(10), aac(11), aac(12), aac(13), aac(14),
  aac(15), aac(16), aac(17), aac(18), aac(19), 
  aac(20), aac(21), aac(22), aac(23), aac(24),
  aac(25), aac(26), aac(27), aac(28), aac(29), 
  aac(30), aac(31), aac(32), aac(33), aac(34),
  aac(35), aac(36), aac(37), aac(38), aac(39), 
  aac(40), aac(41), aac(42), aac(43), aac(44),
  aac(45), aac(46), aac(47), aac(48), aac(49), 
};

long axpb(axpb_options_t opt, float a, float* X, float b) {
  long table_sz = sizeof(axpb_funs_table) / sizeof(axpb_funs_table[0]);
  long c = opt.c;
  assert(c > 0);
  assert(c < table_sz);
  axpb_fun_t f = axpb_funs_table[c].t[opt.algo];
  long clocks = f(opt, a, X, b);
  return clocks;
}

typedef struct {
  algo_t a;
  const char * name;
} algo_table_entry_t;

typedef struct {
  algo_table_entry_t t[algo_invalid];
} algo_table_t;

static algo_table_t algo_table = {
  {
    { algo_scalar, "scalar" },
    { algo_simd,   "simd" },
    { algo_simd_c, "simd_c" },
    { algo_simd_m, "simd_m" },
    { algo_simd_m_nmn, "simd_m_nmn" },
    { algo_simd_m_mnm, "simd_m_mnm" },
    { algo_simd_parallel_m_mnm, "simd_parallel_m_mnm" },
    { algo_cuda, "cuda" },
    { algo_cuda_c, "cuda_c" },
  }
};

algo_t parse_algo(const char * s) {
  for (int i = 0; i < (int)algo_invalid; i++) {
    algo_table_entry_t e = algo_table.t[i];
    if (strcmp(e.name, s) == 0) {
      return e.a;
    }
  }
  fprintf(stderr, "%s:%d:parse_algo: invalid algo %s\n",
          __FILE__, __LINE__, s);
  return algo_invalid;
}

static axpb_options_t default_opts() {
  axpb_options_t opt = {
    .algo_str = "scalar",
    .algo = algo_invalid,
    .bs = 1,
    .w = 32,
    .c = 1,
    .m = 1,
    .n = 1000000,
    .seed = 76843802738543,
    .n_elems_to_show = 1,
    .help = 0,
    .error = 0,
  };
  return opt;
}

static void usage(const char * prog) {
  axpb_options_t o = default_opts();
  fprintf(stderr,
          "usage:\n"
          "\n"
          "  %s [options ...]\n"
          "\n"
          "options:\n"
          "  --help                  show this help\n"
          "  -a,--algo A             use algorithm A (scalar,simd,simd_c,simd_m,simd_mnm,simd_nmn,simd_parallel_mnm,cuda) [%s]\n"
          "  -b,--cuda-block-size N  set cuda block size to N [%ld]\n"
          "  -w,--active-threader-per-warp N  set active threads per warp to N [%ld]\n"
          "  -c,--concurrent-vars N  concurrently update N floats [%ld]\n"
          "  -m,--vars N             update N floats [%ld]\n"
          "  -n,--n N                update each float variable N times [%ld]\n"
          "  -s,--seed N             set random seed to N [%ld]\n"
          ,
          prog,
          o.algo_str,
          o.bs, o.w, o.c, o.m, o.n, o.seed
          );
}

/** 
    @brief command line options
*/
static struct option long_options[] = {
  {"algo",            required_argument, 0, 'a' },
  {"cuda-block-size", required_argument, 0, 'b' },
  {"active-threads-per-warp", required_argument, 0, 'w' },
  {"concurrent-vars", required_argument, 0, 'c' },
  {"vars",            required_argument, 0, 'm' },
  {"n",               required_argument, 0, 'n' },
  {"seed",            required_argument, 0, 's' },
  {"help",            no_argument,       0, 'h'},
  {0,                 0,                 0,  0 }
};

static long make_multiple(long a, long q) {
  if (a == 0) a = 1;
  long b = a + q - 1;
  return b - b % q;
}

/**

 */
static axpb_options_t parse_args(int argc, char ** argv) {
  char * prog = argv[0];
  axpb_options_t opt = default_opts();
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "a:b:w:c:m:n:s:h",
                        long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 0:
      {
        const char * o = long_options[option_index].name;
        fprintf(stderr,
                "bug:%s:%d: should handle option %s\n",
                __FILE__, __LINE__, o);
        opt.error = 1;
        return opt;
      }
    case 'a':
      opt.algo_str = strdup(optarg);
      break;
    case 'b':
      opt.bs = atol(optarg);
      break;
    case 'w':
      opt.w = atol(optarg);
      break;
    case 'c':
      opt.c = atol(optarg);
      break;
    case 'm':
      opt.m = atol(optarg);
      break;
    case 'n':
      opt.n = atol(optarg);
      break;
    case 's':
      opt.seed = atol(optarg);
      break;
    case 'h':
      opt.help = 1;
      break;
    default: /* '?' */
      usage(prog);
      opt.error = 1;
      return opt;
    }
  }
  opt.algo = parse_algo(opt.algo_str);
  if (opt.algo == algo_invalid) {
    opt.error = 1;
    return opt;
  }
  switch (opt.algo) {
  case algo_scalar:
    opt.c = 1;
    opt.m = 1;
    opt.bs = 1;
    break;
  case algo_simd:
    opt.c = 1;
    opt.m = L;
    opt.bs = 1;
    break;
  case algo_simd_c:
    opt.m = opt.c * L;
    opt.bs = 1;
    break;
  case algo_simd_m:
    opt.c = 1;
    opt.m = make_multiple(opt.m, L);
    opt.bs = 1;
    break;
  case algo_simd_m_mnm:
  case algo_simd_m_nmn:
    opt.m = make_multiple(opt.m, opt.c * L);
    opt.bs = 1;
    break;
  case algo_cuda:
    opt.c = 1;
    opt.m = opt.bs;
    break;
  case algo_cuda_c:
    opt.m = make_multiple(opt.m, opt.c * opt.bs);
    break;
  default:
    // other algorithms can update the given number of parameters
    break;
  }
  if (opt.n_elems_to_show >= opt.m) {
    opt.n_elems_to_show = opt.m;
  }
  return opt;
}

/**
   @brief main function
   @param (argc) the number of command line args
   @param (argv) command line args
  */
int main(int argc, char ** argv) {
  axpb_options_t opt = parse_args(argc, argv);
  if (opt.help || opt.error) {
    usage(argv[0]);
    exit(opt.error);
  }

  printf(" algo = %s\n", opt.algo_str);
  printf("    bs = %ld (cuda block size)\n", opt.bs);
  printf("    c = %ld (the number of variables to update in the inner loop)\n", opt.c);
  printf("    m = %ld (the number of FP numbers to update)\n", opt.m);
  printf("    n = %ld (the number of times to update each variable)\n", opt.n);
  printf("    L = %d (SIMD lanes on the CPU)\n", L);
  fflush(stdout);
  
  unsigned short rg[3] = {
    (unsigned short)(opt.seed >> 16),
    (unsigned short)(opt.seed >> 8),
    (unsigned short)(opt.seed)
  };

  float a = erand48(rg);
  float b = erand48(rg);
  float * X = (float *)aligned_alloc(valign, sizeof(float) * opt.m);
  //float * X = (float *)malloc(sizeof(float) * opt.m);
  if (!X) err(1, "malloc");
  for (long j = 0; j < opt.m; j++) { X[j] = j; }
  long flops = 2 * opt.m * opt.n;

  clock_counters_t cc = mk_clock_counters();
  clocks_t c0 = clock_counters_get(cc);
  long long dc = axpb(opt, a, X, b);
  clocks_t c1 = clock_counters_get(cc);
  long cpu  = c1.cpu_clock - c0.cpu_clock;
  long ref  = c1.ref_clock - c0.ref_clock;
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
  printf("\n");
  printf("%f nsec       for performing x=ax+b for %ld variables once\n", wall / (double)opt.n, opt.m);
  printf("%f ref clocks for performing x=ax+b for %ld variables once\n", ref / (double)opt.n, opt.m);
  if (cpu) {
    printf("%f cpu clocks for performing x=ax+b for %ld variables once\n", cpu / (double)opt.n, opt.m);
  } else {
    printf("-------- cpu clocks for performing x=ax+b for %ld variables once\n", opt.m);
  }
  printf("\n");
  printf("%f flops/nsec\n",      flops / (double)wall);
  printf("%f flops/ref clock\n", flops / (double)ref);
  if (cpu) {
    printf("%f flops/cpu clock\n", flops / (double)cpu);
  } else {
    printf("-------- flops/cpu clock\n");
  }
  for (int i = 0; i < opt.n_elems_to_show; i++) {
    long idx = nrand48(rg) % opt.m;
    printf("X[%ld] = %f\n", idx, X[idx]);
  }
  return 0;
}


