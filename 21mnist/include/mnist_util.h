/**
   @file mnist_util.h
   @brief MNIST utility functions/classes
 */
#pragma once
#include <assert.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#if 0
#include <ieee754.h>
#endif

#if __CUDACC__
#include "cuda_util.h"
#else
/** 
    @brief for source compatibility with cuda, define __global__ as empty
*/
#define __global__ 
/** 
    @brief for source compatibility with cuda, define __device__ as empty
*/
#define __device__ 
/** 
    @brief for source compatibility with cuda, define __host__ as empty
*/
#define __host__ 
#endif

/**
   @brief type of array index (either int or long)
 */
typedef int idx_t;
#if !defined(real_type)
/**
   @brief real_type is float if not supplied
 */
#define real_type float
#endif
/**
   @brief type of array elements (may be changed by -Dreal_type=...)
 */
typedef real_type real;

/**
   @brief exit(1) (just for setting breakpoints)
 */
static void bail() {
  exit(1);
}

/**
   @brief signal a fatal error when a CUDA GPU function gets called when it is not compiled 
   by CUDA-enabled compiler setting
 */
static void err_cuda_code_non_cuda_compiler_(const char * file, int line, const char * algo_s) {
  fprintf(stderr,
          "error:%s:%d: a supposedly CUDA function (%s) compiled by non-CUDA compiler and gets called\n",
          file, line, algo_s);
  bail();
}

/**
   @brief signal a fatal error when a CUDA GPU function gets called when it is not compiled 
   by CUDA-enabled compiler setting
 */
#define err_cuda_code_non_cuda_compiler(algo_s) err_cuda_code_non_cuda_compiler_(__FILE__, __LINE__, algo_s)

/**
   @brief an enumeration of implemented algorithms
   @details add your algorithm in this enum
 */
typedef enum {
  algo_cpu_base,
  algo_cuda_base,
  /* add your new algorithm here (name it arbitrarily) */
  /* algo_cpu_simd? */
  /* algo_cpu_omp */
  /* algo_cpu_simd_omp? */
  /* algo_cpu_fast? */
  /* algo_cuda_fast? */
  /* algo_cpu_super_fast? */
  /* algo_cuda_super_fast? */
  
  algo_invalid,
} algo_t;

/**
   @brief convert a string to an algorithm enum 
   @details when you add your algorithm, change this function 
   so that it recognizes your algorithm
 */
static algo_t parse_algo(const char * s) {
  if (strcmp(s, "cpu_base") == 0) {
    return algo_cpu_base;
  } else if (strcmp(s, "cuda_base") == 0) {
    return algo_cuda_base;
    /* add cases here to handle your algorithms
       } else if (strcmp(s, "cpu_fast") == 0) {
       return algo_cpu_fast;
    */
  } else {
    return algo_invalid;
  }
}

/**
   @brief return 1 if the algorithm name (s) or its
   enum value (a) is a CUDA algorithm 
   @details when you add your algorithm, you may need to 
   change this function so that it correctly recognizes 
   whether it is a CUDA algorithm or not
   currently, it considers all and only strings starting 
   with "cuda" to be a CUDA algorithm.
   for a cuda algorithm, the program transfers initial weights
   and training data to gpu.  weights stay on GPU until
   the program finishes.  
  */
static int algo_is_cuda(const char * s, algo_t a) {
  (void)a;
  if (strncmp(s, "cuda", 4) == 0) {
    return 1;
  } else { 
    return 0;
  }
}

/**
   @brief command line options
*/
struct cmdline_opt {
  int verbose;                  /**< verbosity */
  const char * data_dir;        /**< data directory */
  real lr;                      /**< learning rate */
  long epochs;                  /**< number of epochs to process */
  idx_t batch_size;             /**< batch size */
  idx_t train_data_size;        /**< training data size */
  idx_t test_data_size;         /**< test data size */
  long log_interval;            /**< show progress every this batches */
  long weight_seed;             /**< random seed to initialize weights and dropout */
  long dropout_seed_1;          /**< random seed to determine which elements to drop dropout layer 1 */
  long dropout_seed_2;          /**< random seed to determine which elements to drop dropout layer 2 */
  int grad_dbg;                 /**< 1 if we debug gradient */
  const char * algo_s;          /**< string passed to --algo */
  algo_t algo;                  /**< parse_algo(algo_s)  */
  int cuda_algo;                 /**< 1 if this is a CUDA algorithm  */
  const char * log;             /**< log file name */
  int help;                     /**< 1 if -h,--help is given  */
  int error;                    /**< set to one if any option is invalid */
  /**
     @brief initialize a command line object with default vaules
   */
  cmdline_opt() {
    verbose = 1;
    data_dir = "data";
    lr = 1.0;
    epochs = 14;
    batch_size = MAX_BATCH_SIZE;
    train_data_size = -1;
    test_data_size = -1;
    log_interval = 10;
    weight_seed  = 45678901234523L;
    dropout_seed_1 = 56789012345234L;
    dropout_seed_2 = 67890123452345L;
    grad_dbg = 0;
#if __CUDACC__    
    algo_s = "cuda_base";
    cuda_algo = 1;
#else
    algo_s = "cpu_base";
    cuda_algo = 0;
#endif
    algo = algo_invalid;
    log = "mnist.log";
    help = 0;
    error = 0;
  }
};

/**
   @brief command line options for getopt
*/
static struct option long_options[] = {
  {"verbose",           required_argument, 0, 'v' },
  {"data-dir",          required_argument, 0, 'd' },
  {"lr",                required_argument, 0, 'l' },
  {"epochs",            required_argument, 0, 'm' },
  {"batch-size",        required_argument, 0, 'b' },
  {"train-data-size",   required_argument, 0,  0  },
  {"test-data-size",    required_argument, 0,  0  },
  {"log-interval",      required_argument, 0,  0  },
  {"weight-seed",       required_argument, 0,  0  },
  {"dropout-seed-1",    required_argument, 0,  0  },
  {"dropout-seed-2",    required_argument, 0,  0  },
  {"grad-dbg",          required_argument, 0,  0  },
  {"algo",              required_argument, 0, 'a' },
  {"log",               required_argument, 0,  0  },
  {"help",              required_argument, 0, 'h' },
  {0,                   0,                 0,  0  }
};

/**
   @brief show usage
*/
static void usage(const char * prog) {
  cmdline_opt o;
  fprintf(stderr,
          "usage:\n"
          "\n"
          "%s [options]\n"
          "\n"
          " -d,--data-dir D : read data from D [%s]\n"
          " -m,--epochs N : run N epochs [%ld]\n"
          " -b,--batch-size N : set batch size to N [%d]\n"
          " -a,--algo ALGORITHM : set the algorithm (implementation) used [%s]\n"
          " -v,--verbose L : set verbosity level to L [%d]\n"
          " -l,--lr ETA : set learning rate to ETA [%f]\n"
          " --train-data-size N : set training data size to N [%d]\n"
          " --test-data-size N : set test data size to N [%d]\n"
          " --log-interval N : show progress every N batches [%ld]\n"
          " --dropout-seed-1 S : set seed for dropout layer 1 to S [%ld]\n"
          " --dropout-seed-2 S : set seed for dropout layer 2 to S [%ld]\n"
          " --weight-seed S : set seed for initial weights to S [%ld]\n"
          " --grad-dbg 0/1 : debug gradient computation [%d]\n"
          " --log FILE : write log to FILE [%s]\n"
          " -h,--help\n",
          prog,
          o.data_dir,
          o.epochs,
          o.batch_size,
          o.algo_s,
          o.verbose,
          o.lr,
          o.train_data_size,
          o.test_data_size,
          o.log_interval,
          o.dropout_seed_1,
          o.dropout_seed_2,
          o.weight_seed,
          o.grad_dbg,
          o.log
          );
  exit(1);
}

/**
   @brief parse command line args and make a command line object
*/
static cmdline_opt parse_args(int argc, char ** argv) {
  char * prog = argv[0];
  cmdline_opt opt;
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv,
                        "a:b:d:l:m:v:h", long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 0:
      {
        const char * o = long_options[option_index].name;
        if (strcmp(o, "train-data-size") == 0) {
          opt.train_data_size = atol(optarg);
        } else if (strcmp(o, "test-data-size") == 0) {
          opt.test_data_size = atol(optarg);
        } else if (strcmp(o, "log-interval") == 0) {
          opt.log_interval = atol(optarg);
        } else if (strcmp(o, "weight-seed") == 0) {
          opt.weight_seed = atol(optarg);
        } else if (strcmp(o, "dropout-seed-1") == 0) {
          opt.dropout_seed_1 = atol(optarg);
        } else if (strcmp(o, "dropout-seed-2") == 0) {
          opt.dropout_seed_2 = atol(optarg);
        } else if (strcmp(o, "grad-dbg") == 0) {
          opt.grad_dbg = atoi(optarg);
        } else if (strcmp(o, "log") == 0) {
          opt.log = strdup(optarg);
        } else {
          fprintf(stderr,
                  "bug:%s:%d: should handle option %s\n",
                  __FILE__, __LINE__, o);
          opt.error = 1;
          return opt;
        }
      }
      break;
    case 'v':
      opt.verbose = atoi(optarg);
      break;
    case 'd':
      opt.data_dir = strdup(optarg);
      break;
    case 'b':
      opt.batch_size = atoi(optarg);
      break;
    case 'a':
      opt.algo_s = strdup(optarg);
      break;
    case 'l':
      opt.lr = atof(optarg);
      break;
    case 'm':
      opt.epochs = atol(optarg);
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
  if (opt.batch_size > MAX_BATCH_SIZE) {
    fprintf(stderr, "error: cannot specify --batch-sz (%d) > MAX_BATCH_SIZE (%d)\n",
            opt.batch_size, MAX_BATCH_SIZE);
    opt.error = 1;
    return opt;
  }
  opt.algo = parse_algo(opt.algo_s);
  if (opt.algo == algo_invalid) {
    fprintf(stderr, "error: invalid algorithm (%s)\n", opt.algo_s);
    opt.error = 1;
    return opt;
  }
  opt.cuda_algo = algo_is_cuda(opt.algo_s, opt.algo);
#if !__CUDACC__
  if (opt.cuda_algo) {
    fprintf(stderr, "error: --cuda-base 1 allowed only with nvcc\n");
    opt.error = 1;
    return opt;
  }
#endif
  return opt;
}

/**
   @brief maximum of two reals
*/
__device__ __host__
static real max_r(real a, real b) {
  return (a < b ? b : a);
}

/**
   @brief miniumum of two reals
*/
__device__ __host__
static real min_r(real a, real b) {
  return (a < b ? a : b);
}

/**
   @brief maximum of two ints
*/
__device__ __host__
static idx_t max_i(idx_t a, idx_t b) {
  return (a < b ? b : a);
}

/**
   @brief minimum of two ints
*/
__device__ __host__
static idx_t min_i(idx_t a, idx_t b) {
  return (a < b ? a : b);
}

/**
   @brief timestamp 
*/
struct tsc_t {
  long ns;                      /**< nano seconds  */
};

/**
   @brief get timestamp (currently just wallclock time in nano seconds)
*/
static tsc_t get_tsc() {
  struct timespec ts[1];
  tsc_t t;
  if (clock_gettime(CLOCK_REALTIME, ts) == -1) {
    perror("clock_gettime"); bail();
  }
  t.ns = ts->tv_sec * 1000000000L + ts->tv_nsec;
  return t;
}

/**
   @brief pseudo random number generator
   crafted from man erand48 + libc source
*/
struct rnd_gen_t {
  uint64_t x;                   /**< random number state */
  /**
     @brief set next state
   */
  __device__ __host__
  void next() {
    const uint64_t __a = 0x5deece66dull;
    const uint64_t __c = 0xb;
    const uint64_t mask = (1UL << 48) - 1;
    x = (x * __a + __c) & mask;
  }
  /**
     @brief return a random number between 0 and 1
   */
  __device__ __host__
  double rand01() {
#if 1
    next();
    return x / (double)(1UL << 48);
#else
    union ieee754_double temp;
    /* Compute next state.  */
    next();
    /* Construct a positive double with the 48 random bits distributed over
       its fractional part so the resulting FP number is [0.0,1.0).  */
    temp.ieee.negative = 0;
    temp.ieee.exponent = IEEE754_DOUBLE_BIAS;
    temp.ieee.mantissa0 = (x >> 28) & ((1UL << 20) - 1); /* 20 bit */
    temp.ieee.mantissa1 = (x & ((1UL << 28) - 1)) << 4;
    /* Please note the lower 4 bits of mantissa1 are always 0.  */
    return temp.d - 1.0;
#endif
  }
  /**
     @brief return a long between 0 to 2^31 - 1
   */
  __device__ __host__
  long randi32() {
    /* Compute next state.  */
    next();
    /* Store the result.  */
    return (x >> 17) & ((1UL << 31) - 1);
  }
  /**
     @brief return a real between a and b
   */
  __device__ __host__
  double rand(double a, double b) {
    return a + (b - a) * rand01();
  }
  /**
     @brief return a long between a and b
   */
  __device__ __host__
  long randi(long a, long b) {
    return a + randi32() % (b - a);
  }
  /**
     @brief generate a random number from a normal distribution
     of mean=mu and standard deviation=sigma.
     @details see
     https://en.wikipedia.org/wiki/Normal_distribution
     for how the following method works 
  */
  __device__ __host__
  real rand_normal(real mu, real sigma) {
    real u = rand01();
    real v = rand01();
    real x = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
    return mu + x * sigma;
  }
  /**
     @brief return the current state of the generator
  */
  __device__ __host__
  long get_state() {
    return x;
  }
  /**
     @brief set state of the generator
  */
  __device__ __host__
  void seed(uint64_t y) {
    x = y;
  }
};

/**
   @brief if the algorithm is a CUDA algorithm, allocate a device shadow 
   of this object and set dev field of this and all subobjects. otherwise
   it sets all dev fields to null.
   @sa set_dev
   @sa del_dev
*/
template<typename T>
T * make_dev(T * layer, int cuda_algo) {
#if __CUDACC__
  assert(layer->dev == 0);
  if (cuda_algo) {
    T * dev_ = (T*)dev_malloc(sizeof(T));
    layer->set_dev(dev_);
    return dev_;
  } else {
    return 0;
  }
#else
  (void)cuda_algo;
  (void)layer;
  return 0;
#endif
}

/**
   @brief make a copy of this 
   @details if this object has a device pointer, the copy will have
   a device pointer too, but its contents are NOT copied
*/
template<typename T>
T* make_copy(T * layer, int cuda_algo) {
  T * c = new T(*layer);
#if __CUDACC__
  c->dev = 0;
  make_dev(c, cuda_algo);
#else
  (void)cuda_algo;
#endif
  return c;
}
/**
   @brief if the algorithm is a CUDA algorithm, dev field must not
   be null and deallocate it.
   @sa make_dev
   @sa set_dev
*/
template<typename T>
void del_dev(T * layer, int cuda_algo) {
#if __CUDACC__
  if (cuda_algo) {
    if (layer->dev) {
      dev_free(layer->dev);
      layer->dev = 0;
    }
  }
#else
  (void)cuda_algo;
  (void)layer;
#endif
}
/**
   @brief if the algorithm is a CUDA algorithm, dev field must
   not be null and send the host data to the device memory
*/
template<typename T>
void to_dev(T * layer, int cuda_algo) {
#if __CUDACC__
  if (cuda_algo) {
    T* dev_ = layer->dev;
    if (!dev_) {
      dev_ = make_dev(layer, cuda_algo);
    }
    ::to_dev(dev_, layer, sizeof(T));
  }
#else
  (void)cuda_algo;
  (void)layer;
#endif
}
/**
   @brief if the algorithm is a CUDA algorithm, dev field must
   not be null and send the device data to the host memory
*/
template<typename T>
void to_host(T * layer, int cuda_algo) {
#if __CUDACC__
  if (cuda_algo) {
    T * dev_ = layer->dev;
    assert(dev_);
    ::to_host(layer, dev_, sizeof(T));
  }
#else
  (void)cuda_algo;
  (void)layer;
#endif
}




/**
   @brief show various errors 
   @param (gx_gx) ∂L/∂x・∂L/∂x
   @param (dx_dx) dx・dx
   @param (gx_dx) ∂L/∂x・dx
   @param (gw_gw) ∂L/∂w・∂L/∂w
   @param (dw_dw) dw・dw
   @param (gw_dw) ∂L/∂w・dw
   @param (L_minus) L(w-dw,x-dx)
   @param (L) L(w,x)
   @param (L_plus) L(w+dw,x+dx)
 */
__attribute__((unused))
static real show_error(/* double gx_gx, double dx_dx, */ double gx_dx,
                       /* double gw_gw, double dw_dw, */ double gw_dw,
                       double L_minus, double L, double L_plus) {
  //printf("|∂L/∂x|   = %.9f\n", sqrt(gx_gx));
  //printf("|dx|      = %.9f\n", sqrt(dx_dx));
  printf("∂L/∂x・dx = %.9f\n", gx_dx);
  //printf("|∂L/∂w|   = %.9f\n", sqrt(gw_gw));
  //printf("|dw|      = %.9f\n", sqrt(dw_dw));
  printf("∂L/∂w・dw = %.9f\n", gw_dw);
  printf("L- = %.9f\n", L_minus);
  printf("L  = %.9f\n", L);
  printf("L+ = %.9f\n", L_plus);
  double dL = L_plus - L_minus;
  double A = gx_dx + gw_dw;
  double B = dL;
  double e = (A == B ? 0.0 : fabs(A - B) / max_r(fabs(A), fabs(B)));
  printf("A = ∂L/∂x・dx + ∂L/∂w・dw = %.9f\n", gx_dx + gw_dw);
  printf("B = ΔL = %.9f\n", dL);
  printf("relative error = |A-B|/max(|A|,|B|) = %.9f\n", e);
  return e;
}

/**
   @brief logging object
 */
struct logger {
  cmdline_opt opt;              /**< command line options */
  FILE * log_fp;                /**< log file object */
  tsc_t t0;                     /**< the start time stamp */
  /**
     @brief return the current time string like "Wed Jun 30 21:49:08 1993"
   */
  char * cur_time_str() {
    time_t t = time(NULL);
    char * time_s = ctime(&t);
    if (!time_s) {
      perror("ctime_r");
      exit(EXIT_FAILURE);
    }
    int len = strlen(time_s);
    time_s[len-1] = 0;
    return time_s;
  }
  /**
     @brief write a formatted string to the log and may be to standard out
     @param (level) the level of this entry. if opt.verbose>=level, then 
     the string will be output to the standard out (in addition to the log file)
     @param (format) the printf-like format string
   */
  int log(int level, const char * format, ...) {
    tsc_t t = get_tsc();
    long dt = t.ns - t0.ns;
    if (log_fp) {
      va_list ap;
      fprintf(log_fp, "%ld: ", dt);
      va_start(ap, format);
      vfprintf(log_fp, format, ap);
      va_end(ap);
      fprintf(log_fp, "\n");
    }
    if (opt.verbose>=level) {
      va_list ap;
      fprintf(stdout, "%ld: ", dt);
      va_start(ap, format);
      vfprintf(stdout, format, ap);
      va_end(ap);
      fprintf(stdout, "\n");
      fflush(stdout);
    }
    return 1;
  }
  /**
     @brief open a log file and start logging
     @param (opt) command line option
   */
  int start_log(cmdline_opt opt) {
    this->opt = opt;
    log_fp = fopen(opt.log, "wb");
    if (!log_fp) { perror("fopen"); exit(1); }
    t0 = get_tsc();
    log(2, "open a log %s", cur_time_str());
    log_opt();
    log_host();
    log_envs();
    return 1;
  }
  /**
     @brief end logging and close the log file
   */
  int end_log() {
    if (log_fp) {
      log(2, "close a log %s", cur_time_str());
      fclose(log_fp);
      log_fp = 0;
    }
    return 1;
  }
  /**
     @brief log command line options to the log file for the record
   */
  int log_opt() {
    log(2, "verbose=%d", opt.verbose);
    log(2, "data-dir=%s", opt.data_dir);
    log(2, "lr=%f", opt.lr);
    log(2, "epochs=%ld", opt.epochs);
    log(2, "batch-size=%d", opt.batch_size);
    log(2, "train-data-size=%d", opt.train_data_size);
    log(2, "test-data-size=%ld", opt.test_data_size);
    log(2, "weight-seed=%ld", opt.weight_seed);
    log(2, "dropout-seed-1=%ld", opt.dropout_seed_1);
    log(2, "dropout-seed-2=%ld", opt.dropout_seed_2);
    log(2, "grad-dbg=%d", opt.grad_dbg);
    log(2, "algo=%d", opt.algo);
    log(2, "algo_s=%s", opt.algo_s);       // added
    log(2, "cuda_algo=%d", opt.cuda_algo); // added
    log(2, "log=%s", opt.log);
    return 1;
  }
  /**
     @brief log hostname for the record
   */
#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 64
#endif
  int log_host() {
    char name[HOST_NAME_MAX+1];
    name[0] = 0;
    gethostname(name, sizeof(name));
    log(2, "host=%s", name);
    return 1;
  }
  /**
     @brief log an environment var for the record
   */
  int log_env(const char * var) {
    char * s = getenv(var);
    if (s) {
      log(2, "%s=%s", var, s);
    } else {
      log(2, "%s undefined", var);
    }
    return 1;
  }
  /**
     @brief log some environment vars for the record
   */
  int log_envs() {
    log_env("USER");
    log_env("PWD");
    log_env("SLURM_SUBMIT_DIR");
    log_env("SLURM_SUBMIT_HOST");
    log_env("SLURM_JOB_NAME");
    log_env("SLURM_JOB_CPUS_PER_NODE");
    log_env("SLURM_NTASKS");
    log_env("SLURM_NPROCS");
    log_env("SLURM_JOB_ID");
    log_env("SLURM_JOBID");
    log_env("SLURM_NNODES");
    log_env("SLURM_JOB_NUM_NODES");
    log_env("SLURM_NODELIST");
    log_env("SLURM_JOB_PARTITION");
    log_env("SLURM_TASKS_PER_NODE");
    log_env("SLURM_JOB_NODELIST");
    log_env("CUDA_VISIBLE_DEVICES");
    log_env("GPU_DEVICE_ORDINAL");
    log_env("SLURM_CPUS_ON_NODE");
    log_env("SLURM_TASK_PID");
    log_env("SLURM_NODEID");
    log_env("SLURM_PROCID");
    log_env("SLURM_LOCALID");
    log_env("SLURM_JOB_UID");
    log_env("SLURM_JOB_USER");
    log_env("SLURM_JOB_GID");
    log_env("SLURMD_NODENAME");
    return 1;                   /* OK */
  }
  /**
     @brief log the start of a function (f) 
   */
  void log_start_fun_(const char * f) {
    log(4, "%s: starts", f);
  }
  /**
     @brief log the end of a function (f) 
   */
  void log_end_fun_(const char * f, tsc_t t0, tsc_t t1) {
    log(4, "%s: ends. took %ld nsec", f, t1.ns - t0.ns);
  }
};

/**
   @brief log the start of the current function
   @details just log_start_fun(lgr) and you get the caller's function
   name to the log
  */
#define log_start_fun(lgr) lgr->log_start_fun_(__PRETTY_FUNCTION__)
/**
   @brief log the end of the current function
   @details just log_end_fun(lgr, t0, t1) and you get the caller's function
   name to the log along with its execution time
  */
#define log_end_fun(lgr, t0, t1)   lgr->log_end_fun_(__PRETTY_FUNCTION__, t0, t1)

/**
   @brief entry point
 */
int mnist_util_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  if (opt.error || opt.help) usage(argv[0]);
  bail();
  min_i(1,2);
  max_i(3,4);
  min_r(1.2,3.4);
  max_r(5.6,7.8);
  return 0;
}

