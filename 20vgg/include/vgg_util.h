/**
   @file vgg_util.h
   @brief VGG utility functions/classes
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
#include <ieee754.h>

#if __NVCC__
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
   @brief signal a fatal error when a gpu algorithm gets called for cpu only compilation
 */
static void err_gpu_algo_no_gpu_(const char * file, int line, const char * algo_s) {
  fprintf(stderr,
          "error:%s:%d: a GPU algorithm (%s) specified for CPU-only compilation\n",
          file, line, algo_s);
  bail();
}

/**
   @brief signal a fatal error when a gpu algorithm gets called for cpu only compilation
 */
#define err_gpu_algo_no_gpu(algo_s) err_gpu_algo_no_gpu_(__FILE__, __LINE__, algo_s)

/**
   @brief an enumeration of implemented algorithms
   @details add your algorithm in this enum
 */
typedef enum {
  algo_cpu_base,
  algo_gpu_base,
  /* add your new algorithm here (name it arbitrarily) */
  /* algo_cpu_simd? */
  /* algo_cpu_omp */
  /* algo_cpu_simd_omp? */
  /* algo_cpu_super_fast? */
  /* algo_gpu_super_fast? */
  
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
  } else if (strcmp(s, "gpu_base") == 0) {
    return algo_gpu_base;
    /* add cases here to handle your algorithms
       } else if (strcmp(s, "gpu_base") == 0) {
       return algo_gpu_base;
    */
  } else {
    return algo_invalid;
  }
}

/**
   @brief return 1 if the algorithm name (s) or its
   enum value (a) is a gpu algorithm 
   @details when you add your algorithm, you may need to 
   change this function so that it correctly recognizes 
   whether it is a gpu algorithm or not
   currently, it considers all and only strings starting 
   with "gpu" to be a gpu algorithm.
   for a gpu algorithm, the program transfers initial weights
   and training data to gpu.  weights stay on GPU until
   the program finishes.  
  */
int algo_is_gpu(const char * s, algo_t a) {
  (void)a;
  if (strncmp(s, "gpu", 3) == 0) {
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
  const char * cifar_data;      /**< data file */
  const char * cifar_data_dump; /**< prefix of data dump */
  idx_t batch_sz;               /**< batch size */
  real learnrate;               /**< learning rate */
  long iters;                   /**< number of batches to process */
  long partial_data;             /**< choose this number of data in the file (0 for all) */
  int single_batch;             /**< 1 if we choose the same samples every iteration */
  int dropout;                  /**< 1 if we use dropout */
  double validate_ratio;        /**< ratio of data held out for validation */
  double validate_interval;     /**< relative validation frequency */
  long sample_seed;             /**< random seed to draw samples */
  long weight_seed;             /**< random seed to initialize weights and dropout */
  long dropout_seed;            /**< random seed to determine dropout */
  long partial_data_seed;       /**< random seed to determine which data in the file are used for training/validation */
  int grad_dbg;                 /**< 1 if we debug gradient */
  const char * algo_s;          /**< string passed to --algo */
  algo_t algo;                  /**< parse_algo(algo_s)  */
  int gpu_algo;                 /**< 1 if this is a GPU algorithm  */
  const char * log;             /**< log file name */
  int help;                     /**< 1 if -h,--help is given  */
  int error;                    /**< set to one if any option is invalid */
  /**
     @brief initialize a command line object with default vaules
   */
  cmdline_opt() {
    verbose = 1;
    cifar_data = "data/cifar-10-batches-bin/data_batch_1.bin";
    cifar_data_dump = 0; //"cifar-10-imgs/i"
    batch_sz = MAX_BATCH_SIZE;
    learnrate = 1.0e-2;
    iters = 20;
    partial_data = 0;
    single_batch = 0;
    dropout = 0;
    validate_ratio = 0.1;
    validate_interval = 5.0;
    sample_seed  = 34567890123452L;
    weight_seed  = 45678901234523L;
    dropout_seed = 56789012345234L;
    partial_data_seed = 67890123452345L;
    grad_dbg = 0;
#if __NVCC__    
    algo_s = "gpu_base";
    gpu_algo = 1;
#else
    algo_s = "cpu_base";
    gpu_algo = 0;
#endif
    algo = algo_invalid;
    log = "vgg.log";
    help = 0;
    error = 0;
  }
};

/**
   @brief command line options for getopt
*/
static struct option long_options[] = {
  {"batch_sz",          required_argument, 0, 'b' },
  {"iters",             required_argument, 0, 'm' },
  {"verbose",           required_argument, 0, 'v' },
  {"learnrate",         required_argument, 0, 'l' },
  {"algo",              required_argument, 0, 'a' },
  {"cifar_data",        required_argument, 0, 'd' },
  {"cifar_data_dump",   required_argument, 0, 'D' },
  {"partial_data",      required_argument, 0,  0  },
  {"single_batch",      required_argument, 0,  0  },
  {"dropout",           required_argument, 0,  0  },
  {"validate_ratio",    required_argument, 0,  0  },
  {"validate_interval", required_argument, 0,  0  },
  {"sample_seed",       required_argument, 0,  0 },
  {"weight_seed",       required_argument, 0,  0 },
  {"dropout_seed",      required_argument, 0,  0 },
  {"partial_data_seed", required_argument, 0,  0 },
  {"grad_dbg",          required_argument, 0,  0  },
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
          " -m,--iter N : iterate N times [%ld]\n"
          " -b,--batch_sz N : set batch size to N [%d]\n"
          " -a,--algo ALGORITHM : set the algorithm (implementation) used [%s]\n"
          " -v,--verbose L : set verbosity level to L [%d]\n"
          " -d,--cifar_data F : read data from F [%s]\n"
          " -D,--cifar_data_dump F : dump data to Fxxxx.ppm [%s]\n"
          " -l,--learnrate ETA : set learning rate to ETA [%f]\n"
          " --partial_data N : use only random N images from the file (0 for all) [%ld]\n"
          " --single_batch 0/1 : use the same mini batch in every iteration for debugging [%d]\n"
          " --dropout 0/1 : dropout or not [%d]\n"
          " --validate_ratio R : hold out this ratio of data for validation [%f]\n"
          " --validate_interval R : validate every (R * number of data for validation) training samples [%f]\n"
          " --sample_seed S : set seed for sampling to S [%ld]\n"
          " --dropout_seed S : set seed for dropout to S [%ld]\n"
          " --weight_seed S : set seed for initial weights to S [%ld]\n"
          " --partial_data_seed S : set seed for determining which data in the file are used for training/validation [%ld]\n"
          " --grad_dbg 0/1 : debug gradient computation [%d]\n"
          " --log FILE : write log to FILE [%s]\n"
          " -h,--help\n",
          prog,
          o.iters,
          o.batch_sz,
          o.algo_s,
          o.verbose,
          o.cifar_data,
          (o.cifar_data_dump ? o.cifar_data_dump : ""),
          o.learnrate,
          o.partial_data,
          o.single_batch,
          o.dropout,
          o.validate_ratio,
          o.validate_interval,
          o.sample_seed,
          o.dropout_seed,
          o.weight_seed,
          o.partial_data_seed,
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
                        "a:b:d:D:l:m:v:h", long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 0:
      {
        const char * o = long_options[option_index].name;
        if (strcmp(o, "partial_data") == 0) {
          opt.partial_data = atol(optarg);
        } else if (strcmp(o, "single_batch") == 0) {
          opt.single_batch = atoi(optarg);
        } else if (strcmp(o, "dropout") == 0) {
          opt.dropout = (atoi(optarg) != 0);
        } else if (strcmp(o, "validate_ratio") == 0) {
          opt.validate_ratio = atof(optarg);
        } else if (strcmp(o, "validate_interval") == 0) {
          opt.validate_interval = atof(optarg);
        } else if (strcmp(o, "sample_seed") == 0) {
          opt.sample_seed = atol(optarg);
        } else if (strcmp(o, "weight_seed") == 0) {
          opt.weight_seed = atol(optarg);
        } else if (strcmp(o, "dropout_seed") == 0) {
          opt.dropout_seed = atol(optarg);
        } else if (strcmp(o, "partial_data_seed") == 0) {
          opt.partial_data_seed = atol(optarg);
        } else if (strcmp(o, "grad_dbg") == 0) {
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
      opt.cifar_data = strdup(optarg);
      break;
    case 'D':
      opt.cifar_data_dump = strdup(optarg);
      break;
    case 'b':
      opt.batch_sz = atoi(optarg);
      break;
    case 'a':
      opt.algo_s = strdup(optarg);
      break;
    case 'l':
      opt.learnrate = atof(optarg);
      break;
    case 'm':
      opt.iters = atol(optarg);
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
  if (opt.batch_sz > MAX_BATCH_SIZE) {
    fprintf(stderr, "error: cannot specify --batch_sz (%d) > MAX_BATCH_SIZE (%d)\n",
            opt.batch_sz, MAX_BATCH_SIZE);
    opt.error = 1;
    return opt;
  }
  opt.algo = parse_algo(opt.algo_s);
  if (opt.algo == algo_invalid) {
    fprintf(stderr, "error: invalid algorithm (%s)\n", opt.algo_s);
    opt.error = 1;
    return opt;
  }
  opt.gpu_algo = algo_is_gpu(opt.algo_s, opt.algo);
#if !__NVCC__
  if (opt.gpu_algo) {
    fprintf(stderr, "error: --gpu 1 allowed only with nvcc\n");
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
static real show_error(real gx_gx, real dx_dx, real gx_dx,
                       real gw_gw, real dw_dw, real gw_dw,
                       real L_minus, real L, real L_plus) {
  printf("|∂L/∂x|   = %.9f\n", sqrt(gx_gx));
  printf("|dx|      = %.9f\n", sqrt(dx_dx));
  printf("∂L/∂x・dx = %.9f\n", gx_dx);
  printf("|∂L/∂w|   = %.9f\n", sqrt(gw_gw));
  printf("|dw|      = %.9f\n", sqrt(dw_dw));
  printf("∂L/∂w・dw = %.9f\n", gw_dw);
  printf("L- = %.9f\n", L_minus);
  printf("L  = %.9f\n", L);
  printf("L+ = %.9f\n", L_plus);
  real dL = L_plus - L_minus;
  real A = gx_dx + gw_dw;
  real B = dL;
  real e = (A == B ? 0.0 : fabs(A - B) / max_r(fabs(A), fabs(B)));
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
    log(3, "verbose=%d", opt.verbose);
    log(3, "cifar_data=%s", opt.cifar_data);
    log(3, "batch_sz=%d", opt.batch_sz);
    log(3, "learnrate=%f", opt.learnrate);
    log(3, "iters=%ld", opt.iters);
    log(3, "partial_data=%ld", opt.partial_data);
    log(3, "single_batch=%d", opt.single_batch);
    log(3, "dropout=%d", opt.dropout);
    log(3, "validate_ratio=%f", opt.validate_ratio);
    log(3, "validate_interval=%f", opt.validate_interval);
    log(3, "sample_seed=%ld", opt.sample_seed);
    log(3, "weight_seed=%ld", opt.weight_seed);
    log(3, "dropout_seed=%ld", opt.dropout_seed);
    log(3, "partial_data_seed=%ld", opt.partial_data_seed);
    log(3, "grad_dbg=%d", opt.grad_dbg);
    log(3, "algo_s=%s", opt.algo_s);
    log(3, "algo=%d", opt.algo);
    log(3, "gpu_algo=%d", opt.gpu_algo);
    log(3, "log=%s", opt.log);
    return 1;
  }
  /**
     @brief log hostname for the record
   */
  int log_host() {
    char name[HOST_NAME_MAX+1];
    name[0] = 0;
    gethostname(name, sizeof(name));
    log(3, "host=%s", name);
    return 1;
  }
  /**
     @brief log an environment var for the record
   */
  int log_env(const char * var) {
    char * s = getenv(var);
    if (s) {
      log(3, "%s=%s", var, s);
    } else {
      log(3, "%s undefined", var);
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
    log(2, "%s: starts", f);
  }
  /**
     @brief log the end of a function (f) 
   */
  void log_end_fun_(const char * f, tsc_t t0, tsc_t t1) {
    log(2, "%s: ends. took %ld nsec", f, t1.ns - t0.ns);
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
int vgg_util_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  if (opt.error || opt.help) usage(argv[0]);
  rnd_gen_t rg;
  rg.seed(opt.sample_seed);
  bail();
  min_i(1,2);
  max_i(3,4);
  min_r(1.2,3.4);
  max_r(5.6,7.8);
  return 0;
}

/**
   @brief suppress unused function warning
 */
void vgg_util_use_unused_functions() {
  (void)get_tsc;
  (void)show_error;
}
