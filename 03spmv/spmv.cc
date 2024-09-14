/**
   @file spmv.cc
   @brief sparse matrix vector multiplication
   @author Kenjiro Taura
   @date Oct. 14, 2018
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#if __NVCC__
/* cuda_util.h incudes various utilities to make CUDA 
   programming less error-prone. check it before you
   proceed with rewriting it for CUDA */
#include "include/cuda_util.h"
#endif

/** @brief type of matrix index (i,j,...)
    @details 
    for large matrices, we might want to make it 64 bits.
 */
typedef int idx_t;
/** @brief type of a matrix element */
typedef double real;

/** @brief sparse matrix storage format
    @details add more if you want to use another format */
typedef enum {
  sparse_format_coo,        /**< coordinate list */
  sparse_format_coo_sorted, /**< sorted coordinate list */
  sparse_format_csr,        /**< compressed sparse row */
  sparse_format_invalid,    /**< invalid */
} sparse_format_t;

/** @brief type of sparse matrix we work on (how to generate elements) */
typedef enum {
  sparse_matrix_type_random,    /**< uniform random matrix */
  sparse_matrix_type_rmat,      /**< R-MAT (recursive random matrix) */
  sparse_matrix_type_one,       /**< all one on a subset of rows/columns */
  sparse_matrix_type_coo_file,  /**< input from file */
  sparse_matrix_type_invalid,   /**< invalid */
} sparse_matrix_type_t;

/** @brief spmv matrix algorithm */
typedef enum {
  spmv_algo_serial,             /**< serial */
  spmv_algo_parallel,           /**< OpenMP parallel for */
  spmv_algo_cuda,               /**< cuda */
  spmv_algo_task,               /**< task parallel */
  spmv_algo_udr,                /**< user-defined reduction */
  spmv_algo_invalid             /**< invalid */
} spmv_algo_t;

/** @brief an element of coordinate list (i, j, a) */
typedef struct {
  idx_t i;                      /**< row */
  idx_t j;                      /**< column */
  real a;                       /**< element */
} coo_elem_t;

/** @brief an element of compressed sparse row */
typedef struct {
  idx_t j;                      /**< column */
  real a;                       /**< element */
} csr_elem_t;

/** @brief sparse matrix in coodinate list format */
typedef struct {
  coo_elem_t * elems;           /**< elements array */
#ifdef __NVCC__                 /* defined when compiling with nvcc */
  coo_elem_t * elems_dev;       /**< copy of elems on device */
#endif
} coo_t;

/** @brief sparse matrix in compressed row format */
typedef struct {
  idx_t * row_start; /**< elems[row_start[i]] is the first element of row i */
  csr_elem_t * elems;           /**< elements array */
#ifdef __NVCC__
  idx_t * row_start_dev;        /**< copy of row_start on device */
  csr_elem_t * elems_dev;       /**< copy of elems on device */
#endif
} csr_t;

/** @brief sparse matrix (in any format) */
typedef struct {
  sparse_format_t format;  /**< format */
  idx_t M;                 /**< number of rows */
  idx_t N;                 /**< number of columns */
  idx_t nnz;               /**< number of non-zeros */
  union {
    coo_t coo;             /**< coo or sorted coo */
    csr_t csr;             /**< csr */
  };
} sparse_t;

/** @brief vector */
typedef struct {
  idx_t n;                 /**< number of elements */
  real * elems;            /**< array of elements */
#ifdef __NVCC__
  real * elems_dev;        /**< copy of elems on device */
#endif
} vec_t;

/** 
    @brief command line option
*/
typedef struct {
  idx_t M;                 /**< number of rows */
  idx_t N;                 /**< number of columns */
  idx_t nnz;               /**< number of non-zero elements */
  long repeat;             /**< number of iterations (tA (Ax)) */
  char * format_str;       /**< format string (coo, coo_sorted, csr) */
  sparse_format_t format;  /**< format_str converted to enum */

  char * matrix_type_str;  /**< matrix type (random, rmat, file) */
  sparse_matrix_type_t matrix_type; /**< matrix_type_str converted to enum */

  char * algo_str;         /**< algorithm string (serial, parallel, cuda) */
  spmv_algo_t algo;        /**< algo_str converted to enum */

  char * coo_file;         /**< file */
  char * rmat_str;         /**< a,b,c,d probability of rmat */
  double rmat[2][2];       /**< { { a, b }, { c, d } } probability of rmat */
  char * dump;             /**< file name to dump image (gnuplot) data */
  long dump_points;        /**< max number of points in the dump data */
  long dump_seed;          /**< random number seed to randomly choose elements dumped */
  long seed;               /**< random number generator seed */
  int error;               /**< set when we encounter an error */
  int help;                /**< set when -h / --help is given */
} cmdline_options_t;

/** 
    @brief current time in nano second
    @return the current time in nano second
*/
static long cur_time_ns() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec * 1000000000L + ts->tv_nsec;
}

/**
   @brief malloc + check
   @param (sz) size to alloc in bytes
   @return pointer to the allocated memory
   @sa xfree
 */

static void * xalloc(size_t sz) {
  void * a = malloc(sz);
  if (!a) {
    perror("malloc");
    exit(1);
  }
  return a;
}

/**
   @brief wrap free
   @param (a) a pointer returned by calling xalloc
   @sa xalloc
 */
static void xfree(void * a) {
  free(a);
}

/** 
    @brief default values for command line options
*/
static cmdline_options_t default_opts() {
  cmdline_options_t opt = {
    .M = 20000,
    .N = 10000,
    .nnz = 0,
    .repeat = 5,
    .format_str = strdup("coo"),
    .format = sparse_format_invalid,
    .matrix_type_str = strdup("random"),
    .matrix_type = sparse_matrix_type_invalid,
    .algo_str = strdup("serial"),
    .algo = spmv_algo_invalid,
    .coo_file = strdup("mat.txt"),
    .rmat_str = strdup("5,0,1,2"),
    .rmat = { { 0, 0, }, { 0, 0, } },
    .dump = 0,
    .dump_points = 20000,
    .dump_seed = 91807290723,
    .seed = 4567890123,
    .error = 0,
    .help = 0,
  };
  return opt;
}

/** 
    @brief command line options
*/
static struct option long_options[] = {
  {"M",           required_argument, 0, 'M' },
  {"N",           required_argument, 0, 'N' },
  {"nnz",         required_argument, 0, 'z' },
  {"repeat",      required_argument, 0, 'r' },
  {"format",      required_argument, 0, 'f' },
  {"matrix-type", required_argument, 0, 't' },
  {"algo",        required_argument, 0, 'a' },
  {"coo-file",    required_argument, 0,  0  },
  {"rmat",        required_argument, 0,  0  },
  {"dump",        required_argument, 0,  0  },
  {"dump-points", required_argument, 0,  0  },
  {"dump-seed",   required_argument, 0,  0  },
  {"seed",        required_argument, 0, 's'},
  {"help",        required_argument, 0, 'h'},
  {0,             0,                 0,  0 }
};

static char * sparse_format_strs();
static char * sparse_matrix_type_strs();
static char * spmv_algo_strs();

/** 
    @brief release memory for cmdline_options
    @param (opt) the command line option to release the memory of
*/
static void cmdline_options_destroy(cmdline_options_t opt) {
  xfree(opt.format_str);
  xfree(opt.matrix_type_str);
  xfree(opt.algo_str);
  if (opt.coo_file) {
    xfree(opt.coo_file);
  }
  xfree(opt.rmat_str);
  if (opt.dump) {
    xfree(opt.dump);
  }
}

/**
   @brief print usage
   @param (prog) name of the program
  */
static void usage(const char * prog) {
  cmdline_options_t o = default_opts();
  fprintf(stderr,
          "usage:\n"
          "\n"
          "  %s [options ...]\n"
          "\n"
          "options:\n"
          "  --help             show this help\n"
          "  --M N              set the number of rows to N [%ld]\n"
          "  --N N              set the number of columns to N [%ld]\n"
          "  -z,--nnz N         set the number of non-zero elements to N [%ld]\n"
          "  -r,--repeat N      repeat N times [%ld]\n"
          "  -f,--format F      set sparse matrix format to F (%s) [%s]\n"
          "  -t,--matrix-type M set matrix type to T (%s) [%s]\n"
          "  -a,--algo A        set algorithm to A (%s) [%s]\n"
          "  --coo-file F       read matrix from F [%s]\n"
          "  --rmat a,b,c,d     set rmat probability [%s]\n"
          "  -s,--seed S        set random seed to S (use it with -t random or -t rmat) [%ld]\n"
          "  --dump F           dump matrix to a gnuplot file [%s]\n"
          "  --dump-points N    dump up to N points to a gnuplot file (use it with --dump) [%ld]\n"
          "  --dump-seed S      set random number seed to S to choose N points (use it with --dump-points) [%ld]\n"
          ,
          prog,
          (long)o.M,
          (long)o.N,
          (long)o.nnz,
          o.repeat,
          sparse_format_strs(),      o.format_str,
          sparse_matrix_type_strs(), o.matrix_type_str, 
          spmv_algo_strs(),          o.algo_str,        
          (o.coo_file ? o.coo_file : ""),
          o.rmat_str,
          o.seed,
          (o.dump ? o.dump : ""),
          (long)o.dump_points,
          o.dump_seed
          );
  cmdline_options_destroy(o);
}

/** 
    @brief pair of the index value (sparse_format_t) and its name
*/
typedef struct {
  sparse_format_t idx;          /**< index value */ 
  const char * name;            /**< name */
} sparse_format_table_entry_t;

/** 
    @brief table of sparse format and their names
*/
typedef struct {
  sparse_format_table_entry_t t[sparse_format_invalid]; /**< array of index value - name pairs */ 
} sparse_format_table_t;

/** 
    @brief table of index value - sparse format name pairs
*/
static sparse_format_table_t sparse_format_table = {
  {
    { sparse_format_coo,        "coo" },
    { sparse_format_coo_sorted, "coo_sorted" },
    { sparse_format_csr,        "csr" },
  }
};

/** 
    @brief a comma-separated list of available sparse formats
*/
static char * sparse_format_strs() {
  sparse_format_table_entry_t * t = sparse_format_table.t;
  const char * sep = ",";
  size_t n = 0;
  for (int i = 0; i < (int)sparse_format_invalid; i++) {
    if (i > 0) n += strlen(sep);
    n += strlen(t[i].name);
  }
  char * s = (char *)xalloc(n + 1);
  s[0] = 0;
  for (int i = 0; i < (int)sparse_format_invalid; i++) {
    if (i > 0) {
      strncat(s, sep, n - strlen(s));
    }
    strncat(s, t[i].name, n - strlen(s));
  }
  assert(strlen(s) == n);
  return s;
}

/** 
    @brief parse a string for matrix format and return an enum value
    @param (s) the string to parse
*/
static sparse_format_t parse_sparse_format(char * s) {
  sparse_format_table_entry_t * t = sparse_format_table.t;
  for (int i = 0; i < (int)sparse_format_invalid; i++) {
    if (strcasecmp(s, t[i].name) == 0) {
      return t[i].idx;
    }
  }
  fprintf(stderr,
          "error:%s:%d: invalid sparse format (%s)\n",
          __FILE__, __LINE__, s);
  fprintf(stderr, "  must be one of { %s }\n", sparse_format_strs());
  return sparse_format_invalid;
}

/** 
    @brief pair of the index value (matrix_type_t) and its name
*/
typedef struct {
  sparse_matrix_type_t idx;     /**< index value */ 
  const char * name;            /**< name */ 
} sparse_matrix_type_table_entry_t;

/** 
    @brief table of sparse matrix types and their names
*/
typedef struct {
  sparse_matrix_type_table_entry_t t[sparse_matrix_type_invalid]; /**< array of index value - name pairs */ 
} sparse_matrix_type_table_t;

/** 
    @brief table of index value - matrix type name pairs
*/
static sparse_matrix_type_table_t sparse_matrix_type_table = {
  {
    { sparse_matrix_type_random,   "random" },
    { sparse_matrix_type_rmat,     "rmat" },
    { sparse_matrix_type_one,      "one" },
    { sparse_matrix_type_coo_file, "file" },
  }
};

/** 
    @brief a comma-separated list of available matrix types
*/
static char * sparse_matrix_type_strs() {
  sparse_matrix_type_table_entry_t * t = sparse_matrix_type_table.t;
  const char * sep = ",";
  size_t n = 0;
  for (int i = 0; i < (int)sparse_matrix_type_invalid; i++) {
    if (i > 0) n += strlen(sep);
    n += strlen(t[i].name);
  }
  char * s = (char *)xalloc(n + 1);
  s[0] = 0;
  for (int i = 0; i < (int)sparse_matrix_type_invalid; i++) {
    if (i > 0) {
      strncat(s, sep, n - strlen(s));
    }
    strncat(s, t[i].name, n - strlen(s));
  }
  assert(strlen(s) == n);
  return s;
}

/** 
    @brief parse a string for sparse matrix type and return an enum value
    @param (s) the string to parse
*/
static sparse_matrix_type_t parse_sparse_matrix_type(char * s) {
  sparse_matrix_type_table_entry_t * t = sparse_matrix_type_table.t;
  for (int i = 0; i < (int)sparse_matrix_type_invalid; i++) {
    if (strcasecmp(s, t[i].name) == 0) {
      return t[i].idx;
    }
  }
  fprintf(stderr,
          "error:%s:%d: invalid matrix type (%s)\n",
          __FILE__, __LINE__, s);
  fprintf(stderr, "  must be one of { %s }\n", sparse_matrix_type_strs());
  return sparse_matrix_type_invalid;
}


/** 
    @brief pair of the index value (spmv_algo_t) and its name
*/
typedef struct {
  spmv_algo_t idx;              /**< index value */ 
  const char * name;            /**< name */ 
} spmv_algo_table_entry_t;

/** 
    @brief table of spmv algorithms and their names
*/
typedef struct {
  spmv_algo_table_entry_t t[spmv_algo_invalid]; /**< array of index value - name pairs */ 
} spmv_algo_table_t;

/** 
    @brief table of index value - matrix type name pairs
*/
static spmv_algo_table_t spmv_algo_table = {
  {
    { spmv_algo_serial,   "serial" },
    { spmv_algo_parallel, "parallel" },
    { spmv_algo_cuda,     "cuda" },
    { spmv_algo_task,     "task" },
    { spmv_algo_udr,      "udr" },
  }
};

/** 
    @brief print a comma-separated list of available algorithms to the standard error
*/
static char * spmv_algo_strs() {
  spmv_algo_table_entry_t * t = spmv_algo_table.t;
  const char * sep = ",";
  size_t n = 0;
  for (int i = 0; i < (int)spmv_algo_invalid; i++) {
    if (i > 0) n += strlen(sep);
    n += strlen(t[i].name);
  }
  char * s = (char *)xalloc(n + 1);
  s[0] = 0;
  for (int i = 0; i < (int)spmv_algo_invalid; i++) {
    if (i > 0) {
      strncat(s, sep, n - strlen(s));
    }
    strncat(s, t[i].name, n - strlen(s));
  }
  assert(strlen(s) == n);
  return s;
}

/** 
    @brief parse a string for spmv algorithm and return an enum value
    @param (s) the string to parse
*/
static spmv_algo_t parse_spmv_algo(char * s) {
  spmv_algo_table_entry_t * t = spmv_algo_table.t;
  for (int i = 0; i < (int)spmv_algo_invalid; i++) {
    if (strcasecmp(s, t[i].name) == 0) {
      return t[i].idx;
    }
  }
  fprintf(stderr,
          "error:%s:%d: invalid algorithm type (%s)\n",
          __FILE__, __LINE__, s);
  fprintf(stderr, "  must be one of { %s }\n", spmv_algo_strs());
  return spmv_algo_invalid;
}

/** 
    @brief print error meessage during rmat string (a,b,c,d)
*/
static void parse_error_rmat_probability(char * rmat_str) {
  fprintf(stderr,
          "error:%s:%d: argument to --rmat (%s)"
          " must be F,F,F,F where F is a floating point number\n",
          __FILE__, __LINE__, rmat_str);
}
/** 
    @brief parse a string of the form a,b,c,d and put it into 2x2 matrix
    @param (rmat_str) string to parse
    @param (rmat) 2x2 array to put the result into
    @details a string must be comma-separated list of four numbers 
    (integers or floating point numbers), without any spaces.
    it parses the string into four probability numbers (numbers whose
    sum is one).
    e.g., "1,1,1,1" -> { { 0.25, 0.25 }, { 0.25, 0.25 } }
    "1,2,3,4" -> { { 0.1, 0.2 }, { 0.3, 0.4 } }
*/
static int parse_rmat_probability(char * rmat_str, double rmat[2][2]) {
  char * s = rmat_str;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i + j > 0) {
        if (s[0] != ',') {
          parse_error_rmat_probability(rmat_str);
          return 0;
        }
        s++;
      }
      char * next = 0;
      double x = strtod(s, &next);
      if (s == next) {
        /* no conversion performed */
        parse_error_rmat_probability(rmat_str);
        return 0;
      } else {
        rmat[i][j] = x;
        s = next;
      }
    }
  }
  double t = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      t += rmat[i][j];
    }
  }
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      rmat[i][j] /= t;
    }
  }
  return 1;
}

/** 
    @brief parse command line args
    @param (argc) size of argv
    @param (argv) array of argc strings (command line arguments)
*/
static cmdline_options_t parse_args(int argc, char ** argv) {
  char * prog = argv[0];
  cmdline_options_t opt = default_opts();
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "M:N:z:r:f:t:a:s:h",
                        long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 0:
      {
        const char * o = long_options[option_index].name;
        if (strcmp(o, "rmat") == 0) {
          xfree(opt.rmat_str);
          opt.rmat_str = strdup(optarg);
        } else if (strcmp(o, "coo-file") == 0) {
          xfree(opt.coo_file);
          opt.coo_file = strdup(optarg);
        } else if (strcmp(o, "dump") == 0) {
          if (opt.dump) {
            xfree(opt.dump);
          }
          opt.dump = strdup(optarg);
        } else if (strcmp(o, "dump-points") == 0) {
          opt.dump_points = atol(optarg);
        } else if (strcmp(o, "dump-seed") == 0) {
          opt.dump_seed = atol(optarg);
        } else {
          fprintf(stderr,
                  "bug:%s:%d: should handle option %s\n",
                  __FILE__, __LINE__, o);
          opt.error = 1;
          return opt;
        }
      }
      break;
    case 'M':
      opt.M = atol(optarg);
      break;
    case 'N':
      opt.N = atol(optarg);
      break;
    case 'z':
      opt.nnz = atol(optarg);
      break;
    case 'r':
      opt.repeat = atol(optarg);
      break;
    case 'f':
      xfree(opt.format_str);
      opt.format_str = strdup(optarg);
      break;
    case 't':
      xfree(opt.matrix_type_str);
      opt.matrix_type_str = strdup(optarg);
      break;
    case 'a':
      xfree(opt.algo_str);
      opt.algo_str = strdup(optarg);
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
  opt.format = parse_sparse_format(opt.format_str);
  if (opt.format == sparse_format_invalid) {
    opt.error = 1;
    return opt;
  }
  opt.matrix_type = parse_sparse_matrix_type(opt.matrix_type_str);
  if (opt.matrix_type == sparse_matrix_type_invalid) {
    opt.error = 1;
    return opt;
  }
  opt.algo = parse_spmv_algo(opt.algo_str);
  if (opt.algo == spmv_algo_invalid) {
    opt.error = 1;
    return opt;
  }
  if (parse_rmat_probability(opt.rmat_str, opt.rmat) == 0) {
    opt.error = 1;
    return opt;
  }
  return opt;
}

/** 
    @brief make an invalid matrix
*/
static sparse_t mk_sparse_invalid() {
  sparse_t A = { sparse_format_invalid, 0, 0, 0, { } };
  return A;
}

/** 
    @brief destroy coo 
*/
static void coo_destroy(sparse_t A) {
  xfree(A.coo.elems);
}

/** 
    @brief destroy csr
*/
static void csr_destroy(sparse_t A) {
  xfree(A.csr.row_start);
  xfree(A.csr.elems);
}

/** 
    @brief destroy sparse matrix in any format
*/
static void sparse_destroy(sparse_t A) {
  switch (A.format) {
  case sparse_format_coo:
  case sparse_format_coo_sorted:
    coo_destroy(A);
    break;
  case sparse_format_csr:
    csr_destroy(A);
    break;
  default:
    fprintf(stderr,
            "error:%s:%d: sparse_destroy: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    break;
  }
}

/** 
    @brief size (in bytes) of a sparse matrix in coo format
    @param (A) a sparse matrix in coo format
    @return size of the matrix in bytes
*/
static size_t sparse_coo_size(sparse_t A) {
  size_t sz = sizeof(coo_elem_t) * A.nnz;
  return sz;
}

/** 
    @brief size (in bytes) of a sparse matrix in csr format
    @param (A) a sparse matrix in csr format
    @return size of the matrix in bytes
*/
static size_t sparse_csr_size(sparse_t A) {
  size_t nnz_sz = sizeof(csr_elem_t) * A.nnz;
  size_t row_start_sz = sizeof(idx_t) * (A.M + 1);
  return nnz_sz + row_start_sz;
}

/** 
    @brief size (in bytes) of a sparse matrix
    @param (A) a sparse matrix
    @return size of the matrix in bytes
*/
static size_t sparse_size(sparse_t A) {
  switch (A.format) {
  case sparse_format_coo:
  case sparse_format_coo_sorted:
    return sparse_coo_size(A);
  case sparse_format_csr:
    return sparse_csr_size(A);
  default:
    fprintf(stderr,
            "error:%s:%d: sparse_size: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    return 0;
  }
}

/**
   @brief destroy vector
 */
static void vec_destroy(vec_t x) {
  xfree(x.elems);
}

/** 
    @brief make a uniform random coo matrix
    @param (M) number of rows
    @param (N) number of columns
    @param (nnz) number of non-zeros
    @param (rg) random number state (passed to erand48)
    @return a sparse matrix in coo format
*/
static sparse_t mk_coo_random(idx_t M, idx_t N, idx_t nnz,
                              unsigned short rg[3]) {
  printf("%s:%d:mk_coo_random starts ...\n", __FILE__, __LINE__);
  long t0 = cur_time_ns();
  coo_elem_t * elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
  for (idx_t k = 0; k < nnz; k++) {
    idx_t i = nrand48(rg) % M;
    idx_t j = nrand48(rg) % N;
    real  a = erand48(rg);
    coo_elem_t * e = elems + k;
    e->i = i;
    e->j = j;
    e->a = a;
  }
  coo_t coo = { elems };
  sparse_t A = { sparse_format_coo, M, N, nnz, { .coo = coo } };
  long t1 = cur_time_ns();
  printf("%s:%d:mk_coo_random ends. took %.3f sec\n",
         __FILE__, __LINE__, (t1 - t0) * 1.0e-9);
  return A;
}

/**
   @brief a pair of two indices (i and j)
 */
typedef struct {
  idx_t i;                      /**< row number */
  idx_t j;                      /**< column number */
} idx_pair_t;

/**
   @brief choose a pair of 0/1s according to 2x2 probability matrix p[2][2].
   @return a pair of two 0/1s ({0,0}, {0,1}, {1,0} or {1,1})
   @details
   returns {0,0} with probability p[0][0],
   {0,1} with probability p[0][1],
   {1,0} with probability p[1][0] and
   {1,1} with probability p[1][1]
 */
static idx_pair_t rmat_choose_01(double p[2][2], unsigned short rg[3]) {
  double x = erand48(rg);
  double q = 0.0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      q += p[i][j];
      if (x <= q) {
        idx_pair_t ij = { i, j };
        return ij;
      }
    }
  }
  idx_pair_t ij = { 1, 1 };
  return ij;
}

/** 
    @brief choose (i,j) 0 <= i < M and 0 <= j < N
    according to the probability p.
    @param (M) the maximum value i can take plus 1
    @param (N) the maximum value j can take plus 1
    @param (p) 2x2 matrix designating the probability
    @param (rg) random number state (passed to erand48)
    @details
    with probability p[0][0], 0 <= i < M/2 and 0 <= j < N/2.
    with probability p[0][1], 0 <= i < M/2 and N/2 <= j < N.
    with probability p[1][0], M/2 <= i < M and 0 <= j < N/2.
    with probability p[1][1], M/2 <= i < M and N/2 <= j < N.
    we apply this recursively.
*/
static idx_pair_t rmat_choose_pair(idx_t M, idx_t N, double p[2][2],
                                   unsigned short rg[3]) {
  idx_t M0 = 0, M1 = M, N0 = 0, N1 = N;
  while (M1 - M0 > 1 || N1 - N0 > 1) {
    idx_pair_t zo = rmat_choose_01(p, rg);
    if (M1 - M0 > 1) {
      idx_t Mh = (M0 + M1) / 2;
      if (zo.i) {
        M0 = Mh;
      } else {
        M1 = Mh;
      }
    }
    if (N1 - N0 > 1) {
      idx_t Nh = (N0 + N1) / 2;
      if (zo.j) {
        N0 = Nh;
      } else {
        N1 = Nh;
      }
    }
  }
  assert(M0 + 1 == M1);
  assert(N0 + 1 == N1);
  idx_pair_t ij = { M0, N0 };
  return ij;
}

/** 
    @brief make a random R-MAT 
    (https://epubs.siam.org/doi/abs/10.1137/1.9781611972740.43)

    @param (M) the number of rows
    @param (N) the number of columns
    @param (nnz) the number of non-zeros
    @param (p) 2x2 matrix designating the probability
    @param (rg) random number state (passed to erand48)

    @return a sparse matrix in coo fromat
    @details generate R-MAT 
    (https://epubs.siam.org/doi/abs/10.1137/1.9781611972740.43)
    with the specified probability p.
    @sa rmat_choose_pair
*/
static sparse_t mk_coo_rmat(idx_t M, idx_t N, idx_t nnz,
                            double p[2][2], 
                            unsigned short rg[3]) {
  printf("%s:%d:mk_coo_rmat starts ...\n", __FILE__, __LINE__);
  long t0 = cur_time_ns();
  coo_elem_t * elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
  for (idx_t k = 0; k < nnz; k++) {
    idx_pair_t ij = rmat_choose_pair(M, N, p, rg);
    coo_elem_t * e = elems + k;
    e->i = ij.i;
    e->j = ij.j;
    e->a = erand48(rg);
  }
  coo_t coo = { elems };
  sparse_t A = { sparse_format_coo, M, N, nnz, { .coo = coo } };
  long t1 = cur_time_ns();
  printf("%s:%d:mk_coo_rmat ends. took %.3f sec\n",
         __FILE__, __LINE__, (t1 - t0) * 1.0e-9);
  return A;
}

/** 
    @brief make a sparse matrix whose elements are one on certain rows/columns and zero anywhere else
    @param (M) the number of rows
    @param (N) the number of columns
    @param (nnz) the number of non-zeros

    @return a sparse matrix whose elements are all zero or one (see details)

    @details generate a matrix good for debugging. specifically, it
    chooses a certain number of rows and columns and make elements one
    only in the intersection of those rows and columns.  for example,
    if rows chosen are { 2, 5, 8 } and columns chosen are { 1, 4 }, we
    have 3 x 2 = 6 non-zeros, at (2,1), (2,4), (5,1), (5,4), (8,1) and
    (8,4).  more specifically, we first determine the number rows (m)
    and columns (n) as follows. starting from (m,n) = (1,1), we repeat
    incrementing them by one alternatively, until m x n exceeds the
    specified number of non-zeros (nnz) or either one reaches its
    respective limit (i.e., when m reaches M or n reaches N), after
    which we increment only the one that still does not reach the
    limit.  for example, if nnz = 50 and M and N large, we end with m
    = 7 and n = 7.  if nnz = 10^6 M = 10^2 and N = 10^7, we will end
    with m = 10^2 and N 10^4.  note that the actual number of elements
    you got may be slightly smaller than nnz you specified.  it is
    never above the specified nnz.  after the number of these non-zero
    rows and columns are determined, we simply evenly distribute the
    actual set of non-zero rows/columns.  for example, if M = 100 and
    the number of non-zero rows = 8, we divide 99 by (8 - 1),
    obtaining 14, and the set of non-zero rows will be { 0, 14, 28,
    ..., 98 }

*/
static sparse_t mk_coo_one(idx_t M, idx_t N, idx_t nnz) {
  printf("%s:%d:mk_coo_one starts ...\n", __FILE__, __LINE__);
  long t0 = cur_time_ns();
  idx_t nnz_M = 0;
  idx_t nnz_N = 0;
  int cont = 1;
  while (cont) {
    cont = 0;
    if (nnz_M < M && (nnz_M + 1) * nnz_N <= nnz) {
      nnz_M++;
      cont = 1;
    }
    if (nnz_N < N && nnz_M * (nnz_N + 1) <= nnz) {
      nnz_N++;
      cont = 1;
    }
  }
  idx_t real_nnz = nnz_M * nnz_N;
  assert(real_nnz <= nnz);
  assert(nnz_M <= M);
  assert(nnz_N <= N);
  if (real_nnz < nnz) {
    fprintf(stderr,
            "warning:%s:%d: nnz truncated to %ld\n",
            __FILE__, __LINE__, (long)real_nnz);
  }
  coo_elem_t * elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * real_nnz);
  idx_t skip_M = (nnz_M > 1 ? (M - 1) / (nnz_M - 1) : M);
  idx_t skip_N = (nnz_N > 1 ? (N - 1) / (nnz_N - 1) : N);
  idx_t k = 0;
  for (idx_t i = 0; i < nnz_M; i++) {
    for (idx_t j = 0; j < nnz_N; j++) {
      real  a = 1.0;
      coo_elem_t * e = elems + k;
      e->i = i * skip_M;
      e->j = j * skip_N;
      assert(e->i < M);
      assert(e->j < N);
      e->a = a;
      k++;
    }
  }
  assert(k == real_nnz);
  coo_t coo = { elems };
  sparse_t A = { sparse_format_coo_sorted, M, N, real_nnz, { .coo = coo } };
  long t1 = cur_time_ns();
  printf("%s:%d:mk_coo_one ends. took %.3f sec\n",
         __FILE__, __LINE__, (t1 - t0) * 1.0e-9);
  return A;
}

/*********************************************************
 *
 * matrix format conversion (coo <-> coo_sorted <-> csr <-> ...)
 * only "signficant" ones are:
 * coo ---> coo_sorted ---> csr ---> coo_sorted
 * others can be easily built from them.
 * coo ---> any (see above)
 * coo_sorted ---> any (coo_sorted ---> coo is no-op)
 * csr ---> any (csr -> coo_sorted)
 *
 *********************************************************/


/** 
    @brief compare two coo elements 
    @details callback used to sort coo elements in the dictionary order
    @return -1 if a_ < b, 1 if a_ > b and 0 if a_ = b
*/
static int coo_elem_cmp(const void * a_, const void * b_) {
  coo_elem_t * a = (coo_elem_t *)a_;
  coo_elem_t * b = (coo_elem_t *)b_;
  if (a->i < b->i) return -1;
  if (a->i > b->i) return 1;
  if (a->j < b->j) return -1;
  if (a->j > b->j) return 1;
  if (a->a < b->a) return -1;
  if (a->a > b->a) return 1;
  return 0;
}

/** 
    @brief convert coo/coo_sorted matrix A to coo/coo_sorted format.
    @param (A) a sparse matrix in coo format
    @param (format) destination format (coo or coo_sorted)
    @return a sparse matrix in the coo_sorted format
 */

static sparse_t sparse_coo_to_coo(sparse_t A, sparse_format_t format) {
  printf("%s:%d:sparse_coo_to_coo starts ...\n", __FILE__, __LINE__);
  long t0 = cur_time_ns();
  if ((A.format == sparse_format_coo ||
       A.format == sparse_format_coo_sorted) &&
      (format == sparse_format_coo ||
       format == sparse_format_coo_sorted)) {
    int need_sort = (A.format == sparse_format_coo && format == sparse_format_coo_sorted);
    sparse_format_t out_format = (A.format == sparse_format_coo ? format : sparse_format_coo_sorted);
    idx_t nnz = A.nnz;
    idx_t M = A.M;
    coo_elem_t * A_elems = A.coo.elems;
    coo_elem_t * B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
    /* sort if src is coo and dest is coo_sorted. otherwise copy */
    if (need_sort) {
      idx_t * row_p     = (idx_t *)xalloc(sizeof(idx_t) * M);
      idx_t * row_start = (idx_t *)xalloc(sizeof(idx_t) * M);
      idx_t * row_end   = (idx_t *)xalloc(sizeof(idx_t) * M);
      /* count the number of nnz in each row */
#pragma omp parallel for 
      for (idx_t i = 0; i < M; i++) {
        row_p[i] = 0;
      }
#pragma omp parallel for 
      for (idx_t k = 0; k < nnz; k++) {
        idx_t i = A_elems[k].i;
        __sync_fetch_and_add(&row_p[i], 1);
      }
      /* row_count[i] = the number of non-zeros in ith row.
         now calculate where ith row should start. */
      idx_t s = 0;
      for (idx_t i = 0; i < M; i++) {
        idx_t e = s + row_p[i];
        row_start[i] = s;
        row_end[i] = e;
        row_p[i] = s;           /* set p to the start for the next scan */
        s = e;
      }
      assert(s == nnz);
#pragma omp parallel for 
      for (idx_t k = 0; k < nnz; k++) {
        idx_t i = A_elems[k].i;
        idx_t p = __sync_fetch_and_add(&row_p[i], 1);
        assert(p < row_end[i]);
        B_elems[p] = A_elems[k];
      }
      for (idx_t i = 0; i < M; i++) {
        assert(row_p[i] == row_end[i]);
      }
#pragma omp parallel for
      for (idx_t i = 0; i < M; i++) {
        idx_t s = row_start[i];
        idx_t e = row_end[i];
        qsort((void*)&B_elems[s], e - s, sizeof(coo_elem_t), coo_elem_cmp);
      }
      xfree(row_p);
      xfree(row_start);
      xfree(row_end);
    } else {
      /* either already sorted or not asked to sort. just copy */
      memcpy(B_elems, A_elems, sizeof(coo_elem_t) * nnz);
    }
    coo_t coo = { B_elems };
    sparse_t B = { out_format, A.M, A.N, A.nnz, { .coo = coo } };
    long t1 = cur_time_ns();
    printf("%s:%d:sparse_coo_to_coo ends. took %.3f sec\n",
           __FILE__, __LINE__, (t1 - t0) * 1.0e-9);
    return B;
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in coo format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}

/**
   @brief convert a sparse matrix in coo format to csr format.
   @param (A) a sparse matrix in coo format
   @return a sparse matrix in the csr format
 */
static sparse_t sparse_coo_sorted_to_csr(sparse_t A) {
  printf("%s:%d:sparse_coo_sorted_to_csr starts ...\n", __FILE__, __LINE__);
  long t0 = cur_time_ns();
  if (A.format == sparse_format_coo_sorted) {
    idx_t M = A.M;
    idx_t N = A.N;
    idx_t nnz = A.nnz;
    idx_t * row_start = (idx_t *)xalloc(sizeof(idx_t) * (M + 1));
    coo_elem_t * A_elems = A.coo.elems;
    csr_elem_t * B_elems = (csr_elem_t *)xalloc(sizeof(csr_elem_t) * nnz);
    for (idx_t i = 0; i < M + 1; i++) {
      row_start[i] = 0;
    }
    /* scan A's elements (in the dictionary order).
       count the number of non-zeros in each row along the way */
    for (idx_t k = 0; k < nnz; k++) {
      coo_elem_t * e = A_elems + k;
      row_start[e->i]++;
      B_elems[k].j = e->j;
      B_elems[k].a = e->a;
    }
    /* row_start[i] = the number of non-zeros in ith row.
       now calculate where ith row starts. */
    idx_t s = 0;
    for (idx_t i = 0; i < M; i++) {
      idx_t t = s + row_start[i];
      row_start[i] = s;
      s = t;
    }
    row_start[M] = s;
    assert(s == nnz);
    csr_t csr = { row_start, B_elems };
    sparse_t B = { sparse_format_csr, M, N, nnz, { .csr = csr } };
    long t1 = cur_time_ns();
    printf("%s:%d:sparse_coo_sorted_to_csr ends. took %.3f sec\n",
           __FILE__, __LINE__, (t1 - t0) * 1.0e-9);
    return B;
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in coo_sorted format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}

/**
   @brief convert a sparse matrix in coo format to csr format.
   @param (A) a sparse matrix in coo format
   @return a sparse matrix in the csr format
 */
static sparse_t sparse_coo_to_csr(sparse_t A) {
  if (A.format == sparse_format_coo) {
    /* first get coo_sorted format */
    sparse_t B = sparse_coo_to_coo(A, sparse_format_coo_sorted);
    sparse_t C = sparse_coo_sorted_to_csr(B);
    sparse_destroy(B);
    return C;
  } else if (A.format == sparse_format_coo_sorted) {
    sparse_t C = sparse_coo_sorted_to_csr(A);
    return C;
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in coo format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}

/**
   @brief convert sparse matrix in coo format to any specified format.
   @param (A) a sparse matrix in coo format
   @param (format) the destination format
   @return a sparse matrix in the specified format
 */
static sparse_t sparse_coo_to_any(sparse_t A, sparse_format_t format) {
  if (A.format == sparse_format_coo
      || A.format == sparse_format_coo_sorted) {
    switch (format) {
    case sparse_format_coo:
      return sparse_coo_to_coo(A, format);
    case sparse_format_coo_sorted:
      return sparse_coo_to_coo(A, format);
    case sparse_format_csr:
      return sparse_coo_to_csr(A);
    default:
      fprintf(stderr,
              "error:%s:%d: invalid output format %d\n",
              __FILE__, __LINE__, format);
      return mk_sparse_invalid();
    }
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in coo format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}


/**
   @brief convert a sparse matrix in csr format to coo sorted format.
   @param (A) a sparse matrix in csr format
   @return a sparse matrix in coo_sorted format
 */
static sparse_t sparse_csr_to_coo_sorted(sparse_t A) {
  printf("%s:%d:sparse_csr_to_coo_sorted starts ...\n", __FILE__, __LINE__);
  long t0 = cur_time_ns();
  if (A.format == sparse_format_csr) {
    idx_t M = A.M;
    idx_t N = A.N;
    idx_t nnz = A.nnz;
    idx_t * row_start = A.csr.row_start;
    csr_elem_t * A_elems = A.csr.elems;
    coo_elem_t * B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
    for (idx_t i = 0; i < M; i++) {
      idx_t start = row_start[i];
      idx_t end = row_start[i + 1];
      for (idx_t k = start; k < end; k++) {
        csr_elem_t * e = A_elems + k;
        B_elems[k].i = i;
        B_elems[k].j = e->j;
        B_elems[k].a = e->a;
      }
    }
    coo_t coo = { B_elems };
    sparse_t B = { sparse_format_coo_sorted, M, N, nnz, { .coo = coo } };
    long t1 = cur_time_ns();
    printf("%s:%d:sparse_csr_to_coo_sorted ends. took %.3f sec\n",
           __FILE__, __LINE__, (t1 - t0) * 1.0e-9);
    return B;
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in csr format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
}

/**
   @brief convert sparse matrix in csr format to any specified format
   @param (A) a sparse matrix in csr format
   @param (format) the destination format
   @return a sparse format in the specified format
 */
static sparse_t sparse_csr_to_any(sparse_t A, sparse_format_t format) {
  if (A.format == sparse_format_csr) {
    switch (format) {
    case sparse_format_coo:
      return sparse_csr_to_coo_sorted(A);
    case sparse_format_coo_sorted:
      return sparse_csr_to_coo_sorted(A);
    case sparse_format_csr:
      return A;
    default:
      fprintf(stderr,
              "error:%s:%d: invalid output format %d\n",
              __FILE__, __LINE__, format);
      return mk_sparse_invalid();
    }
  } else {
    fprintf(stderr,
            "error:%s:%d: input matrix not in csr format %d\n",
            __FILE__, __LINE__, format);
    return mk_sparse_invalid();
  }
}

/**
   @brief convert a sparse matrix of any format to any specified format
   @param (A) a sparse matrix in csr format
   @param (format) the destination format
   @return a sparse format in the specified format
 */
sparse_t sparse_any_to_any(sparse_t A, sparse_format_t format) {
  switch (A.format) {
  case sparse_format_coo:
    return sparse_coo_to_any(A, format);
  case sparse_format_coo_sorted:
    return sparse_coo_to_any(A, format);
  case sparse_format_csr:
    return sparse_csr_to_any(A, format);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid input format %d\n",
            __FILE__, __LINE__, format);
    return mk_sparse_invalid();
  }
}
/*********************************************************
 *
 * make (generate or read) a sparse matrix
 *
 *********************************************************/

/**
   @brief read a matrix file and return a sparse matrix in coo formated
   @param (M) the number of rows
   @param (N) the number of columns
   @param (nnz) the number of non-zeros
   @param (file) filename
   @return a sparse matrix in coo format
 */
static sparse_t read_coo_file(idx_t M, idx_t N, idx_t nnz, char * file) {
  (void)M;
  (void)N;
  (void)nnz;
  (void)file;
  fprintf(stderr, ":%s:%d:read_coo_file: sorry, not implemented yet\n",
          __FILE__, __LINE__);
  return mk_sparse_invalid();
}

/**
   @brief make a sparse matrix in coo format by the desiginated generation
   method
   @param (opt) command line options specifying matrix type and other
   parameters necessary for some matrix types
   @param (M) the number of rows
   @param (N) the number of columns
   @param (nnz) the number of non-zeros
   @param (rg) random number generator state (passed to erand48)
   @return a sparse matrix in coo format
 */
static sparse_t mk_sparse_matrix_coo(cmdline_options_t opt,
                                     idx_t M, idx_t N, idx_t nnz,
                                     unsigned short rg[3]) {
  switch (opt.matrix_type) {
  case sparse_matrix_type_random:
    return mk_coo_random(M, N, nnz, rg);
  case sparse_matrix_type_rmat:
    return mk_coo_rmat(M, N, nnz, opt.rmat, rg);
  case sparse_matrix_type_one:
    return mk_coo_one(M, N, nnz);
  case sparse_matrix_type_coo_file:
    return read_coo_file(M, N, nnz, opt.coo_file);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid matrix_type %d\n",
            __FILE__, __LINE__, opt.matrix_type);
    return mk_sparse_invalid();
  }
}

/** 
    @brief make (read or generate) a sparse matrix with the specified 
    matrix generation method

    @param (opt) command line options specifying matrix format, 
    matrix type, and other parameters necessary for some matrix types
    @param (M) the number of rows
    @param (N) the number of columns
    @param (nnz) the number of non-zeros
    @param (rg) random number generator state (passed to erand48)
    @return a sparse matrix in coo format
    @sa mk_sparse_matrix_coo
    @sa sparse_coo_to
*/
static sparse_t mk_sparse_matrix(cmdline_options_t opt,
                                 idx_t M, idx_t N, idx_t nnz,
                                 unsigned short rg[3]) {
  sparse_t A = mk_sparse_matrix_coo(opt, M, N, nnz, rg);
  sparse_t B = sparse_coo_to_any(A, opt.format);
  sparse_destroy(A);
  return B;
}

/*********************************************************
 *
 * transpose a sparse matrix
 *
 *********************************************************/

/** 
    @brief transpose a matrix in coordinate list format
    @param (A) a sparse matrix in coo or coo_sorted format
    @return the transposed sparse matrix in coo format
    @sa sparse_transpose
*/
static sparse_t coo_transpose(sparse_t A) {
  printf("%s:%d:coo_transpose starts ...\n", __FILE__, __LINE__);
  long t0 = cur_time_ns();
  assert(A.format == sparse_format_coo
         || A.format == sparse_format_coo_sorted);
  idx_t nnz = A.nnz;
  coo_elem_t * B_elems = 0;
  B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
  memcpy(B_elems, A.coo.elems, sizeof(coo_elem_t) * nnz);
  for (idx_t k = 0; k < nnz; k++) {
    idx_t i = B_elems[k].i;
    idx_t j = B_elems[k].j;
    B_elems[k].i = j;
    B_elems[k].j = i;
  }
  coo_t coo = { B_elems };
  sparse_t B = { sparse_format_coo, A.N, A.M, nnz, { .coo = coo } };
  long t1 = cur_time_ns();
  printf("%s:%d:coo_transpose ends. took %.3f sec\n",
         __FILE__, __LINE__, (t1 - t0) * 1.0e-9);
  return B;
}

/** 
    @brief transpose a matrix in any format
    @param (A) a sparse matrix
    @return the transposed matrix in the same format with A
    @sa coo_transpose
*/
static sparse_t sparse_transpose(sparse_t A) {
  switch (A.format) {
  case sparse_format_coo: {
    return coo_transpose(A);
  }
  case sparse_format_coo_sorted: {
    sparse_t B = coo_transpose(A);
    sparse_t C = sparse_coo_to_coo(B, sparse_format_coo_sorted);
    sparse_destroy(B);
    return C;
  }
  case sparse_format_csr: {
    sparse_t B = sparse_csr_to_coo_sorted(A);
    sparse_t C = coo_transpose(B);
    sparse_t D = sparse_coo_to_csr(C);
    sparse_destroy(B);
    sparse_destroy(C);
    return D;
  }
  default: {
    fprintf(stderr,
            "error:%s:%d: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    return mk_sparse_invalid();
  }
  }
}

/*********************************************************
 *
 * copy data from host to device (CUDA)
 *
 *********************************************************/

#if __NVCC__
static int coo_to_dev(sparse_t& A);
#include "include/coo_to_dev.cc"

static int csr_to_dev(sparse_t& A);
#include "include/csr_to_dev.cc"

/** 
    @brief make a deivce copy of a sparse matrix.
    @param (A) the reference to a matrix whose elem_dev has not 
    been set (i.e., = NULL)
    @return 1 if succeed, 0 if failed.
    @sa coo_to_dev
    @sa csr_to_dev
    @sa vec_to_dev
*/
static int sparse_to_dev(sparse_t& A) {
  switch (A.format) {
  case sparse_format_coo: {
    return coo_to_dev(A);
  }
  case sparse_format_coo_sorted: {
    return coo_to_dev(A);
  }
  case sparse_format_csr: {
    return csr_to_dev(A);
  }
  default: {
    fprintf(stderr,
            "error:%s:%d: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    return 0;
  }
  }
}

static int vec_to_dev(vec_t& v);
#include "include/vec_to_dev.cc"

#endif

/** ************************************************************
    SpMV, finally
    there are matrix of procedures depending on the sparse matrix
    format and the algorithm

spmv
               | serial | parallel | cuda | task | udr  |
    -----------+--------+----------+------+------+------+
    coo        | [T1]   | [M1]     | [M3] | N/S  | N/S  |
    coo_sorted | [T1]   | [M1]     | [M3] | [O5] | [O7] |
    csr        | [T1]   | [M2]     | [M4] | [O6] | [O8] |

vec_norm2, scalar_vec
               | serial | parallel | cuda | task | udr  |
    -----------+--------+----------+------+------+------+
    coo        | [T1]   | [M1]     | [M3] | N/S  | N/S  |
    coo_sorted | [T1]   | [M1]     | [M3] | [M1] | [
    csr        | [T1]   | [M1]     | [M3] | [M1] |

    
*************************************************************/


/** 
    @brief y = A * x in serial for coo format
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @return 1 if succeed, 0 if failed
*/
static int spmv_coo_serial(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  /* initialize y */
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  /* work on all non-zeros */
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    y[i] += ax;
  }
  return 1;                     /* OK */
}

#include "include/spmv_coo_parallel.cc"
#if __NVCC__
#include "include/spmv_coo_cuda.cc"
#endif                                 
#include "include/spmv_coo_task.cc"
#include "include/spmv_coo_udr.cc"

/** 
    @brief y = A * x for coo format, with the specified algorithm
    @param (algo) algorithm
    @param (A) a sparse matrix
    @param (x) a vector
    @param (y) a vector
    @return 1 if succeed, 0 if failed
*/
static int spmv_coo(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  switch (algo) {
  case spmv_algo_serial:
    return spmv_coo_serial(A, x, y);
  case spmv_algo_parallel:
    return spmv_coo_parallel(A, x, y);
#if __NVCC__
  case spmv_algo_cuda:
    return spmv_coo_cuda(A, x, y);
#endif
  case spmv_algo_task:
    return spmv_coo_task(A, x, y);
  case spmv_algo_udr:
    return spmv_coo_udr(A, x, y);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algorithm %d\n",
            __FILE__, __LINE__, algo);
    return 0;
  }
}

/** 
    @brief y = A * x in serial for coo_sorted format
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
*/
static int spmv_coo_sorted_serial(sparse_t A, vec_t vx, vec_t vy) {
  /* the same no matter whether elements are sorted. 
     just call spmv_coo_serial and we are done */
  return spmv_coo_serial(A, vx, vy);
}

#include "include/spmv_coo_sorted_parallel.cc"
#if __NVCC__
#include "include/spmv_coo_sorted_cuda.cc"
#endif                                 
#include "include/spmv_coo_sorted_task.cc"
#include "include/spmv_coo_sorted_udr.cc"


/** 
    @brief y = A * x for coo_sorted format, with the specified algorithm
    @param (algo) algorithm
    @param (A) a sparse matrix
    @param (x) a vector
    @param (y) a vector
    @return 1 if succeed, 0 if failed
*/
static int spmv_coo_sorted(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  switch (algo) {
  case spmv_algo_serial:
    return spmv_coo_sorted_serial(A, x, y);
  case spmv_algo_parallel:
    return spmv_coo_sorted_parallel(A, x, y);
#if __NVCC__
  case spmv_algo_cuda:
    return spmv_coo_sorted_cuda(A, x, y);
#endif
  case spmv_algo_task:
    return spmv_coo_sorted_task(A, x, y);
  case spmv_algo_udr:
    return spmv_coo_sorted_udr(A, x, y);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algorithm %d\n",
            __FILE__, __LINE__, algo);
    return 0;
  }
}

/** 
    @brief y = A * x in serial for csr format
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @return 1 if succeed, 0 if failed
*/
static int spmv_csr_serial(sparse_t A, vec_t vx, vec_t vy) {
  idx_t M = A.M;
  idx_t * row_start = A.csr.row_start;
  csr_elem_t * elems = A.csr.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t i = 0; i < M; i++) {
    idx_t start = row_start[i];
    idx_t end = row_start[i + 1];
    for (idx_t k = start; k < end; k++) {
      csr_elem_t * e = elems + k;
      idx_t j = e->j;
      real  a = e->a;
      y[i] += a * x[j];
    }
  }
  return 1;
}

#include "include/spmv_csr_parallel.cc"
#if __NVCC__
#include "include/spmv_csr_cuda.cc"
#endif
#include "include/spmv_csr_task.cc"
#include "include/spmv_csr_udr.cc"


/** 
    @brief y = A * x for csr format, with the specified algorithm
    @param (algo) algorithm
    @param (A) a sparse matrix
    @param (x) a vector
    @param (y) a vector
    @return 1 if succeed, 0 if failed
*/
static int spmv_csr(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  switch (algo) {
  case spmv_algo_serial:
    return spmv_csr_serial(A, x, y);
  case spmv_algo_parallel:
    return spmv_csr_parallel(A, x, y);
#if __NVCC__
  case spmv_algo_cuda:
    return spmv_csr_cuda(A, x, y);
#endif
  case spmv_algo_task:
    return spmv_csr_task(A, x, y);
  case spmv_algo_udr:
    return spmv_csr_udr(A, x, y);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algorithm %d\n",
            __FILE__, __LINE__, algo);
    return 0;
  }
}

/** 
    @brief y = A * x for any format, with the specified algorithm
    @param (algo) algorithm
    @param (A) a sparse matrix
    @param (x) a vector
    @param (y) a vector
    @return 1 if succeed, 0 if failed
*/
static int spmv(spmv_algo_t algo, sparse_t A, vec_t x, vec_t y) {
  assert(x.n == A.N);
  assert(y.n == A.M);
  switch (A.format) {
  case sparse_format_coo:
    return spmv_coo(algo, A, x, y);
  case sparse_format_coo_sorted:
    return spmv_coo_sorted(algo, A, x, y);
  case sparse_format_csr:
    return spmv_csr(algo, A, x, y);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid format %d\n",
            __FILE__, __LINE__, A.format);
    return 0;
  }
}

/*********************************************************
 *
 * square norm of a vector
 *
 *********************************************************/

/** 
    @brief square norm of a vector in serial
    @param (v) a vector
    @return the square norm of v (v[0]^2 + ... + v[n-1]^2)
*/
static real vec_norm2_serial(vec_t v) {
  real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
  for (idx_t i = 0; i < n; i++) {
    s += x[i] * x[i];
  }
  return s;
}

#include "include/vec_norm2_parallel.cc"
#if __NVCC__
#include "include/vec_norm2_cuda.cc"
#endif
#include "include/vec_norm2_task.cc"
#include "include/vec_norm2_udr.cc"

/** 
    @brief square norm of a vector with the specified algorithm
    @param (algo) algorithm (serial, parallel, task, cuda, ...)
    @param (v) a vector
    @return the square norm of v (v[0]^2 + ... + v[n-1]^2)
*/
static real vec_norm2(spmv_algo_t algo, vec_t v) {
  switch(algo) {
  case spmv_algo_serial:
    return vec_norm2_serial(v);
  case spmv_algo_parallel:
    return vec_norm2_parallel(v);
#if __NVCC__
  case spmv_algo_cuda:
    return vec_norm2_cuda(v);
#endif
  case spmv_algo_task:
    return vec_norm2_task(v);
  case spmv_algo_udr:
    return vec_norm2_udr(v);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algo %d\n",
            __FILE__, __LINE__, algo);
    return -1.0;
  }
}
  
/** 
    @brief k x v in serial
    @param (k) a scalar
    @param (v) a vector
    @return 1 if succeed, 0 if failed
    @details multiply each element of v by k
*/
static int scalar_vec_serial(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems;
  for (idx_t i = 0; i < n; i++) {
    x[i] *= k;
  }
  return 1;
}

#include "include/scalar_vec_parallel.cc"
#if __NVCC__
#include "include/scalar_vec_cuda.cc"
#endif
#include "include/scalar_vec_task.cc"
#include "include/scalar_vec_udr.cc"

/** 
    @brief k x v in parallel with the specified algorithm
    @param (algo) algorithm (serial, parallel, task, cuda, etc.)
    @param (k) a scalar
    @param (v) a vector
    @details multiply each element of v by k
*/
static int scalar_vec(spmv_algo_t algo, real k, vec_t v) {
  switch(algo) {
  case spmv_algo_serial:
    return scalar_vec_serial(k, v);
  case spmv_algo_parallel:
    return scalar_vec_parallel(k, v);
#if __NVCC__
  case spmv_algo_cuda:
    return scalar_vec_cuda(k, v);
#endif
  case spmv_algo_task:
    return scalar_vec_task(k, v);
  case spmv_algo_udr:
    return scalar_vec_udr(k, v);
  default:
    fprintf(stderr,
            "error:%s:%d: invalid algo %d\n",
            __FILE__, __LINE__, algo);
    return 0;
  }
}
  
/** 
    @brief normalize a vector with the specified algortihm
    @param (algo) algorithm (serial, parallel, task, cuda, etc.)
    @param (v) a vector
    @return |v|
    @details make v a unit-length vector (i.e., v = v/|v|)
*/
static real vec_normalize(spmv_algo_t algo, vec_t v) {
  real s2 = vec_norm2(algo, v);
  if (s2 < 0.0) return -1.0;
  real s = sqrt(s2);
  if (!scalar_vec(algo, 1/s, v)) return -1.0;
  return s;
}

/** 
    @brief repeat y = A x; x = tA y; many times, with the
    specified algorithm
    @param (algo)
    @param (A) the reference to a sparse vector
    @param (tA) the reference to A's transpose
    @param (x) the reference to a vector
    @param (y) the reference to a vector
    @param (repeat) the number of times to repeat
    @return the largest singular value of A (= the largest
    eigenvalue of (tA A))
    @details it repeats, (repeat + 1) times, 
        y = Ax; x = tA y; x = x/|x|;
    the first iteration is the "warm-up", which
    checks if it does not encounter an error
    (e.g., kernel launch failure) and makes sure
    to touch all allocated memory once.

    in the end of each iteration it takes |x| and returns that
    of the last iteration, which is the largest
    singular value of A if enough iterations have been made.
    
*/
static real repeat_spmv(spmv_algo_t algo,
                        sparse_t& A, sparse_t& tA,
                        vec_t& x, vec_t& y, idx_t repeat) {
#if __NVCC__
  if (algo == spmv_algo_cuda) {
    /* make device copies of matrix and vectors */
    sparse_to_dev(A);
    sparse_to_dev(tA);
    vec_to_dev(x);
    vec_to_dev(y);
  }
#endif
  
  printf("%s:%d:repeat_spmv: warm up + error check starts\n", __FILE__, __LINE__);
  fflush(stdout);
  long t0 = cur_time_ns();
  /* y = A x and check error */
  if (!spmv(algo, A, x, y))
    return -1.0;
  /* x = tA y and check error */
  if (!spmv(algo, tA, y, x)) 
    return -1.0;
  /* x = x/|x| and check error */
  if (vec_normalize(algo, x) < 0.0)
    return -1.0;
  long t1 = cur_time_ns();
  printf("%s:%d:repeat_spmv: warm up + error check ends. took %.3f sec\n",
         __FILE__, __LINE__, (t1 - t0) * 1.0e-9);

  /* the real iterations to measure */
  printf("%s:%d:repeat_spmv: main loop starts\n", __FILE__, __LINE__);
  fflush(stdout);
  long nnz = A.nnz;
  real lambda = 0.0;
  long flops = (4 * (long)nnz + 3 * (long)x.n) * (long)repeat;
  long t2 = cur_time_ns();
  for (idx_t r = 0; r < repeat; r++) {
    spmv(algo,  A, x, y); /* y = A * x   (2 nnz flops) */
    spmv(algo, tA, y, x); /* x = tA * y  (2 nnz flops) */
    lambda = vec_normalize(algo, x); /* x = x/|x| (and lambda = |x|) */
  }
  long t3 = cur_time_ns();
  long dt = t3 - t2;
  printf("%s:%d:repeat_spmv: main loop ends\n", __FILE__, __LINE__);
  printf("%ld flops in %.6f sec (%.6f GFLOPS)\n",
         flops, dt*1.0e-9, flops/(double)dt);
  return lambda;
}
  
/** 
    @brief make a random vector of n elements
    @param (n) the number of elements of the vector
    @param (rg) random number generator state (passed to erand48)
    @return a vector of n elements
*/
static vec_t mk_vec_random(idx_t n, unsigned short rg[3]) {
  real * x = (real *)xalloc(sizeof(real) * n);
  for (idx_t i = 0; i < n; i++) {
    x[i] = erand48(rg);
  }
  vec_t v = { n, x };
  return v;
}

/** 
    @brief make a random unit-length vector of n elements
    @param (n) the number of elements of the vector
    @param (rg) random number generator state (passed to erand48)
    @return a vector of n elements
*/
static vec_t mk_vec_unit_random(idx_t n, unsigned short rg[3]) {
  vec_t x = mk_vec_random(n, rg);
  vec_normalize(spmv_algo_serial, x);
  return x;
}

/** 
    @brief make a zero vector of n elements
    @param (n) the number of elements of the vector
    @return a zero vector of n elements
*/
static vec_t mk_vec_zero(idx_t n) {
  real * x = (real *)xalloc(sizeof(real) * n);
  for (idx_t i = 0; i < n; i++) {
    x[i] = 0.0;
  }
  vec_t v = { n, x };
  return v;
}

/** 
    @brief compare two elements in an array of idx_t 
    @param (a_) the pointer to an element 1
    @param (b_) the pointer to an element 2
*/
static int cmp_idx_fun(const void * a_, const void * b_) {
  idx_t * a = (idx_t *)a_;
  idx_t * b = (idx_t *)b_;
  return *a - *b;
}

/** 
    @brief dump a sparse matrix A into img_width x img_height bitmap
    (gnuplot file) with the specified filename.
    @param (A) a sparse matrix to dump
    @param (file) the file name to dump A into 
    @param (max_points) the maximum number of points dumped into the file
    @param (seed) the random number seed to choose elements to dump
*/

static int dump_sparse_file(sparse_t A, char * file, idx_t max_points, long seed) {
  printf("%s:%d:dump_sparse_file:"
         " dumping to matrix %ld x %ld (%ld nnz) -> %s\n",
         __FILE__, __LINE__,
         (long)A.M, (long)A.N, (long)A.nnz, file);
  fflush(stdout);
  sparse_t B = sparse_any_to_any(A, sparse_format_coo);
  idx_t M = B.M;
  idx_t N = B.N;
  idx_t nnz = B.nnz;
  coo_elem_t * elems = B.coo.elems;

  idx_t * row_nnz = (idx_t *)malloc(sizeof(idx_t) * M);
  for (idx_t i = 0; i < M; i++) {
    row_nnz[i] = 0;
  }
  /* randomly choose up to max_points */
  unsigned short rg[3] = {
    (unsigned short)((seed >> 32) & ((1 << 16) - 1)),
    (unsigned short)((seed >> 16) & ((1 << 16) - 1)),
    (unsigned short)((seed >> 0 ) & ((1 << 16) - 1)),
  };
  idx_t * chosen = (idx_t *)malloc(sizeof(idx_t) * max_points);
  idx_t n_points = 0;
  for (idx_t k = 0; k < nnz; k++) {
    /* count non-zeros in each row */
    idx_t i = elems[k].i;
    row_nnz[i]++;
    /* choose it with an equal probability */
    if (n_points < max_points) {
      chosen[n_points] = k;
      n_points++;
    } else {
      if (erand48(rg) < n_points / (double)max_points) {
        idx_t replaced = nrand48(rg) % n_points;
        chosen[replaced] = k;
      }
    }
  }
  qsort((void *)chosen, n_points, sizeof(idx_t), cmp_idx_fun);
    
  FILE * wp = fopen(file, "w");
  if (!wp) {
    perror(file);
    return 0;
  }
  fprintf(wp, "# add -e 'term=\"png\"' etc. to the gnuplot commad line to generate a file instead of showing it on the screen. e.g. gnuplot -e 'term=\"png\"' x.gnuplot\n");
  fprintf(wp, "if (exists(\"term\")) set terminal term\n");
  
  fprintf(wp, "# add -e 'nnz_mat=\"FILENAME\"' etc. to the gnuplot commad line to output non-zero matrix to the specified file name . e.g. gnuplot -e 'term=\"png\"' -e 'nnz_mat=\"nnz_mat.png\"' x.gnuplot\n");
  fprintf(wp, "if (exists(\"nnz_mat\")) set output nnz_mat\n");
  fprintf(wp, "set title \"nnz_mat : matrix of non zero elements\"\n");
  fprintf(wp, "set xlabel \"row\"\n");
  fprintf(wp, "set xrange [0:%ld]\n", (long)M);
  fprintf(wp, "set ylabel \"column\"\n");
  fprintf(wp, "set yrange [0:%ld]\n", (long)N);
  fprintf(wp, "$mat << EOD\n");
  for (idx_t i = 0; i < n_points; i++) {
    idx_t k = chosen[i];
    coo_elem_t * e = elems + k;
    fprintf(wp, "%ld %ld %f\n", (long)e->i, (long)e->j, e->a);
  }
  fprintf(wp, "EOD\n");
  fprintf(wp, "plot '$mat' with points\n");
  fprintf(wp, "if (!exists(\"nnz_mat\")) pause -1\n");

  /* reset everything to go ahead to the second graph */
  fprintf(wp, "reset\n");
  
  /* non-zero distribution */
  fprintf(wp, "# add -e 'nnz_row=\"FILENAME\"' etc. to the gnuplot commad line to output non-zero distribution over rows. e.g. gnuplot -e 'term=\"png\"' -e 'nnz_mat=\"nnz_mat.png\"' x.gnuplot\n");
  fprintf(wp, "if (exists(\"nnz_row\")) set output nnz_row\n");
  
  fprintf(wp, "set title \"nnz_row : the number of non zeros in each row\"\n");
  fprintf(wp, "set xlabel \"row\"\n");
  fprintf(wp, "set xrange [0:%ld]\n", (long)M);
  fprintf(wp, "set ylabel \"the number of non zeros\"\n");
  fprintf(wp, "set yrange [0:]\n");
  fprintf(wp, "$row_nnz << EOD\n");
  for (idx_t i = 0; i < M; i++) {
    fprintf(wp, "%ld %ld\n", (long)i, (long)row_nnz[i]);
  }
  fprintf(wp, "EOD\n");
  fprintf(wp, "plot '$row_nnz' with lines title \"\"\n");
  fprintf(wp, "if (!exists(\"nnz_row\")) pause -1\n");
  
  fclose(wp);
  xfree(row_nnz);
  printf("%s:%d:dump_sparse_file: done\n",
         __FILE__, __LINE__);
  fflush(stdout);
  return 1;
}




/** 
    @brief the main function
*/
int main(int argc, char ** argv) {
  cmdline_options_t opt = parse_args(argc, argv);
  if (opt.help || opt.error) {
    usage(argv[0]);
    exit(opt.error);
  }
  idx_t M        = opt.M;
  idx_t N        = (opt.N ? opt.N : M);
  idx_t nnz      = (opt.nnz ? opt.nnz : ((long)M * (long)N + 99L) / 100L);
  long repeat    = opt.repeat;
  unsigned short rg[3] = {
    (unsigned short)((opt.seed >> 32) & ((1 << 16) - 1)),
    (unsigned short)((opt.seed >> 16) & ((1 << 16) - 1)),
    (unsigned short)((opt.seed >> 0 ) & ((1 << 16) - 1)),
  };
  printf("A : %ld x %ld, %ld requested non-zeros\n",
         (long)M, (long)N, (long)nnz);
  printf("repeat : %ld times\n", repeat);
  printf("format : %s\n", opt.format_str);
  printf("matrix : %s\n", opt.matrix_type_str);
  printf("algo : %s\n", opt.algo_str);

  //sparse_t A = mk_sparse_random(opt.format, M, N, nnz, rg);
  sparse_t A = mk_sparse_matrix(opt, M, N, nnz, rg);
  if (opt.dump) {
    dump_sparse_file(A, opt.dump, opt.dump_points, opt.dump_seed);
  }
  sparse_t tA = sparse_transpose(A);
  printf("%s:%d:main A is %ld x %ld, has %ld non-zeros and takes %ld bytes\n",
         __FILE__, __LINE__,
         (long)A.M, (long)A.N, (long)A.nnz, sparse_size(A));
  printf("%s:%d:main tA is %ld x %ld, has %ld non-zeros and takes %ld bytes\n",
         __FILE__, __LINE__,
         (long)tA.M, (long)tA.N, (long)tA.nnz, sparse_size(A));
  vec_t x = mk_vec_unit_random(N, rg);
  vec_t y = mk_vec_zero(M);
  real lambda = repeat_spmv(opt.algo, A, tA, x, y, repeat);
  if (lambda == -1.0) {
    printf("an error ocurred during repeat_spmv\n");
  } else {
    printf("lambda = %.9e\n", lambda);
  }
  vec_destroy(x);
  vec_destroy(y);
  sparse_destroy(A);
  sparse_destroy(tA);
  cmdline_options_destroy(opt);
  return 0;
}

