//% file: mm_main.h
/* 
 * mm_main.h
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <x86intrin.h>
#include "clock.h"

/* type definition */
typedef float real;
typedef long idx_t;

/* a bit of portability across 512 bit and 256 bit SIMD */
#if __AVX512F__
enum { vwidth = 64 };
#elif __AVX__
enum { vwidth = 32 };
#else
#error "__AVX512F__ or __AVX__ must be defined"
#endif
enum {
      valign = sizeof(real),
      //valign = vwidth
};
typedef real realv __attribute__((vector_size(vwidth),aligned(valign)));
enum { L = sizeof(realv) / sizeof(real) };

#if __AVX512F__
realv set1(float a) {
  return _mm512_set1_ps(a);
}
#elif __AVX__
realv set1(float a) {
  return _mm256_set1_ps(a);
}
#else
#error "__AVX512F__ or __AVX__ must be defined"
#endif

/* a simple matrix class which allows you to access elements
   with a(i,j);

   matrix_c<M,N,ld> A;
   for(i=0; i<M; i++) 
     for(j=0; j<N; j++) 
       ... A(i,j) ...

   ld is the number of elements between A(i,x) and A(i+1,x);
   normally it is N, but you may make it different (larger than N)
   if you wish
 */


#define CHECK_IDX 0

/* matrix with constant size and leading dimension */
template<idx_t nR,idx_t nC,idx_t ld>
struct matrix_c {
  //real a[nR][ld] __attribute__((aligned(vwidth)));
  //real a[nR * ld] __attribute__((aligned(vwidth)));
  real * a;
  matrix_c() {
    a = (real *)aligned_alloc(vwidth, sizeof(real) * nR * ld);
  }
  matrix_c(real * _a) {
    a = _a;
  }
  template<idx_t nR0, idx_t nC0>
  matrix_c<nR0,nC0,ld> sub(idx_t i, idx_t j) {
    matrix_c<nR0,nC0,ld> s(&a[i * ld + j]);
    return s;
  }
  /* return a scalar A(i,j) */
  real& operator() (idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < nR);
    assert(j < nC);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return a[i * ld + j];
  }
  /* return a vector at A(i,j) (i.e., A(i,j:j+L) */
  realv& v(idx_t i, idx_t j) {
#if CHECK_IDX
    assert(i < nR);
    assert(j < nC);
    assert(i >= 0);
    assert(j >= 0);
#endif
    return *((realv*)&a[i * ld + j]);
  }
  void rand_init(unsigned short rg[3]) {
    for (idx_t i = 0; i < nR; i++) {
      for (idx_t j = 0; j < nC; j++) {
	(*this)(i,j) = erand48(rg);
      }
    }
  }
  /* initialize all elements by c */
  void const_init(real c) {
    for (idx_t i = 0; i < nR; i++) {
      for (idx_t j = 0; j < nC; j++) {
	(*this)(i,j) = c;
      }
    }
  }
  void zero() {
    const_init(0.0);
  }
};

/* just for randamly checking results.
   compute C(i,j) of C = AB */
template<idx_t M,idx_t N,idx_t K,idx_t lda,idx_t ldb>
static real comp_ij(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B,
                    idx_t i, idx_t j, long times) {
  real s = 0.0;
  for (long t = 0; t < times; t++) {
    asm volatile("# comp_ij K loop begins");
    for (idx_t k = 0; k < K; k++) {
      s += A(i,k) * B(k,j);
    }
    asm volatile("# comp_ij K loop ends");
  }
  return s;
}

/* access a large array to wipe the cache */
static char * wipe_cache(int x) {
  static char * a = 0;
  long n = 100 * 1000 * 1000;
  if (!a) a = (char *)malloc(n);
  memset(a, x, n);
  return a;
}

/* template declaration of gemm */
template<idx_t M,idx_t N,idx_t K,
         idx_t lda,idx_t ldb,idx_t ldc,
         idx_t bM,idx_t bN>
long gemm(matrix_c<M,K,lda>& A, matrix_c<K,N,ldb>& B, matrix_c<M,N,ldc>& C);

int main(int argc, char ** argv) {
  long approx_fmas = (argc > 1 ? atol(argv[1]) : 10L * 1000L * 1000L * 1000L);
  long chk   = (argc > 2 ? atol(argv[2]) : 1);
  long seed  = (argc > 3 ? atol(argv[3]) : 76843802738543);

  const idx_t bM = 8;
  const idx_t bN = 2 * L;       // 16 on xxxbridge, 32 on skylake-x
  const idx_t bK = 240;

  const idx_t M1 = bM * 64;
  const idx_t N1 = bN * 10;
  const idx_t K1 = bK;

  const idx_t M = M1 * 16;
  const idx_t N = N1 * 24;
  const idx_t K = K1 * 1;

  const idx_t lda = K;
  const idx_t ldb = N;
  const idx_t ldc = N;
  
  assert(K <= lda);
  assert(N <= ldb);
  assert(N <= ldc);

#if 0
  matrix_c<M,K,lda> * hA = new matrix_c<M,K,lda>();
  matrix_c<K,N,ldb> * hB = new matrix_c<K,N,ldb>();
  matrix_c<M,N,ldc> * hC = new matrix_c<M,N,ldc>();
#endif
  matrix_c<M,K,lda> A;// = *hA;
  matrix_c<K,N,ldb> B;// = *hB;
  matrix_c<M,N,ldc> C;// = *hC;

  unsigned short rg[3] = { (unsigned short)((seed >> 16) & 65535),
			   (unsigned short)((seed >> 8)  & 65535),
			   (unsigned short)((seed >> 0)  & 65535) };
  const long fmas  = (long)M * (long)N * (long)K;
  const long flops = 2 * fmas;
  const long times = (approx_fmas + fmas - 1) / fmas;
  const long flops_all = flops * times;
  A.rand_init(rg);
  B.rand_init(rg);
  C.zero();
  printf("M = %ld, N = %ld, K = %ld\n", (long)M, (long)N, (long)K);
  printf("L : %ld\n", (long)L);
  printf("A : %ld x %ld (ld=%ld) %ld bytes\n",
         (long)M, (long)K, (long)lda, (long)M * (long)K * sizeof(real));
  printf("B : %ld x %ld (ld=%ld) %ld bytes\n",
         (long)K, (long)N, (long)ldb, (long)K * (long)N * sizeof(real));
  printf("C : %ld x %ld (ld=%ld) %ld bytes\n",
         (long)M, (long)N, (long)ldc, (long)M * (long)N * sizeof(real));
  printf("total = %ld bytes\n",
	 ((long)M * (long)K + (long)K * (long)N + (long)M * (long)N) * sizeof(real));
  char * wipe = wipe_cache(0);
  printf("repeat : %ld times\n", times);
  printf("perform %ld flops ... ", flops_all); fflush(stdout);

  clock_counters_t cc = mk_clock_counters();
  clocks_t c0 = clock_counters_get(cc);

  /* real thing happens here */
  for (long i = 0; i < times; i++) {
    gemm<M,N,K,lda,ldb,ldc,bM,bN>(A, B, C);
  }
  /* real thing ends here */

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

  printf("done \n");
  printf("\n");
  printf("%ld nsec\n", wall);
  printf("%ld ref clocks\n", ref);
  if (cpu) {
    printf("%ld cpu clocks\n", cpu);
  } else {
    printf("-------- cpu clocks\n");
  }
  printf("\n");
  printf("%f flops/nsec\n",      flops_all / (double)wall);
  printf("%f flops/ref clock\n", flops_all / (double)ref);
  if (cpu) {
    printf("%f flops/cpu clock\n", flops_all / (double)cpu);
  } else {
    printf("-------- flops/cpu clock\n");
  }

  if (chk) {
    idx_t i = nrand48(rg) % M;
    idx_t j = nrand48(rg) % N;
    real s = comp_ij(A, B, i, j, times);
    printf("C(%ld,%ld) = %f, ans = %f, |C(%ld,%ld) - s| = %.9f\n",
	   (long)i, (long)j, C(i,j), s,
           (long)i, (long)j, fabs(C(i,j) - s));
  }
  if (wipe) free(wipe);
  clock_counters_destroy(cc);
  return 0;
}
