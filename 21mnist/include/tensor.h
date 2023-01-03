/**
   @file tensor.h
   @brief vectors, matrices, and multi-dimensional arrays
 */

#pragma once

#include <stdio.h>
#ifndef ARRAY_INDEX_CHECK
#define ARRAY_INDEX_CHECK 1
#endif
#include "mnist_util.h"

/**
   @brief aux function for array bounds checking
   @param (a) the lower bound
   @param (x) the index
   @param (b) the upper bound + 1
   @param (a_str) the string representation of a
   @param (x_str) the string representation of x
   @param (b_str) the string representation of b
   @param (file) the file name from which it gets called
   @param (line) the line number from which it gets called
   @details it checks if a<=x<b holds. if not, it shows an 
   error message. on cpu, the error message contains
   the location where the error happened.
 */
__attribute__((unused))
__device__ __host__ 
static void range_chk_(idx_t a, idx_t x, idx_t b,
                       const char * a_str, const char * x_str,
                       const char * b_str, 
                       const char * file, int line) {
#if ! __CUDA_ARCH__
  if (!(a <= x)) {
    fprintf(stderr,
            "error:%s:%d: index check [%s <= %s < %s] failed"
            " (%s = %ld)\n",
            file, line, a_str, x_str, b_str, x_str, (long)x);
  }
#endif
  assert(a <= x);
#if ! __CUDA_ARCH__
  if (!(x < b)) {
    fprintf(stderr,
            "error:%s:%d: index check [%s <= %s < %s] failed"
            " (%s = %ld)\n",
            file, line, a_str, x_str, b_str, x_str, (long)x);
  }
#endif
  assert(x < b);
}

#if ARRAY_INDEX_CHECK
/** 
    @brief array bounds check. turn off (on) if -DARRAY_INDEX_CHECK=0 (1) is given.
    turn it on when you are debugging your code.
    turn it off when you are measuring the performance.
*/
#define range_chk(a, x, b) range_chk_(a, x, b, #a, #x, #b, __FILE__, __LINE__)
#else
/** 
    @brief array bounds check. turn off (on) if -DARRAY_INDEX_CHECK=0 (1) is given.
    turn it on when you are debugging your code.
    turn it off when you are measuring the performance.
*/
#define range_chk(a, x, b) 
#endif


/**
   @brief tensor (multi-dimensional array), up to four dimensions
   @param (maxB) the maximum number of rows (elements along the first dimension)
   @param (C) the number of elements along the second dimension
   @param (H) the number of elements along the third dimension
   @param (W) the number of elements along the fourth dimension
   @details this is essentially BxCxHxW array of reals where B 
   can be a runtime parameter <= maxB.
   throughout the MNIST network, is is used to represent a mini-batch
   of images (B images, each image of which has C channels, each channel
   of which has HxW pixels.
*/
template<typename T,idx_t N0,idx_t N1=1,idx_t N2=1,idx_t N3=1>
struct tensor {
#if __NVCC__
  tensor<T,N0,N1,N2,N3> * dev;     /**< pointer to the device shadow */
#endif
  idx_t n0;                      /**< actual number of elements across the first dimension */
  T w[N0][N1][N2][N3];           /**< elements */
  /**
     @brief access the (b,c,i,j) element
     @param (b) the first index (image index in a mini batch)
     @param (c) the second index (channe index in an image)
     @param (i) the third index (row index in a channel)
     @param (j) the fourth index (column index in a row)
  */
  __device__ __host__ 
  T& operator()(idx_t i0, idx_t i1=0, idx_t i2=0, idx_t i3=0) {
    range_chk(0, i0, n0);
    range_chk(0, i1, N1);
    range_chk(0, i2, N2);
    range_chk(0, i3, N3);
    return w[i0][i1][i2][i3];
  }
  /**
     @brief set the number of elements along the first dimension
     @param (N) the number of elements specified
  */
  __device__ __host__ 
  void set_n0(idx_t n0) {
    assert(n0 <= N0);
    this->n0 = n0;
  }
  /**
     @brief initialize elements of the array  to a single constant value
     @param (B) the number of rows to initialize
     @param (x) the value of each element
  */
  void init_const(idx_t n0, T x) {
    set_n0(n0);
    tensor<T,N0,N1,N2,N3>& a = *this;
    asm volatile("# init_const begins N0=%0 N1=%1 N2=%2 N3=%3" : : "i" (N0), "i" (N1), "i" (N2), "i" (N3));
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) = x;
          }
        }
      }
    }
    asm volatile("# init_const ends N0=%0 N1=%1 N2=%2 N3=%3" : : "i" (N0), "i" (N1), "i" (N2), "i" (N3));
  }
   /**
     @brief randomly initialize elements of the array  uniformly between p and q
     @param (B) the number of rows to initialize
     @param (rg) random number generator
     @param (p) the minimum value
     @param (q) the maximum value
  */
  void init_uniform(idx_t n0, rnd_gen_t& rg, T p, T q) {
    set_n0(n0);
    tensor<T,N0,N1,N2,N3>& a = *this;
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) = rg.rand(p, q);
          }
        }
      }
    }
  }
  /**
     @brief randomly initialize elements of the array  uniformly between p and q
     @param (B) the number of rows to initialize
     @param (rg) random number generator
     @param (p) the minimum value
     @param (q) the maximum value
  */
  void init_uniform_i(idx_t n0, rnd_gen_t& rg, T p, T q) {
    set_n0(n0);
    tensor<T,N0,N1,N2,N3>& a = *this;
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) = rg.randi(p, q);
          }
        }
      }
    }
  }
  /**
     @brief initialize the array with a normal distribution
     @param (B) the number of rows to initialize
     @param (rg) random number generator
     @param (mu) mean of the normal distribution
     @param (sigma) sigma (standard deviation) of the normal distribution
  */
  void init_normal(idx_t n0, rnd_gen_t& rg, real mu, real sigma) {
    set_n0(n0);
    tensor<T,N0,N1,N2,N3>& a = *this;
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) = rg.rand_normal(mu, sigma);
          }
        }
      }
    }
  }
  /**
     @brief in-place update a += alpha * b;
   */
  __device__ __host__
  tensor<T,N0,N1,N2,N3>& add_(T alpha, tensor<T,N0,N1,N2,N3>& b) {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 == b.n0);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) += alpha * b(i0,i1,i2,i3);
          }
        }
      }
    }
    return a;
  }
  /**
     @brief in-place update a *= alpha
   */
  __device__ __host__
  tensor<T,N0,N1,N2,N3>& mul_(T alpha) {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 > 0);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) *= alpha;
          }
        }
      }
    }
    return a;
  }
  /**
     @brief in-place update a *= alpha
   */
  __device__ __host__
  tensor<T,N0,N1,N2,N3>& mul_(tensor<T,N0,N1,N2,N3>& b) {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 == b.n0);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) *= b(i0,i1,i2,i3);
          }
        }
      }
    }
    return a;
  }
  /**
     @brief in-place update a /= b;
   */
  __device__ __host__
  tensor<T,N0,N1,N2,N3>& div_(tensor<T,N0,N1,N2,N3>& b) {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 == b.n0);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) /= b(i0,i1,i2,i3);
          }
        }
      }
    }
    return a;
  }
  /**
     @brief in-place update a = sqrt(a);
   */
  __device__ __host__
  tensor<T,N0,N1,N2,N3>& sqrt_() {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 > 0);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) = sqrt(a(i0,i1,i2,i3));
          }
        }
      }
    }
    return a;
  }
  /**
     @brief in-place update a += alpha * b * c;
   */
  __device__ __host__
  tensor<T,N0,N1,N2,N3>& addcmul_(T alpha, tensor<T,N0,N1,N2,N3>& b, tensor<T,N0,N1,N2,N3>& c) {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 == b.n0);
    assert(a.n0 == c.n0);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            a(i0,i1,i2,i3) += alpha * b(i0,i1,i2,i3) * c(i0,i1,i2,i3);
          }
        }
      }
    }
    return a;
  }
  /**
     @brief b = a + alpha
   */
  __device__ __host__
  tensor<T,N0,N1,N2,N3>& add(T alpha, tensor<T,N0,N1,N2,N3>& b) {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 == b.n0);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            b(i0,i1,i2,i3) = a(i0,i1,i2,i3) + alpha;
          }
        }
      }
    }
    return b;
  }

  /**
     @brief dot product with another array
     @param (a_) the array to take a dot product with
  */
  real sum() {
    tensor<T,N0,N1,N2,N3>& a = *this;
    real s0 = 0.0;
    for (idx_t i0 = 0; i0 < n0; i0++) {
      real s1 = 0.0;
      for (idx_t i1 = 0; i1 < N1; i1++) {
        real s2 = 0.0;
        for (idx_t i2 = 0; i2 < N2; i2++) {
          real s3 = 0.0;
          for (idx_t i3 = 0; i3 < N3; i3++) {
            s3 += a(i0,i1,i2,i3);
          }
          s2 += s3;
        }
        s1 += s2;
      }
      s0 += s1;
    }
    return s0;
  }
  /**
     @brief dot product with another array
     @param (a_) the array to take a dot product with
  */
  double dot(tensor<T,N0,N1,N2,N3>& b) {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 == b.n0);
    double s0 = 0.0;
    for (idx_t i0 = 0; i0 < n0; i0++) {
      double s1 = 0.0;
      for (idx_t i1 = 0; i1 < N1; i1++) {
        double s2 = 0.0;
        for (idx_t i2 = 0; i2 < N2; i2++) {
          double s3 = 0.0;
          for (idx_t i3 = 0; i3 < N3; i3++) {
            s3 += a(i0,i1,i2,i3) * b(i0,i1,i2,i3);
          }
          s2 += s3;
        }
        s1 += s2;
      }
      s0 += s1;
    }
    return s0;
  }
  /**
     
   */
  void print() {
    tensor<T,N0,N1,N2,N3>& a = *this;
    assert(a.n0 > 0);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            printf("[%d][%d][%d][%d] : %f\n",
                   i0, i1, i2, i3, a(i0,i1,i2,i3));
          }
        }
      }
    }
  }
  /**
     @brief set the device shadow of this array
     @param (dev) device address (may be null)
   */
  void set_dev(tensor<T,N0,N1,N2,N3>* dev) {
#if __NVCC__
    this->dev = dev;
#else
    (void)dev;
#endif
  }

#if 1
  /**
     @brief allocate and set the device shadow of this array if
     requested by the parameter
     @param (gpu) 1 to allocate the device shadow
   */
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (tensor<T,N0,N1,N2,N3>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  /**
     @brief deallocate the device shadow of this array
   */
  void del_dev() {
#if __NVCC__
    if (dev) {
      dev_free(dev);
      dev = 0;
    }
#endif
  }
  /**
     @brief send the data to its gpu shadow
   */
  void to_dev() {
#if __NVCC__
    if (dev) {
      ::to_dev(dev, this, sizeof(*this));
    }
#endif
  }
  /**
     @brief get the data back from gpu shadow to host
   */
  void to_host() {
#if __NVCC__
    if (dev) {
      tensor<T,N0,N1,N2,N3> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }

#endif
};

/**
   @brief entry point
 */
int tensor_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  (void)opt;
  const int N = 2;
  const int n = 2;
  tensor<real,N> a1;
  tensor<real,N,N> a2;
  tensor<real,N,N,N> a3;
  tensor<real,N,N,N,N> a4;

  a1.init_const(n, 1);
  a2.init_const(n, 1);
  a3.init_const(n, 1);
  a4.init_const(n, 1);

  real x1 = a1.dot(a1);
  assert(x1 == n);
  real x2 = a2.dot(a2);
  assert(x2 == n * N);
  real x3 = a3.dot(a3);
  assert(x3 == n * N * N);
  real x4 = a4.dot(a4);
  assert(x4 == n * N * N * N);

  const int M = 10000;
  const int a = 3;
  const int b = 8;
  rnd_gen_t rg;
  rg.seed(12345);
  tensor<real,M> c;
  c.init_uniform_i(M, rg, a, b);
  for (long i = 0; i < M; i++) {
    assert(a <= c(i));
    assert(c(i) < b);
  }
  return 0;
}

