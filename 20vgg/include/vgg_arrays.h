/**
   @file vgg_arrays.h
   @brief vectors, matrices, and multi-dimensional arrays
 */

#pragma once

#include <stdio.h>
#ifndef ARRAY_INDEX_CHECK
#define ARRAY_INDEX_CHECK 1
#endif
#include "vgg_util.h"

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
   @brief vector
   @param (N) the maximun number of elements it can hold
 */
template<idx_t N>
struct vec {
#if __NVCC__
  vec<N> * dev;                 /**< pointer to the device shadow */
#endif
  idx_t n;                      /**< actual number of clements (<= N)  */
  real w[N];                    /**< elements */
  /**
     @brief access the i-th element
     @param (i) the index of the element to access
  */
  __device__ __host__ 
  real& operator()(idx_t i) {
    range_chk(0, i, n);
    return w[i];
  }
  /**
     @brief set the number of elements
     @param (n) the number of elements specified
  */
  __device__ __host__ 
  void set_n(idx_t n) {
    this->n = n;
    assert(n <= N);
  }
  /**
     @brief initialize elements of the vector to a single constant value
     @param (n) the number of elements to initialize
     @param (c) the value of each element
  */
  __device__ __host__ 
  void init_const(idx_t n, real c) {
    set_n(n);
    vec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = c;
    }
  }
  /**
     @brief randomly initialize elements of the vector uniformly between p and q
     @param (n) the number of elements to initialize
     @param (rg) random number generator
     @param (p) the minimum value
     @param (q) the maximum value
  */
  void init_uniform(idx_t n, rnd_gen_t& rg, real p, real q) {
    set_n(n);
    vec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = rg.rand(p, q);
    }
  }
  /**
     @brief randomly initialize a randomly chosen single element. other elements are set to zero.
     @param (n) the number of elements to initialize
     @param (rg) random number generator
     @param (p) the minimum value
     @param (q) the maximum value
  */
  void init_single(idx_t n, rnd_gen_t& rg, real p, real q) {
    vec<N>& x = *this;
    x.init_const(n, 0);
    idx_t i = rg.randi(0, n);
    x(i) = rg.rand(p, q);
  }
  /**
     @brief initialize the vector with a normal distribution
     @param (n) the number of elements to initialize
     @param (rg) random number generator
     @param (mu) mean of the normal distribution
     @param (sigma) sigma (standard deviation) of the normal distribution
  */
  void init_normal(idx_t n, rnd_gen_t& rg, real mu, real sigma) {
    set_n(n);
    vec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = rg.rand_normal(mu, sigma);
    }
  }
  /**
     @brief update the vector by eta and dx
     @param (eta) a constant to scale dx
     @param (dx) the increment vector
     @details it performx x += eta * dx
  */
  __device__ __host__
  void update(real eta, vec<N>& dx) {
    // this += eta * dv
    vec<N>& x = *this;
    assert(x.n == dx.n);
    for (idx_t i = 0; i < n; i++) {
      x(i) += eta * dx(i);
    }
  }
  /**
     @brief sum of all values of the vector
  */
  real sum() {
    vec<N>& x = *this;
    real s = 0.0;
    for (idx_t i = 0; i < n; i++) {
      s += x(i);
    }
    return s;
  }
  /**
     @brief dot product with another vector
     @param (y) the vector to take a dot product with
  */
  real dot(vec<N>& y) {
    vec<N>& x = *this;
    assert(x.n == y.n);
    real s = 0.0;
    for (idx_t i = 0; i < n; i++) {
      s += x(i) * y(i);
    }
    return s;
  }
  /**
     @brief set the device shadow of this vector
     @param (dev) device address (may be null)
   */
  void set_dev(vec<N> * dev) {
#if __NVCC__
    this->dev = dev;
#else
    (void)dev;
#endif
  }
  /**
     @brief allocate and set the device shadow of this vector if
     requested by the parameter
     @param (gpu) 1 to allocate the device shadow
   */
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (vec<N>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  /**
     @brief deallocate the device shadow of this vector
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
      vec<N> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief integer vector
   @param (N) the maximun number of elements it can hold
 */
template<idx_t N>
struct ivec {
#if __NVCC__
  ivec<N> * dev;                 /**< pointer to the device shadow */
#endif
  idx_t n;                      /**< actual number of clements (<= N)  */
  idx_t w[N];                    /**< elements */
  /**
     @brief access the i-th element
     @param (i) the index of the element to access
  */
  __device__ __host__ 
  idx_t& operator()(idx_t i) {
    range_chk(0, i, n);
    return w[i];
  }
  /**
     @brief set the number of elements
     @param (n) the number of elements specified
  */
  __device__ __host__ 
  void set_n(idx_t n) {
    this->n = n;
    assert(n <= N);
  }
  /**
     @brief initialize elements of the vector to a single constant value
     @param (n) the number of elements to initialize
     @param (c) the value of each element
  */
  void init_const(idx_t n, idx_t c) {
    set_n(n);
    ivec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = c;
    }
  }
  /**
     @brief randomly initialize elements of the vector uniformly between p and q
     @param (n) the number of elements to initialize
     @param (rg) random number generator
     @param (p) the minimum value
     @param (q) the maximum value
  */
  void init_uniform(idx_t n, rnd_gen_t& rg, idx_t p, idx_t q) {
    set_n(n);
    ivec<N>& x = *this;
    for (idx_t i = 0; i < n; i++) {
      x(i) = rg.randi(p, q);
    }
  }
  /**
     @brief set the device shadow of this vector
     @param (dev) device address (may be null)
   */
  void set_dev(ivec<N> * dev) {
#if __NVCC__
    this->dev = dev;
#else
    (void)dev;
#endif
  }
  /**
     @brief allocate and set the device shadow of this vector if
     requested by the parameter
     @param (gpu) 1 to allocate the device shadow
   */
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (ivec<N>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  /**
     @brief deallocate the device shadow of this vector
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
      ivec<N> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief matrix (2D array)
   @param (M) the maximun number of rows it can hold
   @param (N) the number of elements in each row
 */
template<idx_t M,idx_t N>
struct array2 {
#if __NVCC__
  array2<M,N> * dev;            /**< pointer to the device shadow */
#endif
  idx_t m;                   /**< actual number of clements (<= M)  */
  real w[M][N];              /**< elements */
  /**
     @brief access the (i,j) element
     @param (i) the first index
     @param (j) the second index
  */
  __device__ __host__ 
  real& operator()(idx_t i, idx_t j) {
    range_chk(0, i, m);
    range_chk(0, j, N);
    return w[i][j];
  }
  /**
     @brief set the number of rows
     @param (m) the number of rows specified
  */
  __device__ __host__ 
  void set_n_rows(idx_t m) {
    this->m = m;
    assert(m <= M);
  }
  /**
     @brief initialize elements of the matrix to a single constant value
     @param (m) the number of rows to initialize
     @param (x) the value of each element
  */
  void init_const(idx_t m, real x) {
    set_n_rows(m);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = x;
      }
    }
  }
  /**
     @brief randomly initialize elements of the matrix uniformly between p and q
     @param (m) the number of rows to initialize
     @param (rg) random number generator
     @param (p) the minimum value
     @param (q) the maximum value
  */
  void init_uniform(idx_t m, rnd_gen_t& rg, real p, real q) {
    set_n_rows(m);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = rg.rand(p, q);
      }
    }
  }
  /**
     @brief initialize the matrix with a normal distribution
     @param (m) the number of rows to initialize
     @param (rg) random number generator
     @param (mu) mean of the normal distribution
     @param (sigma) sigma (standard deviation) of the normal distribution
  */
  void init_normal(idx_t m, rnd_gen_t& rg, real mu, real sigma) {
    set_n_rows(m);
    array2<M,N>& a = *this;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) = rg.rand_normal(mu, sigma);
      }
    }
  }
  /**
     @brief update the matrix by eta and dx
     @param (eta) a constant to scale dx
     @param (da) the increment matrix
     @details it performx a += eta * da
  */
  __device__ __host__
  void update(real eta, array2<M,N>& da) {
    // a += eta * da
    array2<M,N>& a = *this;
    assert(a.m == da.m);
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        a(i,j) += eta * da(i,j);
      }
    }
  }
  /**
     @brief dot product with another matrix
     @param (b) the matrix to take a dot product with
  */
  real dot(array2<M,N>& b) {
    array2<M,N>& a = *this;
    assert(a.m == b.m);
    real s0 = 0.0;
    for (idx_t i = 0; i < m; i++) {
      real s1 = 0.0;
      for (idx_t j = 0; j < N; j++) {
        s1 += a(i,j) * b(i,j);
      }
      s0 += s1;
    }
    return s0;
  }
  /**
     @brief set the device shadow of this matrix
     @param (dev) device address (may be null)
   */
  void set_dev(array2<M,N> * dev) {
#if __NVCC__
    this->dev = dev;
#else
    (void)dev;
#endif
  }
  /**
     @brief allocate and set the device shadow of this matrix if
     requested by the parameter
     @param (gpu) 1 to allocate the device shadow
   */
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (array2<M,N>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
#else
    (void)gpu;
#endif
  }
  /**
     @brief deallocate the device shadow of this matrix
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
      array2<M,N> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief 4D array
   @param (maxB) the maximum number of rows (elements along the first dimension)
   @param (C) the number of elements along the second dimension
   @param (H) the number of elements along the third dimension
   @param (W) the number of elements along the fourth dimension
   @details this is essentially BxCxHxW array of reals where B 
   can be a runtime parameter <= maxB.
   throughout the VGG network, is is used to represent a mini-batch
   of images (B images, each image of which has C channels, each channel
   of which has HxW pixels.
*/
template<idx_t maxB,idx_t C,idx_t H,idx_t W>
struct array4 {
#if __NVCC__
  array4<maxB,C,H,W> * dev;     /**< pointer to the device shadow */
#endif
  idx_t B;                      /**< actual number of clements (<= B)  */
  real w[maxB][C][H][W];        /**< elements */
  /**
     @brief access the (b,c,i,j) element
     @param (b) the first index (image index in a mini batch)
     @param (c) the second index (channe index in an image)
     @param (i) the third index (row index in a channel)
     @param (j) the fourth index (column index in a row)
  */
  __device__ __host__ 
  real& operator()(idx_t b, idx_t c, idx_t i, idx_t j) {
    range_chk(0, b, B);
    range_chk(0, c, C);
    range_chk(0, i, H);
    range_chk(0, j, W);
    return w[b][c][i][j];
  }
  /**
     @brief set the number of rows (images)
     @param (B) the number of rows specified
  */
  __device__ __host__ 
  void set_n_rows(idx_t B) {
    this->B = B;
    assert(B <= maxB);
  }
  /**
     @brief initialize elements of the array  to a single constant value
     @param (B) the number of rows to initialize
     @param (x) the value of each element
  */
  void init_const(idx_t B, real x) {
    set_n_rows(B);
    array4<maxB,C,H,W>& a = *this;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) = x;
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
  void init_uniform(idx_t B, rnd_gen_t& rg, real p, real q) {
    set_n_rows(B);
    array4<maxB,C,H,W>& a = *this;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) = rg.rand(p, q);
          }
        }
      }
    }
  }
  /**
     @brief randomly initialize a randomly chosen single element. other elements are set to zero.
     @param (B) the number of rows to initialize
     @param (rg) random number generator
     @param (p) the minimum value
     @param (q) the maximum value
  */
  void init_single(idx_t B, rnd_gen_t& rg, real p, real q) {
    array4<maxB,C,H,W>& a = *this;
    a.init_const(B, 0);
    idx_t b = rg.randi(0, B);
    idx_t c = rg.randi(0, C);
    idx_t i = rg.randi(0, H);
    idx_t j = rg.randi(0, W);
    a(b,c,i,j) = rg.rand(p, q);
  }
  /**
     @brief initialize the array with a normal distribution
     @param (B) the number of rows to initialize
     @param (rg) random number generator
     @param (mu) mean of the normal distribution
     @param (sigma) sigma (standard deviation) of the normal distribution
  */
  void init_normal(idx_t B, rnd_gen_t& rg, real mu, real sigma) {
    set_n_rows(B);
    array4<maxB,C,H,W>& a = *this;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) = rg.rand_normal(mu, sigma);
          }
        }
      }
    }
  }
  /**
     @brief update the array by eta and da
     @param (eta) a constant to scale da
     @param (da) the increment array
     @details it performx a += eta * da
  */
  __device__ __host__
  void update(real eta, array4<maxB,C,H,W>& da) {
    array4<maxB,C,H,W>& a = *this;
    assert(a.B == da.B);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            a(b,c,i,j) += eta * da(b,c,i,j);
          }
        }
      }
    }
  }
  /**
     @brief dot product with another array
     @param (a_) the array to take a dot product with
  */
  real dot(array4<maxB,C,H,W>& a_) {
    array4<maxB,C,H,W>& a = *this;
    assert(a.B == a_.B);
    real s0 = 0.0;
    for (idx_t b = 0; b < B; b++) {
      real s1 = 0.0;
      for (idx_t c = 0; c < C; c++) {
        real s2 = 0.0;
        for (idx_t i = 0; i < H; i++) {
          real s3 = 0.0;
          for (idx_t j = 0; j < W; j++) {
            s3 += a(b,c,i,j) * a_(b,c,i,j);
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
     @brief set the device shadow of this array
     @param (dev) device address (may be null)
   */
  void set_dev(array4<maxB,C,H,W>* dev) {
#if __NVCC__
    this->dev = dev;
#else
    (void)dev;
#endif
  }
  /**
     @brief allocate and set the device shadow of this array if
     requested by the parameter
     @param (gpu) 1 to allocate the device shadow
   */
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (array4<maxB,C,H,W>*)dev_malloc(sizeof(*this));
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
      array4<maxB,C,H,W> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief 4D array for convolution
   @param (OC) the number of rows (elements along the first dimension)
   @param (IC) the number of elements along the second dimension)
   @param (H) 2H+1 is the number of elements along the third dimension
   @param (W) 2W+1 is the number of elements along the third dimension
 */
template<idx_t OC,idx_t IC,idx_t H,idx_t W>
struct warray4 {
#if __NVCC__
  warray4<OC,IC,H,W> * dev;     /**< pointer to the device shadow */
#endif
  real w[OC][IC][2*H+1][2*W+1];                    /**< elements */
  /**
     @brief access the (oc,ic,i,j) element
     @param (oc) the first index
     @param (ic) the second index
     @param (i) the third index
     @param (j) the fourth index
  */
  __device__ __host__ 
  real& operator()(idx_t oc, idx_t ic, idx_t i, idx_t j) {
    range_chk(0, oc, OC);
    range_chk(0, ic, IC);
    range_chk(-H, i, H+1);
    range_chk(-W, j, W+1);
    return w[oc][ic][i+H][j+W];
  }
  /**
     @brief initialize elements of the array to a single constant value
     @param (x) the value of each element
  */
  void init_const(real x) {
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) = x;
          }
        }
      }
    }
  }
  /**
     @brief randomly initialize elements of the array uniformly between p and q
     @param (rg) random number generator
     @param (p) the minimum value
     @param (q) the maximum value
  */
  void init_uniform(rnd_gen_t& rg, real p, real q) {
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) = rg.rand(p, q);
          }
        }
      }
    }
  }
  /**
     @brief initialize the array with a normal distribution
     @param (rg) random number generator
     @param (mu) mean of the normal distribution
     @param (sigma) sigma (standard deviation) of the normal distribution
  */
  void init_normal(rnd_gen_t& rg, real mu, real sigma) {
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) = rg.rand_normal(mu, sigma);
          }
        }
      }
    }
  }
  /**
     @brief update the array by eta and da
     @param (eta) a constant to scale da
     @param (da) the increment array
     @details it performx a += eta * da
  */
  __device__ __host__
  void update(real eta, warray4<OC,IC,H,W>& da) {
    // a += eta * da
    warray4<OC,IC,H,W>& a = *this;
    for (idx_t oc = 0; oc < OC; oc++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = -H; i <= H; i++) {
          for (idx_t j = -W; j <= W; j++) {
            a(oc,ic,i,j) += eta * da(oc,ic,i,j);
          }
        }
      }
    }
  }
  /**
     @brief dot product with another array
     @param (a_) the array to take a dot product with
  */
  real dot(warray4<OC,IC,H,W>& a_) {
    warray4<OC,IC,H,W>& a = *this;
    real s0 = 0.0;
    for (idx_t oc = 0; oc < OC; oc++) {
      real s1 = 0.0;
      for (idx_t ic = 0; ic < IC; ic++) {
        real s2 = 0.0;
        for (idx_t i = -H; i <= H; i++) {
          real s3 = 0.0;
          for (idx_t j = -W; j <= W; j++) {
            s3 += a(oc,ic,i,j) * a_(oc,ic,i,j);
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
     @brief set the device shadow of this array
     @param (dev) device address (may be null)
   */
  void set_dev(warray4<OC,IC,H,W>* dev) {
#if __NVCC__
    this->dev = dev;
#else
    (void)dev;
#endif
  }
  /**
     @brief allocate and set the device shadow of this array if
     requested by the parameter
     @param (gpu) 1 to allocate the device shadow
   */
  void make_dev(int gpu) {
#if __NVCC__
    if (gpu) {
      dev = (warray4<OC,IC,H,W>*)dev_malloc(sizeof(*this));
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
      warray4<OC,IC,H,W> * dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
};

/**
   @brief entry point
 */
int vgg_arrays_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const int N = 10;
  const int n = 8;
  vec<N> v;
  ivec<N> iv;
  array2<N,N> a2;
  array4<N,N,N,N> a4;
  warray4<N,N,N,N> w4;
  v.init_const(n, 1);
  iv.init_const(n, 2);
  a2.init_const(n, 3);
  a4.init_const(n, 4);
  w4.init_const(5);
  v(n-1) = opt.iters;
  iv(n-1) = 6;
  a2(n-1,n-1) = 7;
  a4(n-1,n-1,n-1,n-1) = 8;
  w4(n-1,n-1,n-1,n-1) = 9;
  return 0;
}

/**
   @brief suppress unused function warning
 */
void vgg_arrays_use_unused_functions() {
  (void)range_chk_;
}
