/**
   @file linear.h
   @brief linear (fully connected) layer
 */
#pragma once

#include <math.h>
#include "mnist_util.h"
#include "tensor.h"
#include "ada_delta.h"
#include "grad_check.h"

/**
   @brief configuration data for Linear
   @details no configuration currently exist
*/
struct LinearCfg { };

/**
   @brief linear (fully connected) layer

   @param (M) 
   @param (N) 
   @param (K0)
   @param (K1)
   @param (K2)

   @details this layer takes as input a M x (K0 x K1 x K2) tensor x and
   outputs a M x N matrix. essentially it is a matrix multiply y = x * w
 */
template<idx_t M,idx_t N,idx_t K0,idx_t K1=1,idx_t K2=1>
struct Linear {
#if __CUDACC__
  Linear<M,N,K0,K1,K2>* dev;      /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger  */
  tensor<real,M,K0,K1,K2>* x_ptr; /**< address of input passed to forward */
  tensor<real,K0,K1,K2,N> w;      /**< weight of the matrix (y = w x + bias) */
  tensor<real,N> b;               /**< bias */
  tensor<real,M,N> y;             /**< output of the forward  */
  tensor<real,K0,K1,K2,N> gw;     /**< gradient of loss wrt to w */
  tensor<real,N> gb;              /**< bias */
  tensor<real,M,K0,K1,K2> gx;  /**< gradient of loss wrt to input x */
  AdaDelta<K0,K1,K2,N> opt_w;  /**< AdaDelta optimizer for w */
  AdaDelta<N> opt_b;           /**< AdaDelta optimizer for b */
  /**
     @brief initialize the layer
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
     @param (cfg) configuration parameters
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, LinearCfg cfg) {
    this->opt = opt;
    this->lgr = lgr;
    (void)cfg;
    real bound = 1.0 / sqrt(K0 * K1 * K2);
    w.init_uniform(K0, rg, -bound, bound);
    b.init_uniform(N, rg, -bound, bound);
    opt_w.init(opt.lr);
    opt_b.init(opt.lr);
  }
  /**
     @brief set the device pointer for this and all subobjects
     @param (dev) a device memory or null

     @details if dev is not null, dev fields of all subojects 
     point to the corresponding subjects in the device memory.
     if dev is not null, all dev fields become null.
  */
  void set_dev(Linear<M,N,K0,K1,K2>* dev) {
#if __CUDACC__
    this->dev = dev;
    w.set_dev(dev ? &dev->w : 0);
    b.set_dev(dev ? &dev->b : 0);
    y.set_dev(dev ? &dev->y : 0);
    gw.set_dev(dev ? &dev->gw : 0);
    gb.set_dev(dev ? &dev->gb : 0);
    gx.set_dev(dev ? &dev->gx : 0);
    opt_w.set_dev(dev ? &dev->opt_w : 0);
    opt_b.set_dev(dev ? &dev->opt_b : 0);
#else
    (void)dev;
#endif
  }


  /**
     @brief the baseline (serial) implementation of update

     @details called both by cpu implementation (update_cpu_base) and
     cuda implementation (update_cuda_base). the call sequence update
     -> update_cpu_base -> update_base on cpu and and is update ->
     update_cuda_base -> update_cuda_base_global ->
     update_cuda_base_device -> update_base

     @sa update
     @sa update_cpu_base
     @sa update_cuda_base
     @sa update_cuda_base_global
     @sa update_cuda_base_device
  */
  __device__ __host__
  void update_base() {
    opt_w.update(w, gw);
    opt_b.update(b, gb);
  }
  /**
     @brief the device function of update called from the 
     global (non-member) function
     @sa update
     @sa update_cuda_base
     @sa update_cuda_base_global
     @sa update_base
  */
  __device__
  void update_cuda_base_device() {
    update_base();
  }
  /**
     @brief a cuda version of baseline code called from the 
     entry function (update)
     @sa update
     @sa update_cuda_base_device
     @sa update_cuda_base_global
     @sa update_base
  */
  void update_cuda_base() {
#if __CUDACC__
    launch_and_sync((update_cuda_base_global<<<1,1>>>(dev)));
#else
    err_cuda_code_non_cuda_compiler(opt.algo_s);
#endif
  }
  /**
     @brief a cpu version of baseline code called from the 
     entry function (update)
     @sa update
     @sa update_base
  */
  void update_cpu_base() {
    update_base();
  }
  /**
     @brief update weights of all sublayers with gradients
     that must have been computed
     @sa update_cpu_base
     @sa update_cuda_base
     @sa update_cuda_base_global
     @sa update_cuda_base_device
     @sa update_base
     @sa forward
     @sa backward
  */
  void update() {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      update_cpu_base(); break;
    case algo_cuda_base:
      update_cuda_base(); break;
    default:
      if (opt.cuda_algo) {
        update_cuda_base();
      } else {
        update_cpu_base();
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
  }
  /**
     @brief the baseline (serial) implementation of forward
     @param (x) input images
     @param (training) 1 if it is called in training not testing

     @details called both by cpu implementation (forward_cpu_base) and
     cuda implementation (forward_cuda_base). the call sequence
     forward -> forward_cpu_base -> forward_base on cpu and and is
     forward -> forward_cuda_base -> forward_cuda_base_global ->
     forward_cuda_base_device -> forward_base

     @sa forward
     @sa forward_cpu_base
     @sa forward_cuda_base
     @sa forward_cuda_base_global
     @sa forward_cuda_base_device
  */
  __device__ __host__
  void forward_base(tensor<real,M,K0,K1,K2>& x, int training) {
    (void)training;
    const idx_t m = x.n0;
    y.set_n0(m);
    x_ptr = &x;
    for (idx_t i = 0; i < m; i++) {
      for (idx_t j = 0; j < N; j++) {
        real v = 0.0;
        for (idx_t k0 = 0; k0 < K0; k0++) {
          for (idx_t k1 = 0; k1 < K1; k1++) {
            for (idx_t k2 = 0; k2 < K2; k2++) {
              v += x(i,k0,k1,k2) * w(k0,k1,k2,j);
            }
          }
        }
        y(i,j) = v + b(j);
      }
    }
  }
  /**
     @brief the device function of forward called from the 
     global (non-member) function
     @param (x) input images
     @param (training) 1 if it is called in training not testing
     @sa forward
     @sa forward_cuda_base
     @sa forward_cuda_base_global
     @sa forward_base
  */
  __device__
  void forward_cuda_base_device(tensor<real,M,K0,K1,K2>& x, int training) {
    forward_base(x, training);
  }
  /**
     @brief a cuda version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @param (training) 1 if it is called in training not testing
     @sa forward
     @sa forward_cuda_base_global
     @sa forward_cuda_base_device
     @sa forward_base
  */
  void forward_cuda_base(tensor<real,M,K0,K1,K2>& x, int training) {
#if __CUDACC__
    launch_and_sync((forward_cuda_base_global<<<1,1>>>(dev, x.dev, training)));
#else
    (void)x;
    (void)training;
    err_cuda_code_non_cuda_compiler(opt.algo_s);
#endif
  }
  /**
     @brief a cpu version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @param (training) 1 if it is called in training not testing
     @sa forward
     @sa forward_base
  */
  void forward_cpu_base(tensor<real,M,K0,K1,K2>& x, int training) {
    forward_base(x, training);
  }
  /**
     @brief forward phase of the layer
     @param (x) input images
     @param (training) 1 if it is called in training not testing
     @sa forward_base
     @sa forward_cpu_base
     @sa forward_cuda_base
     @sa forward_cuda_base_global
     @sa forward_cuda_base_device
     @sa backward
     @sa update
  */
  tensor<real,M,N>& forward(tensor<real,M,K0,K1,K2>& x, int training) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu_base(x, training); break;
    case algo_cuda_base:
      forward_cuda_base(x, training); break;
    default:
      if (opt.cuda_algo) {
        forward_cuda_base(x, training);
      } else {
        forward_cpu_base(x, training);
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
    return y;
  }
  /**
     @brief the baseline (serial) implementation of backward
     @param (gy) gradient of loss with respect to the output
     @details called both by cpu implementation (backward_cpu_base)
     and cuda implementation (backward_cuda_base). the call sequence
     backward -> backward_cpu_base -> backward_base on cpu and and is
     backward -> backward_cuda_base -> backward_cuda_base_global ->
     backward_cuda_base_device -> backward_base
     @sa backward
     @sa backward_cpu_base
     @sa backward_cuda_base
     @sa backward_cuda_base_global
     @sa backward_cuda_base_device
     @sa backward_base
  */
  __device__ __host__
  void backward_base(tensor<real,M,N>& gy) {
    const idx_t m = gy.n0;
    gw.set_n0(K0);
    gb.set_n0(N);
    gx.set_n0(m);
    tensor<real,M,K0,K1,K2>& x = *x_ptr;
    for (idx_t k0 = 0; k0 < K0; k0++) {
      for (idx_t k1 = 0; k1 < K1; k1++) {
        for (idx_t k2 = 0; k2 < K2; k2++) {
          for (idx_t j = 0; j < N; j++) {
            real v = 0.0;
            for (idx_t i = 0; i < m; i++) {
              v += gy(i,j) * x(i,k0,k1,k2);
            }
            gw(k0,k1,k2,j) = v;
          }
        }
      }
    }
    for (idx_t j = 0; j < N; j++) {
      real v = 0.0;
      for (idx_t i = 0; i < m; i++) {
        v += gy(i, j);
      }
      gb(j) = v;
    }
    for (idx_t i = 0; i < m; i++) {
      for (idx_t k0 = 0; k0 < K0; k0++) {
        for (idx_t k1 = 0; k1 < K1; k1++) {
          for (idx_t k2 = 0; k2 < K2; k2++) {
            real v = 0.0;
            for (idx_t j = 0; j < N; j++) {
              v += gy(i,j) * w(k0,k1,k2,j);
            }
            gx(i,k0,k1,k2) = v;
          }
        }
      }
    }
  }
  /**
     @brief the device function of backward called from the 
     global (non-member) function
     @param (gy) gradient of loss with respect to the output
     @sa backward
     @sa backward_cuda_base
     @sa backward_cuda_base_global
     @sa backward_base
  */
  __device__
  void backward_cuda_base_device(tensor<real,M,N>& gy) {
    backward_base(gy);
  }
  /**
     @brief a cuda version of baseline code called from the 
     entry function (backward)
     @param (gy) gradient of loss with respect to the output
     @sa backward
     @sa backward_cuda_base_global
     @sa backward_cuda_base_device
     @sa backward_base
  */
  void backward_cuda_base(tensor<real,M,N>& gy) {
#if __CUDACC__
    launch_and_sync((backward_cuda_base_global<<<1,1>>>(dev, gy.dev)));
#else
    (void)gy;
    err_cuda_code_non_cuda_compiler(opt.algo_s);
#endif
  }
  /**
     @brief a cpu version of baseline code called from the 
     entry function (backward)
     @param (gy) gradient of loss with respect to the output
     @sa backward
     @sa backward_base
  */
  void backward_cpu_base(tensor<real,M,N>& gy) {
    backward_base(gy);
  }
  /**
     @brief calc the gradient of loss wrt the input (x)
     @param (gy) gradient of loss with respect to the output
     @details calc the gradient of loss wrt the input. along the way,
     it also calculates the gradient of loss wrt weights for
     all sublayers that have weights. since this is the entire
     network, gy is actually a vector whose components are all 1.
     (loss = sum of losses of each data).
     @sa backward_cpu_base
     @sa backward_cuda_base
     @sa backward_cuda_base_global
     @sa backward_cuda_base_device
     @sa backward_base
     @sa forward
     @sa update
  */
  tensor<real,M,K0,K1,K2>& backward(tensor<real,M,N>& gy) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      backward_cpu_base(gy); break;
    case algo_cuda_base:
      backward_cuda_base(gy); break;
    default:
      if (opt.cuda_algo) {
        backward_cuda_base(gy);
      } else {
        backward_cpu_base(gy);
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
    return gx;
  }
  /**
     @brief randomly set all gradients to values between p and q
     @param (rg) random number generator
     @param (p) minimum value of a component
     @param (q) maximum value of a component
     @details only used for checking gradient computation
  */
  void rand_grad(rnd_gen_t& rg, real p, real q) {
    gw.init_uniform(K0, rg, p, q);
    gb.init_uniform(N, rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object o
     @param (o) the object from which gradients get copied
     @details only used for checking gradient computation
  */
  void copy_grad(Linear<M,N,K0,K1,K2>& o) {
    gw = o.gw;
    gb = o.gb;
  }
  /**
     @brief w += alpha * gw
     @param (alpha) alpha of w += alpha * gw
  */
  void add_grad(real alpha) {
    w.add_(alpha, gw);
    b.add_(alpha, gb);
  }
  /**
     @brief take the inner product of gradients
     @param (o) the object to take the inner product with
     @details take the inner product of this object's gradient and b's
     gradient. only used for checking gradient computation
  */
  double grad_dot_grad(Linear<M,N,K0,K1,K2>& o) {
    return gw.dot(o.gw) + gb.dot(o.gb);
  }
};

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
   @sa grad_check
   @details if this header file is included from
   a main C++ file and define linear_main to be main
   (e.g., with -Dlinear_main=main), then this
   function becomes th main function of the executable.
   it calls grad_check repeatedly to test
   the implementation of backward of linear.
*/
int linear_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t M = min_i(maxB, opt.batch_size);
  const idx_t N = 10;
  const idx_t K = 128;
  const int n_checks = opt.epochs;
  /* logger */
  logger lgr;
  lgr.start_log(opt);
  /* initialize random number generator */
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  /* check errors */
  double max_e = 0.0;
  double sum_e = 0.0;
  LinearCfg cfg;
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    real e = grad_check<Linear<maxB,N,K>,
                        tensor<real,maxB,K>,
                        tensor<real,maxB,N>,
                        LinearCfg>(opt, &lgr, rg, cfg, M);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}
