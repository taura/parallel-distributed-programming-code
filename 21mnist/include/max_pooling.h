/**
   @file max_pooling.h
   @brief max pooling layer
 */
#pragma once

#include "mnist_util.h"
#include "tensor.h"
#include "grad_check.h"

/**
   @brief configuration data for Maxpooling2D
   @details no configuration currently exist
*/
struct MaxPooling2DCfg { };

/**
   @brief max pooling layer

   @param (maxB) the maximum number of images (batch size)
   @param (C) the number of channels per input image
   @param (H) height of an image
   @param (W) width of an image
   @param (S) shrink factor of pooling layers (2)

   @details this layer implements max pooling. it takes SxS patch of 
   input images and take the maximum of it. given an HxW image,
   output (H/S)x(W/S) image

 */
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
struct MaxPooling2D {
#if __CUDACC__
  MaxPooling2D<maxB,C,H,W,S>* dev; /**< device shadow  */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger  */
  tensor<real,maxB,C,H/S,W/S> y;     /**< output of the forward */
  tensor<idx_t,maxB,C,H/S,W/S> argmax_i; /**< the index that gave the maximum of each output pixel */
  tensor<idx_t,maxB,C,H/S,W/S> argmax_j; /**< the index that gave the maximum of each output pixel */
  tensor<real,maxB,C,H,W> gx;          /**< gradient of loss wrt to input x */
  /**
     @brief initialize the layer
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
     @param (cfg) configuration parameters
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, MaxPooling2DCfg cfg) {
    this->opt = opt;
    this->lgr = lgr;
    (void)rg;
    (void)cfg;
  }
  /**
     @brief set the device pointer for this and all subobjects
     @param (dev) a device memory or null

     @details if dev is not null, dev fields of all subojects 
     point to the corresponding subjects in the device memory.
     if dev is not null, all dev fields become null.
  */
  void set_dev(MaxPooling2D<maxB,C,H,W,S>* dev) {
#if __CUDACC__
    this->dev = dev;
    y.set_dev(dev ? &dev->y : 0);
    argmax_i.set_dev(dev ? &dev->argmax_i : 0);
    argmax_j.set_dev(dev ? &dev->argmax_j : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#else
    (void)dev;
#endif
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
  void forward_base(tensor<real,maxB,C,H,W>& x, int training) {
    (void)training;
    const idx_t B = x.n0;
    y.set_n0(B);
    argmax_i.set_n0(B);
    argmax_j.set_n0(B);
    for (idx_t s = 0; s < B; s++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H/S; i++) {
          for (idx_t j = 0; j < W/S; j++) {
            idx_t max_i = S * i;
            idx_t max_j = S * j;
            real v = x(s,c,max_i,max_j);
            for (idx_t i_ = S * i; i_ < S * (i + 1); i_++) {
              for (idx_t j_ = S * j; j_ < S * (j + 1); j_++) {
                if (v < x(s,c,i_,j_)) {
                  max_i = i_;
                  max_j = j_;
                  v = x(s,c,max_i,max_j);
                }
              }
            }
            y(s,c,i,j) = v;
            argmax_i(s,c,i,j) = max_i;
            argmax_j(s,c,i,j) = max_j;
          }
        }
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
  void forward_cuda_base_device(tensor<real,maxB,C,H,W>& x, int training) {
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
  void forward_cuda_base(tensor<real,maxB,C,H,W>& x, int training) {
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
  void forward_cpu_base(tensor<real,maxB,C,H,W>& x, int training) {
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
  tensor<real,maxB,C,H/S,W/S>& forward(tensor<real,maxB,C,H,W>& x, int training) {
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
  void backward_base(tensor<real,maxB,C,H/S,W/S>& gy) {
    const idx_t B = gy.n0;
    gx.set_n0(B);
    for (idx_t s = 0; s < B; s++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            gx(s,c,i,j) = 0;
          }
        }
      }
    }
    for (idx_t s = 0; s < B; s++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H/S; i++) {
          for (idx_t j = 0; j < W/S; j++) {
            idx_t i_ = argmax_i(s,c,i,j);
            idx_t j_ = argmax_j(s,c,i,j);
            gx(s,c,i_,j_) = gy(s,c,i,j);
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
  void backward_cuda_base_device(tensor<real,maxB,C,H/S,W/S>& gy) {
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
  void backward_cuda_base(tensor<real,maxB,C,H/S,W/S>& gy) {
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
  void backward_cpu_base(tensor<real,maxB,C,H/S,W/S>& gy) {
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
  tensor<real,maxB,C,H,W>& backward(tensor<real,maxB,C,H/S,W/S>& gy) {
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
     @details as this layer has no weights, it's noop
  */
  void rand_grad(rnd_gen_t& rg, real p, real q) {
    (void)rg;
    (void)p;
    (void)q;
  }
  /**
     @brief set all gradients to gradients of another object o
     @param (o) the object from which gradients get copied
     @details as this layer has no weights, it's noop
  */
  void copy_grad(MaxPooling2D<maxB,C,H,W,S>& o) {
    (void)o;
  }
  /**
     @brief w += alpha * gw
     @param (alpha) alpha of w += alpha * gw
     @details as this layer has no weights, it's noop
  */
  void add_grad(real alpha) {
    (void)alpha;
  }
  /**
     @brief take the inner product of gradients
     @param (o) the object to take the inner product with
     @details as this layer has no weights, it returns zero
  */
  double grad_dot_grad(MaxPooling2D<maxB,C,H,W,S>& o) {
    (void)o;
    return 0.0;
  }
};

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
   @sa grad_check
   @details if this header file is included from
   a main C++ file and define max_pooling_main to be main
   (e.g., with -Dmax_pooling_main=main), then this
   function becomes th main function of the executable.
   it calls grad_check repeatedly to test
   the implementation of backward of max_pooling.
*/
int max_pooling_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_size);
  const idx_t C = 1; // 3;
  const idx_t H = 8; // 32
  const idx_t W = 8; // 32;
  const idx_t S = 2;
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
  MaxPooling2DCfg cfg;
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    double e = grad_check<MaxPooling2D<maxB,C,H,W,S>,
                          tensor<real,maxB,C,H,W>,
                          tensor<real,maxB,C,H/S,W/S>,
                          MaxPooling2DCfg>(opt, &lgr, rg, cfg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

