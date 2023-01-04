/**
   @file nll_log_softmax.h
   @brief negative log likelihood loss applied to log softmax
 */
#pragma once

#include <math.h>
#include "mnist_util.h"
#include "tensor.h"
#include "grad_check.h"

/**
   @brief configuration data for NLLLogSoftmaxCfg
   @details no configuration currently exist
*/
struct NLLLogSoftmaxCfg { };
  
/**
   @brief negative log likelihood loss of log softmax

   @param (maxB) the maximum number of images (batch size)
   @param (nC) number of classes (10)

   @details input is essentially a two dimensional vector
   describing the score for each image and each class.
   the score for image i and a class c is a likelihood
   that the image i belongs to the class c.
   based on this matrix, it first converts the vector
   for each image to the probability vector using 
   the softmax function. it then compares the probability
   score with the true label and calculate the loss
   using the negative log likelihood
 */
template<idx_t maxB,idx_t nC>
struct NLLLogSoftmax {
#if __CUDACC__
  NLLLogSoftmax<maxB,nC>* dev; /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger */
  tensor<real,maxB,nC> y;       /**< y = log softmax(x)  */
  tensor<real,maxB> l;          /**< NLL loss(y) */
  tensor<real,maxB,nC> gx;     /**< gradient of loss wrt to input x */

  /**
     @brief initialize the layer
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
     @param (cfg) configuration parameters
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, NLLLogSoftmaxCfg cfg) {
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
  void set_dev(NLLLogSoftmax<maxB,nC>* dev) {
#if __CUDACC__
    this->dev = dev;
    y.set_dev(dev ? &dev->y : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#else
    (void)dev;
#endif
  }

  /**
     @brief compute y = log(softmax(x))
     @param (x) a matrix 
     @param (y) output vector
     @details for 1D vector x x = (x_0, ..., x_{n-1}), 
     
                           (exp(x_0)     / Σ_j exp(x_j))
     logsoftmax(x)   = log (exp(x_1)     / Σ_j exp(x_j))
                           (   ...       / Σ_j exp(x_j))
                           (exp(x_{n-1}) / Σ_j exp(x_j))

     the input to this function is essentially a two 
     dimensional matrix (4D array whose last two axes
     have only one element), which is simply a set of
     vectors.

 */
  __device__ __host__
  void log_softmax(tensor<real,maxB,nC>& x, tensor<real,maxB,nC>& y) {
    const idx_t B = x.n0;
    y.set_n0(B);
    for (long b = 0; b < B; b++) {
      long m = 0;
      for (long c = 0; c < nC; c++) {
        m = (x(b,m) < x(b,c) ? c : m);
      }
      real v = 0.0;
      for (long c = 0; c < nC; c++) {
        y(b,c) = x(b,c) - x(b,m);
        v += exp(y(b,c));
      }
      real logv = log(v);
      for (long c = 0; c < nC; c++) {
        y(b,c) -= logv;
      }
    }
  }
  /**
     @brief the baseline (serial) implementation of forward
     @param (x) input images
     @param (t) true label
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
  void forward_base(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
    (void)training;
    const idx_t B = x.n0;
    l.set_n0(B);
    log_softmax(x, y);
    for (idx_t b = 0; b < B; b++) {
      l(b) = -y(b,t(b));
    }
  }
  /**
     @brief the device function of forward called from the 
     global (non-member) function
     @param (x) input images
     @param (t) true label
     @param (training) 1 if it is called in training not testing
     @sa forward
     @sa forward_cuda_base
     @sa forward_cuda_base_global
     @sa forward_base
  */
  __device__
  void forward_cuda_base_device(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
    forward_base(x, t, training);
  }
  /**
     @brief a cuda version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @param (t) true label
     @param (training) 1 if it is called in training not testing
     @sa forward
     @sa forward_cuda_base_global
     @sa forward_cuda_base_device
     @sa forward_base
  */
  void forward_cuda_base(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
#if __CUDACC__
    launch_and_sync((forward_cuda_base_global<<<1,1>>>(dev, x.dev, t.dev, training)));
#else
    (void)x;
    (void)t;
    (void)training;
    err_cuda_code_non_cuda_compiler(opt.algo_s);
#endif
  }
  /**
     @brief a cpu version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @param (t) true label
     @param (training) 1 if it is called in training not testing
     @sa forward
     @sa forward_base
  */
  void forward_cpu_base(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
    forward_base(x, t, training);
  }
  /**
     @brief forward phase of the layer
     @param (x) input images
     @param (t) true label
     @param (training) 1 if it is called in training not testing
     @sa forward_base
     @sa forward_cpu_base
     @sa forward_cuda_base
     @sa forward_cuda_base_global
     @sa forward_cuda_base_device
     @sa backward
     @sa update
  */
  tensor<real,maxB>& forward(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu_base(x, t, training); break;
    case algo_cuda_base:
      forward_cuda_base(x, t, training); break;
    default:
      if (opt.cuda_algo) {
        forward_cuda_base(x, t, training);
      } else {
        forward_cpu_base(x, t, training);
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
    return l;
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
  void backward_base(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
    const idx_t B = gy.n0;
    gx.set_n0(B);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < nC; c++) {
        if (c == t(b)) {
          gx(b,c) = gy(b) * (-1 + exp(y(b,c)));
        } else {
          gx(b,c) = gy(b) *       exp(y(b,c));
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
  void backward_cuda_base_device(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
    backward_base(gy, t);
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
  void backward_cuda_base(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
#if __CUDACC__
    launch_and_sync((backward_cuda_base_global<<<1,1>>>(dev, gy.dev, t.dev)));
#else
    (void)gy;
    (void)t;
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
  void backward_cpu_base(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
    backward_base(gy, t);
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
  tensor<real,maxB,nC>& backward(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      backward_cpu_base(gy, t); break;
    case algo_cuda_base:
      backward_cuda_base(gy, t); break;
    default:
      if (opt.cuda_algo) {
        backward_cuda_base(gy, t);
      } else {
        backward_cpu_base(gy, t);
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
  void copy_grad(NLLLogSoftmax<maxB,nC>& o) {
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
  double grad_dot_grad(NLLLogSoftmax<maxB,nC>& o) {
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
   a main C++ file and define nll_log_softmax_main to be main
   (e.g., with -Dnll_log_softmax_main=main), then this
   function becomes th main function of the executable.
   it calls grad_check repeatedly to test
   the implementation of backward of nll_log_softmax.
*/
int nll_log_softmax_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_size);
  const idx_t nC = 10;
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
  NLLLogSoftmaxCfg cfg;
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    double e = grad_check_loss<NLLLogSoftmax<maxB,nC>,
                               tensor<real,maxB,nC>,
                               tensor<idx_t,maxB>,
                               tensor<real,maxB>,
                               NLLLogSoftmaxCfg>(opt, &lgr, rg, cfg, B, nC);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

