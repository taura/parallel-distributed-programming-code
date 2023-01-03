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
struct NLLLogSoftmaxCfg {
};
  
template<idx_t maxB,idx_t nC>
struct NLLLogSoftmax {
#if __NVCC__
  NLLLogSoftmax<maxB,nC>* dev; /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger */
  tensor<real,maxB,nC> y;       /**< y = log softmax(x)  */
  tensor<real,maxB> l;          /**< NLL loss(y) */
  tensor<real,maxB,nC> gx;     /**< gradient of loss wrt to input x */

  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
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
     @sa make_dev
     @sa del_dev
     @details if dev is not null, dev fields of all subojects 
     point to the corresponding subjects in the device memory.
     if dev is not null, all dev fields become null.
  */
  void set_dev(NLLLogSoftmax<maxB,nC>* dev) {
#if __NVCC__
    this->dev = dev;
    y.set_dev(dev ? &dev->y : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#else
    (void)dev;
#endif
  }

  /**
     @brief compute log(softmax(x))
     @param (x) a matrix 
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
     called both by cpu implementation (forward_cpu) and 
     gpu implementation (forward_dev). the call sequence
     forward -> forward_cpu -> forward_base on cpu and
     and is forward -> forward_gpu -> forward_global -> forward_dev -> forward_base
     @param (x) input images
     @param (t) true labels
     @sa forward
     @sa forward_gpu
     @sa forward_global
     @sa forward_dev
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
#if __NVCC__
  /**
     @brief the device function of forward called from the 
     global (non-member) function
     @param (x) input images
     @param (t) true labels
     @sa forward
     @sa forward_gpu
     @sa forward_global
     @sa forward_base
  */
  __device__
  void forward_dev(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
    forward_base(x, t, training);
  }
  /**
     @brief a gpu version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @param (t) true labels
     @sa forward
     @sa forward_global
     @sa forward_dev
     @sa forward_base
  */
  void forward_gpu(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev, t.dev, training)));
  }
#endif
  /**
     @brief a cpu version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @param (t) true labels
     @sa forward
     @sa forward_base
  */
  void forward_cpu(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
    forward_base(x, t, training);
  }
  /**
     @brief calc the loss function of a mini-batch (x,t)
     @param (x) input images
     @param (t) true labels of images
     @sa backward
     @sa update
  */
  tensor<real,maxB>& forward(tensor<real,maxB,nC>& x, tensor<idx_t,maxB>& t, int training) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu(x, t, training); break;
#if __NVCC__
    case algo_gpu_base:
      forward_gpu(x, t, training); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        forward_gpu(x, t, training);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        forward_cpu(x, t, training);
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
    return l;
  }
  /**
     @brief the baseline (serial) implementation of backward
     called both by cpu implementation (backward_cpu) and 
     gpu implementation (backward_dev). the call sequence
     backward -> backward_cpu -> backward_base on cpu and
     and is backward -> backward_gpu -> backward_global -> backward_dev -> backward_base
     @param (gy) gradient of loss with respect to the output
     @sa backward
     @sa backward_gpu
     @sa backward_global
     @sa backward_dev

     y_i = log(exp x_i / Σ_k exp(x_k))
         = x_i - log(Σ_k exp(x_k))
     
     ∂y_i/∂x_j =   - exp(x_j) / Σ_k exp(x_k) (i != j)
                 1 - exp(x_j) / Σ_k exp(x_k) (i == j)

     ∂L/∂x_j = Σ_i ∂L/∂y_i ∂y_i/∂x_j

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
#if __NVCC__
  /**
     @brief the device function of backward called from the 
     global (non-member) function
     @param (gy) gradient of loss with respect to the output
     @sa backward
     @sa backward_gpu
     @sa backward_global
     @sa backward_base
  */
  __device__
  void backward_dev(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
    backward_base(gy, t);
  }
  /**
     @brief a gpu version of baseline code called from the 
     entry function (backward)
     @param (gy) gradient of loss with respect to the output
     @sa backward
     @sa backward_global
     @sa backward_dev
     @sa backward_base
  */
  void backward_gpu(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
    launch_and_sync((backward_global<<<1,1>>>(dev, gy.dev, t.dev)));
  }
#endif
  /**
     @brief a cpu version of baseline code called from the 
     entry function (backward)
     @param (gy) gradient of loss with respect to the output
     @sa backward
     @sa backward_base
  */
  void backward_cpu(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
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
     @sa forward
     @sa update
  */
  tensor<real,maxB,nC>& backward(tensor<real,maxB>& gy, tensor<idx_t,maxB>& t) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      backward_cpu(gy, t); break;
#if __NVCC__
    case algo_gpu_base:
      backward_gpu(gy, t); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        backward_gpu(gy, t);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        backward_cpu(gy, t);
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
  */
  void rand_grad(rnd_gen_t& rg, real p, real q) {
    (void)rg;
    (void)p;
    (void)q;
  }
  /**
     @brief set all gradients to gradients of another object
     @param (o) the object from which gradients get copied
     @details transfer gradients of o to this object
  */
  void copy_grad(NLLLogSoftmax<maxB,nC>& o) {
    (void)o;
  }
  /**
     @brief w += alpha * gw
  */
  void add_grad(real alpha) {
    (void)alpha;
  }
  /**
     @brief take the inner product of gradients
     @param (o) the object to take the inner product with
     @details take the inner product of this object's 
     gradients and b's gradients
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
   @sa nll_log_softmax_grad_check_rand
   @details if this header file is included from
   a main C++ file and define softmaxcrossentropy_main to be main
   (e.g., with -Dsoftmaxcrossentropy_main=main), then this
   function becomes th main function of the executable.
   it calls softmaxcrossentropy_grad_check_rand repeatedly to test
   the implementation of VGG network.
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

