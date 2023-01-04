/**
   @file dropout.h
   @brief dropout layer
 */
#pragma once

#include "mnist_util.h"
#include "tensor.h"
#include "grad_check.h"

/**
   @brief configuration data for Dropout
*/
struct DropoutCfg {
  real ratio;                   /**< the probability to drop (zero) an element */
  long seed;                    /**< random number seed */
};

/**
   @brief dropout layer

   @param (N0) first dimension
   @param (N1) second dimension
   @param (N2) third dimension
   @param (N3) fourth dimension

   @details y(i0,i1,i2,i3) = 0 with a specified probability 
                             x(i0,i1,i2,i3) otherwise
   for all i0, i1, i2 and i3.

 */
template<idx_t N0,idx_t N1,idx_t N2=1,idx_t N3=1>
struct Dropout {
#if __CUDACC__
  Dropout<N0,N1,N2,N3>* dev;     /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger */
  rnd_gen_t rg;                 /**< random number generator to choose dropout */
  tensor<real,N0,N1,N2,N3> y;        /**< output of the forward */
  tensor<real,N0,N1,N2,N3> gx;      /**< gradient of loss wrt to input x */
  real drop_ratio;              /**< drop probability */
  long state_forward;           /**< random number state at the forward function */
  /**
     @brief initialize the layer
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
     @param (cfg) configuration parameters
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, DropoutCfg cfg) {
    this->opt = opt;
    this->lgr = lgr;
    (void)rg;
    this->drop_ratio = cfg.ratio;
    this->rg.seed(cfg.seed);
  }
  /**
     @brief set the device pointer for this and all subobjects
     @param (dev) a device memory or null

     @details if dev is not null, dev fields of all subojects 
     point to the corresponding subjects in the device memory.
     if dev is not null, all dev fields become null.
  */
  void set_dev(Dropout<N0,N1,N2,N3>* dev) {
#if __CUDACC__
    this->dev = dev;
    y.set_dev(dev ? &dev->y : 0);
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
  void forward_base(tensor<real,N0,N1,N2,N3>& x, int training) {
    const idx_t n0 = x.n0;
    y.set_n0(n0);
    /* zero elements with probability of ratio and
       scale others by 1/(1-ratio) so that the sum 
       will stay approximately the same */
    state_forward = rg.get_state();
    real p = training ? drop_ratio : 0.0;
    real scale = 1.0 / (1 - p);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            if (rg.rand01() < p) {
              y(i0,i1,i2,i3) = 0.0;
            } else {
              y(i0,i1,i2,i3) = x(i0,i1,i2,i3) * scale;
            }
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
  void forward_cuda_base_device(tensor<real,N0,N1,N2,N3>& x, int training) {
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
  void forward_cuda_base(tensor<real,N0,N1,N2,N3>& x, int training) {
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
  void forward_cpu_base(tensor<real,N0,N1,N2,N3>& x, int training) {
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
  tensor<real,N0,N1,N2,N3>& forward(tensor<real,N0,N1,N2,N3>& x, int training) {
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
  void backward_base(tensor<real,N0,N1,N2,N3>& gy) {
    const idx_t n0 = gy.n0;
    gx.set_n0(n0);
    rg.seed(state_forward);
    real scale = 1.0 / (1 - drop_ratio);
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            if (rg.rand01() < drop_ratio) {
              gx(i0,i1,i2,i3) = 0.0;
            } else {
              gx(i0,i1,i2,i3) = scale * gy(i0,i1,i2,i3);
            }
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
  void backward_cuda_base_device(tensor<real,N0,N1,N2,N3>& gy) {
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
  void backward_cuda_base(tensor<real,N0,N1,N2,N3>& gy) {
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
  void backward_cpu_base(tensor<real,N0,N1,N2,N3>& gy) {
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
  tensor<real,N0,N1,N2,N3>& backward(tensor<real,N0,N1,N2,N3>& gy) {
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
  void copy_grad(Dropout<N0,N1,N2,N3>& o) {
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
  double grad_dot_grad(Dropout<N0,N1,N2,N3>& o) {
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
   a main C++ file and define dropout_main to be main
   (e.g., with -Ddropout_main=main), then this
   function becomes th main function of the executable.
   it calls grad_check repeatedly to test
   the implementation of backward of dropout.
*/
int dropout_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_size);
  const idx_t C = 2;
  const idx_t H = 16;
  const idx_t W = 16;
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
  DropoutCfg cfg = { .ratio = 0.5, .seed = opt.dropout_seed_1 };
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    double e = grad_check<Dropout<maxB,C,H,W>,
                          tensor<real,maxB,C,H,W>,
                          tensor<real,maxB,C,H,W>,
                          DropoutCfg>(opt, &lgr, rg, cfg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

