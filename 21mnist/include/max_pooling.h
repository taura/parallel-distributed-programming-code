/**
   @file maxpooling.h
   @brief max pooling layer
 */
#pragma once

#include "mnist_util.h"
#include "tensor.h"
#include "grad_check.h"

/**
   @brief max pooling layer

   @param (maxB) the maximum number of images (batch size)
   @param (C) the number of channels per input image (the 
               original input has typically three channels for RGB. 
               in hidden layers, it starts from 64 and goes up 
               to 512 in the last hidden layer)
   @param (H) height of an image (32 for an input image, down to 1 in
              the last hidden layer)
   @param (W) width of an image (32 for an input image, down to 1 in
              the last hidden layer)
   @param (S) shrink factor of pooling layers (2)

   @details this layer implements max pooling. it takes SxS patch of 
   input images and take the maximum of it. given an HxW image,
   output (H/S)x(W/S) image

 */
struct MaxPooling2DCfg {
};

template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
struct MaxPooling2D {
#if __NVCC__
  MaxPooling2D<maxB,C,H,W,S>* dev; /**< device shadow  */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger  */
  tensor<real,maxB,C,H/S,W/S> y;     /**< output of the forward */
  tensor<idx_t,maxB,C,H/S,W/S> argmax_i; /**< the index that gave the maximum of each output pixel */
  tensor<idx_t,maxB,C,H/S,W/S> argmax_j; /**< the index that gave the maximum of each output pixel */
  tensor<real,maxB,C,H,W> gx;          /**< gradient of loss wrt to input x */
  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
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
     @sa make_dev
     @sa del_dev
     @details if dev is not null, dev fields of all subojects 
     point to the corresponding subjects in the device memory.
     if dev is not null, all dev fields become null.
  */
  void set_dev(MaxPooling2D<maxB,C,H,W,S>* dev) {
#if __NVCC__
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
     called both by cpu implementation (forward_cpu) and 
     gpu implementation (forward_dev). the call sequence
     forward -> forward_cpu -> forward_base on cpu and
     and is forward -> forward_gpu -> forward_global -> forward_dev -> forward_base
     @param (x) input images
     @sa forward
     @sa forward_gpu
     @sa forward_global
     @sa forward_dev
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
#if __NVCC__
  /**
     @brief the device function of forward called from the 
     global (non-member) function
     @param (x) input images
     @sa forward
     @sa forward_gpu
     @sa forward_global
     @sa forward_base
  */
  __device__
  void forward_dev(tensor<real,maxB,C,H,W>& x, int training) {
    forward_base(x, training);
  }
  /**
     @brief a gpu version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @sa forward
     @sa forward_global
     @sa forward_dev
     @sa forward_base
  */
  void forward_gpu(tensor<real,maxB,C,H,W>& x, int training) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev, training)));
  }
#endif
  /**
     @brief a cpu version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @sa forward
     @sa forward_base
  */
  void forward_cpu(tensor<real,maxB,C,H,W>& x, int training) {
    forward_base(x, training);
  }
  /**
     @brief calc the loss function of a mini-batch (x)
     @param (x) input images
     @sa backward
     @sa update
  */
  tensor<real,maxB,C,H/S,W/S>& forward(tensor<real,maxB,C,H,W>& x, int training) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu(x, training); break;
#if __NVCC__
    case algo_gpu_base:
      forward_gpu(x, training); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        forward_gpu(x, training);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        forward_cpu(x, training);
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
    return y;
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
  void backward_dev(tensor<real,maxB,C,H/S,W/S>& gy) {
    backward_base(gy);
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
  void backward_gpu(tensor<real,maxB,C,H/S,W/S>& gy) {
    launch_and_sync((backward_global<<<1,1>>>(dev, gy.dev)));
  }
#endif
  /**
     @brief a cpu version of baseline code called from the 
     entry function (backward)
     @param (gy) gradient of loss with respect to the output
     @sa backward
     @sa backward_base
  */
  void backward_cpu(tensor<real,maxB,C,H/S,W/S>& gy) {
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
     @sa forward
     @sa update
  */
  tensor<real,maxB,C,H,W>& backward(tensor<real,maxB,C,H/S,W/S>& gy) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      backward_cpu(gy); break;
#if __NVCC__
    case algo_gpu_base:
      backward_gpu(gy); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        backward_gpu(gy);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        backward_cpu(gy);
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
  void copy_grad(MaxPooling2D<maxB,C,H,W,S>& o) {
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
  double grad_dot_grad(MaxPooling2D<maxB,C,H,W,S>& o) {
    (void)o;
    return 0.0;
  }
};

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
   @sa maxpooling_grad_check_rand
   @details if this header file is included from
   a main C++ file and define maxpooling_main to be main
   (e.g., with -Dmaxpooling_main=main), then this
   function becomes th main function of the executable.
   it calls maxpooling_grad_check_rand repeatedly to test
   the implementation of VGG network.
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

