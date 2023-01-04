/**
   @file convolution.h
   @brief convolution layer
 */
#pragma once

#include "mnist_util.h"
#include "tensor.h"
#include "ada_delta.h"
#include "grad_check.h"

/**
   @brief configuration data for Convolution2D
   @details no configuration currently exist
*/
struct Convolution2DCfg { };

/**
   @brief convolution of images

   @param (maxB) the maximum number of images it can handle at a time (batch size)
   @param (IC) the number of channels per input image (the 
               original input image for MNIST has is grey scale
               and therefore has a single channel. 
               hidden layers have 32 or 64 channels.
   @param (H) height of an image (28 for an input image, 26 after the first 
              convolution layer and 24 after the second)
   @param (W) width of an image (same as H)
   @param (K) convolution kernel size (3 for MNIST). filter is K x K
   @param (OC) the number of channels per an output image

   @details this layer converts each ICxWxH image to
   OCx(W-K+1)x(H-K+1) image, applying ICxKxK stencil to each pixel
 */
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
struct Convolution2D {
#if __CUDACC__
  Convolution2D<maxB,IC,H,W,K,OC> * dev; /**< device shadow */
#endif
  cmdline_opt opt;                 /**< command line option  */
  logger * lgr;                    /**< logger */
  tensor<real,maxB,IC,H,W>* x_ptr;    /**< pointer to the input to forward (x) */
  tensor<real,OC,IC,K,K> w;           /**< weight (y = w ＊ x + b) */ 
  tensor<real,OC> b;                  /**< bias (y = w ＊ x + b) */ 
  tensor<real,maxB,OC,H-K+1,W-K+1> y; /**< layer output */
  tensor<real,OC,IC,K,K> gw;          /**< ∂L/∂w */
  tensor<real,OC> gb;                 /**< ∂L/∂b */
  tensor<real,maxB,IC,H,W> gx;        /**< ∂L/∂x */
  AdaDelta<OC,IC,K,K> opt_w;          /**< optimizer for w */
  AdaDelta<OC> opt_b;                 /**< optimizer for b */
  /**
     @brief initialize the layer
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
     @param (cfg) configuration parameters
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, Convolution2DCfg cfg) {
    this->opt = opt;
    this->lgr = lgr;
    (void)cfg;
    real bound = 1.0 / sqrt(IC * K * K);
    /* init weight and bias */
    w.init_uniform(OC, rg, -bound, bound);
    b.init_uniform(OC, rg, -bound, bound);
    /* init optimizers */
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
  void set_dev(Convolution2D<maxB,IC,H,W,K,OC>* dev) {
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
    /* let the optimizer update w and b based on their gradients */
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
    assert(dev);
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
      /* fallback to base */
      if (opt.cuda_algo) {
        update_cuda_base();
      } else {
        update_cpu_base();
      }
      break;
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
  void forward_base(tensor<real,maxB,IC,H,W>& x, int training) {
    (void)training;
    idx_t B = x.n0;             // batch size
    y.set_n0(B);
    x_ptr = &x;                 // save pointer to input for backward
    for (idx_t s = 0; s < B; s++) {       // for each sample
      for (idx_t oc = 0; oc < OC; oc++) { // for each output channel
        for (idx_t i = 0; i < H - K + 1; i++) {   // for each output pixel
          for (idx_t j = 0; j < W - K + 1; j++) { // for each output pixel
            // calculate a single output pixel
            real v = 0.0;
            for (idx_t ic = 0; ic < IC; ic++) { // input channel
              for (idx_t di = 0; di < K; di++) {
                for (idx_t dj = 0; dj < K; dj++) {
                  v += w(oc,ic,di,dj) * x(s,ic,i+di,j+dj);
                }
              }
            }
            y(s,oc,i,j) = v + b(oc);
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
  void forward_cuda_base_device(tensor<real,maxB,IC,H,W>& x, int training) {
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
  void forward_cuda_base(tensor<real,maxB,IC,H,W>& x, int training) {
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
  void forward_cpu_base(tensor<real,maxB,IC,H,W>& x, int training) {
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
  tensor<real,maxB,OC,H-K+1,W-K+1>& forward(tensor<real,maxB,IC,H,W>& x, int training) {
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
  void backward_base(tensor<real,maxB,OC,H-K+1,W-K+1>& gy) {
    idx_t B = gy.n0;
    gw.set_n0(OC);
    gb.set_n0(OC);
    gx.set_n0(B);
    tensor<real,maxB,IC,H,W>& x = *x_ptr;
    for (idx_t oc = 0; oc < OC; oc++) {   // output channel
      for (idx_t ic = 0; ic < IC; ic++) { // input channel
        for (idx_t di = 0; di < K; di++) { // kernel pixel
          for (idx_t dj = 0; dj < K; dj++) { // kernel pixel
            real v = 0.0;
            for (idx_t s = 0; s < B; s++) { // training samples
              for (idx_t i = 0; i < H - K + 1; i++) { // sample pixel
                for (idx_t j = 0; j < W - K + 1; j++) { // sample pixel
                  v += gy(s,oc,i,j) * x(s,ic,i+di,j+dj);
                }
              }
            }
            gw(oc,ic,di,dj) = v;
          }
        }
      }
    }
    for (idx_t oc = 0; oc < OC; oc++) {
      real v = 0.0;
      for (idx_t s = 0; s < B; s++) {
        for (idx_t i = 0; i < H - K + 1; i++) {
          for (idx_t j = 0; j < W - K + 1; j++) {
            v += gy(s,oc,i,j);
          }
        }
      }
      gb(oc) = v;
    }
    for (idx_t s = 0; s < B; s++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            real v = 0.0;
            for (idx_t oc = 0; oc < OC; oc++) {
              for (idx_t di = 0; di < K; di++) {
                for (idx_t dj = 0; dj < K; dj++) {
                  if (0 <= i - di && i - di < H - K + 1
                      && 0 <= j - dj && j - dj < W - K + 1) {
                    v += gy(s,oc,i-di,j-dj) * w(oc,ic,di,dj);
                  }
                }
              }
            }
            gx(s,ic,i,j) = v;
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
  void backward_cuda_base_device(tensor<real,maxB,OC,H-K+1,W-K+1>& gy) {
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
  void backward_cuda_base(tensor<real,maxB,OC,H-K+1,W-K+1>& gy) {
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
  void backward_cpu_base(tensor<real,maxB,OC,H-K+1,W-K+1>& gy) {
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
  tensor<real,maxB,IC,H,W>& backward(tensor<real,maxB,OC,H-K+1,W-K+1>& gy) {
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
  /* member functions below assume data are on the host.
     they are only for checking (debugging) implementations */
  /**
     @brief randomly set all gradients to values between p and q
     @param (rg) random number generator
     @param (p) minimum value of a component
     @param (q) maximum value of a component
     @details only used for checking gradient computation
  */
  void rand_grad(rnd_gen_t& rg, real p, real q) {
    gw.init_uniform(OC, rg, p, q);
    gb.init_uniform(OC, rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object o
     @param (o) the object from which gradients get copied
     @details only used for checking gradient computation
  */
  void copy_grad(Convolution2D<maxB,IC,H,W,K,OC>& o) {
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
  double grad_dot_grad(Convolution2D<maxB,IC,H,W,K,OC>& o) {
    return gw.dot(o.gw) + gb.dot(o.gb);
  }
};

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
   @sa convolution_grad_check_rand
   @details if this header file is included from
   a main C++ file and define convolution_main to be main
   (e.g., with -Dconvolution_main=main), then this
   function becomes th main function of the executable.
   it calls grad_check repeatedly to test
   the implementation of backward of convolution.
*/
int convolution_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  if (opt.error || opt.help) usage(argv[0]);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_size);
  const idx_t IC = 1;
  const idx_t H = 28;
  const idx_t W = 28;
  const idx_t K = 3;
  const idx_t OC = 32;
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
  Convolution2DCfg cfg;
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    double e = grad_check<Convolution2D<maxB,IC,H,W,K,OC>,
                          tensor<real,maxB,IC,H,W>,
                          tensor<real,maxB,OC,H-K+1,W-K+1>,
                          Convolution2DCfg>(opt, &lgr, rg, cfg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}
