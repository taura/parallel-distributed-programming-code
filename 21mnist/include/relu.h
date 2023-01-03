/**
   @file relu.h
   @brief rectified linear activation layer (relu(x) = max(0,x))
 */
#pragma once

#include "mnist_util.h"
#include "tensor.h"
#include "grad_check.h"

/**
   @brief rectified linear layer

   @param (maxB) the maximum number of images (batch size)
   @param (C) the number of channels per input image (the 
               original input has typically three channels for RGB. 
               in hidden layers, it starts from 64 and goes up 
               to 512 in the last hidden layer)
   @param (H) height of an image (32 for an input image, down to 1 in
              the last hidden layer)
   @param (W) width of an image (32 for an input image, down to 1 in
              the last hidden layer)

   @details this layer normalizes a batch of images 

 */
struct ReluCfg {
};

template<idx_t N0,idx_t N1,idx_t N2=1,idx_t N3=1>
struct Relu {
#if __NVCC__
  Relu<N0,N1,N2,N3>* dev;        /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger */
  tensor<real,N0,N1,N2,N3>* x_ptr; /**< pointer to input passed to forward */
  tensor<real,N0,N1,N2,N3> y;      /**< output of the forward */
  tensor<real,N0,N1,N2,N3> gx;     /**< gradient of loss wrt input x */
  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, ReluCfg cfg) {
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
  void set_dev(Relu<N0,N1,N2,N3>* dev) {
#if __NVCC__
    this->dev = dev;
    y.set_dev(dev ? &dev->y : 0);
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
  void forward_base(tensor<real,N0,N1,N2,N3>& x, int training) {
    (void)training;
    const idx_t n0 = x.n0;
    y.set_n0(n0);
    x_ptr = &x;
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            y(i0,i1,i2,i3) = max_r(0, x(i0,i1,i2,i3));
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
  void forward_dev(tensor<real,N0,N1,N2,N3>& x, int training) {
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
  void forward_gpu(tensor<real,N0,N1,N2,N3>& x, int training) {
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
  void forward_cpu(tensor<real,N0,N1,N2,N3>& x, int training) {
    forward_base(x, training);
  }
  /**
     @brief calc the loss function of a mini-batch (x)
     @param (x) input images
     @sa backward
     @sa update
  */
  tensor<real,N0,N1,N2,N3>& forward(tensor<real,N0,N1,N2,N3>& x, int training) {
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
  void backward_base(tensor<real,N0,N1,N2,N3>& gy) {
    const idx_t n0 = gy.n0;
    gx.set_n0(n0);
    tensor<real,N0,N1,N2,N3>& x = *x_ptr;
    for (idx_t i0 = 0; i0 < n0; i0++) {
      for (idx_t i1 = 0; i1 < N1; i1++) {
        for (idx_t i2 = 0; i2 < N2; i2++) {
          for (idx_t i3 = 0; i3 < N3; i3++) {
            gx(i0,i1,i2,i3) = (x(i0,i1,i2,i3) >= 0 ? gy(i0,i1,i2,i3) : 0);
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
  void backward_dev(tensor<real,N0,N1,N2,N3>& gy) {
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
  void backward_gpu(tensor<real,N0,N1,N2,N3>& gy) {
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
  void backward_cpu(tensor<real,N0,N1,N2,N3>& gy) {
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
  tensor<real,N0,N1,N2,N3>& backward(tensor<real,N0,N1,N2,N3>& gy) {
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
  void copy_grad(Relu<N0,N1,N2,N3>& o) {
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
  double grad_dot_grad(Relu<N0,N1,N2,N3>& o) {
    (void)o;
    return 0.0;
  }
};

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
   @sa relu_grad_check_rand
   @details if this header file is included from
   a main C++ file and define relu_main to be main
   (e.g., with -Drelu_main=main), then this
   function becomes th main function of the executable.
   it calls relu_grad_check_rand repeatedly to test
   the implementation of VGG network.
*/
int relu_main(int argc, char ** argv) {
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
  ReluCfg cfg;
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    double e = grad_check<Relu<maxB,C,H,W>,
                          tensor<real,maxB,C,H,W>,
                          tensor<real,maxB,C,H,W>,
                          ReluCfg>(opt, &lgr, rg, cfg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

