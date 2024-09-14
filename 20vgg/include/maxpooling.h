/**
   @file maxpooling.h
   @brief max pooling layer
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
  struct MaxPooling2D;

/**
   @brief a global CUDA function that implements the baseline 
   forward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (x_dev) the address of the device shadow of the input matrix
   @sa forward_dev
   @sa forward_gpu
  */
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
  __global__ void forward_global(MaxPooling2D<maxB,C,H,W,S>* dev,
                                 array4<maxB,C,H,W>* x_dev) {
  dev->forward_dev(*x_dev);
}

/**
   @brief a global CUDA function that implements the baseline 
   backward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (gy_dev) the address of the device shadow of the input matrix
   @sa backward_dev
   @sa backward_gpu
  */
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
  __global__ void backward_global(MaxPooling2D<maxB,C,H,W,S>* dev,
                                  array4<maxB,C,H/S,W/S>* gy_dev) {
  dev->backward_dev(*gy_dev);
}
#endif

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
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
struct MaxPooling2D {
#if __NVCC__
  MaxPooling2D<maxB,C,H,W,S>* dev; /**< device shadow  */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger  */
  array4<maxB,C,H/S,W/S> y;     /**< output of the forward */
  array4<maxB,C,H/S,W/S> max_idx; /**< the index that gave the maximum of each output pixel */
  array4<maxB,C,H,W> gx;          /**< gradient of loss wrt to input x */
  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
  */
  void init(cmdline_opt opt, logger * lgr) {
    this->opt = opt;
    this->lgr = lgr;
  }
  /**
     @brief make a copy of this 
     @details if this object has a device pointer, the copy will have
     a device pointer too, but its contents are NOT copied
  */
  MaxPooling2D<maxB,C,H,W,S>* copy() {
    MaxPooling2D<maxB,C,H,W,S>* c = new MaxPooling2D<maxB,C,H,W,S>(*this);
    c->make_dev();
    return c;
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
    max_idx.set_dev(dev ? &dev->max_idx : 0);
    gx.set_dev(dev ? &dev->gx : 0);
#else
    (void)dev;
#endif
  }
  /**
     @brief if the algorithm is a gpu algorithm, allocate a device shadow 
     of this object and set dev field of this and all subobjects. otherwise
     it sets all dev fields to null.
     @sa set_dev
     @sa del_dev
  */
  void make_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      dev = (MaxPooling2D<maxB,C,H,W,S>*)dev_malloc(sizeof(*this));
    } else {
      dev = 0;
    }
    set_dev(dev);
#endif
  }
  /**
     @brief if the algorithm is a gpu algorithm, dev field must not
     be null and deallocate it.
     @sa make_dev
     @sa set_dev
  */
  void del_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      assert(dev);
      dev_free(dev);
      dev = 0;
    }
#endif
  }
  /**
     @brief if the algorithm is a gpu algorithm, dev field must
     not be null and send the host data to the device memory
  */
  void to_dev() {
#if __NVCC__
    if (opt.gpu_algo) {
      assert(dev);
      ::to_dev(dev, this, sizeof(*this));
    }
#endif
  }
  /**
     @brief if the algorithm is a gpu algorithm, dev field must
     not be null and send the device data to the host memory
  */
  void to_host() {
#if __NVCC__
    if (opt.gpu_algo) {
      assert(dev);
      MaxPooling2D<maxB,C,H,W,S>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
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
  void forward_base(array4<maxB,C,H,W>& x) {
    const idx_t B = x.B;
    y.set_n_rows(B);
    max_idx.set_n_rows(B);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H/S; i++) {
          for (idx_t j = 0; j < W/S; j++) {
            real s = x(b,c,S*i,S*j);
            idx_t idx = W * S * i  + S * j;
            for (idx_t i_ = S * i; i_ < S * (i + 1); i_++) {
              for (idx_t j_ = S * j; j_ < S * (j + 1); j_++) {
                if (s < x(b,c,i_,j_)) {
                  s = x(b,c,i_,j_);
                  idx = W * i_ + j_;
                }
              }
            }
            y(b,c,i,j) = s;
            max_idx(b,c,i,j) = idx;
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
  void forward_dev(array4<maxB,C,H,W>& x) {
    forward_base(x);
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
  void forward_gpu(array4<maxB,C,H,W>& x) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev)));
  }
#endif
  /**
     @brief a cpu version of baseline code called from the 
     entry function (forward)
     @param (x) input images
     @sa forward
     @sa forward_base
  */
  void forward_cpu(array4<maxB,C,H,W>& x) {
    forward_base(x);
  }
  /**
     @brief calc the loss function of a mini-batch (x)
     @param (x) input images
     @sa backward
     @sa update
  */
  array4<maxB,C,H/S,W/S>& forward(array4<maxB,C,H,W>& x) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu(x); break;
#if __NVCC__
    case algo_gpu_base:
      forward_gpu(x); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        forward_gpu(x);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        forward_cpu(x);
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
  void backward_base(array4<maxB,C,H/S,W/S>& gy) {
    const idx_t B = gy.B;
    gx.set_n_rows(B);
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < C; c++) {
        for (idx_t i = 0; i < H/S; i++) {
          for (idx_t j = 0; j < W/S; j++) {
            for (idx_t i_ = S * i; i_ < S * (i + 1); i_++) {
              for (idx_t j_ = S * j; j_ < S * (j + 1); j_++) {
                gx(b,c,i_,j_) = 0;
              }
            }
            idx_t idx = max_idx(b,c,i,j);
            idx_t i_ = idx / W;
            idx_t j_ = idx % W;
            gx(b,c,i_,j_) = gy(b,c,i,j);
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
  void backward_dev(array4<maxB,C,H/S,W/S>& gy) {
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
  void backward_gpu(array4<maxB,C,H/S,W/S>& gy) {
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
  void backward_cpu(array4<maxB,C,H/S,W/S>& gy) {
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
  array4<maxB,C,H,W>& backward(array4<maxB,C,H/S,W/S>& gy) {
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
};

/**
   @brief check the gradient computation of a maxpooling layer
   @param (opt) command line option
   @param (lgr) logger 
   @param (rg) random number generator
   @param (B) the number of images
   @sa maxpooling_main
   @details it first makes a layer object with initial weights W 
   and generates an input (x and t).
   it then creates two layers whose weights are slightly different
   from the original one by dw/2 (i.e., w-dw/2 and w+dw/2), as well as
   two inputs slighly different from the original inputs by dx/2
   (x-dx/2 and x+dx/2).  it then computes L(w,x), L(x-dw/2,x-dx/2) and
   L(w+dw/2,x+dw/2) and check if L(x+dw/2,x+dx/2)-L(x-dw/2,x-dx/2)
   is close to ∂L/∂x dx + ∂L/∂w dw.  ∂L/∂x and ∂L/∂w are obtained
   by backward computation. This is essentially checking if
   the gradients obtained by backward computation correctly approximates
   the diff of the output.
*/
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t S>
  real maxpooling_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize maxpooling parameters */
  MaxPooling2D<maxB,C,H,W,S> * mp = new MaxPooling2D<maxB,C,H,W,S>();
  mp->init(opt, lgr);
  mp->make_dev();
  mp->to_dev();

  /* make copies */
  MaxPooling2D<maxB,C,H,W,S> * mp_minus = mp->copy();
  MaxPooling2D<maxB,C,H,W,S> * mp_plus  = mp->copy();
  /* make coefficients to make the single loss value */
  array4<maxB,C,H/S,W/S> * alpha = new array4<maxB,C,H/S,W/S>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,C,H,W> * x = new array4<maxB,C,H,W>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* forward and backward */
  array4<maxB,C,H/S,W/S>& y = mp->forward(*x);
  array4<maxB,C,H,W>& gx = mp->backward(*alpha);
  mp->to_host();

  /* make dx */
  real e = 1.0e-4;
  array4<maxB,C,H,W> * dx = new array4<maxB,C,H,W>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,C,H,W> * x_minus = new array4<maxB,C,H,W>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,C,H,W> * x_plus  = new array4<maxB,C,H,W>(*x);
  x_plus->make_dev(opt.gpu_algo);
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* send copies to gpu */
  mp_minus->to_dev();
  mp_plus->to_dev();
  /* make y(x-dx/2), y(x+dx/2) */
  array4<maxB,C,H/S,W/S>& y_minus = mp_minus->forward(*x_minus);
  array4<maxB,C,H/S,W/S>& y_plus  = mp_plus->forward(*x_plus);
  /* get the result back to host */
  y_minus.to_host();
  y_plus.to_host();

  /* get the single loss values */
  real L_minus = alpha->dot(y_minus);
  real L       = alpha->dot(y);
  real L_plus  = alpha->dot(y_plus);
  /* various inner products */
  real gx_gx = gx.dot(gx);                         /* ∂L/∂x・∂L/∂x */
  real dx_dx = dx->dot(*dx);                       /* ∂L/∂x・dx */
  real gx_dx = gx.dot(*dx);                        /* dx・dx */
  real gw_gw = 0;                                  /* ∂L/∂w・∂L/∂w */
  real dw_dw = 0;                                  /* ∂L/∂w・dw */
  real gw_dw = 0;                                  /* dw・dw */

  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  mp->del_dev();
  mp_minus->del_dev();
  mp_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete mp;
  delete mp_minus;
  delete mp_plus;
  delete alpha;
  delete x;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

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
int maxpooling_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t C = 3;
  const idx_t H = 32;
  const idx_t W = 32;
  const idx_t S = 2;
  const int n_checks = opt.iters;
  /* logger */
  logger lgr;
  lgr.start_log(opt);
  /* initialize random number generator */
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  /* check errors */
  real max_e = 0.0;
  real sum_e = 0.0;
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    real e = maxpooling_grad_check_rand<maxB,C,H,W,S>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

