/**
   @file convolution.h
   @brief convolution layer
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
  struct Convolution2D;

/**
   @brief a global CUDA function that implements the baseline 
   forward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (x_dev) the address of the device shadow of the input matrix
   @sa forward_dev
   @sa forward_gpu
  */
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
__global__ void forward_global(Convolution2D<maxB,IC,H,W,K,OC>* dev,
                               array4<maxB,IC,H,W>* x_dev) {
  /* call the member function */
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
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
__global__ void backward_global(Convolution2D<maxB,IC,H,W,K,OC>* dev,
                                array4<maxB,OC,H,W>* gy_dev) {
  dev->backward_dev(*gy_dev);
}

/**
   @brief a global CUDA function that implements the baseline 
   update function for GPU
   @param (dev) the address of the device shadow of the object
   @param (eta) the address of the device shadow of the input matrix
   @sa update_dev
   @sa update_gpu
  */
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
  __global__ void update_global(Convolution2D<maxB,IC,H,W,K,OC>* dev, real eta) {
  dev->update_dev(eta);
}
#endif

/**
   @brief convolution of images

   @param (maxB) the maximum number of images (batch size)
   @param (IC) the number of channels per input image (the 
               original input has typically three channels for RGB. 
               in hidden layers, it starts from 64 and goes up 
               to 512 in the last hidden layer)
   @param (H) height of an image (32 for an input image, down to 1 in
              the last hidden layer)
   @param (W) width of an image (32 for an input image, down to 1 in
              the last hidden layer)
   @param (K) convolution kernel size (1). filter array has (2K+1)*(2K+1) elems)
   @param (OC) the number of channels per an output image

   @details this layer converts each ICxWxH image
   to OCxWxH image, applying ICx(2K+1)x(2K+1) stencil to each pixel

 */
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
struct Convolution2D {
#if __NVCC__
  Convolution2D<maxB,IC,H,W,K,OC> * dev; /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option  */
  logger * lgr;                 /**< logger */
  array4<maxB,IC,H,W>* x_ptr;      /**< pointer to the input to forward (x) */
  warray4<OC,IC,K,K> w;            /**< y = w * x (convolution) */ 
  array4<maxB,OC,H,W> y;           /**< y = forward(x) */
  warray4<OC,IC,K,K> gw;           /**< ∂L/∂w */
  array4<maxB,IC,H,W> gx;          /**< ∂L/∂x */
  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg) {
    this->opt = opt;
    this->lgr = lgr;
    w.init_normal(rg, 0.0, 1 / sqrt((2 * K + 1) * (2 * K + 1) * IC));
  }
  /**
     @brief make a copy of this 
     @details if this object has a device pointer, the copy will have
     a device pointer too, but its contents are NOT copied
  */
  Convolution2D<maxB,IC,H,W,K,OC>* copy() {
    Convolution2D<maxB,IC,H,W,K,OC> * c = new Convolution2D<maxB,IC,H,W,K,OC>(*this);
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
  void set_dev(Convolution2D<maxB,IC,H,W,K,OC>* dev) {
#if __NVCC__
    this->dev = dev;
    w.set_dev(dev ? &dev->w : 0);
    y.set_dev(dev ? &dev->y : 0);
    gw.set_dev(dev ? &dev->gw : 0);
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
      dev = (Convolution2D<maxB,IC,H,W,K,OC>*)dev_malloc(sizeof(*this));
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
      Convolution2D<maxB,IC,H,W,K,OC>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
  /**
     @brief the baseline (serial) implementation of update
     called both by cpu implementation (update_cpu) and 
     gpu implementation (update_dev). the call sequence
     update -> update_cpu -> update_base on cpu and
     and is update -> update_gpu -> update_global -> update_dev -> update_base
     @param (eta) the learning rate
     @sa update
     @sa update_gpu
     @sa update_global
     @sa update_dev
  */
  __device__ __host__
  void update_base(real eta) {
    w.update(eta, gw);
  }
#if __NVCC__
  /**
     @brief the device function of update called from the 
     global (non-member) function
     @param (eta) the learning rate
     @sa update
     @sa update_gpu
     @sa update_global
     @sa update_base
  */
  __device__
  void update_dev(real eta) {
    update_base(eta);
  }
  /**
     @brief a gpu version of baseline code called from the 
     entry function (update)
     @param (eta) the learning rate
     @sa update
     @sa update_global
     @sa update_dev
     @sa update_base
  */
  void update_gpu(real eta) {
    launch_and_sync((update_global<<<1,1>>>(dev, eta)));
  }
#endif
  /**
     @brief a cpu version of baseline code called from the 
     entry function (update)
     @param (eta) the learning rate
     @sa update
     @sa update_base
  */
  void update_cpu(real eta) {
    update_base(eta);
  }
  /**
     @brief update weights of all sublayers with gradients
     that must have been computed
     @param (eta) the learning rate
     @sa forward
     @sa backward
  */
  void update(real eta) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      update_cpu(eta); break;
#if __NVCC__
    case algo_gpu_base:
      update_gpu(eta); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        update_gpu(eta);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        update_cpu(eta);
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
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
  void forward_base(array4<maxB,IC,H,W>& x) {
    idx_t B = x.B;
    y.set_n_rows(B);
    x_ptr = &x;                 /* save pointer to input */
    for (idx_t b = 0; b < B; b++) {       // samples
      for (idx_t oc = 0; oc < OC; oc++) { // output channels
        for (idx_t i = 0; i < H; i++) {   // width
          for (idx_t j = 0; j < W; j++) { // height
            real s = 0.0;
            for (idx_t ic = 0; ic < IC; ic++) { // input channel
              /* -K<=i_<=K, 0<=i+i_<H => -K<=i_&-i<i_; i_<=K&i_<H-i*/
              for (idx_t i_ = max_i(-K,-i); i_ <= min_i(K,H-i-1); i_++) {
                for (idx_t j_ = max_i(-K,-j); j_ <= min_i(K,W-j-1); j_++) {
                  s += w(oc,ic,i_,j_) * x(b,ic,i+i_,j+j_);
                }
              }
            }
            y(b,oc,i,j) = s;
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
  void forward_dev(array4<maxB,IC,H,W>& x) {
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
  void forward_gpu(array4<maxB,IC,H,W>& x) {
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
  void forward_cpu(array4<maxB,IC,H,W>& x) {
    forward_base(x);
  }
  /**
     @brief calc the loss function of a mini-batch (x)
     @param (x) input images
     @sa backward
     @sa update
  */
  array4<maxB,OC,H,W>& forward(array4<maxB,IC,H,W>& x) {
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
  void backward_base(array4<maxB,OC,H,W>& gy) {
    idx_t B = gy.B;
    gx.set_n_rows(B);
    array4<maxB,IC,H,W>& x = *x_ptr;
    for (idx_t oc = 0; oc < OC; oc++) { // output channel
      for (idx_t ic = 0; ic < IC; ic++) { // input channel
        for (idx_t i_ = -K; i_ <= K; i_++) {
          for (idx_t j_ = -K; j_ <= K; j_++) {
            real s = 0.0;
            for (idx_t b = 0; b < B; b++) { // samples
              for (idx_t i = max_i(0,-i_); i < min_i(H,H-i_); i++) {   // width
                for (idx_t j = max_i(0,-j_); j < min_i(W,W-j_); j++) { // height
                  s += gy(b,oc,i,j) * x(b,ic,i+i_,j+j_);
                }
              }
            }
            gw(oc,ic,i_,j_) = s;
          }
        }
      }
    }
    for (idx_t b = 0; b < B; b++) { // samples
      for (idx_t ic = 0; ic < IC; ic++) { // input channel
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            real s = 0.0;
            for (idx_t oc = 0; oc < OC; oc++) { // output channels
              for (idx_t i_ = max_i(-K,i-H+1); i_ <= min_i(K,i); i_++) {
                for (idx_t j_ = max_i(-K,j-W+1); j_ <= min_i(K,j); j_++) {
                  /* max(-K,i-H+1) <= i_ <= min(K,i)
                     i-H+1 <= i_ <= i
                     -i+H-1 >= -i_ >= -i
                     H-1 >= i-i_ >= 0
                  */
                  s += gy(b,oc,i-i_,j-j_) * w(oc,ic,i_,j_);
                }
              }
            }
            gx(b,ic,i,j) = s;
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
  void backward_dev(array4<maxB,OC,H,W>& gy) {
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
  void backward_gpu(array4<maxB,OC,H,W>& gy) {
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
  void backward_cpu(array4<maxB,OC,H,W>& gy) {
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
  array4<maxB,IC,H,W>& backward(array4<maxB,OC,H,W>& gy) {
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
  /* member functions below assume data are on the host.
     they are only for checking (debugging) implementations */
  /**
     @brief randomly set all gradients to values between p and q
     @param (rg) random number generator
     @param (p) minimum value of a component
     @param (q) maximum value of a component
  */
  void rand_grad(rnd_gen_t& rg, real p, real q) {
    gw.init_uniform(rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object
     @param (o) the object from which gradients get copied
     @details transfer gradients of o to this object
  */
  void set_grad(Convolution2D<maxB,IC,H,W,K,OC>& o) {
    gw = o.gw;
  }
  /**
     @brief take the inner product of gradients
     @param (o) the object to take the inner product with
     @details take the inner product of this object's 
     gradients and b's gradients
  */
  real gw_dot_gw(Convolution2D<maxB,IC,H,W,K,OC>& o) {
    return gw.dot(o.gw);
  }
};

/**
   @brief check the gradient computation of a convolution layer
   @param (opt) command line option
   @param (lgr) logger 
   @param (rg) random number generator
   @param (B) the number of images
   @sa convolution_main
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
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
  static real convolution_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* make weight and transfer to gpu if working on gpu */
  Convolution2D<maxB,IC,H,W,K,OC> * conv = new Convolution2D<maxB,IC,H,W,K,OC>();
  conv->init(opt, lgr, rg);
  conv->make_dev();
  conv->to_dev();
  /* make w - dw/2 and w + dw/2 */
  Convolution2D<maxB,IC,H,W,K,OC> * conv_minus = conv->copy();
  Convolution2D<maxB,IC,H,W,K,OC> * conv_plus = conv->copy();
  /* make coefficients to make the single loss value */
  array4<maxB,OC,H,W> * alpha = new array4<maxB,OC,H,W>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,IC,H,W> * x = new array4<maxB,IC,H,W>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* forward and backward */
  array4<maxB,OC,H,W>& y = conv->forward(*x);
  array4<maxB,IC,H,W>& gx = conv->backward(*alpha);
  /* ensure the gradient is back to host */
  conv->to_host();
  
  /* make dx */
  real e = 1.0e-4;
  array4<maxB,IC,H,W> * dx = new array4<maxB,IC,H,W>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,IC,H,W> * x_minus = new array4<maxB,IC,H,W>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,IC,H,W> * x_plus  = new array4<maxB,IC,H,W>(*x);
  x_plus->make_dev(opt.gpu_algo);
  /* update on the host and send the to gpu */
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
  
  /* set gw to a random vector */
  conv_minus->rand_grad(rg, -e, e);
  conv_plus->set_grad(*conv_minus);
  /* send them to gpu */
  conv_minus->to_dev();
  conv_plus->to_dev();
  /* update weights using gw (update runs on gpu) */
  conv_minus->update(-0.5);      /* w -= dw/2 */
  conv_plus->update(0.5);        /* w += dw/2 */
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  array4<maxB,OC,H,W>& y_minus = conv_minus->forward(*x_minus);
  array4<maxB,OC,H,W>& y_plus  = conv_plus->forward(*x_plus);
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
  real gw_gw = conv->gw_dot_gw(*conv);             /* ∂L/∂w・∂L/∂w */
  real dw_dw = conv_minus->gw_dot_gw(*conv_minus); /* ∂L/∂w・dw */
  real gw_dw = conv->gw_dot_gw(*conv_minus);       /* dw・dw */
  
  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  conv->del_dev();
  conv_minus->del_dev();
  conv_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();
  
  delete conv;
  delete conv_minus;
  delete conv_plus;
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
   @sa convolution_grad_check_rand
   @details if this header file is included from
   a main C++ file and define convolution_main to be main
   (e.g., with -Dconvolution_main=main), then this
   function becomes th main function of the executable.
   it calls convolution_grad_check_rand repeatedly to test
   the implementation of VGG network.
*/
int convolution_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  if (opt.error || opt.help) usage(argv[0]);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t IC = 3;
  const idx_t H = 32;
  const idx_t W = 32;
  const idx_t K = 1;
  const idx_t OC = 64;
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
    real e = convolution_grad_check_rand<maxB,IC,H,W,K,OC>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}
