/**
   @file linear.h
   @brief linear (fully connected) layer
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t IC,idx_t nC>
  struct Linear;

/**
   @brief a global CUDA function that implements the baseline 
   forward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (x_dev) the address of the device shadow of the input matrix
   @sa forward_dev
   @sa forward_gpu
  */
template<idx_t maxB,idx_t IC,idx_t nC>
__global__ void forward_global(Linear<maxB,IC,nC>* dev,
                               array4<maxB,IC,1,1>* x_dev) {
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
template<idx_t maxB,idx_t IC,idx_t nC>
  __global__ void backward_global(Linear<maxB,IC,nC>* dev,
                                  array4<maxB,nC,1,1>* gy_dev) {
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
template<idx_t maxB,idx_t IC,idx_t nC>
  __global__ void update_global(Linear<maxB,IC,nC>* dev, real eta) {
  dev->update_dev(eta);
}
#endif

/**
   @brief linear (fully connected) layer

   @param (maxB) the maximum number of images (batch size)
   @param (IC) the number of channels per input image (the 
               original input has typically three channels for RGB. 
               in hidden layers, it starts from 64 and goes up 
               to 512 in the last hidden layer)
   @param (nC) number of classes (10)

 */
template<idx_t maxB,idx_t IC,idx_t nC>
struct Linear {
#if __NVCC__
  Linear<maxB,IC,nC>* dev;      /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger  */
  array4<maxB,IC,1,1>* x_ptr;   /**< address of input passed to forward */
  array2<IC,nC> w;              /**< weight of the matrix */
  array4<maxB,nC,1,1> y;        /**< output of the forward  */
  array2<IC,nC> gw;             /**< gradient of loss wrt to w */
  array4<maxB,IC,1,1> gx;       /**< gradient of loss wrt to input x */
  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg) {
    this->opt = opt;
    this->lgr = lgr;
    w.init_normal(IC, rg, 0.0, 1 / sqrt(IC));
  }
  /**
     @brief make a copy of this 
     @details if this object has a device pointer, the copy will have
     a device pointer too, but its contents are NOT copied
  */
  Linear<maxB,IC,nC>* copy() {
    Linear<maxB,IC,nC>* c = new Linear<maxB,IC,nC>(*this);
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
  void set_dev(Linear<maxB,IC,nC>* dev) {
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
      dev = (Linear<maxB,IC,nC>*)dev_malloc(sizeof(*this));
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
      Linear<maxB,IC,nC>* dev_ = dev;
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
  void forward_base(array4<maxB,IC,1,1>& x) {
    const idx_t B = x.B;
    y.set_n_rows(B);
    x_ptr = &x;
    /* y = x * maxB (x : maxBxIC, w : ICxnC -> y : maxBxnC) */
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < nC; c++) {
        real s = 0.0;
        for (idx_t ic = 0; ic < IC; ic++) {
          s += x(b,ic,0,0) * w(ic,c);
        }
        y(b,c,0,0) = s;
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
  void forward_dev(array4<maxB,IC,1,1>& x) {
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
  void forward_gpu(array4<maxB,IC,1,1>& x) {
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
  void forward_cpu(array4<maxB,IC,1,1>& x) {
    forward_base(x);
  }
  /**
     @brief calc the loss function of a mini-batch (x)
     @param (x) input images
     @sa backward
     @sa update
  */
  array4<maxB,nC,1,1>& forward(array4<maxB,IC,1,1>& x) {
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
  void backward_base(array4<maxB,nC,1,1>& gy) {
    const idx_t B = gy.B;
    gw.set_n_rows(IC);
    gx.set_n_rows(B);
    array4<maxB,IC,1,1>& x = *x_ptr;
    for (idx_t ic = 0; ic < IC; ic++) {
      for (idx_t c = 0; c < nC; c++) {
        real s = 0.0;
        for (idx_t b = 0; b < B; b++) {
          s += gy(b,c,0,0) * x(b,ic,0,0);
        }
        gw(ic,c) = s;
      }
    }
    for (idx_t b = 0; b < B; b++) {
      for (idx_t ic = 0; ic < IC; ic++) {
        real s = 0.0;
        for (idx_t c = 0; c < nC; c++) {
          s += gy(b,c,0,0) * w(ic,c);
        }
        gx(b,ic,0,0) = s;
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
  void backward_dev(array4<maxB,nC,1,1>& gy) {
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
  void backward_gpu(array4<maxB,nC,1,1>& gy) {
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
  void backward_cpu(array4<maxB,nC,1,1>& gy) {
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
  array4<maxB,IC,1,1>& backward(array4<maxB,nC,1,1>& gy) {
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
    gw.init_uniform(IC, rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object
     @param (o) the object from which gradients get copied
     @details transfer gradients of o to this object
  */
  void set_grad(Linear<maxB,IC,nC>& o) {
    gw = o.gw;
  }
  /**
     @brief take the inner product of gradients
     @param (o) the object to take the inner product with
     @details take the inner product of this object's 
     gradients and b's gradients
  */
  real gw_dot_gw(Linear<maxB,IC,nC>& o) {
    return gw.dot(o.gw);
  }
};

/**
   @brief check the gradient computation of a linear layer
   @param (opt) command line option
   @param (lgr) logger 
   @param (rg) random number generator
   @param (B) the number of images
   @sa linear_main
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
template<idx_t maxB,idx_t IC,idx_t nC>
  real linear_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize linear parameters */
  Linear<maxB,IC,nC> * linear = new Linear<maxB,IC,nC>();
  linear->init(opt, lgr, rg);
  linear->make_dev();
  linear->to_dev();
  /* make w - dw/2 and w + dw/2 */
  Linear<maxB,IC,nC> * linear_minus = linear->copy();
  Linear<maxB,IC,nC> * linear_plus  = linear->copy();
  /* make coefficients to make the single loss value */
  array4<maxB,nC,1,1> * alpha = new array4<maxB,nC,1,1>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,IC,1,1> * x = new array4<maxB,IC,1,1>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* forward and backward */
  array4<maxB,nC,1,1>& y = linear->forward(*x);
  array4<maxB,IC,1,1>& gx = linear->backward(*alpha);
  /* ensure the gradient is back to host */
  linear->to_host();

  /* make dx */
  real e = 1.0e-4;
  array4<maxB,IC,1,1> * dx = new array4<maxB,IC,1,1>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,IC,1,1> * x_minus = new array4<maxB,IC,1,1>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,IC,1,1> * x_plus  = new array4<maxB,IC,1,1>(*x);
  x_plus->make_dev(opt.gpu_algo);
  /* update on the host and send the to gpu */
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* set gw to a random vector */
  linear_minus->rand_grad(rg, -e, e);
  linear_plus->set_grad(*linear_minus);
  /* send them to gpu */
  linear_minus->to_dev();
  linear_plus->to_dev();
  /* update weights using gw (update runs on gpu) */
  linear_minus->update(-0.5);      /* w -= dw/2 */
  linear_plus->update(0.5);        /* w += dw/2 */
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  array4<maxB,nC,1,1>& y_minus = linear_minus->forward(*x_minus);
  array4<maxB,nC,1,1>& y_plus  = linear_plus->forward(*x_plus);
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
  real gw_gw = linear->gw_dot_gw(*linear);             /* ∂L/∂w・∂L/∂w */
  real dw_dw = linear_minus->gw_dot_gw(*linear_minus); /* ∂L/∂w・dw */
  real gw_dw = linear->gw_dot_gw(*linear_minus);       /* dw・dw */

  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  linear->del_dev();
  linear_minus->del_dev();
  linear_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete linear;
  delete linear_minus;
  delete linear_plus;
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
   @sa linear_grad_check_rand
   @details if this header file is included from
   a main C++ file and define linear_main to be main
   (e.g., with -Dlinear_main=main), then this
   function becomes th main function of the executable.
   it calls linear_grad_check_rand repeatedly to test
   the implementation of VGG network.
*/
int linear_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t IC = 512;
  const idx_t nC = 10;
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
    real e = linear_grad_check_rand<maxB,IC,nC>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}
