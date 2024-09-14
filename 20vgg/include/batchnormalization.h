/**
   @file batchnormalization.h
   @brief batch normalization layer
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
  struct BatchNormalization;

/**
   @brief a global CUDA function that implements the baseline 
   forward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (x_dev) the address of the device shadow of the input matrix
   @sa forward_dev
   @sa forward_gpu
  */
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
__global__ void forward_global(BatchNormalization<maxB,IC,H,W>* dev,
                               array4<maxB,IC,H,W>* x_dev) {
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
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
__global__ void backward_global(BatchNormalization<maxB,IC,H,W>* dev,
                                array4<maxB,IC,H,W>* gy_dev) {
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
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
  __global__ void update_global(BatchNormalization<maxB,IC,H,W>* dev, real eta) {
  dev->update_dev(eta);
}
#endif

/**
   @brief batch normalization

   @param (maxB) the maximum number of images (batch size)
   @param (IC) the number of channels per input image (the 
               original input has typically three channels for RGB. 
               in hidden layers, it starts from 64 and goes up 
               to 512 in the last hidden layer)
   @param (H) height of an image (32 for an input image, down to 1 in
              the last hidden layer)
   @param (W) width of an image (32 for an input image, down to 1 in
              the last hidden layer)

   @details this layer normalizes a batch of images 

 */
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
struct BatchNormalization {
#if __NVCC__
  BatchNormalization<maxB,IC,H,W> * dev; /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line options */
  logger * lgr;                 /**< logger */
  vec<IC> gamma;                /**< gamma parameter */
  vec<IC> beta;                 /**< beta parameter  */
  array4<maxB,IC,H,W> x_hat;    /**< normalized x */
  vec<IC> inv_std;              /**< inverse of standard deviation */
  array4<maxB,IC,H,W> y;        /**< output of the forward */
  vec<IC> ggamma;               /**< gradient of loss wrt gamma */
  vec<IC> gbeta;                /**< gradient of loss wrt beta  */
  array4<maxB,IC,H,W> gx;       /**< gradient of loss wrt x  */
  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg) {
    this->opt = opt;
    this->lgr = lgr;
    gamma.init_uniform(IC, rg, 0.0, 1.0);
    beta.init_uniform(IC, rg, 0.0, 1.0);
  }
  /**
     @brief make a copy of this 
     @details if this object has a device pointer, the copy will have
     a device pointer too, but its contents are NOT copied
  */
  BatchNormalization<maxB,IC,H,W>* copy() {
    BatchNormalization<maxB,IC,H,W>* c = new BatchNormalization<maxB,IC,H,W>(*this);
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
  void set_dev(BatchNormalization<maxB,IC,H,W>* dev) {
#if __NVCC__
    this->dev = dev;
    gamma.set_dev(dev ? &dev->gamma : 0);
    beta.set_dev(dev ? &dev->beta : 0);
    x_hat.set_dev(dev ? &dev->x_hat : 0);
    inv_std.set_dev(dev ? &dev->inv_std : 0);
    y.set_dev(dev ? &dev->y : 0);
    ggamma.set_dev(dev ? &dev->ggamma : 0);
    gbeta.set_dev(dev ? &dev->gbeta : 0);
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
      dev = (BatchNormalization<maxB,IC,H,W>*)dev_malloc(sizeof(*this));
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
      BatchNormalization<maxB,IC,H,W>* dev_ = dev;
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
    gamma.update(eta, ggamma);
    beta.update(eta, gbeta);
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
     @brief calc a mean of each input channel
     @param (x) input images
     @sa forward
     @sa backward
     @details this is an auxiliary function called
     from forward. mean(i) = average of pixel
     values over all pixels of all images in layer i
  */
  __device__ __host__
  vec<IC> mean_bij(array4<maxB,IC,H,W>& x) {
    const idx_t B = x.B;
    vec<IC> mean;
    mean.init_const(IC, 0);
    for (idx_t ic = 0; ic < IC; ic++) {
      real s = 0.0;
      for (idx_t b = 0; b < B; b++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            s += x(b,ic,i,j);
          }
        }
      }
      mean(ic) = s / (B * H * W);
    }
    return mean;
  }
  /**
     @brief calc a standard deviation of each input channel
     @param (x) input images
     @param (mu) mean 
     @sa forward
     @sa backward
     @details this is an auxiliary function called
     from forward. inv_std(i) = 1 / sqrt(standard deviation of pixel
     values over all pixels of all images in layer i)
  */
  __device__ __host__
  vec<IC>& inv_std_bij(array4<maxB,IC,H,W>& x, vec<IC>& mu) {
    const idx_t B = x.B;
    const real epsilon = 2.0e-5;
    const real l_BHW = 1 / (real)(B * H * W);
    inv_std.set_n(IC);
    for (idx_t ic = 0; ic < IC; ic++) {
      real s = 0.0;
      for (idx_t b = 0; b < B; b++) {
        for (idx_t i = 0; i < H; i++) {
          for (idx_t j = 0; j < W; j++) {
            real ds = x(b,ic,i,j) - mu(ic);
            s += ds * ds;
          }
        }
      }
      inv_std(ic) = 1.0 / sqrt(s * l_BHW + epsilon);
    }
    return inv_std;
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
    const idx_t B = x.B;
    x_hat.set_n_rows(B);
    y.set_n_rows(B);
    if (B * H * W > 1) {
      vec<IC> mu = mean_bij(x);
      inv_std = inv_std_bij(x, mu);
      for (idx_t b = 0; b < B; b++) {
        for (idx_t ic = 0; ic < IC; ic++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              x_hat(b,ic,i,j) = (x(b,ic,i,j) - mu(ic)) * inv_std(ic);
              y(b,ic,i,j) = gamma(ic) * x_hat(b,ic,i,j) + beta(ic);
            }
          }
        }
      }
    } else {
      for (idx_t b = 0; b < B; b++) {
        for (idx_t ic = 0; ic < IC; ic++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              y(b,ic,i,j) = x(b,ic,i,j);
            }
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
     @brief calc the loss function of a mini-batch (x,t)
     @param (x) input images
     @sa backward
     @sa update
  */
  array4<maxB,IC,H,W>& forward(array4<maxB,IC,H,W>& x) {
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
     @param (gy) the gradient of loss wrt y 
     @sa backward
     @sa backward_gpu
     @sa backward_global
     @sa backward_dev
  */
  __device__ __host__
  void backward_base(array4<maxB,IC,H,W>& gy) {
    const idx_t B = gy.B;
    gx.set_n_rows(B);
    gbeta.set_n(IC);
    ggamma.set_n(IC);
    if (B * H * W > 1) {
      for (idx_t ic = 0; ic < IC; ic++) {
        real s = 0.0, t = 0.0;
        for (idx_t b = 0; b < B; b++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              s += gy(b,ic,i,j);
              t += gy(b,ic,i,j) * x_hat(b,ic,i,j);
            }
          }
        }
        gbeta(ic) = s;
        ggamma(ic) = t;
      }
      real l_BHW = 1 / (real)(B * H * W);
      for (idx_t ic = 0; ic < IC; ic++) {
        real a = gamma(ic) * inv_std(ic);
        real gg = ggamma(ic);
        real gb = gbeta(ic);
        for (idx_t b = 0; b < B; b++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              gx(b,ic,i,j) = a * (gy(b,ic,i,j) - l_BHW * (gg * x_hat(b,ic,i,j) + gb));
            }
          }
        }
      }
    } else {
      for (idx_t b = 0; b < B; b++) {
        for (idx_t ic = 0; ic < IC; ic++) {
          for (idx_t i = 0; i < H; i++) {
            for (idx_t j = 0; j < W; j++) {
              gx(b,ic,i,j) = gy(b,ic,i,j);
            }
          }
        }
      }
    }
  }
#if __NVCC__
  /**
     @brief the device function of backward called from the 
     global (non-member) function
     @param (gy) the gradient of loss wrt y 
     @sa backward
     @sa backward_gpu
     @sa backward_global
     @sa backward_base
  */
  __device__
  void backward_dev(array4<maxB,IC,H,W>& gy) {
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
  void backward_gpu(array4<maxB,IC,H,W>& gy) {
    launch_and_sync((backward_global<<<1,1>>>(dev, gy.dev)));
  }
#endif
  /**
     @brief a cpu version of baseline code called from the 
     entry function (backward)
     @param (gy) the gradient of loss wrt y 
     @sa backward
     @sa backward_base
  */
  void backward_cpu(array4<maxB,IC,H,W>& gy) {
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
  array4<maxB,IC,H,W>& backward(array4<maxB,IC,H,W>& gy) {
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
    ggamma.init_uniform(IC, rg, p, q);
    gbeta.init_uniform(IC, rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object
     @param (o) the object from which gradients get copied
     @details transfer gradients of o to this object
  */
  void set_grad(BatchNormalization<maxB,IC,H,W>& o) {
    ggamma = o.ggamma;
    gbeta = o.gbeta;
  }
  /**
     @brief take the inner product of gradients
     @param (b) the object to take the inner product with
     @details take the inner product of this object's 
     gradients and b's gradients
  */
  real gw_dot_gw(BatchNormalization<maxB,IC,H,W>& b) {
    BatchNormalization<maxB,IC,H,W>& a = *this;
    real s = 0.0;
    s += a.ggamma.dot(b.ggamma);
    s += a.gbeta.dot(b.gbeta);
    return s;
  }
};

/**
   @brief check the gradient computation of a batch normalization layer
   @param (opt) command line option
   @param (lgr) logger 
   @param (rg) random number generator
   @param (B) the number of images
   @sa batchnormalization_main
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
template<idx_t maxB,idx_t IC,idx_t H,idx_t W>
  static real batchnormalization_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize batch normalization parameters */
  BatchNormalization<maxB,IC,H,W> * bn = new BatchNormalization<maxB,IC,H,W>();
  bn->init(opt, lgr, rg);
  bn->make_dev();
  bn->to_dev();
  /* make w - dw/2 and w + dw/2 */
  BatchNormalization<maxB,IC,H,W> * bn_minus = bn->copy();
  BatchNormalization<maxB,IC,H,W> * bn_plus = bn->copy();
  /* make coefficients to make the single loss value */
  array4<maxB,IC,H,W> * alpha = new array4<maxB,IC,H,W>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,IC,H,W> * x = new array4<maxB,IC,H,W>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* forward and backward */
  array4<maxB,IC,H,W>& y = bn->forward(*x);
  array4<maxB,IC,H,W>& gx = bn->backward(*alpha);
  /* ensure the gradient is back to host */
  bn->to_host();

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
  bn_minus->rand_grad(rg, -e, e);
  bn_plus->set_grad(*bn_minus);
  /* send them to gpu */
  bn_minus->to_dev();
  bn_plus->to_dev();
  /* update weights using gw (update runs on gpu) */
  bn_minus->update(-0.5);      /* w -= dw/2 */
  bn_plus->update(0.5);        /* w += dw/2 */
  /* make y(w-dw,x-dx), y(w+dw,x+dx) */
  array4<maxB,IC,H,W>& y_minus = bn_minus->forward(*x_minus);
  array4<maxB,IC,H,W>& y_plus  = bn_plus->forward(*x_plus);
  /* get the result back to host */
  y_minus.to_host();
  y_plus.to_host();

  /* get the single loss values */
  real L_minus = alpha->dot(y_minus);
  real L       = alpha->dot(y);
  real L_plus  = alpha->dot(y_plus);
  /* various inner products */
  real gx_gx = gx.dot(gx);                       /* ∂L/∂x・∂L/∂x */
  real dx_dx = dx->dot(*dx);                     /* ∂L/∂x・dx */
  real gx_dx = gx.dot(*dx);                      /* dx・dx */
  real gw_gw = bn->gw_dot_gw(*bn);               /* ∂L/∂w・∂L/∂w */
  real dw_dw = bn_minus->gw_dot_gw(*bn_minus);   /* ∂L/∂w・dw */
  real gw_dw = bn->gw_dot_gw(*bn_minus);         /* dw・dw */

  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  bn_minus->del_dev();
  bn_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();
  
  delete bn_minus;
  delete bn_plus;
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
   @sa batchnormalization_grad_check_rand
   @details if this header file is included from
   a main C++ file and define batchnormalization_main to be main
   (e.g., with -Dbatchnormalization_main=main), then this
   function becomes th main function of the executable.
   it calls batchnormalization_grad_check_rand repeatedly to test
   the implementation of VGG network.
*/
int batchnormalization_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const idx_t IC = 64;
  const idx_t H = 32;
  const idx_t W = 32;
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
    real e = batchnormalization_grad_check_rand<maxB,IC,H,W>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

