/**
   @file softmaxcrossentropy.h
   @brief softmax + cross entropy layer
 */
#pragma once

#include <math.h>
#include "vgg_util.h"
#include "vgg_arrays.h"

#if __NVCC__
template<idx_t maxB,idx_t nC>
  struct SoftmaxCrossEntropy;

/**
   @brief a global CUDA function that implements the baseline 
   forward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (x_dev) the address of the device shadow of the input matrix
   @param (t_dev) the address of the device shadow of the true labels
   @sa forward_dev
   @sa forward_gpu
  */
template<idx_t maxB,idx_t nC>
  __global__ void forward_global(SoftmaxCrossEntropy<maxB,nC>* dev,
                                 array4<maxB,nC,1,1>* x_dev, ivec<maxB>* t_dev) {
  dev->forward_dev(*x_dev, *t_dev);
}

/**
   @brief a global CUDA function that implements the baseline 
   backward function for GPU
   @param (dev) the address of the device shadow of the object
   @param (gy_dev) the address of the device shadow of the input matrix
   @sa backward_dev
   @sa backward_gpu
  */
template<idx_t maxB,idx_t nC>
  __global__ void backward_global(SoftmaxCrossEntropy<maxB,nC>* dev,
                                  vec<maxB>* gy_dev) {
  dev->backward_dev(*gy_dev);
}
#endif

/**
   @brief dropout layer

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
   using the cross entropy function. 
 */
template<idx_t maxB,idx_t nC>
struct SoftmaxCrossEntropy {
#if __NVCC__
  SoftmaxCrossEntropy<maxB,nC>* dev; /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger */
  ivec<maxB>* t_ptr;            /**< pointer to the true labels passed to forward */
  array2<maxB,nC> lsm;          /**< record log(softmax)  */
  vec<maxB> y;                  /**< output of the forward */
  array4<maxB,nC,1,1> gx;       /**< gradient of loss wrt to input x */

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
  SoftmaxCrossEntropy<maxB,nC>* copy() {
    SoftmaxCrossEntropy<maxB,nC>* c = new SoftmaxCrossEntropy<maxB,nC>(*this);
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
  void set_dev(SoftmaxCrossEntropy<maxB,nC>* dev) {
#if __NVCC__
    this->dev = dev;
    y.set_dev(dev ? &dev->y : 0);
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
      dev = (SoftmaxCrossEntropy<maxB,nC>*)dev_malloc(sizeof(*this));
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
      SoftmaxCrossEntropy<maxB,nC>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }

  /**
     @brief compute log(softmax(x))
     @param (x) a matrix 
     @details for 1D vector x x = (x_0, ..., x_{n-1}), 
     
                           (exp(x_0)     / Σ_j exp(x_j))
     logsoftmax(x)_i = log (exp(x_1)     / Σ_j exp(x_j))
                           (   ...       / Σ_j exp(x_j))
                           (exp(x_{n-1}) / Σ_j exp(x_j))

     the input to this function is essentially a two 
     dimensional matrix (4D array whose last two axes
     have only one element), which is simply a set of
     vectors.

 */
  __device__ __host__
  array2<maxB,nC>& logsoftmax(array4<maxB,nC,1,1>& x) {
    const idx_t B = x.B;
    for (long b = 0; b < B; b++) {
      long m = 0;
      for (long c = 0; c < nC; c++) {
        m = (x(b,m,0,0) < x(b,c,0,0) ? c : m);
      }
      real s = 0.0;
      for (long c = 0; c < nC; c++) {
        lsm(b,c) = x(b,c,0,0) - x(b,m,0,0);
        s += exp(lsm(b,c));
      }
      for (long c = 0; c < nC; c++) {
        lsm(b,c) -= log(s);
      }
    }
    return lsm;
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
  void forward_base(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    const idx_t B = x.B;
    lsm.set_n_rows(B);
    y.set_n(B);
    t_ptr = &t;

    logsoftmax(x);
    for (idx_t b = 0; b < B; b++) {
      y(b) = -lsm(b,t(b));
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
  void forward_dev(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    forward_base(x, t);
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
  void forward_gpu(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev, t.dev)));
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
  void forward_cpu(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    forward_base(x, t);
  }
  /**
     @brief calc the loss function of a mini-batch (x,t)
     @param (x) input images
     @param (t) true labels of images
     @sa backward
     @sa update
  */
  vec<maxB>& forward(array4<maxB,nC,1,1>& x, ivec<maxB>& t) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu(x, t); break;
#if __NVCC__
    case algo_gpu_base:
      forward_gpu(x, t); break;
#endif
    default:
      if (opt.gpu_algo) {
#if __NVCC__
        forward_gpu(x, t);
#else
        err_gpu_algo_no_gpu(opt.algo_s);
#endif
      } else {
        forward_cpu(x, t);
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
  void backward_base(vec<maxB>& gy) {
    const idx_t B = gy.n;
    gx.set_n_rows(B);
    ivec<maxB>& t = *t_ptr;
    for (idx_t b = 0; b < B; b++) {
      for (idx_t c = 0; c < nC; c++) {
        if (c == t(b)) {
          gx(b,c,0,0) = gy(b) * (-1 + exp(lsm(b,c)));
        } else {
          gx(b,c,0,0) = gy(b) * exp(lsm(b,c));
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
  void backward_dev(vec<maxB>& gy) {
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
  void backward_gpu(vec<maxB>& gy) {
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
  void backward_cpu(vec<maxB>& gy) {
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
  array4<maxB,nC,1,1>& backward(vec<maxB>& gy) {
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
   @brief check the gradient computation of a softmaxcrossentropy layer
   @param (opt) command line option
   @param (lgr) logger 
   @param (rg) random number generator
   @param (B) the number of images
   @sa softmaxcrossentropy_main
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
template<idx_t maxB,idx_t nC>
  static real softmaxcrossentropy_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize softmax */
  SoftmaxCrossEntropy<maxB,nC> * smxe = new SoftmaxCrossEntropy<maxB,nC>();
  smxe->init(opt, lgr);
  smxe->make_dev();
  smxe->to_dev();
  /* make copies */
  SoftmaxCrossEntropy<maxB,nC>* smxe_minus = smxe->copy();
  SoftmaxCrossEntropy<maxB,nC>* smxe_plus  = smxe->copy();
  /* make coefficients to make the single loss value */
  vec<maxB> * alpha = new vec<maxB>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,nC,1,1> * x = new array4<maxB,nC,1,1>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* make input (t) */
  ivec<maxB> * t = new ivec<maxB>();
  t->make_dev(opt.gpu_algo);
  t->init_uniform(B, rg, 0, nC);
  t->to_dev();
  /* forward and backward */
  vec<maxB>& y = smxe->forward(*x, *t);
  array4<maxB,nC,1,1>& gx = smxe->backward(*alpha);
  smxe->to_host();

  /* make dx */
  real e = 1.0e-4;
  array4<maxB,nC,1,1> * dx = new array4<maxB,nC,1,1>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,nC,1,1> * x_minus = new array4<maxB,nC,1,1>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,nC,1,1> * x_plus  = new array4<maxB,nC,1,1>(*x);
  x_plus->make_dev(opt.gpu_algo);
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* send copies to gpu */
  smxe_minus->to_dev();
  smxe_plus->to_dev();
  /* make y(x-dx/2), y(x+dx/2) */
  vec<maxB>& y_minus = smxe_minus->forward(*x_minus, *t);
  vec<maxB>& y_plus  = smxe_plus->forward(*x_plus, *t);
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
  smxe->del_dev();
  smxe_minus->del_dev();
  smxe_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  t->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete smxe;
  delete smxe_minus;
  delete smxe_plus;
  delete alpha;
  delete x;
  delete t;
  delete dx;
  delete x_minus;
  delete x_plus;
  return rel_e;
}

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
   @sa softmaxcrossentropy_grad_check_rand
   @details if this header file is included from
   a main C++ file and define softmaxcrossentropy_main to be main
   (e.g., with -Dsoftmaxcrossentropy_main=main), then this
   function becomes th main function of the executable.
   it calls softmaxcrossentropy_grad_check_rand repeatedly to test
   the implementation of VGG network.
*/
int softmaxcrossentropy_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
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
    real e = softmaxcrossentropy_grad_check_rand<maxB,nC>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

