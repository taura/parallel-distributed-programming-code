/**
   @file block.h
   @brief a block of three layers (convolution; batch normalization; relu)
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"
#include "convolution.h"
#include "batchnormalization.h"
#include "relu.h"

/**
   @brief a block of three layers (convolution; batch normalization; relu)

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

   @details this layer applies convolution, batch normalization and relu
   in this order.

 */
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
struct Block {
#if __NVCC__
  Block<maxB,IC,H,W,K,OC>* dev; /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger  */
  Convolution2D     <maxB,IC,H,W,K,OC> conv; /**< convolution layer */
  BatchNormalization<maxB,OC,H,W>      bn;   /**< batch normalization layer */
  Relu              <maxB,OC,H,W>      relu; /**< rectified linear layer  */
  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg) {
    this->opt = opt;
    this->lgr = lgr;
    conv.init(opt, lgr, rg);
    bn.init(opt, lgr, rg);
    relu.init(opt, lgr);
  }
  /**
     @brief make a copy of this 
     @details if this object has a device pointer, the copy will have
     a device pointer too, but its contents are NOT copied
  */
  Block<maxB,IC,H,W,K,OC>* copy() {
    Block<maxB,IC,H,W,K,OC>* c = new Block<maxB,IC,H,W,K,OC>(*this);
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
  void set_dev(Block<maxB,IC,H,W,K,OC>* dev) {
#if __NVCC__
    this->dev = dev;
    conv.set_dev(dev ? &dev->conv : 0);
    bn.set_dev(dev ? &dev->bn : 0);
    relu.set_dev(dev ? &dev->relu : 0);
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
      dev = (Block<maxB,IC,H,W,K,OC>*)dev_malloc(sizeof(*this));
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
      Block<maxB,IC,H,W,K,OC>* dev_ = dev;
      ::to_host(this, dev_, sizeof(*this));
      assert(dev_ == dev);
    }
#endif
  }
  /**
     @brief update weights of all sublayers with gradients
     that must have been computed
     @param (eta) the learning rate
     @sa forward
     @sa backward
  */
  void update(real eta) {
    conv.update(eta);
    bn.update(eta);
  }
  /**
     @brief calc the loss function of a mini-batch (x)
     @param (x) input images
     @sa backward
     @sa update
  */
  array4<maxB,OC,H,W>& forward(array4<maxB,IC,H,W>& x) {
    array4<maxB,OC,H,W>& x1 = conv.forward(x);
    array4<maxB,OC,H,W>& x2 = bn.forward(x1);
    array4<maxB,OC,H,W>&  y = relu.forward(x2);
    return y;
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
    array4<maxB,OC,H,W>& g2 = relu.backward(gy);
    array4<maxB,OC,H,W>& g1 = bn.backward(g2);
    array4<maxB,IC,H,W>& gx = conv.backward(g1);
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
    conv.rand_grad(rg, p, q);
    bn.rand_grad(rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object
     @param (o) the object from which gradients get copied
     @details transfer gradients of o to this object
  */
  void set_grad(Block<maxB,IC,H,W,K,OC>& o) {
    conv.set_grad(o.conv);
    bn.set_grad(o.bn);
  }
  /**
     @brief take the inner product of gradients
     @param (b) the object to take the inner product with
     @details take the inner product of this object's 
     gradients and b's gradients
  */
  real gw_dot_gw(Block<maxB,IC,H,W,K,OC>& b) {
    Block<maxB,IC,H,W,K,OC>& a = *this;
    real s = 0.0;
    s += a.conv.gw_dot_gw(b.conv);
    s += a.bn.gw_dot_gw(b.bn);
    return s;
  }
};

/**
   @brief check the gradient computation of a block
   @param (opt) command line option
   @param (lgr) logger 
   @param (rg) random number generator
   @param (B) the number of images
   @sa block_main
   @details it first makes a block with initial weights W 
   and generates an input (x and t).
   it then creates two BLOCK networks whose weights are slightly different
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
  static real block_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize block parameters */
  Block<maxB,IC,H,W,K,OC> * block = new Block<maxB,IC,H,W,K,OC>();
  block->init(opt, lgr, rg);
  block->make_dev();
  block->to_dev();
  /* make w - dw/2 and w + dw/2 */
  Block<maxB,IC,H,W,K,OC> * block_minus = block->copy();
  Block<maxB,IC,H,W,K,OC> * block_plus  = block->copy();
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
  array4<maxB,OC,H,W>& y = block->forward(*x);
  array4<maxB,IC,H,W>& gx = block->backward(*alpha);
  /* ensure the gradient is back to host */
  block->to_host();

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
  block_minus->rand_grad(rg, -e, e);
  block_plus->set_grad(*block_minus);
  /* send them to gpu */
  block_minus->to_dev();
  block_plus->to_dev();
  /* update weights using gw (update runs on gpu) */
  block_minus->update(-0.5);      /* w -= dw/2 */
  block_plus->update(0.5);        /* w += dw/2 */
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  array4<maxB,OC,H,W>& y_minus = block_minus->forward(*x_minus);
  array4<maxB,OC,H,W>& y_plus  = block_plus->forward(*x_plus);
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
  real gw_gw = block->gw_dot_gw(*block);             /* ∂L/∂w・∂L/∂w */
  real dw_dw = block_minus->gw_dot_gw(*block_minus); /* ∂L/∂w・dw */
  real gw_dw = block->gw_dot_gw(*block_minus);       /* dw・dw */

  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  block->del_dev();
  block_minus->del_dev();
  block_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete block;
  delete block_minus;
  delete block_plus;
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
   @sa block_grad_check_rand
   @details if this header file is included from
   a main C++ file and define block_main to be main
   (e.g., with -Dblock_main=main), then this
   function becomes th main function of the executable.
   it calls block_grad_check_rand repeatedly to test
   the implementation of block.
*/
int block_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
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
    real e = block_grad_check_rand<maxB,IC,H,W,K,OC>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

