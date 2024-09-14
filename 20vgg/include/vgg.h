/**
   @file vgg.h
   @brief VGG network
 */
#pragma once

#include "vgg_util.h"
#include "vgg_arrays.h"
#include "block.h"
#include "linear.h"
#include "dropout.h"
#include "maxpooling.h"
#include "softmaxcrossentropy.h"

/**
   @brief VGG network
   @param (maxB) maximum batch size it can accommodate (64)
   @param (C0) the number of channels in an input image (3)
   @param (H) height of an input image (32)
   @param (W) width of an input image (32)
   @param (K) convolution kernel size (1). filter array has (2K+1)*(2K+1) elems)
   @param (S) shrink factor of pooling layers (2)
   @param (C1) number of channels in the first hidden layer (64). those in the following layers are 2xC1, 4xC1, 8xC1, ..., 
   @param (nC) number of classes (10)
 */
template<idx_t maxB,idx_t C0,idx_t H,idx_t W,idx_t K,idx_t S,idx_t C1,idx_t nC>
struct VGG {
#if __NVCC__
  VGG<maxB,C0,H,W,K,S,C1,nC>* dev; /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger */
  array4<maxB,C0,H,W> x;        /**< input images */
  ivec<maxB> t;                 /**< true labels of images */
  ivec<maxB> idxs;              /**< indexes of images */
  vec<maxB> gy;                 /**< gradient of the loss wrt the output */
  
  /* group 1 : (C0,H1,W1)->(C1,H2,W2) */
  static const idx_t H1 = H,    /**< intermediate image size */
    W1 = W;                     /**< intermediate image size */
  Block       <maxB,C0,H1,W1,K,C1> block1_1;        /**< sublayer */
  Dropout     <maxB,C1,H1,W1>      dropout1_1;      /**< sublayer */
  Block       <maxB,C1,H1,W1,K,C1> block1_2;        /**< sublayer */
  MaxPooling2D<maxB,C1,H1,W1,S>    max_pooling_2d1; /**< sublayer */

  static const idx_t H2 = H/S,  /**< intermediate image size */
    W2 = W/S,                   /**< intermediate image size */
    C2 = S*C1;                  /**< intermediate channels */
  /* group 2 : (C1,H2,W2)->(C2,H3,W3) */
  Block       <maxB,C1,H2,W2,K,C2> block2_1;        /**< sublayer */
  Dropout     <maxB,C2,H2,W2>      dropout2_1;      /**< sublayer */
  Block       <maxB,C2,H2,W2,K,C2> block2_2;        /**< sublayer */
  MaxPooling2D<maxB,C2,H2,W2,S>    max_pooling_2d2; /**< sublayer */

  static const idx_t H3 = H2/S, /**< intermediate image size */
    W3 = W2/S,                  /**< intermediate image size */
    C3 = S*C2;                  /**< intermediate channels */
  /* group 3 : (C2,H3,W3)->(C3,H4,W4) */
  Block       <maxB,C2,H3,W3,K,C3> block3_1;        /**< sublayer */
  Dropout     <maxB,C3,H3,W3>      dropout3_1;      /**< sublayer */
  Block       <maxB,C3,H3,W3,K,C3> block3_2;        /**< sublayer */
  Dropout     <maxB,C3,H3,W3>      dropout3_2;      /**< sublayer */
  Block       <maxB,C3,H3,W3,K,C3> block3_3;        /**< sublayer */
  MaxPooling2D<maxB,C3,H3,W3,S>    max_pooling_2d3; /**< sublayer */
  
  static const idx_t H4 = H3/S, /**< intermediate image size */
    W4 = W3/S,                  /**< intermediate image size */
    C4 = S*C3;                  /**< intermediate channels */
  /* group 4 : (C3,H4,W4)->(C4,H5,W5) */
  Block       <maxB,C3,H4,W4,K,C4> block4_1;        /**< sublayer */
  Dropout     <maxB,C4,H4,W4>      dropout4_1;      /**< sublayer */
  Block       <maxB,C4,H4,W4,K,C4> block4_2;        /**< sublayer */
  Dropout     <maxB,C4,H4,W4>      dropout4_2;      /**< sublayer */
  Block       <maxB,C4,H4,W4,K,C4> block4_3;        /**< sublayer */
  MaxPooling2D<maxB,C4,H4,W4,S>    max_pooling_2d4; /**< sublayer */
  
  static const idx_t H5 = H4/S, /**< intermediate image size */
    W5 = W4/S;                  /**< intermediate image size */
  /* group 5 : (C4,H5,W5)->(C4,H6,W6) */
  Block       <maxB,C4,H5,W5,K,C4> block5_1;        /**< sublayer */
  Dropout     <maxB,C4,H5,W5>      dropout5_1;      /**< sublayer */
  Block       <maxB,C4,H5,W5,K,C4> block5_2;        /**< sublayer */
  Dropout     <maxB,C4,H5,W5>      dropout5_2;      /**< sublayer */
  Block       <maxB,C4,H5,W5,K,C4> block5_3;        /**< sublayer */
  MaxPooling2D<maxB,C4,H5,W5,S>    max_pooling_2d5; /**< sublayer */
  
  /* group 6 : (C4,H6,W6) -> classification -> loss */
  static const idx_t H6 = H5/S, /**< intermediate image size */
    W6 = W5/S;                  /**< intermediate image size */
  Dropout           <maxB,C4,H6,W6> dropout6_1; /**< sublayer */
  Linear            <maxB,C4,C4>    fc1;        /**< sublayer */
  BatchNormalization<maxB,C4,H6,W6> bn_fc1;     /**< sublayer */
  Relu              <maxB,C4,H6,W6> relu;       /**< sublayer */
  Dropout           <maxB,C4,H6,W6> dropout6_2; /**< sublayer */
  Linear            <maxB,C4,nC>    fc2;        /**< sublayer */
  SoftmaxCrossEntropy<maxB,nC>      softmax_cross_entropy; /**< sublayer */

  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg) {
    this->opt = opt;
    this->lgr = lgr;
    assert(H6 == 1);
    assert(W6 == 1);

    long seed = opt.dropout_seed;
    real dropout_ratio1 = 0.3 * (opt.dropout != 0);
    real dropout_ratio2 = 0.4 * (opt.dropout != 0);
    real dropout_ratio3 = 0.5 * (opt.dropout != 0);
    
    block1_1.init(opt, lgr, rg);
    dropout1_1.init(opt, lgr, dropout_ratio1, seed += 100);
    block1_2.init(opt, lgr, rg);
    max_pooling_2d1.init(opt, lgr);

    block2_1.init(opt, lgr, rg);
    dropout2_1.init(opt, lgr, dropout_ratio2, seed += 100);
    block2_2.init(opt, lgr, rg);
    max_pooling_2d2.init(opt, lgr);

    block3_1.init(opt, lgr, rg);
    dropout3_1.init(opt, lgr, dropout_ratio2, seed += 100);
    block3_2.init(opt, lgr, rg);
    dropout3_2.init(opt, lgr, dropout_ratio2, seed += 100);
    block3_3.init(opt, lgr, rg);
    max_pooling_2d3.init(opt, lgr);

    block4_1.init(opt, lgr, rg);
    dropout4_1.init(opt, lgr, dropout_ratio2, seed += 100);
    block4_2.init(opt, lgr, rg);
    dropout4_2.init(opt, lgr, dropout_ratio2, seed += 100);
    block4_3.init(opt, lgr, rg);
    max_pooling_2d4.init(opt, lgr);

    block5_1.init(opt, lgr, rg);
    dropout5_1.init(opt, lgr, dropout_ratio2, seed += 100);
    block5_2.init(opt, lgr, rg);
    dropout5_2.init(opt, lgr, dropout_ratio2, seed += 100);
    block5_3.init(opt, lgr, rg);
    max_pooling_2d5.init(opt, lgr);

    dropout6_1.init(opt, lgr, dropout_ratio3, seed += 100);
    fc1.init(opt, lgr, rg);
    bn_fc1.init(opt, lgr, rg);
    relu.init(opt, lgr);
    dropout6_2.init(opt, lgr, dropout_ratio3, seed += 100);
    fc2.init(opt, lgr, rg);
    
    softmax_cross_entropy.init(opt, lgr);
  }
  /**
     @brief make a copy of this 
     @details if this object has a device pointer, the copy will have
     a device pointer too, but its contents are NOT copied
  */
  VGG<maxB,C0,H,W,K,S,C1,nC>* copy() {
    VGG<maxB,C0,H,W,K,S,C1,nC>* c = new VGG<maxB,C0,H,W,K,S,C1,nC>(*this);
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
  void set_dev(VGG<maxB,C0,H,W,K,S,C1,nC>* dev) {
#if __NVCC__
    this->dev = dev;
    x.set_dev(dev ? &dev->x : 0);
    t.set_dev(dev ? &dev->t : 0);
    gy.set_dev(dev ? &dev->gy : 0);

    block1_1.set_dev(dev ? &dev->block1_1 : 0);
    dropout1_1.set_dev(dev ? &dev->dropout1_1 : 0);
    block1_2.set_dev(dev ? &dev->block1_2 : 0);
    max_pooling_2d1.set_dev(dev ? &dev->max_pooling_2d1 : 0);
    
    block2_1.set_dev(dev ? &dev->block2_1 : 0);
    dropout2_1.set_dev(dev ? &dev->dropout2_1 : 0);
    block2_2.set_dev(dev ? &dev->block2_2 : 0);
    max_pooling_2d2.set_dev(dev ? &dev->max_pooling_2d2 : 0);
    
    block3_1.set_dev(dev ? &dev->block3_1 : 0);
    dropout3_1.set_dev(dev ? &dev->dropout3_1 : 0);
    block3_2.set_dev(dev ? &dev->block3_2 : 0);
    dropout3_2.set_dev(dev ? &dev->dropout3_2 : 0);
    block3_3.set_dev(dev ? &dev->block3_3 : 0);
    max_pooling_2d3.set_dev(dev ? &dev->max_pooling_2d3 : 0);
    
    block4_1.set_dev(dev ? &dev->block4_1 : 0);
    dropout4_1.set_dev(dev ? &dev->dropout4_1 : 0);
    block4_2.set_dev(dev ? &dev->block4_2 : 0);
    dropout4_2.set_dev(dev ? &dev->dropout4_2 : 0);
    block4_3.set_dev(dev ? &dev->block4_3 : 0);
    max_pooling_2d4.set_dev(dev ? &dev->max_pooling_2d4 : 0);
    
    block5_1.set_dev(dev ? &dev->block5_1 : 0);
    dropout5_1.set_dev(dev ? &dev->dropout5_1 : 0);
    block5_2.set_dev(dev ? &dev->block5_2 : 0);
    dropout5_2.set_dev(dev ? &dev->dropout5_2 : 0);
    block5_3.set_dev(dev ? &dev->block5_3 : 0);
    max_pooling_2d5.set_dev(dev ? &dev->max_pooling_2d5 : 0);
    
    dropout6_1.set_dev(dev ? &dev->dropout6_1 : 0);
    fc1.set_dev(dev ? &dev->fc1 : 0);
    bn_fc1.set_dev(dev ? &dev->bn_fc1 : 0);
    relu.set_dev(dev ? &dev->relu : 0);
    dropout6_2.set_dev(dev ? &dev->dropout6_2 : 0);
    fc2.set_dev(dev ? &dev->fc2 : 0);
    
    softmax_cross_entropy.set_dev(dev ? &dev->softmax_cross_entropy : 0);
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
      dev = (VGG<maxB,C0,H,W,K,S,C1,nC>*)dev_malloc(sizeof(*this));
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
      /* make sure dev field does not get broken */
      VGG<maxB,C0,H,W,K,S,C1,nC>* dev_ = dev;
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
    block1_1.update(eta);
    block1_2.update(eta);

    block2_1.update(eta);
    block2_2.update(eta);

    block3_1.update(eta);
    block3_2.update(eta);
    block3_3.update(eta);
  
    block4_1.update(eta);
    block4_2.update(eta);
    block4_3.update(eta);
  
    block5_1.update(eta);
    block5_2.update(eta);
    block5_3.update(eta);
  
    fc1.update(eta);
    bn_fc1.update(eta);
    fc2.update(eta);
  }
  /**
     @brief calc the loss function of a mini-batch (x,t)
     @param (x) input images
     @param (t) true labels of images
     @sa backward
     @sa update
  */
  vec<maxB>& forward(array4<maxB,C0,H,W>& x, ivec<maxB>& t) {
    /* group 1 : (C0,H1,W1)->(C1,H2,W2) */
    array4<maxB,C1,H1,W1>&  x1 = block1_1.forward(x);
    array4<maxB,C1,H1,W1>&  x2 = dropout1_1.forward(x1);
    array4<maxB,C1,H1,W1>&  x3 = block1_2.forward(x2);
    array4<maxB,C1,H2,W2>&  x4 = max_pooling_2d1.forward(x3);
    /* group 2 : (C1,H2,W2)->(C2,H3,W3) */
    array4<maxB,C2,H2,W2>&  x5 = block2_1.forward(x4);
    array4<maxB,C2,H2,W2>&  x6 = dropout2_1.forward(x5);
    array4<maxB,C2,H2,W2>&  x7 = block2_2.forward(x6);
    array4<maxB,C2,H3,W3>&  x8 = max_pooling_2d2.forward(x7);
    /* group 3 : (C2,H3,W3)->(C3,H4,W4) */
    array4<maxB,C3,H3,W3>&  x9 = block3_1.forward(x8);
    array4<maxB,C3,H3,W3>& x10 = dropout3_1.forward(x9);
    array4<maxB,C3,H3,W3>& x11 = block3_2.forward(x10);
    array4<maxB,C3,H3,W3>& x12 = dropout3_2.forward(x11);
    array4<maxB,C3,H3,W3>& x13 = block3_3.forward(x12);
    array4<maxB,C3,H4,W4>& x14 = max_pooling_2d3.forward(x13);
    /* group 4 : (C3,H4,W4)->(C4,H5,W5) */
    array4<maxB,C4,H4,W4>& x15 = block4_1.forward(x14);
    array4<maxB,C4,H4,W4>& x16 = dropout4_1.forward(x15);
    array4<maxB,C4,H4,W4>& x17 = block4_2.forward(x16);
    array4<maxB,C4,H4,W4>& x18 = dropout4_2.forward(x17);
    array4<maxB,C4,H4,W4>& x19 = block4_3.forward(x18);
    array4<maxB,C4,H5,W5>& x20 = max_pooling_2d4.forward(x19);
    /* group 5 : (C4,H5,W5)->(C4,H6,W6) */
    array4<maxB,C4,H5,W5>& x21 = block5_1.forward(x20);
    array4<maxB,C4,H5,W5>& x22 = dropout5_1.forward(x21);
    array4<maxB,C4,H5,W5>& x23 = block5_2.forward(x22);
    array4<maxB,C4,H5,W5>& x24 = dropout5_2.forward(x23);
    array4<maxB,C4,H5,W5>& x25 = block5_3.forward(x24);
    array4<maxB,C4,H6,W6>& x26 = max_pooling_2d5.forward(x25);
    /* group 6 : (C4,H6,W6) -> classification -> loss */
    array4<maxB,C4,H6,W6>& x27 = dropout6_1.forward(x26);
    array4<maxB,C4,H6,W6>& x28 = fc1.forward(x27);
    array4<maxB,C4,H6,W6>& x29 = bn_fc1.forward(x28);
    array4<maxB,C4,H6,W6>& x30 = relu.forward(x29);
    array4<maxB,C4,H6,W6>& x31 = dropout6_2.forward(x30);
    array4<maxB,nC,H6,W6>& x32 = fc2.forward(x31);
    vec<maxB>& y = softmax_cross_entropy.forward(x32, t);
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
  array4<maxB,C0,H1,W1>& backward(vec<maxB>& gy) {
    /* group 6 */
    array4<maxB,nC,H6,W6>& g32 = softmax_cross_entropy.backward(gy);
    array4<maxB,C4,H6,W6>& g31 = fc2.backward(g32);
    array4<maxB,C4,H6,W6>& g30 = dropout6_2.backward(g31);
    array4<maxB,C4,H6,W6>& g29 = relu.backward(g30);
    array4<maxB,C4,H6,W6>& g28 = bn_fc1.backward(g29);
    array4<maxB,C4,H6,W6>& g27 = fc1.backward(g28);
    array4<maxB,C4,H6,W6>& g26 = dropout6_1.backward(g27);
    /* group 5 */
    array4<maxB,C4,H5,W5>& g25 = max_pooling_2d5.backward(g26);
    array4<maxB,C4,H5,W5>& g24 = block5_3.backward(g25);
    array4<maxB,C4,H5,W5>& g23 = dropout5_2.backward(g24);
    array4<maxB,C4,H5,W5>& g22 = block5_2.backward(g23);
    array4<maxB,C4,H5,W5>& g21 = dropout5_1.backward(g22);
    /* group 4 */
    array4<maxB,C4,H5,W5>& g20 = block5_1.backward(g21);
    array4<maxB,C4,H4,W4>& g19 = max_pooling_2d4.backward(g20);
    array4<maxB,C4,H4,W4>& g18 = block4_3.backward(g19);
    array4<maxB,C4,H4,W4>& g17 = dropout4_2.backward(g18);
    array4<maxB,C4,H4,W4>& g16 = block4_2.backward(g17);
    array4<maxB,C4,H4,W4>& g15 = dropout4_1.backward(g16);
    /* group 3 */
    array4<maxB,C3,H4,W4>& g14 = block4_1.backward(g15);
    array4<maxB,C3,H3,W3>& g13 = max_pooling_2d3.backward(g14);
    array4<maxB,C3,H3,W3>& g12 = block3_3.backward(g13);
    array4<maxB,C3,H3,W3>& g11 = dropout3_2.backward(g12);
    array4<maxB,C3,H3,W3>& g10 = block3_2.backward(g11);
    array4<maxB,C3,H3,W3>&  g9 = dropout3_1.backward(g10);
    /* group 2 */
    array4<maxB,C2,H3,W3>&  g8 = block3_1.backward(g9);
    array4<maxB,C2,H2,W2>&  g7 = max_pooling_2d2.backward(g8);
    array4<maxB,C2,H2,W2>&  g6 = block2_2.backward(g7);
    array4<maxB,C2,H2,W2>&  g5 = dropout2_1.backward(g6);
    /* group 1 */
    array4<maxB,C1,H2,W2>&  g4 = block2_1.backward(g5);
    array4<maxB,C1,H1,W1>&  g3 = max_pooling_2d1.backward(g4);
    array4<maxB,C1,H1,W1>&  g2 = block1_2.backward(g3);
    array4<maxB,C1,H1,W1>&  g1 = dropout1_1.backward(g2);
    array4<maxB,C0,H1,W1>&  g0 = block1_1.backward(g1);
    return g0;
  }
  int log_minibatch(idx_t start_offset) {
    array2<maxB,nC>& lsm = softmax_cross_entropy.lsm;
    lsm.to_host();
    const idx_t B = idxs.n;
    int correct = 0;
    for (idx_t b = 0; b < B; b++) {
      /* get the prediction from logsoftmax */
      idx_t pred_class = 0;
      for (idx_t c = 0; c < nC; c++) {
        if (lsm(b,pred_class) < lsm(b,c)) {
          pred_class = c;
        }
      }
      if (pred_class == t(b)) {
        correct++;
      }
      lgr->log(1, "sample %d image %d pred %d truth %d",
               start_offset + b, idxs(b), pred_class, t(b));
    }
    return correct;
  }

  /**
     @brief perform an entire iteration (= forward; backward; update)
     @param (x) input images (a mini batch)
     @param (t) true labels
     @param (eta) learning rate
     @sa forward
     @sa backward
     @sa update
     @details do everything on a mini-batch. forward calculates the 
     loss wrt x and t; backward calculates the gradient 
     of loss wrt x and weights; update updates weights with the
     gradients and the learning rate.
  */
  real forward_backward_update(array4<maxB,C0,H,W>& x, ivec<maxB>& t, real eta) {
    const idx_t B = x.B;
    /* forward */
    vec<maxB>& y = forward(x, t);
    /* a vector (1,1,1,...) to make the single loss value from loss of each sample */
    gy.init_const(B, 1.0);
    gy.to_dev();
    /* backward (set weights of all sublayers) */
    backward(gy);
    /* update */
    real e = eta / B;
    update(-e);
    /* get the loss of each sample back to host if we are working on GPU */
    y.to_host();
    real L = gy.dot(y);
    return L;
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
    block1_1.rand_grad(rg, p, q);
    block1_2.rand_grad(rg, p, q);

    block2_1.rand_grad(rg, p, q);
    block2_2.rand_grad(rg, p, q);

    block3_1.rand_grad(rg, p, q);
    block3_2.rand_grad(rg, p, q);
    block3_3.rand_grad(rg, p, q);
  
    block4_1.rand_grad(rg, p, q);
    block4_2.rand_grad(rg, p, q);
    block4_3.rand_grad(rg, p, q);
  
    block5_1.rand_grad(rg, p, q);
    block5_2.rand_grad(rg, p, q);
    block5_3.rand_grad(rg, p, q);
  
    fc1.rand_grad(rg, p, q);
    bn_fc1.rand_grad(rg, p, q);
    fc2.rand_grad(rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object
     @param (o) the object from which gradients get copied
     @details transfer gradients of o to this object
  */
  void set_grad(VGG<maxB,C0,H,W,K,S,C1,nC>& o) {
    block1_1.set_grad(o.block1_1);
    block1_2.set_grad(o.block1_2);
    
    block2_1.set_grad(o.block2_1);
    block2_2.set_grad(o.block2_2);

    block3_1.set_grad(o.block3_1);
    block3_2.set_grad(o.block3_2);
    block3_3.set_grad(o.block3_3);
  
    block4_1.set_grad(o.block4_1);
    block4_2.set_grad(o.block4_2);
    block4_3.set_grad(o.block4_3);
  
    block5_1.set_grad(o.block5_1);
    block5_2.set_grad(o.block5_2);
    block5_3.set_grad(o.block5_3);
  
    fc1.set_grad(o.fc1);
    bn_fc1.set_grad(o.bn_fc1);
    fc2.set_grad(o.fc2);
  }
  /**
     @brief take the inner product of gradients
     @param (b) the object to take the inner product with
     @details take the inner product of this object's 
     gradients and b's gradients
  */
  real gw_dot_gw(VGG<maxB,C0,H,W,K,S,C1,nC>& b) {
    VGG<maxB,C0,H,W,K,S,C1,nC>& a = *this;
    real s = 0.0;
    s += a.block1_1.gw_dot_gw(b.block1_1);
    s += a.block1_2.gw_dot_gw(b.block1_2);
    s += a.block2_1.gw_dot_gw(b.block2_1);
    s += a.block2_2.gw_dot_gw(b.block2_2);
    s += a.block3_1.gw_dot_gw(b.block3_1);
    s += a.block3_2.gw_dot_gw(b.block3_2);
    s += a.block3_3.gw_dot_gw(b.block3_3);
    s += a.block4_1.gw_dot_gw(b.block4_1);
    s += a.block4_2.gw_dot_gw(b.block4_2);
    s += a.block4_3.gw_dot_gw(b.block4_3);
    s += a.block5_1.gw_dot_gw(b.block5_1);
    s += a.block5_2.gw_dot_gw(b.block5_2);
    s += a.block5_3.gw_dot_gw(b.block5_3);
    s += a.fc1.gw.dot(b.fc1.gw);
    s += a.bn_fc1.gw_dot_gw(b.bn_fc1);
    s += a.fc2.gw.dot(b.fc2.gw);
    return s;
  }
};

/**
   @brief check the gradient computation of a VGG network
   @param (opt) command line option
   @param (lgr) logger 
   @param (rg) random number generator
   @param (B) the number of images
   @sa vgg_main
   @details it first makes a VGG network with initial weights W 
   and generates an input (x and t).
   it then creates two VGG networks whose weights are slightly different
   from the original one by dw/2 (i.e., w-dw/2 and w+dw/2), as well as
   two inputs slighly different from the original inputs by dx/2
   (x-dx/2 and x+dx/2).  it then computes L(w,x), L(x-dw/2,x-dx/2) and
   L(w+dw/2,x+dw/2) and check if L(x+dw/2,x+dx/2)-L(x-dw/2,x-dx/2)
   is close to ∂L/∂x dx + ∂L/∂w dw.  ∂L/∂x and ∂L/∂w are obtained
   by backward computation. This is essentially checking if
   the gradients obtained by backward computation correctly approximates
   the diff of the output.
*/
template<idx_t maxB,idx_t C0,idx_t H,idx_t W,idx_t K,idx_t S,idx_t C1,idx_t nC>
  static real vgg_grad_check_rand(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, idx_t B) {
  /* initialize vgg parameters */
  VGG<maxB,C0,H,W,K,S,C1,nC> * vgg = new VGG<maxB,C0,H,W,K,S,C1,nC>();
  vgg->init(opt, lgr, rg);
  vgg->make_dev();
  vgg->to_dev();
  /* make w - dw/2 and w + dw/2 */
  VGG<maxB,C0,H,W,K,S,C1,nC> * vgg_minus = vgg->copy();
  VGG<maxB,C0,H,W,K,S,C1,nC> * vgg_plus  = vgg->copy();
  /* make coefficients to make the single loss value */
  vec<maxB> * alpha = new vec<maxB>();
  alpha->make_dev(opt.gpu_algo);
  alpha->init_uniform(B, rg, 0.0, 1.0);
  alpha->to_dev();
  /* make input (x) */
  array4<maxB,C0,H,W> * x = new array4<maxB,C0,H,W>();
  x->make_dev(opt.gpu_algo);
  x->init_uniform(B, rg, 0.0, 1.0);
  x->to_dev();
  /* make input (t) */
  ivec<maxB>* t = new ivec<maxB>();
  t->make_dev(opt.gpu_algo);
  t->init_uniform(B, rg, 0, nC);
  t->to_dev();
  /* forward and backward */
  vec<maxB>& y = vgg->forward(*x, *t);
  array4<maxB,C0,H,W>& gx = vgg->backward(*alpha);
  /* ensure the gradient is back to host */
  vgg->to_host();
    
  /* make dx */
  real e = 1.0e-4;
  array4<maxB,C0,H,W> * dx = new array4<maxB,C0,H,W>();
  dx->init_uniform(B, rg, -e, e);
  /* make x - dx/2 and x + dx/2 */
  array4<maxB,C0,H,W> * x_minus = new array4<maxB,C0,H,W>(*x);
  x_minus->make_dev(opt.gpu_algo);
  array4<maxB,C0,H,W> * x_plus  = new array4<maxB,C0,H,W>(*x);
  x_plus->make_dev(opt.gpu_algo);
  /* update on the host and send the to gpu */
  x_minus->update(-0.5, *dx);
  x_plus->update( 0.5, *dx);
  x_minus->to_dev();
  x_plus->to_dev();
    
  /* set gw to a random vector */
  vgg_minus->rand_grad(rg, -e, e);
  vgg_plus->set_grad(*vgg_minus);
  /* send them to gpu */
  vgg_minus->to_dev();
  vgg_plus->to_dev();
  /* update weights using gw (update runs on gpu) */
  vgg_minus->update(-0.5);      /* w -= dw/2 */
  vgg_plus->update(0.5);        /* w += dw/2 */
  /* make y(w-dw/2,x-dx/2), y(w+dw/2,x+dx/2) */
  vec<maxB>& y_minus = vgg_minus->forward(*x_minus, *t);
  vec<maxB>& y_plus  = vgg_plus->forward(*x_plus, *t);
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
  real gw_gw = vgg->gw_dot_gw(*vgg);             /* ∂L/∂w・∂L/∂w */
  real dw_dw = vgg_minus->gw_dot_gw(*vgg_minus); /* ∂L/∂w・dw */
  real gw_dw = vgg->gw_dot_gw(*vgg_minus);       /* dw・dw */
  
  real rel_e = show_error(gx_gx, dx_dx, gx_dx, gw_gw, dw_dw, gw_dw, L_minus, L, L_plus);
  /* clean up */
  vgg->del_dev();
  vgg_minus->del_dev();
  vgg_plus->del_dev();
  alpha->del_dev();
  x->del_dev();
  dx->del_dev();
  x_minus->del_dev();
  x_plus->del_dev();

  delete vgg;
  delete vgg_minus;
  delete vgg_plus;
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
   @sa vgg_grad_check_rand
   @details if this header file is included from
   a main C++ file and define vgg_main to be main
   (e.g., with -Dvgg_main=main), then this
   function becomes th main function of the executable.
   it calls vgg_grad_check_rand repeatedly to test
   the implementation of VGG network.
*/
int vgg_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_sz);
  const int C0 = 3;
  const int H = 32;
  const int W = 32;
  const int K = 1;
  const int S = 2;
  const int C1 = 64;
  const int nC = 10;
  const int n_checks = opt.iters;
  /* logger */
  logger lgr;
  lgr.start_log(opt);
  /* initialize random number generator */
  rnd_gen_t rg;
  rg.seed(opt.sample_seed);
  /* check errors */
  real max_e = 0.0;
  real sum_e = 0.0;
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    real e = vgg_grad_check_rand<maxB,C0,H,W,K,S,C1,nC>(opt, &lgr, rg, B);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

