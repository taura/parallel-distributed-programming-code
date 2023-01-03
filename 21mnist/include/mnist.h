/**
   @file mnist.h
   @brief MNIST network
 */
#pragma once

#include "mnist_util.h"
#include "tensor.h"
#include "convolution.h"
#include "relu.h"
#include "max_pooling.h"
#include "dropout.h"
#include "linear.h"
#include "nll_log_softmax.h"
#include "grad_check.h"

/**
   @file mnist.h
   @brief MNIST network
 */
#pragma once

/**
   @brief MNIST network
   @param (maxB) maximum batch size it can accommodate (64)
   @param (C) number of channels in the input
   @param (H) image height
   @param (W) image width
   @param (nC) number of classes
 */
struct MNISTCfg {
  Convolution2DCfg conv1;
  ReluCfg relu1;
  Convolution2DCfg conv2;
  ReluCfg relu2;
  MaxPooling2DCfg max_pooling_2d;
  DropoutCfg dropout1;
  LinearCfg fc1;
  ReluCfg relu3;
  DropoutCfg dropout2;
  LinearCfg fc2;
  NLLLogSoftmaxCfg nll_log_softmax;
};

template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t nC>
struct MNIST {
#if __NVCC__
  MNIST<maxB,C,H,W,nC>* dev;    /**< device shadow */
#endif
  cmdline_opt opt;              /**< command line option */
  logger * lgr;                 /**< logger */
  tensor<real,maxB,C,H,W> x;    /**< input images */
  tensor<idx_t,maxB> t;         /**< true labels of images */
  tensor<idx_t,maxB> idxs;      /**< indexes of images */
  tensor<idx_t,maxB> pred;      /**< predicted labels of images */
  tensor<real,maxB> gy;         /**< gradient of the loss wrt the output */
  static const idx_t K = 3;     /**< kernel size = 3 x 3 */
  static const idx_t H1 =  H - K + 1, W1 =  W - K + 1;
  static const idx_t H2 = H1 - K + 1, W2 = W1 - K + 1;
  static const idx_t H3 = H2 / 2, W3 = W2 / 2;
  static const idx_t C1 = 32;   /* channels after first convolution */
  static const idx_t C2 = 64;   /* channels after second convolution */
  static const idx_t nF = 128;  /* features output by first fully-connected */
  Convolution2D<maxB,C,H,W,K,C1> conv1;
  Relu<maxB,C1,H1,W1> relu1;
  Convolution2D<maxB,C1,H1,W1,K,C2> conv2;
  Relu<maxB,C2,H2,W2> relu2;
  MaxPooling2D<maxB,C2,H2,W2,2> max_pooling_2d;
  Dropout<maxB,C2,H3,W3> dropout1;
  Linear<maxB,nF,C2,H3,W3> fc1;
  Relu<maxB,nF> relu3;
  Dropout<maxB,nF> dropout2;
  Linear<maxB,nC,nF> fc2;
  NLLLogSoftmax<maxB,nC> nll_log_softmax;
  
  /**
     @brief initialize 
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
  */
  void init(cmdline_opt opt, logger * lgr, rnd_gen_t& rg, MNISTCfg cfg) {
    this->opt = opt;
    this->lgr = lgr;
    conv1.init(opt, lgr, rg, cfg.conv1);
    relu1.init(opt, lgr, rg, cfg.relu1);
    conv2.init(opt, lgr, rg, cfg.conv2);
    relu2.init(opt, lgr, rg, cfg.relu2);
    max_pooling_2d.init(opt, lgr, rg, cfg.max_pooling_2d);
    dropout1.init(opt, lgr, rg, cfg.dropout1);
    fc1.init(opt, lgr, rg, cfg.fc1);
    relu3.init(opt, lgr, rg, cfg.relu3);
    dropout2.init(opt, lgr, rg, cfg.dropout2);
    fc2.init(opt, lgr, rg, cfg.fc2);
    nll_log_softmax.init(opt, lgr, rg, cfg.nll_log_softmax);
  }
  /**
     @brief make a copy of this 
     @details if this object has a device pointer, the copy will have
     a device pointer too, but its contents are NOT copied
  */
  MNIST<maxB,C,H,W,nC>* copy() {
    MNIST<maxB,C,H,W,nC>* c = new MNIST<maxB,C,H,W,nC>(*this);
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
  void set_dev(MNIST<maxB,C,H,W,nC>* dev) {
#if __NVCC__
    this->dev = dev;
    x.set_dev(dev ? &dev->x : 0);
    t.set_dev(dev ? &dev->t : 0);
    idxs.set_dev(dev ? &dev->idxs : 0);
    pred.set_dev(dev ? &dev->pred : 0);
    gy.set_dev(dev ? &dev->gy : 0);
    conv1.set_dev(dev ? &dev->conv1 : 0);
    relu1.set_dev(dev ? &dev->relu1 : 0);
    conv2.set_dev(dev ? &dev->conv2 : 0);
    relu2.set_dev(dev ? &dev->relu2 : 0);
    max_pooling_2d.set_dev(dev ? &dev->max_pooling_2d : 0);
    dropout1.set_dev(dev ? &dev->dropout1 : 0);
    fc1.set_dev(dev ? &dev->fc1 : 0);
    relu3.set_dev(dev ? &dev->relu3 : 0);
    dropout2.set_dev(dev ? &dev->dropout2 : 0);
    fc2.set_dev(dev ? &dev->fc2 : 0);
    nll_log_softmax.set_dev(dev ? &dev->nll_log_softmax : 0);
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
      dev = (MNIST<maxB,C,H,W,nC>*)dev_malloc(sizeof(*this));
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
      MNIST<maxB,C,H,W,nC>* dev_ = dev;
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
  void update() {
    conv1.update();
    conv2.update();
    fc1.update();
    fc2.update();
  }
  /**
     @brief calc the loss function of a mini-batch (x,t)
     @param (x) input images
     @param (t) true labels of images
     @sa backward
     @sa update
  */
  tensor<real,maxB>& forward(tensor<real,maxB,C,H,W>& x, tensor<idx_t,maxB>& t) {
    tensor<real,maxB,C1,H1,W1>& x1  = conv1.forward(x);
    tensor<real,maxB,C1,H1,W1>& x2  = relu1.forward(x1);
    tensor<real,maxB,C2,H2,W2>& x3  = conv2.forward(x2);
    tensor<real,maxB,C2,H2,W2>& x4  = relu2.forward(x3);
    tensor<real,maxB,C2,H3,W3>& x5  = max_pooling_2d.forward(x4);
    tensor<real,maxB,C2,H3,W3>& x6  = dropout1.forward(x5);
    tensor<real,maxB,nF,1,1>&   x7  = fc1.forward(x6);
    tensor<real,maxB,nF,1,1>&   x8  = relu3.forward(x7);
    tensor<real,maxB,nF,1,1>&   x9  = dropout2.forward(x8);
    tensor<real,maxB,nC,1,1>&   x10 = fc2.forward(x9);
    tensor<real,maxB>&          l   = nll_log_softmax.forward(x10, t);
    return l;
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
  tensor<real,maxB,C,H,W>& backward(tensor<real,maxB>& gl, tensor<idx_t,maxB>& t) {
    tensor<real,maxB,nC,1,1>&   gx10 = nll_log_softmax.backward(gl, t);
    tensor<real,maxB,nF,1,1>&   gx9  = fc2.backward(gx10);
    tensor<real,maxB,nF,1,1>&   gx8  = dropout2.backward(gx9);
    tensor<real,maxB,nF,1,1>&   gx7  = relu3.backward(gx8);
    tensor<real,maxB,C2,H3,W3>& gx6  = fc1.backward(gx7);
    tensor<real,maxB,C2,H3,W3>& gx5  = dropout1.backward(gx6);
    tensor<real,maxB,C2,H2,W2>& gx4  = max_pooling_2d.backward(gx5);
    tensor<real,maxB,C2,H2,W2>& gx3  = relu2.backward(gx4);
    tensor<real,maxB,C1,H1,W1>& gx2  = conv2.backward(gx3);
    tensor<real,maxB,C1,H1,W1>& gx1  = relu1.backward(gx2);
    tensor<real,maxB,C,H,W>&    gx   = conv1.backward(gx1);
    return gx;
  }
  /**
     @brief log the result of prediction
  */
  void predict(tensor<idx_t,maxB>& pred) {
    tensor<real,maxB,nC,1,1>& y = nll_log_softmax.y;
    y.to_host();
    const idx_t B = idxs.n0;
    pred.set_n0(B);
    for (idx_t s = 0; s < B; s++) {
      /* get the prediction from logsoftmax */
      idx_t pred_class = 0;
      for (idx_t c = 0; c < nC; c++) {
        if (y(s,pred_class,0,0) < y(s,c,0,0)) {
          pred_class = c;
        }
      }
      pred(s) = pred_class;
    }
  }

  /**
     @brief log the result of prediction
  */
  idx_t log_prediction(idx_t start_offset, tensor<idx_t,maxB>& pred, tensor<idx_t,maxB>& t) {
    const idx_t B = idxs.n0;
    idx_t correct = 0;
    for (idx_t s = 0; s < B; s++) {
      lgr->log(3, "sample %d image %d pred %d truth %d",
               start_offset + s, idxs(s), pred(s), t(s));
      if (pred(s) == t(s)) {
        correct++;
      }
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
  real forward_backward_update(tensor<real,maxB,C,H,W>& x, tensor<idx_t,maxB>& t) {
    const idx_t B = x.n0;
    /* forward */
    tensor<real,maxB>& L = forward(x, t);
    /* a vector (1,1,1,...) to make the single loss value from loss of each sample */
    gy.init_const(B, 1.0);
    gy.to_dev();
    /* backward (set weights of all sublayers) */
    backward(gy, t);
    /* update */
    update();
    /* get the loss of each sample back to host if we are working on GPU */
    L.to_host();
    double Lsum = gy.dot(L);
    return Lsum;
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
    conv1.rand_grad(rg, p, q);
    conv2.rand_grad(rg, p, q);
    fc1.rand_grad(rg, p, q);
    fc2.rand_grad(rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object
     @param (o) the object from which gradients get copied
     @details transfer gradients of o to this object
  */
  void copy_grad(MNIST<maxB,C,H,W,nC>& o) {
    conv1.copy_grad(o.conv1);
    conv2.copy_grad(o.conv2);
    fc1.copy_grad(o.fc1);
    fc2.copy_grad(o.fc2);
  }
  /**
     @brief perform w += alpha * gw 
   */
  void add_grad(real alpha) {
    conv1.add_grad(alpha);
    conv2.add_grad(alpha);
    fc1.add_grad(alpha);
    fc2.add_grad(alpha);
  }
  /**
     @brief take the inner product of gradients
     @param (b) the object to take the inner product with
     @details take the inner product of this object's 
     gradients and b's gradients
  */
  double grad_dot_grad(MNIST<maxB,C,H,W,nC>& b) {
    MNIST<maxB,C,H,W,nC>& a = *this;
    double s = 0.0;
    s += a.conv1.grad_dot_grad(b.conv1);
    s += a.conv2.grad_dot_grad(b.conv2);
    s += a.fc1.grad_dot_grad(b.fc1);
    s += a.fc2.grad_dot_grad(b.fc2);
    return s;
  }
};

/**
   @brief entry point of this header file
   @param (argc) the number of command line args
   @param (argv) command line args
   @sa mnist_grad_check_rand
   @details if this header file is included from
   a main C++ file and define mnist_main to be main
   (e.g., with -Dmnist_main=main), then this
   function becomes th main function of the executable.
   it calls mnist_grad_check_rand repeatedly to test
   the implementation of MNIST network.
*/
int mnist_main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  const idx_t maxB = MAX_BATCH_SIZE;
  const idx_t B = min_i(maxB, opt.batch_size);
  const int C = 1;
  const int H = 28;
  const int W = 28;
  const int nC = 10;
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
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
    //double e = mnist_grad_check_rand<maxB,C,H,W,nC>(opt, &lgr, rg, B);
    long seed = opt.dropout_seed;
    MNISTCfg cfg = {
      .conv1 = {},
      .relu1 = {},
      .conv2 = {},
      .relu2 = {},
      .max_pooling_2d = {},
      .dropout1 = { .ratio = 0.25f * (seed != 0), .seed = seed += 100 },
      .fc1 = {},
      .relu3 = {},
      .dropout2 = { .ratio =  0.5f * (seed != 0), .seed = seed += 100 },
      .fc2 = {},
      .nll_log_softmax = {}
    };
    double e = grad_check_loss<MNIST<maxB,C,H,W,nC>,
                               tensor<real,maxB,C,H,W>,
                               tensor<idx_t,maxB>,
                               tensor<real,maxB>,
                               MNISTCfg>(opt, &lgr, rg, cfg, B, nC);
    max_e = max_r(max_e, e);
    sum_e += e;
  }
  printf("max relative error = %.9f\n", max_e);
  printf("avg relative error = %.9f\n", sum_e / n_checks);
  lgr.end_log();
  return 0;
}

