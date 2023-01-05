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
   @brief configuration data for MNIST
*/
struct MNISTCfg {
  Convolution2DCfg conv1;       /**< conv1's cfg parameter */
  ReluCfg relu1;                /**< relu1's cfg parameter */
  Convolution2DCfg conv2;       /**< conv2's cfg parameter */
  ReluCfg relu2;                /**< relu2's cfg parameter */
  MaxPooling2DCfg max_pooling_2d; /**< max_pooling_2d's cfg parameter */
  DropoutCfg dropout1;            /**< dropout1's cfg parameter */
  LinearCfg fc1;                  /**< fc1's cfg parameter */
  ReluCfg relu3;                  /**< relu3's cfg parameter */
  DropoutCfg dropout2;            /**< dropout2's cfg parameter */
  LinearCfg fc2;                  /**< fc2's cfg parameter */
  NLLLogSoftmaxCfg nll_log_softmax; /**< nll_log_softmax's cfg parameter */
};

/**
   @brief MNIST network
   @param (maxB) maximum batch size it can accommodate (64)
   @param (C) number of channels in the input
   @param (H) image height
   @param (W) image width
   @param (nC) number of classes
 */
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t nC>
struct MNIST {
#if __CUDACC__
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
     @brief initialize everything
     @param (opt) command line options
     @param (lgr) logger
     @param (rg) random number generator for initializing weights
     @param (cfg) configuration parameters
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
     @brief set the device pointer for this and all subobjects
     @param (dev) a device memory or null

     @details if dev is not null, dev fields of all subojects 
     point to the corresponding subjects in the device memory.
     if dev is not null, all dev fields become null.
  */
  void set_dev(MNIST<maxB,C,H,W,nC>* dev) {
#if __CUDACC__
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
    conv1.update();
    conv2.update();
    fc1.update();
    fc2.update();
  }
  /**
     @brief forward phase of the network
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
  tensor<real,maxB>& forward(tensor<real,maxB,C,H,W>& x, tensor<idx_t,maxB>& t, int training) {
    tensor<real,maxB,C1,H1,W1>& x1  = conv1.forward(x, training);
    tensor<real,maxB,C1,H1,W1>& x2  = relu1.forward(x1, training);
    tensor<real,maxB,C2,H2,W2>& x3  = conv2.forward(x2, training);
    tensor<real,maxB,C2,H2,W2>& x4  = relu2.forward(x3, training);
    tensor<real,maxB,C2,H3,W3>& x5  = max_pooling_2d.forward(x4, training);
    tensor<real,maxB,C2,H3,W3>& x6  = dropout1.forward(x5, training);
    tensor<real,maxB,nF>&       x7  = fc1.forward(x6, training);
    tensor<real,maxB,nF>&       x8  = relu3.forward(x7, training);
    tensor<real,maxB,nF>&       x9  = dropout2.forward(x8, training);
    tensor<real,maxB,nC>&       x10 = fc2.forward(x9, training);
    tensor<real,maxB>&          l   = nll_log_softmax.forward(x10, t, training);
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
     @sa backward_cpu_base
     @sa backward_cuda_base
     @sa backward_cuda_base_global
     @sa backward_cuda_base_device
     @sa backward_base
     @sa forward
     @sa update
  */
  tensor<real,maxB,C,H,W>& backward(tensor<real,maxB>& gl, tensor<idx_t,maxB>& t) {
    tensor<real,maxB,nC>&       gx10 = nll_log_softmax.backward(gl, t);
    tensor<real,maxB,nF>&       gx9  = fc2.backward(gx10);
    tensor<real,maxB,nF>&       gx8  = dropout2.backward(gx9);
    tensor<real,maxB,nF>&       gx7  = relu3.backward(gx8);
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
     @brief write the predicted class of all samples of the batch into pred
     @param (pred) the vector to which the predicted classes are written to
  */
  void predict(tensor<idx_t,maxB>& pred) {
    tensor<real,maxB,nC>& y = nll_log_softmax.y;
    to_host(&y, opt.cuda_algo);
    const idx_t B = idxs.n0;
    pred.set_n0(B);
    for (idx_t s = 0; s < B; s++) {
      /* get the prediction from logsoftmax */
      idx_t pred_class = 0;
      for (idx_t c = 0; c < nC; c++) {
        if (y(s,pred_class) < y(s,c)) {
          pred_class = c;
        }
      }
      pred(s) = pred_class;
    }
  }
  /**
     @brief write the predicted classes into the log and returns the number
     of correctly predicted samples
     @param (start_offset) the sequence number of the first sample of 
     the batch
     @param (pred) the vector of predicted classes for each sample
     the batch
     @param (t) the vector of true labels for each sample
  */
  idx_t log_prediction(idx_t start_offset,
                       tensor<idx_t,maxB>& pred, tensor<idx_t,maxB>& t) {
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
     @sa forward
     @sa backward
     @sa update
     @details do everything on a mini-batch. forward calculates the 
     loss wrt x and t; backward calculates the gradient 
     of loss wrt x and weights; update updates weights with the
     gradients
  */
  real forward_backward_update(tensor<real,maxB,C,H,W>& x, tensor<idx_t,maxB>& t) {
    const idx_t B = x.n0;
    /* forward */
    tensor<real,maxB>& L = forward(x, t, 1);
    /* a vector (1,1,1,...) to make the single loss value from loss of each sample */
    gy.init_const(B, 1.0);
    to_dev(&gy, opt.cuda_algo);
    /* backward (set weights of all sublayers) */
    backward(gy, t);
    /* update */
    update();
    /* get the loss of each sample back to host if we are working on GPU */
    to_host(&L, opt.cuda_algo);
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
     @details only used for checking gradient computation
  */
  void rand_grad(rnd_gen_t& rg, real p, real q) {
    conv1.rand_grad(rg, p, q);
    conv2.rand_grad(rg, p, q);
    fc1.rand_grad(rg, p, q);
    fc2.rand_grad(rg, p, q);
  }
  /**
     @brief set all gradients to gradients of another object o
     @param (o) the object from which gradients get copied
     @details only used for checking gradient computation
  */
  void copy_grad(MNIST<maxB,C,H,W,nC>& o) {
    conv1.copy_grad(o.conv1);
    conv2.copy_grad(o.conv2);
    fc1.copy_grad(o.fc1);
    fc2.copy_grad(o.fc2);
  }
  /**
     @brief w += alpha * gw
     @param (alpha) alpha of w += alpha * gw
  */
  void add_grad(real alpha) {
    conv1.add_grad(alpha);
    conv2.add_grad(alpha);
    fc1.add_grad(alpha);
    fc2.add_grad(alpha);
  }
  /**
     @brief take the inner product of gradients
     @param (o) the object to take the inner product with
     @details take the inner product of this object's gradient and b's
     gradient. only used for checking gradient computation
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
   it calls grad_check repeatedly to test
   the implementation of backward of mnist.
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
  long seed1 = opt.dropout_seed_1;
  long seed2 = opt.dropout_seed_2;
  MNISTCfg cfg = {
    .conv1 = {},
    .relu1 = {},
    .conv2 = {},
    .relu2 = {},
    .max_pooling_2d = {},
    .dropout1 = { .ratio = 0.25f * (seed1 != 0), .seed = seed1 },
    .fc1 = {},
    .relu3 = {},
    .dropout2 = { .ratio =  0.5f * (seed2 != 0), .seed = seed2 },
    .fc2 = {},
    .nll_log_softmax = {}
  };
  for (int iter = 0; iter < n_checks; iter++) {
    printf("==== %d ====\n", iter);
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

