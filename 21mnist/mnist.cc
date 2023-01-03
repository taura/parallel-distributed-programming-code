/**
   @file mnist.cc --- a C++ implemention of MNIST
 */

#include "include/mnist_util.h"
#include "include/mnist_data.h"
#include "include/mnist.h"

/**
   @brief grab a mini batch (B training samples), forward, backward and update.
   @return the average loss of the mini batch.
 */
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t nC>
static void train(MNIST<maxB,C,H,W,nC> * mnist,
                  mnist_dataset<maxB,C,H,W>& data, idx_t B, long epoch, long log_interval) {
  data.rewind();
  long n_samples = 0;
  for (long batch_idx = 0; data.get_data(mnist->x, mnist->t, mnist->idxs, B); batch_idx++) {
    real Lsum = mnist->forward_backward_update(mnist->x, mnist->t);
    real L = Lsum / mnist->idxs.n0;
    mnist->predict(mnist->pred);
    mnist->log_prediction(n_samples, mnist->pred, mnist->t);
    if (batch_idx % log_interval == 0) {
      printf("Train Epoch: %ld [%ld/%ld (%.0f%%)]\tLoss: %.6f\n",
             epoch, n_samples, data.n_data,
             100. * n_samples / data.n_data, L);
    }
    n_samples += mnist->idxs.n0;
  }
}

/**
   @brief forward compute B_validate validation samples 
   (taking several mini batches if necessary)
   @return the average loss of the validation data
 */
template<idx_t maxB,idx_t C,idx_t H,idx_t W,idx_t nC>
static void test(MNIST<maxB,C,H,W,nC> * mnist,
                 mnist_dataset<maxB,C,H,W>& data, idx_t B) {
  real Lsum = 0.0;
  long n_samples = 0;
  long n_correct = 0;
  data.rewind();
  for (long batch_idx = 0; data.get_data(mnist->x, mnist->t, mnist->idxs, B); batch_idx++) {
    tensor<real,maxB>& y = mnist->forward(mnist->x, mnist->t);
    y.to_host();
    mnist->predict(mnist->pred);
    Lsum += y.sum();
    n_samples += y.n0;
    n_correct += mnist->log_prediction(n_samples, mnist->pred, mnist->t);
  }
  assert(n_samples == data.n_data);
  printf("Test set: Average loss: %.4f, Accuracy: %ld/%ld (%.0f%%)\n",
         Lsum / n_samples, n_correct, n_samples, (100. * n_correct) / n_samples);
}

/**
   @brief main function of MNIST
   @details Train MNIST network with data from the file specified by
   --mnist_data/-d (default: cifar-10-batches-bin/data_batch_1.bin).
   If you want to use only a part of data, you can specify a range
   by --start_data and --end_data. e.g., --start
   Take a number of samples specified by --batch_sz/-d
   samples at a time for training.
   Occasionally evaluate the network with validation data.
   @return the average loss of the validation data
 */
int main(int argc, char ** argv) {
  cmdline_opt opt = parse_args(argc, argv);
  if (opt.error || opt.help) usage(argv[0]);
  const idx_t maxB = MAX_BATCH_SIZE;  /**< max batch size (constant) */
  const idx_t C = 1;                  /**< channels in input */
  const idx_t H = 28;                 /**< input image height */
  const idx_t W = 28;                 /**< input image width */
  const idx_t nC = 10;                /**< num classes */
  const idx_t B  = opt.batch_size; /**< true batch size (<= maxB) */
  assert(B <= maxB);
  /* logger */
  logger lgr;
  lgr.start_log(opt);
  /* random number */
  rnd_gen_t rg;
  rg.seed(opt.weight_seed);
  /* build model and initialize weights */
  lgr.log(1, "model building starts");
  long dropout_seed = opt.dropout_seed;
  MNISTCfg cfg = {
    .dropout1 = { .ratio = 0.25 * (dropout_seed != 0), .seed = (dropout_seed += 100) },
    .dropout2 = { .ratio =  0.5 * (dropout_seed != 0), .seed = (dropout_seed += 100) }
  };
  MNIST<maxB,C,H,W,nC> * mnist = new MNIST<maxB,C,H,W,nC>();
  mnist->init(opt, &lgr, rg, cfg);
  make_dev(mnist, opt.gpu_algo);
  to_dev(mnist, opt.gpu_algo);
  lgr.log(1, "model building ends");
  /* load data */
  mnist_dataset<maxB,C,H,W> train_data;
  mnist_dataset<maxB,C,H,W> test_data;
  real mean = 0.1307;           // pytorch
  real std = 0.3081;            // pytorch
  train_data.load(lgr, opt.data_dir, opt.train_data_size, mean, std, 1);
  test_data.load(lgr, opt.data_dir, opt.test_data_size, mean, std, 0);
  /* training loop */
  lgr.log(1, "training starts");
  for (long i = 0; i < opt.epochs; i++) {
    train(mnist, train_data, B, i + 1, opt.log_interval);
    test(mnist, test_data, B);
    //scheduler.step();
  }
  lgr.log(1, "training ends");
  lgr.end_log();

  train_data.close();
  test_data.close();
  delete mnist;
  return 0;
}

