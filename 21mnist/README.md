Overview
=============

* This program trains a neural network for hand-written character recognition
* It is a very basic implementation without vectorization or parallelization
  * CPU and CUDA version are provided
  * CPU version does not use SIMD or threads
  * CUDA version uses only a single(!) CUDA thread

Neural Network Model
==================

* The model is a translation of mnist model in pytorch examples https://github.com/pytorch/examples/tree/main/mnist
* The original pytorch model is written in python but the baseline code provided herein is purely C++
* You may want to run the orignal pytorch implementation as follows, just to get a sense of how the learning should progress
```
$ git clone git@github.com:pytorch/examples.git 
# if the above does not work, try this
# git clone https://github.com/pytorch/examples.git
$ cd examples/mnist
$ python3 main.py 
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../data/MNIST/raw/train-images-idx3-ubyte.gz
100%|███████████████████████████████████████████████████████████████████████████████████| 9912422/9912422 [00:00<00:00, 19001166.44it/s]
Extracting ../data/MNIST/raw/train-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../data/MNIST/raw/train-labels-idx1-ubyte.gz
100%|███████████████████████████████████████████████████████████████████████████████████████| 28881/28881 [00:00<00:00, 47226391.35it/s]
Extracting ../data/MNIST/raw/train-labels-idx1-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw/t10k-images-idx3-ubyte.gz
100%|███████████████████████████████████████████████████████████████████████████████████| 1648877/1648877 [00:00<00:00, 16007747.99it/s]
Extracting ../data/MNIST/raw/t10k-images-idx3-ubyte.gz to ../data/MNIST/raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz
100%|█████████████████████████████████████████████████████████████████████████████████████████| 4542/4542 [00:00<00:00, 22333562.45it/s]
Extracting ../data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/MNIST/raw

Train Epoch: 1 [0/60000 (0%)]	Loss: 2.305400
Train Epoch: 1 [640/60000 (1%)]	Loss: 1.358980
Train Epoch: 1 [1280/60000 (2%)]	Loss: 0.862885
Train Epoch: 1 [1920/60000 (3%)]	Loss: 0.594271
Train Epoch: 1 [2560/60000 (4%)]	Loss: 0.339464
Train Epoch: 1 [3200/60000 (5%)]	Loss: 0.444732
  ...
```

Dataset: MNIST
==================

* The dataset is in `data` directory
* If you execute the above command to run the pytorch model, the same data should have been downloaded into `examples/data/MNIST/raw`; the contents of the `raw` directory and `data` directory should be identical

Compile: 
==================

* `compile.mk` is a makefile template you will base your work on 
* copy it to `Makefile` and run `make`
* it builds executable(s) into `exe` directory
* in the default setting, it will build `exe/mnist_cpu_base` and `exe/mnist_cuda_base`
* the former is made by `clang++` and the latter by `nvcc`

```
$ cp compile.mk Makefile # first-time only
$ make
mkdir -p exe/dir
clang++ -O3 -DMAX_BATCH_SIZE=64 -DARRAY_INDEX_CHECK=0 -Dreal_type=float -Wall -Wextra -Wno-strict-overflow   -o exe/mnist_cpu_base mnist.cc
nvcc -O3 -DMAX_BATCH_SIZE=64 -DARRAY_INDEX_CHECK=0 -Dreal_type=float -x cu --gpu-code sm_80 --gpu-architecture compute_80   -o exe/mnist_cuda_base mnist.cc
```
* `make clean` removes `exe` directory

```
$ make clean
rm -rf exe
```

* You can add your executables by changing the `Makefile` (see comments in it for how)

Run: 
==================

* Make sure you have the following files
  * `data/train-images-idx3-ubyte` : training images
  * `data/train-labels-idx1-ubyte` : training labels
  * `data/t10k-images-idx3-ubyte` : test images
  * `data/t10k-labels-idx1-ubyte` : test labels

CPU:
------------------

```
$ ./exe/mnist_cpu_base [options]
```

(or whichever executable name you obtained, if you changed build settings in `Makefile`)

* Run on a compute node, through srun

```
$ srun -p cpu -t 0:20:00 ./exe/mnist_cpu_base [options]
```

GPU:
------------------

* Run on login node.  You cannot run GPU code on the login node, so it's almost pointless

```
$ ./exe/mnist_cuda_base [options]
```

* Run on a compute node having GPUs. Do not forget `--gres gpu:1` option

```
$ srun -p gpu -t 0:20:00 --gres gpu:1 ./exe/mnist_cuda_base [options]
```

Runtime Options
=============

Help (`-v`, `--verbose`)
--------------------------

```
$ ./exe/mnist_cpu_base --help
./exe/mnist_cpu_base: option '--help' requires an argument
usage:

./exe/mnist_cpu_base [options]

 -d,--data-dir D : read data from D [data]
 -m,--epochs N : run N epochs [14]
 -b,--batch-size N : set batch size to N [64]
 -a,--algo ALGORITHM : set the algorithm (implementation) used [cpu_base]
 -v,--verbose L : set verbosity level to L [1]
 -l,--lr ETA : set learning rate to ETA [1.000000]
 --train-data-size N : set training data size to N [-1]
 --test-data-size N : set test data size to N [-1]
 --log-interval N : show progress every N batches [10]
 --dropout-seed-1 S : set seed for dropout layer 1 to S [56789012345234]
 --dropout-seed-2 S : set seed for dropout layer 2 to S [67890123452345]
 --weight-seed S : set seed for initial weights to S [45678901234523]
 --grad-dbg 0/1 : debug gradient computation [0]
 --log FILE : write log to FILE [mnist.log]
 -h,--help
```


Verbosity (`-v`, `--verbose`)
--------------------------

* Give `-v 2` option and you will see the progress more frequently.  You can also know which functions are taking much time.

```
$ ./exe/mnist_cpu_base -v 2
57492: open a log Thu Jan  5 01:18:44 2023
116634: model building starts
34305813: model building ends
34314251: loading data from data
176462711: use 60000 data items out of 60000
177693306: loading data from data
201820701: use 10000 data items out of 10000
202036708: training starts
202086438: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 1, 28, 28, 3, 32>::forward(tensor<real, maxB, IC, H, W> &, int) [maxB = 64, IC = 1, H = 28, W = 28, K = 3, OC = 32]: starts
205420643: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 1, 28, 28, 3, 32>::forward(tensor<real, maxB, IC, H, W> &, int) [maxB = 64, IC = 1, H = 28, W = 28, K = 3, OC = 32]: ends. took 3329884 nsec
205426445: tensor<real, N0, N1, N2, N3> &Relu<64, 32, 26, 26>::forward(tensor<real, N0, N1, N2, N3> &, int) [N0 = 64, N1 = 32, N2 = 26, N3 = 26]: starts
206214594: tensor<real, N0, N1, N2, N3> &Relu<64, 32, 26, 26>::forward(tensor<real, N0, N1, N2, N3> &, int) [N0 = 64, N1 = 32, N2 = 26, N3 = 26]: ends. took 786295 nsec
206254235: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 32, 26, 26, 3, 64>::forward(tensor<real, maxB, IC, H, W> &, int) [maxB = 64, IC = 32, H = 26, W = 26, K = 3, OC = 64]: starts
707303715: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 32, 26, 26, 3, 64>::forward(tensor<real, maxB, IC, H, W> &, int) [maxB = 64, IC = 32, H = 26, W = 26, K = 3, OC = 64]: ends. took 501034856 nsec
707320868: tensor<real, N0, N1, N2, N3> &Relu<64, 64, 24, 24>::forward(tensor<real, N0, N1, N2, N3> &, int) [N0 = 64, N1 = 64, N2 = 24, N3 = 24]: starts
709169505: tensor<real, N0, N1, N2, N3> &Relu<64, 64, 24, 24>::forward(tensor<real, N0, N1, N2, N3> &, int) [N0 = 64, N1 = 64, N2 = 24, N3 = 24]: ends. took 1846468 nsec
 ...
```

* `-v 3` shows the result of every sample

Execution log (`--log`)
--------------------------

* Detailed execution records are saved into a file (default: `mnist.log`).  The file includes everything you will see with `-v 3` and more.  You can specify the filename with --log option.  When you execute many instances concurrently, make sure you specify a unique log file to each process.

Batch size (`-b`, `--batch-size`)
--------------------------

* For training, it repeats taking a number of samples and updating the model parameters (weights) to the direction that decreases the loss (the difference between the model prediction and the true label).  In each iteration, it takes a number of samples specified by `--batch-size` (`-b`).

```
$ ./exe/mnist_cpu_base -b 1
  ...
```

* This number is called the _mini-batch size_.  The default is `MAX_BATCH_SIZE` specified by a compile-time option `-DMAX_BATCH_SIZE=N`.  For example, `clang++ -DMAX_BATCH_SIZE=64 ... mnist.cc` sets the mini-batch size to 64.  A usual value is 64 but you may consider changing it for performance tuning.

```
$ ... (edit the Makefile at the line `flags += -DMAX_BATCH_SIZE=xxx`) ...
$ make
mkdir -p exe/dir
clang++ -O3 -DMAX_BATCH_SIZE=128 -DARRAY_INDEX_CHECK=0 -Dreal_type=float -Wall -Wextra -Wno-strict-overflow   -o exe/mnist_cpu_base mnist.cc     
nvcc -O3 -DMAX_BATCH_SIZE=128 -DARRAY_INDEX_CHECK=0 -Dreal_type=float -x cu --gpu-code sm_80 --gpu-architecture compute_80   -o exe/mnist_cuda_base mnist.cc     
```

* Note that `MAX_BATCH_SIZE` affects the memory footprint.  An instance of MNIST object holds all intermediate data within the instance and its size is roughly proportional to `MAX_BATCH_SIZE`.  Note that specifying a small batch size at runtime (via `--batch-size`) does not change the size of an instance.

* The batch size significantly affects the time of a single iteration, especially in an unoptimized baseline code.  The baseline code will take a time proportional to the batch size for a single iteration.

* For a quick experiment, you will want to make it small (e.g., `-b 1`).

* For easier debugging, you may also want to consider compiling the program with `-DMAX_BATCH_SIZE=1`

The number of iterations (`-m`,`--epochs`)
--------------------------

* `-m` option specifies the number of epochs to run.  Default 14.

* A single epoch scans the entire data (60000 samples) once

* Therefore, by default, it processes 60000 x 14 = 840000 samples

The number of data (`--train-data-size N` and `--test-data-size N`)
--------------------------

* You can specify the number of training and test samples used in the data file by `--train-data-size N` and `--test-data-size N`, respectively

* They use the first `N` samples in the training data and test data file

* Therefore, by setting all of `--train-data-size` `--test-data-size` and `--epochs` to very small numbers, you can experiment with the program quickly

* A meaningless example that set all of them to one (train with a single sample and test it with another sample)

```
$ ./exe/mnist_cpu_base -m 1 --train-data-size 1 --test-data-size 1
59257: model building starts
1067454326: model building ends
1067490294: loading 9000/1000 training/validation data from cifar-10-batches-bin/data_batch_1.bin starts
1507158800: loading data ends
1507168085: training starts
1507170300: === train 0 - 1 ===
2551752614: train loss = 1.770801783
2551766476: training ends
```
dropout (`--dropout-seed-1 X` and `--dropout-seed-2 X`)
--------------------------

* There are two _dropout_ in the network

* A dropout layer takes a number of inputs and produces the same number of cells

* For each input, it either zeros or passes through the input randomly (i.e., output_i = 0 with some probability and input_i otherwise) during the training

* You can turn off the dropout by giving `--dropout-seed-1 0` `--dropout-seed-2 0` (default is ON)

* If the argument to `--dropout-seed-1` is not zero, the first dropout layer drops (zeros) the output with probability 0.25; if the argument to `--dropout-seed-2` is not zero, the second dropout layer drops (zeros) the output with probability 0.5

* Non-zero arguments to `--dropout-seed-1` or `--dropout-seed-2` seed the random number generator that chooses which outputs are zeroed in the respective dropout layers

Data directory (`-d DIR`, `--data-dir DIR`)
--------------------------

* specifies directory from which data will be read (default: `data`)
* file names in the directory are hardcoded
  * `DIR/train-images-idx3-ubyte` : training images
  * `DIR/train-labels-idx1-ubyte` : training labels
  * `DIR/t10k-images-idx3-ubyte` : test images
  * `DIR/t10k-labels-idx1-ubyte` : test labels

Learning rate (`--lr R`)
--------------------------

* In each iteration, the backward phase calculates the gradient of the averaged loss (the average taken over the mini-batch) with respect to all the weights of the network.  The update phase that follows then changes the weights using the gradients.  The specific algorithm used is called [AdaDelta](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html)

* The argument to `--lr` option specifies the learning rate (the parameter $\gamma$ in the description of the above page)

* The default value is 1.0 (the same as pytorch implementation)

* In practice, learning rate is an important hyper parameter to affect the stability or speed of convergence, but in this exercise, you do not have to play with it as the main theme is to optimize performance of the computation

Change seeds
--------------------------

There are a few places in which the program uses random numbers, namely,

 * when it initializes weight parameters,
 * when it chooses which cells to dropout in a dropout layer
 
_ALL COMPONENTS BEHAVE DETERMINISTICALLY._  That is, if you repeat executions with the same configuration repeatedly, it starts from the same weight parameters, uses the same data for validation and training, picks up the same training data and drops out the same cells.

Unless your algorithm behave undeterministically, the results should be always the same.  This should help you debug your code.

You can change these things by giving different seeds for each of these random number generators.  Specifically,

 * `--weight-seed` changes initial weight parameters
 * `--dropout-seed-1` and `--dropout-seed-2` change which cells are dropped out

Simply give an arbitrary number to any of them to make sure your algorithm is not sensitive to any of them.

A general remark: when using a pseudo random number for a randomized algorithm such as this one, _ALWAYS_ give it a seed you chose and make it behave deterministically given the same seed.  This is a tremendous help for debugging.  After you have done initial debugging, you can test your algorithm across different sequences of random numbers just by giving different seeds.  Note that virtually all pseudo random number generators are deterministic after a seed is given.  A random number generator without any seed generates different sequences every time simply because they use a different seed each time; when you do not seed it, it simply takes a value from an external source (e.g., the current time) and uses it as a seed.  Nothing else is different.  In this sense, there is almost no point in not giving a seed of your choice (give the current time as a seed if you want to purposefully make it behave differently each time).  Without giving a seed, your algorithm can NEVER behave deterministically, which is a nightmare for debugging and the nightmare is nothing but unnecessary.

GPU execution (`-a gpu_base`)
--------------------------

* The program compiled with nvcc supports GPU execution and this is the default behavior. It also supports CPU execution with `-a cpu_base` is given

* Here are the default behavior of baseline implementation

```
$ exe/mnist_cpu_base -a cpu_base  # (1) baseline code on CPU 
$ exe/mnist_cpu_base -a gpu_base  # (2) error
$ exe/mnist_cpu_base              # (3) same as (1)
$ exe/mnist_cuda_base -a cpu_base # (4) baseline code on CPU 
$ exe/mnist_cuda_base -a gpu_base # (5) baseline code on GPU
$ exe/mnist_cuda_base             # (6) same as (5)
```

* `nvc++` and `g++` versions behave the same as `clang++` version

* Note that baseline code is neither vectorized nor parallelized.  In particular, it uses only a single CUDA thread on GPU (!)

Algorithm choice (`-a`)
--------------------------

* The `-a` option described above is an option that chooses an algorithm from available repertories.  In the given code, only baseline algorithms for GPU and CPU are implemented.  You will add your implementation as another available choice here.


Controlled experiments
--------------------------

* After you did a bit of work, you want to make sure you got it done right.  Especially, you may be afraid that you broke a function.  To make sure the network is still functioning, you might want to do a small and controlled experiment.

* You probably want to start with something like this.

```
$ ./exe/mnist_cpu_base -a YOUR_ALGORITHM --train-data-size 1 --test-data-size 0
```

* This uses only a single training sample and no test data
* So it keeps taking a single data and updating the weight for that data
* If the displayed `Loss:` value does not quickly decrease (from somewhere around 2.3), something is fundamentally broken
* Here is a sample output

```
$ ./exe/mnist_cpu_base --train-data-size 1 --test-data-size 0
70439: model building starts
29138567: model building ends
29151547: loading data from data
178925205: use 1 data items out of 60000
180095306: loading data from data
205702860: use 0 data items out of 10000
205865807: training starts
246484743: Train Epoch: 1 [0/1 (0%)]	Loss: 2.286959
287260933: Train Epoch: 2 [0/1 (0%)]	Loss: 0.927964
326199836: Train Epoch: 3 [0/1 (0%)]	Loss: 0.004244
366033056: Train Epoch: 4 [0/1 (0%)]	Loss: 0.001433
406566019: Train Epoch: 5 [0/1 (0%)]	Loss: 0.002503
447044652: Train Epoch: 6 [0/1 (0%)]	Loss: 0.000037
487002791: Train Epoch: 7 [0/1 (0%)]	Loss: 0.000096
527712430: Train Epoch: 8 [0/1 (0%)]	Loss: 0.000678
569439645: Train Epoch: 9 [0/1 (0%)]	Loss: 0.000182
609997880: Train Epoch: 10 [0/1 (0%)]	Loss: 0.000268
651483114: Train Epoch: 11 [0/1 (0%)]	Loss: 0.001442
692094341: Train Epoch: 12 [0/1 (0%)]	Loss: 0.000163
731295247: Train Epoch: 13 [0/1 (0%)]	Loss: 0.000023
772726750: Train Epoch: 14 [0/1 (0%)]	Loss: 0.000006
772747086: training ends
```

* Recall that this will repeat processing only a single sample and the "Loss" refers to the loss against this particular sample.  If the loss does not decrease, you are very likely to have introduced a bug in your gradient calculation (backward) or somewhere else.

* If it succeeds with a single training sample, try it with a larger but still a small number of samples next (e.g., 16) and gradually increase the number of samples


How you interpret the loss?
==========================

If you are curious what the value of the loss actually means, here it is.  In the final stage of the network, it ends up with a vector of 10 components (the number of classes) for each image, each component of which represents the probability that the model thinks the image belongs to a particular class.  This 10-element vector, say P, is then compared with the true label (class) for it.  If the true class of that image is c, then the loss for this particular image is 

   -log(P[c])

where P[c] is the probability that the model says the image belongs to the true class c.

Therefore, if the network is a random guess that returns 1/10 for every class, the average loss for an image is

   -log(1/10) = 2.3025850...

which is just about what you observe in the first iteration.

* The loss becomes zero if the network says the probability is 1 for the correct class and 0 for all other classes.  But as far as the classification performance  is concerned, the network outputs a correct class as long as the probability for the true class is larger than for any other class.  A sufficient condition for this is P[c] > 0.5, as it guarantees that P[c] is maximum among P[0], ..., P[9], whose sum is one.  When P[c] = 0.5, the loss would be 

   -log(1/2) = 0.69314...

Navigating the source code
==========================

1. open `docs/doxy/html/index.html` with your browser to see documentation of source files
1. open `docs/tags/HTML/index.html` with your browser to jump between functions
1. compile it with -O0 -g (edit Makefile) and run it under the debugger (gdb or cuda-gdb)

Guide for development
==========================

Source code structure
--------------------------

 * `mnist.cc` -- the main file
 * `include/`

  (nuts and bolts)

  - `cuda_util.h` -- helpers for CUDA
  - `mnist_util.h`  -- trivial utilities
  - `tensors.h` -- vectors, matrix and multidimensional tensors
  - `mnist_data.h` -- data handling

  (primitive layers)

  - `convolution.h` -- convolution
  - `linear.h` -- linear (or fully connected) layer
  - `relu.h` -- rectified linear activation
  - `dropout.h` -- dropout
  - `max_pooling.h` -- max pooling
  - `nll_log_softmax.h` -- log softmax + negative log-likelihood

  (composite layers)

  - `mnist.h` -- the entire MNIST

The main function in mnist.cc instantiates a MNIST network, which is defined in `mnist.h`
It repeats processing training data, occasionally processing validation data.

Each layer defines a class whose name is similar to the file name.  e.g., `convolution.h` defines `Convolution2D` class.

All classes for primitive and composite layers have two important functions, among others.
 * `forward` -- take an input from the previous (downstream) layer and computes its output
 * `backward` -- take a gradient of loss wrt the upstream layer and computes the gradient wrt its input and weights

In addition, classes that have parameters (convolution, linear and batchnormalization) have another function.

 * `update` -- take a learning rate parameter and update its weights, using the gradient computed in the backward phase.

**********************************************************************************
YOUR MAIN JOB WILL BE TO IMPLEMENT A HIGH PERFORMANCE VERSION OF THESE FUNCTIONS.
**********************************************************************************

You eventually want to work on all six files (`convolution.h`, `linear.h`, `relu.h`, `dropout.h`, `max_pooling.h`, `nll_log_softmax.h`) but you can work incrementally.  You can make one layer faster while leaving all others intact.  You can know which functions are taking much time by `-v 2` option.

Stepping through the code using gdb (or cuda-gdb)
--------------------------

When working on details, you want to step through the code using gdb (or cuda-gdb to step through GPU code).  They also help you get an idea about how things work.  For that, compile the code with `-O0 -g`.  Also add `-Xptxas -O0` if you compile with nvcc.

The structure of the baseline implementations (and switching between different algorithms)
--------------------------

As I mentioned above, functions you will primarily be working on are `forward` and `backward` functions on the six classes and update functions for the three classes.  Each of them has a structure to switch between GPU code and CPU code (currently, a single execution can run either entirely on CPU or entirely on GPU; you cannot have some layers executed by CPU and others on GPU in the same execution).  Let's look at the `forward` function of `Convolution2D` class, for example.

The member function named `forward` is the entry point of the forwarding phase.  It only executes a switch statement to decide which implementation to use (cpu or gpu in the baseline code).

```
  tensor<real,maxB,OC,H-K+1,W-K+1>& forward(tensor<real,maxB,IC,H,W>& x, int training) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_base:
      forward_cpu_base(x, training); break;
    case algo_cuda_base:
      forward_cuda_base(x, training); break;
    default:
      if (opt.cuda_algo) {
        forward_cuda_base(x, training);
      } else {
        forward_cpu_base(x, training);
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
    return y;
  }
```

The above code, depending on the algorithm chosen at the command line (-a option), calls either `forward_cpu_base` or `forward_cuda_base`.  The former simply calls another function, `forward_base`, which does the real job.

```
  void forward_cpu_base(tensor<real,maxB,IC,H,W>& x, int training) {
    forward_base(x, training);
  }
```

The latter calls into GPU.  Since nvcc does not allow a class member function to be a global function (a GPU function callable from a host), we need to define a global function outside the class (`forward_cuda_base_global`), which then calls back a member function (`forward_cuda_base_device`).  This is the baseline implementation of `forward_cuda_base`.

```
  void forward_cuda_base(tensor<real,maxB,IC,H,W>& x, int training) {
#if __CUDACC__
    launch_and_sync((forward_cuda_base_global<<<1,1>>>(dev, x.dev, training)));
#else
    (void)x;
    (void)training;
    err_cuda_code_non_cuda_compiler(opt.algo_s);
#endif
  }
```

The global function, `forward_cuda_base_global`, is defined outside the class as follows.  Note that it launches only a single CUDA-thread, something you definitely want to do differently in your high performance version.

```
template<typename T, typename I>
__global__ void forward_cuda_base_global(T* dev, I* x_dev, int training) {
  /* call the member function */
  dev->forward_cuda_base_device(*x_dev, training);
}
```

The member function `forward_cuda_base_device` actually calls the same forward_base function that does the real job.

```
  __device__
  void forward_cuda_base_device(tensor<real,maxB,IC,H,W>& x, int training) {
    forward_base(x, training);
  }
```

This same pattern appears for backward and update too.  In this way, the baseline code shares the same piece of code between CPU and GPU.  The trick makes sense only for the baseline code.  In your high performance implementations, you are probably going to have separate pieces of code for CPU and GPU anyways.

How to add your implementation
--------------------------

Here is how you change the code when working on a new implementation.  As already mentioned, there are two implementations already in place, cpu_base and gpu_base.

Before starting the real work, there are some work for preparation.

 * Come up with a name of the new implementation.  Let's say it is cpu_ultra_fast (in reality, you want to have a name that better represents what it does, like cpu_simd).

 * Add a new symbol to the enum algo_t defined in mnist_util.h
``` 
typedef enum {
  algo_cpu_base,
  algo_gpu_base,
  /* add your new algorithm here (name it arbitrarily) */

  algo_cpu_ultra_fast, <----  YOU ADD THIS

  algo_invalid,
} algo_t;
```

 * Change the parse_algo function right below it so that it recognizes the new name.  Obviously, the baseline code recognizes only "cpu_base" and "gpu_base".  You simply add an appropriate "else if" branch to handle your name.

```
algo_t parse_algo(const char * s) {
  if (strcmp(s, "cpu_base") == 0) {
    return algo_cpu_base;
  } else if (strcmp(s, "gpu_base") == 0) {
    return algo_gpu_base;
  } else if (strcmp(s, "cpu_ultra_fast") == 0) {  <---- YOU ADD THIS
    return algo_cpu_ultra_fast;                   <---- YOU ADD THIS
  } else {
    return algo_invalid;
  }
}
```

 * You might also need to change the function `algo_is_cuda` so that the program correctly recognizes whether it is a CUDA algorithm.  By default, it simply assumes all and only names starting with "cuda" are CUDA algorithms.  You need to change this only when your algorithm name does not conform to this convention (e.g., a CUDA algorithm named "v100_only").  It will be a good idea to stick to the convention rather than modifying this function.

```
static int algo_is_cuda(const char * s, algo_t a) {
  (void)a;
  if (strncmp(s, "cuda", 4) == 0) {
    return 1;
  } else { 
    return 0;
  }
}
```

At this point, the program at least recognizes your algorithm and calls GPU base code or CPU base code depending on your algorithm is a GPU algorithm or not (judged by algo_is_gpu function above).  Recall that the switch statement falls back to forward_cuda_base or forward_cpu_base when the switch statement does not have a specific case for your algorithm.

Now you are ready to add a real implementation.  Thanks to the structure just mentioned, you can do so incrementally (you do not have to implement all functions to get your version used).  To start off, let's say you want to implement a `forward` function of `Convolution2D` class.  The first thing you need to do is to add an appropriate case in the switch statement.

```
  tensor<real,maxB,OC,H-K+1,W-K+1>& forward(tensor<real,maxB,IC,H,W>& x, int training) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_ultra_fast:
      forward_cpu_ultra_fast(x); break;
      /* implementations in the orignal version below */
    case algo_cpu_base:
      forward_cpu_base(x, training); break;
    case algo_cuda_base:
      forward_cuda_base(x, training); break;
    default:
      if (opt.cuda_algo) {
        forward_cuda_base(x, training);
      } else {
        forward_cpu_base(x, training);
      }        
    }
    tsc_t t1 = get_tsc();
    log_end_fun(lgr, t0, t1);
    return y;
  }
```

Your real job is, of course, to implement `forward_cpu_ultra_fast` function.  Use SIMD, OpenMP or whatever is necessary to make it faster.  You probably start by copy-pasting the `forward_base` implementation.

If you work on CUDA implementation, you need to implement two functions.  Let's say your algorithm name is `cuda_ultra_fast`.  After adding another case in the switch statement like this

```
    case algo_cuda_ultra_fast:
      forward_cuda_ultra_fast(x); break;
```

your `forward_cuda_ultra_fast` function will launch a global function with a more sensible value of the thread block size.

```
  void forward_cuda_ultra_fast(array4<maxB,IC,H,W>& x) {
    int block_sz = ...;
    int num_blocks = ...;
    launch_and_sync((forward_cuda_ultra_fast_global<<<num_blocks,block_sz>>>(dev, x.dev)));
  }
```

Next you define `forward_cuda_ultra_fast_global` function outside the class.  You may want to copy the template definition for `forward_cuda_base_global` in `cuda_util.h`.  This will be a boilerplate code.

```
template<typename T, typename I>
__global__ void forward_cuda_ultra_fast_global(T* dev, I* x_dev, int training) {
  /* call the member function */
  dev->forward_cuda_ultra_fast_device(*x_dev, training);
}
```

Finally, you define `forward_ultra_fast_device` member function, which does the real job.

In `forward_cuda_base_device` function that is supposed to do a real job, you compute the output and put them in the 'y' variable, which is already defined for you as a member field.  This convention is used throughout the program.  All classes have a member field named 'y' to which you should put the results.  

```
  __device__ __host__ 
  void forward_base(tensor<real,maxB,IC,H,W>& x, int training) {
    (void)training;
    idx_t B = x.n0;             // batch size
    y.set_n0(B);
    x_ptr = &x;                 // save pointer to input for backward
    for (idx_t s = 0; s < B; s++) {       // for each sample
      for (idx_t oc = 0; oc < OC; oc++) { // for each output channel
        for (idx_t i = 0; i < H - K + 1; i++) {   // for each output pixel
          for (idx_t j = 0; j < W - K + 1; j++) { // for each output pixel
            // calculate a single output pixel
            real v = 0.0;
            for (idx_t ic = 0; ic < IC; ic++) { // input channel
              for (idx_t di = 0; di < K; di++) {
                for (idx_t dj = 0; dj < K; dj++) {
                  v += w(oc,ic,di,dj) * x(s,ic,i+di,j+dj);
                }
              }
            }
            y(s,oc,i,j) = v + b(oc);
          }
        }
      }
    }
  }
```

Similarly, a backward implementation is supposed to put the results into another member variable named 'gx' (∂L/∂x, gradients with respect to x) and 'gw' if the layer has weights.

There is one thing to note here.  The input typically is an array (single- or multi-dimensional) whose primary (leftmost) index refers to a particular sample in a mini-batch.  In the above example, x is a four dimensional array and thus has a type `tensor<maxB,IC,H,W>&`. maxB is a _compile time_ constant you specified by `-DMAX_BATCH_SIZE=xxx` at compile time.  _The actual number of samples in this array may be smaller and is passed via a field variable of the input.  You have to process only the actual number of samples passed in the array.

In this example, x.n0 has the actual number of rows in the array.  Thus,
 * the outermost loop iterates x.n0 number of times rather than maxB times.
```
    idx_t B = x.n0;             // batch size
      ...
    for (idx_t s = 0; s < B; s++) {       // for each sample
```
 * it also sets the actual number of rows in the output y, by doing
```
    y.set_n0(B);
```

Debugging a layer
--------------------------

After you change implementation of a layer you will want to make sure you got it right.  It may not happen immediately, however.  Several mechanisms are in place to help you debug them efficiently.

Catching basic coding errors
--------------------------

First, after you change an implementation of a layer, make sure you turn a compile-time option `-DARRAY_INDEX_CHECK=1` on.  This will check array index every time you access an element of a vector, matrix or tensor.  It will catch obvious errors such as looping with wrong bounds or indexing arrays with wrong variables.

```
$ make
clang++ -O3 -DMAX_BATCH_SIZE=64 -DARRAY_INDEX_CHECK=1 -Dreal_type=float -Wall -Wextra -Wno-strict-overflow   -o exe/mnist_cpu_base mnist.cc
```

Catching logical (mathematics) errors
--------------------------

After you have a code that at least is not caught by array indexing errors, you now want to check if the code really does the job.  The first command line you want to test your code with is this.

```
$ ./exe/mnist_cpu_base --train-data-size 1 --test-data-size 1 -a name_of_your_algorithm
```
Like I introduced already, this processes only a single sample repeatedly.  The error thus should steadily decrease over iterations and the loss should quickly drop to almost zero.  

It should also be noted that your algorithm and the baseline should behave almost identically (e.g., Loss value at each line should be almost identical), as they are processing the same data, initialize weights identically (as they use the same seed for random number generators), and dropping the same elements at dropout layers.

If this is not the case, it is a sign that you introduced a bug.  You need to shoot down where you introduce a bug, for which you will want to debug a single layer at a time.

Each layer is implemented in a single header file.

 * `convolution.h`
 * `linear.h`
 * `relu.h`
 * `dropout.h`
 * `max_pooling.h`
 * `nll_log_softmax.h`

Each header file actually contains an entry point function so that it can compile and run alone.  For example, `convolution.h` has a function `convolution_main` that runs only a convolution.  Therefore, if you include this file from any C++ file and compile it with -Dconvolution_main=main, you get an executable that only runs that layer.

Indeed,

```
$ cd include
$ cp compile.mk Makefile
   ... edit Makefile to include your algorithm in the same as before ...
$ make
```

just does that for all layers.  You will obtain `exe/convolution_cuda_base`, etc.

The entry point function checks if the gradients obtained by forward/backward computation indeed approximate the change of the output value.  Specifically, let's say we have a layer implementing a function F(W, X), where W is the weights and X the input to the layer.  We check if the following approximation holds.

    F(W + ΔW/2, X + ΔX/2) - F(W - ΔW/2, X - ΔX/2) ≒ ∂F/∂W・ΔW + ∂F/∂X・ΔX

(There are many layers that do not have weight parameters.  For such layers, we simply check if F(X + ΔX/2) - F(X - ΔX/2) ≒ ∂F/∂X・ΔX holds).

In implementation terms, we

 * generate inputs (X) and weights (W) randomly
 * generate small changes to inputs (ΔX) and weights (ΔW) randomly
 * perform forward and backward computation to obtain ∂F/∂W and ∂F/∂X and thus to obtain ∂F/∂W・ΔW + ∂F/∂X・ΔX
 * apply changes to X and W to obtain X±ΔX/2 and W±ΔW/2
 * perform forward computation on both of them, to obtain F(W + ΔW/2, X + ΔX/2) and F(W - ΔW/2, X - ΔX/2)
 * compare F(W + ΔW/2, X + ΔX/2) - F(W - ΔW/2, X - ΔX/2) and ∂F/∂W・ΔW + ∂F/∂X・ΔX and report their relative difference.  The relative difference between A and B is |A-B|/max(|A|,|B|)

Here is an output of the linear layer.

```
tau@xps13:include$ ./exe/convolution_cpu_base 
==== 0 ====
==== 0 ====
∂L/∂x・dx = -0.031560765
∂L/∂w・dw = -0.924178811
L- = 24.717748625
L  = 24.239772473
L+ = 23.762027670
A = ∂L/∂x・dx + ∂L/∂w・dw = -0.955739576
B = ΔL = -0.955720955
relative error = |A-B|/max(|A|,|B|) = 0.000019483
==== 1 ====
∂L/∂x・dx = -0.133630750
∂L/∂w・dw = -0.346542146
L- = 307.584572267


    ...


max relative error = 0.000675920
avg relative error = 0.000144470
```

In the end of the execution, it reports that the maximum and average relative errors are 0.000675920 and 0.000144470, respectively.

Note that linear layer implements a linear function, for which an equation

    F(W + ΔW/2, X + ΔX/2) - F(W - ΔW/2, X - ΔX/2) = ∂F/∂W・ΔW + ∂F/∂X・ΔX

should strictly hold if all elementary computations are done without rounding errors.  Any error should be due to rounding errors, which should be small if you are not accumulating too many numbers of numbers of significantly different magnitudes.

If the reported relative error is small enough, it means that moving the weights to the opposite direction of the computed gradient should decrease the loss function, which is the very purpose of the optimization process.  As long as this holds, you do not have be concerned too much about the difference to the baseline code, which has its own rounding errors.  

How small is small enough?  It actually depends on layers and the type of floating point numbers.  Especially when using a single precision floating point numbers (`-Dreal_type=float`), rounding errors easily become significant.  Average relative errors of 10% or even 30% do not necessarily indicate a bug.  Double precision numbers are much less prone to rounding errors.  For the purpose of checking if your code faithfully computes what it should compute, consider testing the double-precision version of it, which should report tiny relative errors.  Here are tables summarizing the maximum/average errors for a single sample.

| layer             | max (SP)    | avg (SP)    | max (DP)    | max (DB)    |
| ------------------|:-----------:|:-----------:|:-----------:|:-----------:|
|`convolution`      | 0.000675920 | 0.000144470 | 0.000000000 | 0.000000000 |
|`linear`           | 0.000052440 | 0.000027702 | 0.000000000 | 0.000000000 |
|`relu`             | 0.007696916 | 0.003426107 | 0.008433753 | 0.003538515 |
|`dropout`          | 0.001975660 | 0.000319423 | 0.000000003 | 0.000000000 |
|`max_pooling`      | 0.223777667 | 0.034783347 | 0.223725769 | 0.035110420 |
|`nll_log_softmax`  | 0.004039020 | 0.000634289 | 0.000000049 | 0.000000014 |



UNDER CONSTRUCTION BELOW. STAY tuned
=============

A record submission tool and record viewers
=============

* Here is a tool to submit a result of executing mnist and a web page to see results submitted by all
* You are required to submit at least the final result you report in your final term paper, but you are encouraged to submit your results whenever you think you made a progress.  Don't wait until you think you are finished
* You can submit your results as many times as you want; you can also (though not encouraged to) delete them if you think there are too many to comfortably see (you can filter out unnecessary records, so you should not have to do this)

Submit your run
-------------

* Running an executable leaves a log file, called mnist.log by default (you can change it with `--log` option)
* Data can be viewed at https://taulec.zapto.org/viewer/
* Do the following on taulec to submit your result 
```
submit < mnist.log
```
* `submit` is a hand-made tool installed (globally) on taulec.  You do not find it in the repository

View your run
-------------

* visit https://taulec.zapto.org/viewer/ to see the results
* it is going to evolve (translation: I am still working on it)

Details you should not (but occasionally might) have to know
-------------

* `submit` is a tailor-made command installed at /usr/local/bin/submit on taulec.zapto.org
* data are stored at /home/share/mnist_records on taulec.zapto.org
* to allow you to write to it, which you normally cannot, `submit` is a setuid program that effectively runs as tau 

Performance criteria and the regulation
==========================


