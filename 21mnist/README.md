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

* `make` in this directory (`21mnist`) builds executables into `exe` directory
* in the default setting, it will build two executables
  * `exe/mnist.0.float.clang++` (CPU version): the executable built by `clang++`; it uses `float` for floating point numbers; it does not perform array bounds checking
  * `exe/mnist.0.float.nvcc` (CUDA version): the executable built by `nvcc`; it uses `float` for floating point numbers; it does not perform array bounds checking

```
$ make -j
mkdir -p exe/dir
clang++ -O0 -g -DMAX_BATCH_SIZE=64  -o exe/mnist.0.float.clang++ -Dreal_type=float -DARRAY_INDEX_CHECK=0 mnist.cc
nvcc -O0 -g -DMAX_BATCH_SIZE=64 --gpu-code sm_80 --gpu-architecture compute_80 -x cu -o exe/mnist.0.float.nvcc -Dreal_type=float -DARRAY_INDEX_CHECK=0 mnist.cc
```
* `make clean` removes `exe` directory

```
$ make clean
rm -rf exe
```

* You can build other executables by changing the `Makefile`

Build Options: 
==================



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
$ ./exe/mnist.0.float.clang++ [options]
```

(or whichever executable name you obtained, if you changed build settings in `Makefile`)

* Run on a compute node, through srun

```
$ srun -p cpu -t 0:20:00 ./exe/mnist.0.float.clang++ [options]
```

GPU:
------------------

* Run on login node.  You cannot run GPU code on the login node, so it's almost pointless

```
$ ./exe/mnist.0.float.nvcc [options]
```

* Run on a compute node having GPUs. Do not forget `--gres gpu:1` option

```
$ srun -p gpu -t 0:20:00 --gres gpu:1 ./exe/mnist.0.float.nvcc [options]
```

Runtime Options
=============

Verbosity (`-v`, `--verbose`)
--------------------------

Give `-v 2` option and you will see the progress more frequently.  You can also know which functions are taking much time.

```
$ ./exe/mnist.0.float.clang++ -v 2
35902: open a log Tue Jan  3 18:39:44 2023
55311: model building starts
29999216: model building ends
30009726: loading data from data
170580095: use 60000 data items out of 60000
172385699: loading data from data
195709866: use 10000 data items out of 10000
195877847: training starts
195924222: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 1, 28, 28, 3, 32>::forward(tensor<real, maxB, IC, H, W> &) [maxB = 64, IC = 1, H = 28, W = 28, K = 3, OC = 32]: starts
199234495: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 1, 28, 28, 3, 32>::forward(tensor<real, maxB, IC, H, W> &) [maxB = 64, IC = 1, H = 28, W = 28, K = 3, OC = 32]: ends. took 3308279 nsec
199242133: tensor<real, N0, N1, N2, N3> &Relu<64, 32, 26, 26>::forward(tensor<real, N0, N1, N2, N3> &) [N0 = 64, N1 = 32, N2 = 26, N3 = 26]: starts
200090476: tensor<real, N0, N1, N2, N3> &Relu<64, 32, 26, 26>::forward(tensor<real, N0, N1, N2, N3> &) [N0 = 64, N1 = 32, N2 = 26, N3 = 26]: ends. took 846270 nsec
```

* `-v 3` shows the result of every element

Execution log (`--log`)
--------------------------

* Detailed execution records are saved into a file (default: mnist.log).  The file includes everything you will see with -v 3 and more.  You can specify the filename with --log option.  When you execute many instances concurrently, make sure you specify a unique log file to each process.

Batch size (`-b`, `--batch-size`)
--------------------------

* For training, it repeats taking a number of samples and updating the model parameters (weights) to the direction that decreases the loss (the difference between the model prediction and the true label).  In each iteration, it takes a number of samples specified by `--batch-size` (`-b`).

```
$ ./exe/mnist.0.float.clang++ -b 1
  ...
```

* This number is called the _mini-batch size_.  The default is `MAX_BATCH_SIZE` specified by a compile-time option `-DMAX_BATCH_SIZE=N`.  For example, `clang++ -DMAX_BATCH_SIZE=64 ... mnist.cc` sets the mini-batch size to 64.  A usual value is 64 but you may consider changing it for performance tuning.

```
$ ... (edit the Makefile at the line `flags += -DMAX_BATCH_SIZE=xxx`) ...
$ make -Bj
mkdir -p exe/dir
clang++ -O3 -DMAX_BATCH_SIZE=128  -o exe/mnist.0.float.clang++ -Dreal_type=float -DARRAY_INDEX_CHECK=0 mnist.cc
nvcc -O3 -DMAX_BATCH_SIZE=128 --gpu-code sm_80 --gpu-architecture compute_80 -x cu -o exe/mnist.0.float.nvcc -Dreal_type=float -DARRAY_INDEX_CHECK=0 mnist.cc
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
$ ./exe/mnist.0.float.clang++ -m 1 --train-data-size 1 --test-data-size 1
59257: model building starts
1067454326: model building ends
1067490294: loading 9000/1000 training/validation data from cifar-10-batches-bin/data_batch_1.bin starts
1507158800: loading data ends
1507168085: training starts
1507170300: === train 0 - 1 ===
2551752614: train loss = 1.770801783
2551766476: training ends
```
dropout (`--dropout-seed1 X` and `--dropout-seed2 X`)
--------------------------

* There are two layers called _dropout_, which randomly turns off (zeros) output of the previous layer (i.e., output_i = 0 with some probability and input_i otherwise) during the training.

* Dropout is ON by default 

* You can turn off the dropout by giving `--dropout-seed 0`

* If the argument to `--dropout-seed` is not zero, it is used to seed a random number generator

* There are two dropout layers in the 

(i.e., output_i = input_i) for the reproducibility of the results.  You may turn it on by giving --dropout 1.

Dropout is generally believed to improve generalization.  The reason that dropout is nevertheless off by default is to make the network behavior more predictable/deterministic and to make the convergence for small training data faster.

Fix a batch (--single_batch 1)
--------------------------

During development, you may want to repeat processing the same mini-batch again and again, to make sure that the network is at least adjusting to the particular mini-batch.  Combine this with --dropout 0 (default) and perhaps with a small batch size, like -b 16 or even -b 1).  In those cases the loss should steadily decrease over iterations.  If it does not happen, suspect your bug, particularly in your backward phase.

```
## pick 16 samples and keep using it in every iteration
$ ./vgg.g++ -b 16 --single_batch 1  
61817: model building starts
1069297283: model building ends
1069327665: loading 9000/1000 training/validation data from cifar-10-batches-bin/data_batch_1.bin starts
1511663981: loading data ends
1511674321: training starts
1511678684: === train 0 - 16 ===
17625595995: train loss = 2.878649473
17625609968: === train 16 - 32 ===
34872185986: train loss = 1.450479388
34872200747: === train 32 - 48 ===
52187145425: train loss = 0.543023527
52187161277: === train 48 - 64 ===
69594766356: train loss = 0.186343119
  ...
```

Data file (-d, --partial_data and --partial_data_seed)
--------------------------

It reads data from the file specified by --cifar_data (-d) option (default: data/cifar-10-batches-bin/data_batch_1.bin).  The original data can be obtained from https://www.cs.toronto.edu/~kriz/cifar.html (get "CIFAR-10 binary version (suitable for C programs)" or https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz).  It contains 5 datasets and each one has 10000 images.

If you want to use only a part of data, you can specify the number of data used by --partial_data option.  --partial_data N randomly chooses N images from the data file.  You can seed the random number generator to choose those images by --partial_data_seed X.  If N is zero, then the whole data set in the file are used.

```
## fix 160 samples and only picking data from there
$ ./vgg.g++ --partial_data 160
```


Data for training and validation (--validate_ratio and --validate_interval)
--------------------------

A portion of data is reserved for validation and not used for training.  The ratio of validation data relative to the data set is specified by --validate_ratio option (default: 0.1).  Note that the number of data you specified with --partial_data N counts both training and validation data.  For example, --partial_data 1000 --validate_ratio 0.1 uses 1000 * 0.1 = 100 images for validation and leaves 900 images for training.

Fractions are rounded down.  If the number of validation data becomes zero as a result, the validation won't be performed at all.

It occasionally evaluates the loss against the validation data.  The frequency can be adjusted by --validate_interval option, which specifies the relative number of samples to process for training and that for validation.  For example, if it is set to 5.0 and the number of data for validation is 100, it runs a validation for every 5.0 * 100 = 500 training samples processed.  In other words, --validate_interval x sets the time to run a validation below 1/(1+x) of the total processing time.

Learning rate
--------------------------

In each iteration, the backward phase calculates the gradient of the averaged loss (the average taken over the mini-batch) with respect to all the weights of the network.  The update phase that follows then changes the weights to the opposite direction of the gradient.  Mathematically, we do

    W = W - (η/B) ∂L/∂W

where L is the summation of the loss function over the mini batch and B the batch size (hence 1/B ∂L/∂W represents the gradient of the average loss with respect to weights W).  We multiply it by a constant η, which is called a learning rate.

The reason why the above update will decrease the loss is as follows.  In general, 

    L(W + ΔW) ≒ L(W) + ∂L/∂W・ΔW

holds, where ・ represents the inner product (summation of all component-wise products).

So if ΔW is taken as an opposite direction of ∂L/∂W (i.e., ΔW = -η∂L/∂W), then

    L(W + ΔW) ≒ L(W) + ∂L/∂W・ΔW
               = L(W) + ∂L/∂W ・(-η∂L/∂W)
               = L(W) -η ∂L/∂W・∂L/∂W
               < L(W)

The value of learning rate can be specified with --learnrate option.  Default: 1.0e-2

The learning rate does not affect the time of each iteration, but may affect the time until convergence.

Since the first-order approximation

    L(W + ΔW) ≒ L(W) + ∂L/∂W・ΔW

holds only for small ΔW, picking a too large η may break the very basic property that an update will decrease the loss.  

On the other hand, the larger the value of η, the "faster" weights move to the opposite direction of the gradient and the faster L(w) decreases.  You may be able to reach the same point with a fewer number of iterations.

In summary, choosing the right value for η is important but not necessarily easy.

Since our primary focus in the exercise is on optimizing each stage of computation, we are not concerned about the choice of η too much.  Specifically, our (primary) performance criteria will be the average time to process a single sample.

Change seeds
--------------------------

There are a few places in which the program uses random numbers, namely,

 * when it initializes weight parameters,
 * when it initially chooses the subset of data from the input file divides them into into validation and training,
 * when it chooses training samples in each iteration and
 * when it uses which cells to dropout in a dropout layer
 
_ALL COMPONENTS BEHAVE DETERMINISTICALLY._  That is, if you repeat executions with the same configuration repeatedly, it starts from the same weight parameters, uses the same data for validation and training, picks up the same training data and drops out the same cells.

Unless your algorithm behave undeterministically, the results should be always the same.  This should help you debug your code.

You can change these things by giving different seeds for each of these random number generators.  Specifically,

 * --weight_seeds changes initial weight parameters
 * --partial_data_seed changes which data in the file are used
 * --sample_seed changes which data are picked for training
 * --dropout_seed changes which cells are dropped out

Simply give an arbitrary number to any of them to make sure your algorithm is not sensitive to any of them.

A general remark: when using a pseudo random number for a randomized algorithm such as stochastic gradient descent, _ALWAYS_ give it a seed you chose and make it behave deterministically given the same seed.  This is a tremendous help for debugging.  After you have done initial debugging, you can test your algorithm across different sequences of random numbers just by giving different seeds.  Note that virtually all pseudo random number generators are deterministic after a seed is given.  A random number generator without any seed generates different sequences every time simply because they use different seeds every time; when you do not give it a seed, it simply takes a value from an external source (e.g., the current time) and uses it as a seed.  Nothing else is different.  In this sense, there is almost no point in not giving a seed of your choice (give the current time as a seed if you want to purposefully make it behave differently each time).  Without giving a seed, your algorithm can NEVER behave deterministically, which is a nightmare for debugging and the nightmare is nothing but unnecessary.

GPU execution (-a gpu_base)
--------------------------

The program compiled with nvcc, vgg.nvcc, supports GPU execution and this is the default behavior. vgg.nvcc also supports CPU execution with -a cpu_base option.

```
$ vgg.nvcc -a cpu_base  # (1) baseline code on CPU 
$ vgg.nvcc -a gpu_base  # (2) baseline code on GPU 
$ vgg.nvcc              # (3) same as (2)
$ vgg.g++  -a cpu_base  # (4) baseline code on CPU 
$ vgg.g++  -a gpu_base  # (5) error
$ vgg.g++               # (6) same as (4)
```

Note that baseline code is neither vectorized nor parallelized.  In particular, it uses only a single CUDA thread on GPU (!)

Algorithm choice (-a)
--------------------------

The -a option described above is an option that chooses an algorithm from available repertories.  In the given code, only baseline algorithms for GPU and CPU are implemented.  You will add your implementation as another available choice here.


Controlled experiments
--------------------------

After you did a bit of work, you want to make sure you got it done right.  Especially, you may be afraid that you broke a function.  To make sure the network is still functioning, you might want to do a small and controlled experiment.

You probably want to start with something like this.

```
$ ./vgg.gcc --dropout 0 --single_batch 1 -b 1
```

(Note: --dropout 0 is the default, so you actually do not have to give it explicitly)

They together (1) turn off dropout to avoid fluctuating losses across iterations due to the changing network (--dropout 0), (2) process the same mini-batch at every iteration to avoid fluctuating losses due to different data in different iterations, and (3) make the mini-batch size extremely small (1, in this particular case) to have a quick turn around time.

Here is a sample output.  With the default value of learning rate (1.0e-2), the loss seems decreasing quickly (perhaps so quickly that the remaining iterations are useless).

```
$ ./vgg.gcc --dropout 0 --single_batch 1 -b 1
=== train 0 - 1 ===
train loss = 2.064959049
=== train 1 - 2 ===
train loss = 0.002221737
=== train 2 - 3 ===
train loss = 0.002113967
=== train 3 - 4 ===
train loss = 0.002022009
=== train 4 - 5 ===
train loss = 0.001938014
=== train 5 - 6 ===
train loss = 0.001864006
=== train 6 - 7 ===
train loss = 0.001799989
=== train 7 - 8 ===
train loss = 0.001746916
  ...
```

Remember that this will repeat processing only a single sample and the "train loss" refers to the loss against this particular sample.  If the loss does not decrease, you are very likely to have introduced a bug in your gradient calculation (backward) or somewhere else.

You may play with small learning rates to see that it will keep decreasing for a larger number of iterations.

```
$ ./vgg.gcc --dropout 0 --single_batch 1 -b 1 --learnrate 1.0e-3
=== train 0 - 1 ===
train loss = 2.064959049
=== train 1 - 2 ===
train loss = 0.937229633
=== train 2 - 3 ===
train loss = 0.503207982
=== train 3 - 4 ===
train loss = 0.290159106
=== train 4 - 5 ===
train loss = 0.209925532
=== train 5 - 6 ===
train loss = 0.163231462
  ...
```

How you interpret the loss?
==========================

If you are curious what the value of the loss actually means, here it is.  In the final stage of the network, it ends up with a vector of 10 components (the number of classes) for each image, each component of which represents the probability that the model thinks the image belongs to a particular class.  This 10 elements vector, say p, is then compared with the true label (class) for it.  If the true class of that image is c, then the loss for this particular image is 

   -log(p[c])

where p[c] is the probability that the model says the image belongs to the true class c.

Therefore, if the network is a random guess that returns 1/10 for every class, the average loss for an image is

   -log(1/10) = 2.3025850...

which is just about what you observe in the first iteration.

The loss becomes zero if the network says the probability is 1 for the correct class and 0 for all other classes.  But as far as the classification performance  is concerned, the network outputs a correct class as long as the probability for the true class is larger than for any other class.  A sufficient condition for this is p[c] > 0.5, as it guarantees that p[c] is maximum among p[0], ..., p[9], whose sum is one.  When p[c] = 0.5, the loss would be 

   -log(1/2) = 0.69314...

Navigating the source code
==========================

1. open HTML/index.html in your browser
1. compile it with -O0 -g (edit Makefile) run it under the debugger

<font color="red">(Deprecated Jan 3, 2021)</font> Performance profiling
==========================

1. after you run, you will get vgg.log that has a detailed execution log
1. do the following
``
$ ./vgg.g++
$ ls
  ... vgg.log ...
$ cd records
$ ./parse_log ../vgg.log
$ open index.html with your browser
``

<font color="red">This tool is superseded by another tool with which you can compare multiple runs by you and friends.</font>  See the next section for the new tool.

This tool is now meaningful only to have fun by seeing how correctly/incorrectly labeled pictures change over time. 

<font color="red">(Added on Jan 3 2021)</font> A record submission tool and record viewers
=============

* Here is a tool to submit a result of executing vgg and a web page to see results submitted by all
* You are required to submit at least the final result you report in your final term paper, but you are encouraged to submit your results whenever you think you made a progress.  Don't wait until you think you are finished
* You can submit your results as many times as you want; you can also (though not encouraged to) delete them if you think there are too many to comfortably see (you can filter out unnecessary records, so you should not have to do this)

Submit your run
-------------

* Running a vgg executable leaves a log file, called vgg.log by default (you can change it with --log option)
* Data is stored and viewed on taulec, as IST cluster cannot host a user's web page
* Assume you ran vgg on IST cluster.  
* Do the following to submit your result from IST cluster.
```
ssh YOUR-USER-ID-ON-TAULEC@taulec.zapto.org submit < vgg.log
```
* `submit` is a hand-made tool installed (globally) on taulec.  You do not find it in the repository

* Note that in order to run the above command, you should be able to ssh from IST cluster to taulec.  If you are not, see <a href="https://www.eidos.ic.i.u-tokyo.ac.jp/~tau/lecture/parallel_distributed/html/ist_cluster.html#ist-to-taulec" target="_blank">this page</a>.

View your run
-------------

* visit https://taulec.zapto.org/viewer/ to see the results
* it is going to evolve (translation: I am still working on it)

Details you should not (but occasionally might) have to know
-------------

* `submit` is a tailor-made command installed at /usr/local/bin/submit on taulec.zapto.org
* data are stored at /home/tau/public_html/lecture/parallel_distributed/parallel-distributed-handson/20vgg/records/vgg_records/ on taulec.zapto.org
* to allow you to write to it, which you normally cannot, `submit` is a setuid program that effectively runs as tau 

(Updated on Jan 3 2021) Performance criteria and the regulation
==========================

The ultimate goal for training is to get a good classification performance (accuracy) as fast as possible.  So, ideally, our performance criteria should be the time until you meet a certain accuracy (or conversely, the achieved accuracy in a given amount of time).  It is a bit challenging to define a good regulation along this line, because meeting any meaningful accuracy would take a large amount of time (at least until you get a very high performance code) that makes your experiments very time consuming.  Accuracy also depends on many heuristics such as learning rate or how you update weights, which are orthogonal to the main objective of our study (making a given computation fast).  To simplify your goal and make the amount of time for experiments shorter, we set our performance criteria and the regulation as follows.

 * The performance criteria is the number of trained samples per second.  For example, if you train the network with 256 images in 50 seconds, your score will be 256/50 = 5.12 samples/sec.  Your goal is simply to maximize this number.
 * You can get a relevant number easily in the above viewer page https://taulec.zapto.org/viewer/
   * in "Loss/accuracy evolution with samples/time" by setting x-axis to t and y-axis samples; the samples per second is simply the gradient of the curve (which should be almost straight)
   * in "Execution time breakdown", the total height of the stack is the execution time (in ns) per sample.  Samples per second is simply the reciprocal of the height, multiplied by $10^9$. The shorter the bar is, the higher the throughput of your program is.
 * These data are taken from the execution log, so you may want to look at them yourself.
 * For example, the following log

```
815859250: === train 0 - 64 ===    (the first line of this form)



  ...



1186702243923: === train 1216 - 1280 ===  (the last line of this form)

  ...

1245300919526: train loss = 2.518893003   (the last line of this form)
```

indicates the training started at time 815859250 and ended at time 1245300919526 (times are shown in nano seconds) and it processed 1280 images.  The score is then 1280 / (1245.300919526 - 0.815859250) = 1.02853 images/sec.
 * In order to get a good throughput with vectorization and/or parallelization, you almost certainly want to process many images at a time (a fairly large batch size).  Try to play with it.
 * While the number of images per second is the goal, following constraints are imposed to preclude meaningless optimizations
  - You can use only a subset of the dataset (otherwise the training may take too much time until you get a noticeable improvement on the loss), but your data must have at least 64 images for training
  - When you report the result, your network must achieve the average a loss <= 1.0 for at least the images used for training.  The easiest setting is to use only 64 images for training, like this.
```
./vgg.g++ --partial_data 64 --validate_ratio 0
```
  - The baseline code should achieve a training loss = 0.851912320 after 4 iterations (256 images have been processed) and took about 4 minutes on my laptop.  Your code should achieve a similar loss in a similar setting unless you somehow break it.  It should be much faster, of course.

Remarks: The setting (of achieving loss = 0.85.. for the same 64 images used for training) is not very meaningful as a machine learning task.  First, it achieves good classification performance only for data used for training and not for validation data.  If you check the network against a validation data, the loss is still very large, so clearly the mission has not been accomplished.  Having a large gap between training data and validation data means the network may be over-fitting.  Second, the dataset is obviously too small to be meaningful.  We nevertheless allow such settings to make a single experiment finish quickly and make it still possible to observe that the loss is decreasing (so you got the job done correctly).  Having a more stringent condition, e.g., achieve a similar loss for validation data, would make the training time much longer.

* With all that said, you are encouraged to run the program in more practically useful settings (e.g., use the full data set, leaving a tenth of it for validation and run long enough to achieve a good loss/accuracy) once you have a reasonably fast code.

Guide for development
==========================

Source code structure
--------------------------

 * vgg.cc -- the main file
 * include/

  (nuts and bolts)

  - vgg_util.h  -- trivial utilities
  - cuda_util.h -- helpers for CUDA
  - vgg_arrays.h -- vectors, matrix and multidimensional tensors
  - cifar.h -- data loader

  (primitive layers)

  - convolution.h -- convolution
  - batchnormalization.h -- batch normalization
  - relu.h -- rectified linear activation
  - dropout.h -- dropout
  - linear.h -- linear (or fully connected) layer
  - maxpooling.h -- max pooling
  - softmaxcrossentropy.h -- softmax + cross entropy

  (composite layers)

  - block.h -- convolution; batch normalization; relu
  - vgg.h -- the entire VGG

The main function in vgg.cc instantiates a VGG network, which is defined in vgg.h.
It repeats processing training data, occasionally processing validation data.

Each layer defines a class whose name is similar to the file name.  e.g., convolution.h defines Convolution2D class.

All classes for primitive and composite layers have two important functions, among others.
 * forward -- take an input from the previous (downstream) layer and computes its output
 * backward -- take a gradient of loss wrt the upstream layer and computes the gradient wrt its input and weights

In addition, classes that have parameters (convolution, linear and batchnormalization) have another function.

 * update -- take a learning rate parameter and update its weights, using the gradient computed in the backward phase.

**********************************************************************************
YOUR MAIN JOB WILL BE TO IMPLEMENT A HIGH PERFORMANCE VERSION OF THESE FUNCTIONS.
**********************************************************************************

You eventually want to work on all seven files (convolution.h, batchnormalization.h, relu.h, dropout.h, linear.h, maxpooling.h, softmaxcrossentropy.h) but you can work incrementally.  You can make one layer faster while leaving all others intact.  You can know which functions are taking much time by -v 2 option.

Stepping through the code using gdb (or cuda-gdb)
--------------------------

When working on details, you want to step through the code using gdb (or cuda-gdb to step through GPU code).  They also help you get an idea about how things work.  For that, compile the code with -O0 -g.  Also add -Xptxas -O0 if you compile with nvcc.

The structure of the baseline implementations (and switching between different algorithms)
--------------------------

As I mentioned above, functions you will primarily be working on are forward and backward functions on the seven classes and update functions for the three classes.  Each of them has a structure to switch between GPU code and CPU code (currently, a single execution can run either entirely on CPU or entirely on GPU; you cannot have some layers executed by CPU and others on GPU in the same execution).  Let's look at the forward function of Convolution class, for example.

The member function named "forward" is the entry point of the forwarding phase.  It only executes a switch statement to decide which implementation to use (cpu or gpu in the baseline code).

```
  array4<maxB,OC,H,W>& forward(array4<maxB,IC,H,W>& x) {
    if (opt.verbose>=2) { print_start(); }
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
    if (opt.verbose>=2) { print_end(t0, t1); }
    return y;
  }
```

Depending on the algorithm chosen at the command line (-a option), it calls either forward_cpu or forward_gpu.  The former simply calls another function, forward_base, which does the real job.

```
  void forward_cpu(array4<maxB,IC,H,W>& x) {
    forward_base(x);
  }
```

The latter calls into a GPU code.  Since nvcc does not allow a class member function to be a global function (a GPU function callable from a host), we need to define a global function outside the class (forward_global), which then calls back a member function (forward_dev).  This is the baseline implementation of forward_gpu.

```
  void forward_gpu(array4<maxB,IC,H,W>& x) {
    launch_and_sync((forward_global<<<1,1>>>(dev, x.dev)));
  }
```

The global function, forward_global, is defined outside the class as follows.  Note that it launches only a single CUDA-thread, something you definitely want to do differently in your high performance version.

```
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
__global__ void forward_global(Convolution2D<maxB,IC,H,W,K,OC>* dev,
                               array4<maxB,IC,H,W>* x_dev) {
  dev->forward_dev(*x_dev);
}
```

The member function forward_dev actually calls the same forward_base function that does the real job.

```
  __device__ __host__ 
  void forward_base(array4<maxB,IC,H,W>& x) {
    ... do the real job ...
  }
```

This same pattern appears for backward and update too.  In this way, the baseline code shares the same piece of code between CPU and GPU.  The trick makes sense only for the baseline code.  In your high performance implementations, you are probably going to have separate pieces of code for CPU and GPU anyways.

How to add your implementation
--------------------------

Here is how you change the code when working on a new implementation.  As already mentioned, there are two implementations already in place, cpu_base and gpu_base.

Before starting the real work, there are some work for preparation.

 * Come up with a name of the new implementation.  Let's say it is cpu_ultra_fast (in reality, you want to have a name that better represents what it does, like cpu_simd).

 * Add a new symbol to the enum algo_t defined in vgg_util.h
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

 * You might also need to change the function algo_is_gpu so that the program correctly recognizes whether it is a CPU algorithm or a GPU algorithm.  By default, it simply assumes all and only names starting with "gpu" are GPU algorithms.  You need to change this only when your algorithm name does not conform to this convention (e.g., a GPU algorithm named "v100_only").  It will be a good idea to stick to the convention rather than modifying this function.

```
int algo_is_gpu(const char * s, algo_t a) {
  (void)a;
  if (strncmp(s, "gpu", 3) == 0) {
    return 1;
  } else { 
    return 0;
  }
}
```

At this point, the program at least recognizes your algorithm and calls GPU base code or CPU base code depending on your algorithm is a GPU algorithm or not (judged by algo_is_gpu function above).  Recall that the switch statement falls back to forward_gpu or forward_cpu when the switch statement does not have a specific case for your algorithm.

Now you are ready to add a real implementation.  Thanks to the structure just mentioned, you can do so incrementally (you do not have to implement all functions to get your version used).  To start off, let's say you want to implement a forward function of Convolution2D class.  The first thing you need to do is to add an appropriate case in the switch statement.

```
  array4<maxB,OC,H,W>& forward(array4<maxB,IC,H,W>& x) {
    if (opt.verbose>=2) { print_start(); }
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_ultra_fast:
      forward_cpu_ultra_fast(x); break;

      ...
    }
    tsc_t t1 = get_tsc();
    if (opt.verbose>=2) { print_end(t0, t1); }
    return y;
  }
```

Your real job is, of course, to implement forward_cpu_ultra_fast function.  Use SIMD, OpenMP or whatever is necessary to make it faster.  You probably start by copy-pasting the forward_base implementation.

If you work on GPU implementation, you need to implement two functions.  Let's say your algorithm name is gpu_ultra_fast.  After adding another case in the switch statement like this

```
    case algo_gpu_ultra_fast:
      forward_gpu_ultra_fast(x); break;
```

your forward_gpu_ultra_fast function will launch a global function with a more sensible value of the thread block size.

```
  void forward_gpu_ultra_fast(array4<maxB,IC,H,W>& x) {
    int block_sz = ...;
    int num_blocks = ...;
    launch_and_sync((forward_ultra_fast_global<<<num_blocks,block_sz>>>(dev, x.dev)));
  }
```

Next you define forward_ultra_fast_global function outside the class, near the beginning of the file.  This will be a boilerplate code.

```
template<idx_t maxB,idx_t IC,idx_t H,idx_t W,idx_t K,idx_t OC>
__global__ void forward_ultra_fast_global(Convolution2D<maxB,IC,H,W,K,OC>* dev,
                                          array4<maxB,IC,H,W>* x_dev) {
  dev->forward_ultra_fast_dev(*x_dev);
}
```

Finally, you define forward_ultra_fast_dev member function, which does the real job.

In forward_base function that is supposed to do a real job, you compute the output and put them in the 'y' variable, which is already defined for you as a member field.  This convention is used throughout the program.  All classes have a member field named 'y' to which you should put the results.  

```
  __device__ __host__ 
  void forward_base(array4<maxB,IC,H,W>& x) {
    idx_t B = x.B;
    y.set_n_rows(B);
    ...
    for (idx_t b = 0; b < B; b++) {       // samples
      ...
         ...
            y(b,oc,i,j) = s;
         ...
      ...
    }
  }
```

Similarly, a backward implementation is supposed to put the results into another member variable named 'gx' (∂L/∂x, gradients with respect to x) and 'gw' if the layer has weights.

There is one thing to note here.  The input typically is an array (single- or multi-dimensional) whose primary (leftmost) index refers to a particular sample in a mini-batch.  In the above example, x is a four dimensional array and thus has a type array4<maxB,IC,H,W>&. maxB is a _compile time_ constant you specified by -DMAX_BATCH_SIZE=xxx.  _The actual number of samples in this array may be smaller and is passed via a field variable of the input._  You have to process only the actual number of samples passed in the array.

In this example, x.B has the actual number of rows in the array.  Thus,
 * the outermost loop iterates x.B number of times rather than maxB times.
```
    idx_t B = x.B;
     ...
    for (idx_t b = 0; b < B; b++) {       // samples
```
 * it also sets the actual number of rows in the output y, by doing
```
    idx_t B = x.B;
    y.set_n_rows(B);
```

Debugging a layer
--------------------------

After you change implementation of a layer you will want to make sure you got it right.  It may not happen immediately, however.  Several mechanisms are in place to help you debug them efficiently.

Catching basic coding errors
--------------------------

First, after you change an implementation of a layer, make sure you turn a compile-time option -DARRAY_INDEX_CHECK=1 on.  This will check array index every time you access an element of a vector, matrix or tensor.  It will catch obvious errors such as looping with wrong bounds or indexing arrays with wrong variables.

```
$ g++ ... -DARRAY_INDEX_CHECK=1 ... -o vgg.g++ vgg.cc
```

Catching logical (mathematics) errors
--------------------------

After you have a code that at least is not caught by array indexing errors, you now want to check if the code really does the job.  The first command line you want to test your code with is this.

```
$ ./vgg.gcc --dropout 0 --single_batch 1 -b 1 -a name_of_your_algorithm
```

Like I introduced already, this processes only a single sample repeatedly, without any dropout that would introduce different behaviors between iterations.  The error thus should steadily decrease over iterations, in almost exactly the same pace with the baseline implementation.  Try different learning rate (--learnrate 1.0e-3, 1.0e-4, etc.) and confirm they behave very similarly for each one.

Remember that both are doing exactly the same computation.  There is a randomness in how it chooses samples, but _it is deterministic_ as I already mentioned; the sample picked should be the same unless you change the seed of the random number generator (by --sample_seed).  Were there any difference between the two implementations, it should indicate that your algorithm outputs results different from the other implementation for the same input.  If the difference is slight, it may be indicating that the two has different rounding errors for computing mathematically equivalent expressions.  In particular, summing up many numbers in a different order affect rounding errors very much.  When you parallelize or vectorize your code, you almost certainly change the order in which numbers are accumulated.  Therefore, a slight difference in the loss may not be worth looking into.

If you observe a significant change, you need to shoot down where you introduce a bug, for which you will want to debug a single layer at a time.

Each layer is implemented in a single header file.

 * batchnormalization.h
 * convolution.h
 * dropout.h
 * linear.h
 * maxpooling.h
 * relu.h
 * softmaxcrossentropy.h

Each header file actually contains an entry point function so that it can compile and run alone.  For example, convolution.h has a function convolution_main that runs only a convolution.  Therefore, if you include this file from any C++ file and compile it with -Dconvolution_main=main, you get an executable that only runs that layer.

Indeed,

```
$ cd include
$ make
```

builds all such executables.  You will obtain batchnormalization.{float,double}.{g++,nvcc}, etc.

The entry point function checks if the gradients obtained by forward/backward computation indeed approximate the change of the output value.  Specifically, let's say we have a layer implementing a function F(W, X).  We check if the following approximation holds.

    F(W + ΔW/2, X + ΔX/2) - F(W - ΔW/2, X - ΔX/2) ≒ ∂F/∂W・ΔW + ∂F/∂X・ΔX

(There are many layers that do not have weight parameters.  For such layers, we simply check if F(X + ΔX/2) - F(X - ΔX/2) ≒ ∂F/∂X・ΔX holds).

In implementation terms, we

 * generate inputs (X) and weights (W) randomly
 * generate small changes to inputs (ΔX) and weights (ΔW) randomly
 * perform forward and backward computation to obtain ∂F/∂W and ∂F/∂X and thus to obtain ∂F/∂W・ΔW + ∂F/∂X・ΔX
 * apply changes to X and W to obtain X±ΔX and W±ΔW
 * perform forward computation on both of them, to obtain F(W + ΔW/2, X + ΔX/2) and F(W - ΔW/2, X - ΔX/2)
 * compare F(W + ΔW/2, X + ΔX/2) and F(W - ΔW/2, X - ΔX/2) and ∂F/∂W・ΔW + ∂F/∂X・ΔX and report their relative difference.  The relative difference between A and B is |A-B|/max(|A|,|B|)

Here is an output of the linear layer.

```
$ ./linear.float.g++ -b 1
==== 0 ====
|∂L/∂x|   = 2.067968130
|dx|      = 0.001285600
∂L/∂x・dx = -0.000187458
|∂L/∂w|   = 27.588932037
|dw|      = 0.004125366
∂L/∂w・dw = 0.000541298
L- = -0.982513964
L  = -0.982336760
L+ = -0.982159197
A = ∂L/∂x・dx + ∂L/∂w・dw = 0.000353840
B = ΔL = 0.000354767
relative error = |A-B|/max(|A|,|B|) = 0.002611878
==== 1 ====
|∂L/∂x|   = 2.037896156
|dx|      = 0.001299710
    ...


max relative error = 0.009731923
avg relative error = 0.001512904
```

In the end of the execution, it reports that the maximum and average relative errors are 0.009731923 and 0.001512904, respectively.

Note that linear layer implements a linear function, for which an equation

    F(W + ΔW/2, X + ΔX/2) - F(W - ΔW/2, X - ΔX/2) = ∂F/∂W・ΔW + ∂F/∂X・ΔX

should strictly hold if all elementary computations are done without rounding errors.  Any error should be due to rounding errors, which should be small if you are not accumulating too many numbers of numbers of significantly different magnitudes.

If the reported relative error is small enough, it means that moving the weights to the opposite direction of the computed gradient should decrease the loss function, which is the very purpose of the optimization process.  As long as this holds, you do not have be concerned too much about the difference to the baseline code, which has its own rounding errors.  

How small is small enough?  It actually depends on layers and the type of floating point numbers.  Especially when using a single precision floating point numbers (executables *.float.*} do so), rounding errors easily become significant.  Average relative errors of 10% or even 30% do not necessarily indicate a bug.  Double precision numbers are much less prone to rounding errors.  For the purpose of checking if your code faithfully computes what it should compute, consider testing the double-precision version of it (*.double.*), which should report tiny relative errors.  Here are tables summarizing the maximum/average errors for a single sample.

| layer             | max (SP)    | avg (SP)    | max (DP)    | max (DB)    |
| ------------------|:-----------:|:-----------:|:-----------:|:-----------:|
|batchnormalization | 0.321599394 | 0.047267061 | 0.000000012 | 0.000000001 |
|convolution        | 0.005766520 | 0.001314429 | 0.000000000 | 0.000000000 |
|dropout            | 0.927670479 | 0.083410397 | 0.000000003 | 0.000000000 |
|linear             | 0.009731923 | 0.001512904 | 0.000000000 | 0.000000000 |
|maxpooling         | 1.918741345 | 0.126148313 | 1.013361927 | 0.052107410 |
|relu               | 0.170680821 | 0.019997066 | 0.136701009 | 0.014832921 |
|softmaxcrossentropy| 0.034060795 | 0.007438810 | 0.000000000 | 0.000000000 |



## code

```
make 
```

## data

```
git clone git@github.com:pytorch/examples.git
cd examples/mnist
python3 mnist.py
```

will download data into examples/data/MNIST/raw

make a symlink to raw directory as data

```
cd ../..
ln -s examples/data/MNIST/raw data
```

# Pascal Vincent format

data[3] D (dimension)
data[2] T (element type) {8: torch.uint8, 9: torch.int8, 11: torch.int16, 12: torch.int32, 13: torch.float32, 14: torch.float64}

next D ints (4 bytes x D)
  size of each dimension
  
divide each byte by 255
subtract mean 0.3..
divide by std 0.4..
  
convert cifar10 data to Pascal Vincent format and MNIST can be used for Cifar-10 as well

/usr/local/lib/python3.10/dist-packages/torch/optim/
_single_tensor_adadelta
(Pdb) p lr
1.0
(Pdb) p rho
0.9
(Pdb) p eps
1e-06
(Pdb) p weight_decay
0
(Pdb) p maximize
False

square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)

square_avg -> vt のこと

