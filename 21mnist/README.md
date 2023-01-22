Overview
=============

* This program trains a neural network for hand-written character recognition
* I provide a very basic implementation without vectorization or parallelization
  * CPU and CUDA version are provided
  * CPU version does not use SIMD or threads
  * CUDA version uses only a single(!) CUDA thread

The Neural Network Model
==================

* The model is a C++ translation of `mnist` model in pytorch examples https://github.com/pytorch/examples/tree/main/mnist
* The original pytorch model is written in python but the baseline code provided herein is purely C++ that does not rely on any external library
  * mnist is actually the name of the database, not a neural network model, so the name `mnist` is a bit of misnomer, but this is what pytorch examples name it and we will use the same name
* If you want, you may run the original pytorch implementation as follows, just to get a sense of how the learning should progress
* My C++ translation tries to faithfully replicate what the pytorch implementation does, so you can compare your implementation with the pytorch implementation
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

Download:
==================

* Download my baseline C++ implementation from the github

* If you have registered your SSH key to the github, 

```
git clone git@github.com:taura/parallel-distributed.git
```

* Or,

```
git clone https://github.com/taura/parallel-distributed.git
```

**Remarks:** _Please anticipate this repository (especially, this `README.md` file) will get updated later to give you more info and instructions later.  Source code changes will be minor (hopefully none), but please still prepare for incorporating future changes by `git pull`.  Git hopefully merges my changes and yours, but I recommend you to make your changes easy to identify (e.g., by enclosing them by `#ifdef ... #endif`) in case you have to merge changes manually.  I will provide more instruction about how you are supposed to modify the code and submit it later._


Dataset: MNIST
==================

* The dataset is in `data/` directory
* Make sure you have the following files
  * `data/train-images-idx3-ubyte` : training images
  * `data/train-labels-idx1-ubyte` : training labels
  * `data/t10k-images-idx3-ubyte` : test images
  * `data/t10k-labels-idx1-ubyte` : test labels
* If you executed the pytorch implementation, the same data should have been downloaded into `examples/data/MNIST/raw/`; the contents of the `raw/` directory and `data/` directory should be identical
* If you are curious what kind of images they are, you can take a look at [the first 500 images](http://taulec.zapto.org/~share/parallel-distributed/21mnist/imgs/train-pgms/00000_00500/) in the training dataset

Compile: 
==================

* `make` in this folder (`21mnist`) builds executable(s) into `exe` directory
* in the default setting, it will build two executables
  * `exe/mnist_cpu_base`, made by `clang++` 
  * `exe/mnist_cuda_base`, made by `nvcc`

```
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

_Details about srun command will be given later (please ignore it for a moment)_

GPU:
------------------

* Run on tauleg000

```
$ ./exe/mnist_cuda_base [options]
```

* Run on a compute node having GPUs. Do not forget `--gres gpu:1` option

```
$ srun -p gpu -t 0:20:00 --gres gpu:1 ./exe/mnist_cuda_base [options]
```

_Details about srun command will be given later (please ignore it for a moment)_


Default Behavior:
------------------

* The default behavior below is the same as the original pytorch implementation
* It reads data from files in the `./data` directory in the current working directory (= the directory you executed the command from) 
* It runs 14 epochs (= scans the entire training dataset (60000 samples) 14 times = uses 60000 x 14 = 840000 samples in total)
* The mini-batch size (the number of samples used at a time to update weights) is 64
* Every 10 mini-batches (= 640 samples), it reports the loss of a mini-batch
* Between epochs, it scans the entire test dataset (10000 samples) once and reports the accuracy


Runtime Options
=============

Help (`-h`,`--help` or any invalid option, for that matter)
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

* Give `-v 2` option shows various settings at command line and environment variables

```
$ ./exe/mnist_cpu_base -v 2
22990: open a log Thu Jan  5 14:59:00 2023
31146: verbose=2
32073: data-dir=data
32900: lr=1.000000
35868: epochs=14
36567: batch-size=64
37239: train-data-size=-1
37975: test-data-size=4294967295
38717: weight-seed=45678901234523
39419: dropout-seed-1=56789012345234
40137: dropout-seed-2=67890123452345
40753: grad-dbg=0
41437: algo=0
42067: log=mnist.log
59566: host=fv
60731: USER=tau
61502: PWD=/home/tau/public_html/lecture/parallel_distributed/parallel-distributed/21mnist
62530: SLURM_SUBMIT_DIR undefined
63272: SLURM_SUBMIT_HOST undefined
   ...
91256: SLURM_JOB_GID undefined
91946: SLURMD_NODENAME undefined
93087: model building starts
27729075: model building ends
27746412: loading data from data
172280759: use 60000 data items out of 60000
173330293: loading data from data
197724941: use 10000 data items out of 10000
197876526: training starts
2320860096: Train Epoch: 1 [0/60000 (0%)]	Loss: 2.316252
   ...
```

* Give `-v 3` option shows the result (predicted and true labels) of all samples

```
$ ./exe/mnist_cpu_base -v 3
28172: open a log Thu Jan  5 15:01:03 2023
36131: verbose=3
37242: data-dir=data
37990: lr=1.000000
41821: epochs=14
42458: batch-size=64
43099: train-data-size=-1
43813: test-data-size=4294967295
44533: weight-seed=45678901234523
45226: dropout-seed-1=56789012345234
45874: dropout-seed-2=67890123452345
46571: grad-dbg=0
47277: algo=0
47895: log=mnist.log
62483: host=fv
63864: USER=tau
64635: PWD=/home/tau/public_html/lecture/parallel_distributed/parallel-distributed/21mnist
65792: SLURM_SUBMIT_DIR undefined
66510: SLURM_SUBMIT_HOST undefined

    ...
    
80714: SLURM_JOB_GID undefined
81319: SLURMD_NODENAME undefined
81926: model building starts
30049565: model building ends
30068902: loading data from data
182503763: use 60000 data items out of 60000
183485130: loading data from data
209477720: use 10000 data items out of 10000
209628269: training starts
2517480573: sample 0 image 0 pred 6 truth 5
2517491709: sample 1 image 1 pred 5 truth 0
2517493022: sample 2 image 2 pred 6 truth 4
2517494213: sample 3 image 3 pred 5 truth 1

    ...
    
2517580873: sample 61 image 61 pred 6 truth 4
2517582222: sample 62 image 62 pred 4 truth 6
2517583434: sample 63 image 63 pred 4 truth 0
2517584718: Train Epoch: 1 [0/60000 (0%)]	Loss: 2.316252
4837309739: sample 64 image 64 pred 4 truth 4
4837324018: sample 65 image 65 pred 4 truth 5
4837325090: sample 66 image 66 pred 1 truth 6

    ...
```

* `-v 4` shows all layers called and their elapsed time

```
$ ./exe/mnist_cpu_base -v 4
26295: open a log Thu Jan  5 15:02:43 2023
34947: verbose=4
35908: data-dir=data
36734: lr=1.000000
39803: epochs=14
40504: batch-size=64
41237: train-data-size=-1
41965: test-data-size=4294967295
42730: weight-seed=45678901234523
43508: dropout-seed-1=56789012345234
45434: dropout-seed-2=67890123452345
46206: grad-dbg=0
46893: algo=0
47542: log=mnist.log
49705: host=fv
51243: USER=tau
53524: PWD=/home/tau/public_html/lecture/parallel_distributed/parallel-distributed/21mnist
60221: SLURM_SUBMIT_DIR undefined
61242: SLURM_SUBMIT_HOST undefined

   ...

82311: SLURM_JOB_GID undefined
83001: SLURMD_NODENAME undefined
84134: model building starts
28704401: model building ends
28714696: loading data from data
177709188: use 60000 data items out of 60000
178715128: loading data from data
203401720: use 10000 data items out of 10000
203663521: training starts
203697529: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 1, 28, 28, 3, 32>::forward(tensor<real, maxB, IC, H, W> &, int) [maxB = 64, IC = 1, H = 28, W = 28, K = 3, OC = 32]: starts
208063761: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 1, 28, 28, 3, 32>::forward(tensor<real, maxB, IC, H, W> &, int) [maxB = 64, IC = 1, H = 28, W = 28, K = 3, OC = 32]: ends. took 4363831 nsec
208080391: tensor<real, N0, N1, N2, N3> &Relu<64, 32, 26, 26>::forward(tensor<real, N0, N1, N2, N3> &, int) [N0 = 64, N1 = 32, N2 = 26, N3 = 26]: starts
209284742: tensor<real, N0, N1, N2, N3> &Relu<64, 32, 26, 26>::forward(tensor<real, N0, N1, N2, N3> &, int) [N0 = 64, N1 = 32, N2 = 26, N3 = 26]: ends. took 1202192 nsec
209291784: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 32, 26, 26, 3, 64>::forward(tensor<real, maxB, IC, H, W> &, int) [maxB = 64, IC = 32, H = 26, W = 26, K = 3, OC = 64]: starts
769120455: tensor<real, maxB, OC, H - K + 1, W - K + 1> &Convolution2D<64, 32, 26, 26, 3, 64>::forward(tensor<real, maxB, IC, H, W> &, int) [maxB = 64, IC = 32, H = 26, W = 26, K = 3, OC = 64]: ends. took 559827015 nsec
769135387: tensor<real, N0, N1, N2, N3> &Relu<64, 64, 24, 24>::forward(tensor<real, N0, N1, N2, N3> &, int) [N0 = 64, N1 = 64, N2 = 24, N3 = 24]: starts
771112988: tensor<real, N0, N1, N2, N3> &Relu<64, 64, 24, 24>::forward(tensor<real, N0, N1, N2, N3> &, int) [N0 = 64, N1 = 64, N2 = 24, N3 = 24]: ends. took 1975174 nsec

  ...
```

Execution log (`--log`)
--------------------------

* Detailed execution records are saved into a file (default: `mnist.log`).  The file includes everything you will see with `-v 4`.  You can specify the filename with --log option.  When you execute many instances concurrently, make sure you specify a unique log file to each process.

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
109363: model building starts
30447215: model building ends
30458449: loading data from data
168557228: use 1 data items out of 60000
170495562: loading data from data
194254783: use 1 data items out of 10000
194575069: training starts
232683143: Train Epoch: 1 [0/1 (0%)]	Loss: 2.286959
244354120: Test set: Average loss: 2.7689, Accuracy: 0/1 (0%)
244366675: training ends
```

Initialization seed (`--weight-seed S`)
--------------------------

* It seeds a random number generator so the model will start from different initial weights
* Giving the same seed guarantees the network starts from the same initial weights
* If not specified, it uses a fixed default value

Dropout seed (`--dropout-seed-1 X` and `--dropout-seed-2 X`)
--------------------------

* There are two _dropout_ in the network

* A dropout layer takes a number of inputs and produces the same number of cells

* For each input, it either zeros or passes through the input randomly (i.e., output_i = 0 with some probability and input_i otherwise) during the training

* You can turn off the dropout by giving `--dropout-seed-1 0` `--dropout-seed-2 0` (default is ON)

* If the argument to `--dropout-seed-1` is not zero, the first dropout layer drops (zeros) the output with probability 0.25; if the argument to `--dropout-seed-2` is not zero, the second dropout layer drops (zeros) the output with probability 0.5

* Non-zero arguments to `--dropout-seed-1` or `--dropout-seed-2` seed the random number generator that chooses which outputs are zeroed in the respective dropout layers

* Again, default seeds are fixed

Remarks about seeding random number generators
--------------------------

There are a few places in which the program uses random numbers, namely,

 * when it initializes weight parameters (`--weight-seed`),
 * when it chooses which cells to dropout in a dropout layer (`--dropout-seed-1` and `--dropout-seed-2`)
 
_ALL COMPONENTS BEHAVE DETERMINISTICALLY._  That is, if you repeat executions with the same configuration repeatedly, it starts from the same weight parameters, uses the same training data in exactly the same order and drops out the same cells.

Unless your algorithm behaves non-deterministically, the results should be always the same when you repeat the same command line.  This should help you debug your code.

You can change these things by giving different seeds for each of these random number generators.  Specifically,

 * `--weight-seed` changes initial weight parameters
 * `--dropout-seed-1` and `--dropout-seed-2` change which cells are dropped out

Simply give an arbitrary number to any of them to make sure your algorithm are not sensitive to initial weights or particular dropout decisions

A general remark: when using a pseudo random number for a randomized algorithm such as this one, _ALWAYS_ give it a seed you chose and make it behave deterministically given the same seed.  This is a tremendous help for debugging.  After you have done initial debugging, you can test your algorithm across different sequences of random numbers just by giving different seeds.  Note that virtually all pseudo random number generators are deterministic after a seed is given.  A random number generator without any seed generates different sequences every time simply because they use a different seed each time; when you do not seed it, it simply takes a value from an external source (e.g., the current time) and uses it as a seed.  Nothing else is different.  In this sense, there is almost no point in not giving a seed of your choice (give the current time as a seed if you want to purposefully make it behave differently each time).  Without giving a seed, your algorithm can NEVER behave deterministically, which is a nightmare for debugging and the nightmare is nothing but unnecessary.


Data directory (`-d DIR`, `--data-dir DIR`)
--------------------------

* specifies directory from which data will be read (default: `data`)
* file names in the directory are hard-coded
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

GPU execution (`-a cuda_base`)
--------------------------

* The program compiled with nvcc supports GPU execution and this is the default behavior. It also supports CPU execution with `-a cpu_base` is given

* Here are the default behavior of baseline implementation

```
$ exe/mnist_cpu_base -a cpu_base  # (1) baseline code on CPU 
$ exe/mnist_cpu_base -a cuda_base  # (2) error
$ exe/mnist_cpu_base              # (3) same as (1)
$ exe/mnist_cuda_base -a cpu_base # (4) baseline code on CPU 
$ exe/mnist_cuda_base -a cuda_base # (5) baseline code on GPU
$ exe/mnist_cuda_base             # (6) same as (5)
```

* Note that baseline code is neither vectorized nor parallelized
* In particular, it uses only a single CUDA thread on GPU (!)

Algorithm choice (`-a`)
--------------------------

* The `-a` option described above is an option that chooses an algorithm from available repertories.  In the given code, only baseline algorithms for GPU and CPU are implemented.  
* You can (and should) add your implementation as another available choice here.


Controlled experiments
--------------------------

* After you roll your version, you want to make sure you got it done right.  Especially, you may be afraid that you broke a function.  To make sure the network is still functioning, you will want to start with a small and controlled experiment.

* You probably want to start with something like this.

```
$ ./exe/mnist_cpu_base -a YOUR_ALGORITHM --train-data-size 1 --test-data-size 0
```

* This uses only a single training sample and no test data at all
* So it keeps taking a single data and updating the weight for that data
* If the displayed `Loss:` value does not quickly decrease (from somewhere around 2.3 to almost zero after a few iterations), something will be fundamentally broken
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


How to interpret the loss value?
==========================

If you are curious what the value of the loss actually means, here it is.  In the final stage of the network, it ends up with a vector of 10 components (the number of classes) for each image, each component of which represents the probability that the model thinks the image belongs to a particular class.  This 10-element vector, say P, is then compared with the true label (class) for it.  If the true class of that image is c, then the loss for this particular image is 

   -log(P[c])

where P[c] is the probability that the model says the image belongs to the true class c.

Therefore, if the network is a random guess that returns 1/10 for every class, the average loss for an image is

   -log(1/10) = 2.3025850...

which is just about what you observe in the first iteration.

* The loss becomes zero if the network says the probability is 1 for the correct class and 0 for all other classes.  But as far as the classification performance  is concerned, the network outputs a correct class as long as the probability for the true class is larger than any other class.  A sufficient condition for this is P[c] > 0.5, as it guarantees that P[c] is greatest among P[0], ..., P[9], whose sum is one.  When P[c] = 0.5, the loss would be 

   -log(1/2) = 0.69314...

Navigating the source code
==========================

Three options to know how things actually work

1. open https://taulec.zapto.org/~share/parallel-distributed/21mnist/docs/doxy/html/files.html to see function/method/struct documentation (the same documents are in the source code, too, but this one is easier to navigate)
1. open https://taulec.zapto.org/~share/parallel-distributed/21mnist/docs/tags/HTML/index.html with your browser to jump between functions (where this function is defined and where this function is called)
1. compile it with -O0 -g (edit Makefile) and run it under the debugger (gdb or cuda-gdb) (useful to know how things exactly work, from reading data to training to calling layers to updating weights)

A guide for developing your code
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

  (the whole network)

  - `mnist.h` -- the entire MNIST

* The main function in `mnist.cc` instantiates a MNIST network, which is defined in `mnist.h`
* It repeats processing training data, occasionally processing test data.

Each layer defines a class whose name is similar to the file name.  e.g., `convolution.h` defines `Convolution2D` class.

All classes for primitives and the whole network have two important functions, among others.
 * `forward` -- take an input from the previous (downstream) layer and computes its output
 * `backward` -- take a gradient of loss wrt the upstream layer and computes the gradient wrt its input and weights

In addition, classes that have weight parameters (`Convolution2D` and `Linear`) have another function.

 * `update` -- update its weights, using the gradient computed in the backward phase.

**********************************************************************************
YOUR MAIN JOB WILL BE TO IMPLEMENT A HIGH PERFORMANCE VERSION OF THESE FUNCTIONS.
**********************************************************************************

You will eventually want to work on all six files (`convolution.h`, `linear.h`, `relu.h`, `dropout.h`, `max_pooling.h`, `nll_log_softmax.h`) but you can work incrementally.  You can make one layer faster while leaving all others intact.  You can know which functions are taking much time by `-v 4` option.

Stepping through the code using gdb (or cuda-gdb)
--------------------------

When working on details, you want to step through the code using gdb (or cuda-gdb to step through GPU code).  They also help you get an idea about how things work.  For that, compile the code with `-O0 -g`.  Also add `-Xptxas -O0` if you compile with nvcc.

The structure of the baseline implementations (and switching between different algorithms)
--------------------------

As I mentioned above, functions you will primarily be working on are `forward` and `backward` functions on the six classes and `update` functions for the two classes.  Each of them has the same structure to switch between GPU code and CPU code (currently, a single execution can run either entirely on CPU or entirely on GPU; you cannot have some layers executed by CPU and others on GPU in the same execution).  Let's look at the `forward` function of `Convolution2D` class, for example.

The member function named `forward` is the entry point of the forwarding phase.  It only executes a switch statement to decide which implementation to use.

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

The latter calls into GPU.  Since `nvcc` does not allow a class member function to be a global function (a GPU function callable from a host), we need to define a global function outside the class (`forward_cuda_base_global`), which then calls back a member function (`forward_cuda_base_device`).  This is the baseline implementation of `forward_cuda_base`.

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

Here is how you change the code when working on a new implementation.  As already mentioned, there are two implementations already in place, cpu_base and cuda_base.

Before starting the real work, there are some work for preparation.

 * Come up with a name of the new implementation.  Let's say it is `cpu_awesome` (in reality, you want to have a name that better represents what it does, like `cpu_simd`).

 * Add a new symbol to the enum `algo_t` defined in `mnist_util.h`
``` 
typedef enum {
  algo_cpu_base,
  algo_cuda_base,
  /* add your new algorithm here (name it arbitrarily) */

  algo_cpu_awesome, <----  YOU ADD THIS

  algo_invalid,
} algo_t;
```

 * Change the `parse_algo` function right below it so that it recognizes the new name.  Obviously, the baseline code recognizes only "cpu_base" and "cuda_base".  You simply add an appropriate "else if" branch to handle your name.

```
algo_t parse_algo(const char * s) {
  if (strcmp(s, "cpu_base") == 0) {
    return algo_cpu_base;
  } else if (strcmp(s, "cuda_base") == 0) {
    return algo_cuda_base;
  } else if (strcmp(s, "cpu_awesome") == 0) {  <---- YOU ADD THIS
    return algo_cpu_awesome;                   <---- YOU ADD THIS
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

At this point, the program at least recognizes your algorithm and calls GPU base code or CPU base code depending on your algorithm is a GPU algorithm or not (judged by algo_is_gpu function above).  Recall that the switch statement falls back to `forward_cuda_base` or `forward_cpu_base` when the switch statement does not have a specific case for your algorithm.

Now you are ready to add a real implementation.  Thanks to the structure just mentioned, you can do so incrementally (you do not have to implement all functions to get your version used).  To start off, let's say you want to implement a `forward` function of `Convolution2D` class.  The first thing you need to do is to add an appropriate case in the switch statement.

```
  tensor<real,maxB,OC,H-K+1,W-K+1>& forward(tensor<real,maxB,IC,H,W>& x, int training) {
    log_start_fun(lgr);
    tsc_t t0 = get_tsc();
    switch (opt.algo) {
      /* add case for your implementations here */
    case algo_cpu_awesome:
      forward_cpu_awesome(x); break;
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

Your real job is, of course, to implement `forward_cpu_awesome` function.  Use SIMD, OpenMP or whatever is necessary to make it faster.  You probably start by copy-pasting the `forward_base` implementation.

If you work on CUDA implementation, you need to implement two functions.  Let's say your algorithm name is `cuda_awesome`.  After adding another case in the switch statement like this

```
    case algo_cuda_awesome:
      forward_cuda_awesome(x); break;
```

your `forward_cuda_awesome` function will launch a global function with a more sensible value of the thread block size.

```
  void forward_cuda_awesome(array4<maxB,IC,H,W>& x) {
    int block_sz = ...;
    int num_blocks = ...;
    launch_and_sync((forward_cuda_awesome_global<<<num_blocks,block_sz>>>(dev, x.dev)));
  }
```

Next you define `forward_cuda_awesome_global` function outside the class.  You may want to copy the template definition for `forward_cuda_base_global` in `cuda_util.h`.  This will be a boilerplate code.

```
template<typename T, typename I>
__global__ void forward_cuda_awesome_global(T* dev, I* x_dev, int training) {
  /* call the member function */
  dev->forward_cuda_awesome_device(*x_dev, training);
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
$ ./exe/mnist_cpu_base --train-data-size 1 --test-data-size 0 -a name_of_your_algorithm
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



A record submission tool and record viewers (updated: 23 Jan, 2023)
=============

* Here is a tool to submit a result of executing mnist and a web page to see results submitted by all
* You are required to submit at least the final result you report in your final term paper, but you are encouraged to submit your results whenever you think you made a progress.  Don't wait until you think you are finished
* You can submit your results as many times as you want; you can also (though not encouraged to) delete them if you think there are too many to comfortably see (you can filter out unnecessary records, so you should not have to do this)

Submit your run
-------------

* Running an executable leaves a log file, called mnist.log by default (you can change it with `--log` option)
* Data can be viewed at https://taulec.zapto.org/mnist_viewer/
* Do the following on taulec to submit your result 
```
submit mnist.log
```
* You are recommended to do the following before you submit the result to make sure the log is well-formed
```
submit --dryrun mnist.log
```
* If the log file is not on taulec (e.g., you are working on your laptop), do the following
```
ssh uXXXX@taulec.zapto.org submit --dryrun < mnist.log
ssh uXXXX@taulec.zapto.org submit < mnist.log
```

View your run
-------------

* visit https://taulec.zapto.org/mnist_viewer/ to see the results

Details you should not (but occasionally might) have to know
-------------

* `submit` is a tailor-made command installed at /usr/local/bin/submit on taulec.zapto.org

Performance criteria and the regulation
==========================


