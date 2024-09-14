This directory explains how to run CUDA programs.


Build
================
```
$ make
nvcc -o hello_gpu hello_gpu.cu 
nvcc -o hello_gpu2 hello_gpu2.cu 
```

Run (submit)
================

```
$ srun -p p --gres gpu:1 ./hello_gpu
hello I am CUDA thread 96 out of 128
hello I am CUDA thread 97 out of 128
hello I am CUDA thread 98 out of 128
hello I am CUDA thread 99 out of 128
hello I am CUDA thread 0 out of 128
    ...
hello I am CUDA thread 93 out of 128
hello I am CUDA thread 94 out of 128
hello I am CUDA thread 95 out of 128
OK
```

When submitting a job, be sure

 * you specify a partition of GPU nodes (-p p or -p v)
 * specify --gres gpu:1, which requests a GPU on the node.  If omitted, the program raises the following error.

```
$ srun -p p ./hello_gpu
NG: no CUDA-capable device is detected
srun: error: p103: task 0: Exited with exit code 1
```

Note that this error appears only because the program checks an error after calling CUDA API or launching a kernel.

Otherwise the program continues running and crashes after a while or exits with a wrong result.  Either case the program becomes harder to debug.  Make sure you always check errors as demonstrated by this example.

