
Build
=================

```
$ make
g++ -o spmv.gcc spmv.cc  -Wall -Wextra -O3 -fopenmp  
/usr/local/cuda/bin/nvcc -o spmv.nvcc -x cu spmv.cc  --gpu-code sm_60 --gpu-architecture compute_60 -O3 -Xptxas -O3,-v
   ...
```

do this on the login node.

Run
=================

Example:

```
$ srun -p big -t 0:01:00 ./spmv.gcc
A : 100000 x 100000, 100000000 non-zeros 800000000 bytes for non-zeros
repeat : 5 times
format : coo
matrix : random
algo : serial
2000000000 flops for spmv
generating 100000 x 100000 matrix (100000000 non-zeros) ...  done
repeat_spmv : warm up + error check
repeat_spmv : start
2002500000 flops in 1.023463911e+01 sec (1.956590729e-01 GFLOPS)
lambda = 5.006385423e+02
```

srun should be used to run any executable on a compute node.

But for a very small/short run for the purpose of quick correctness check, you can directly run it on the login node.

```
$ ./spmv.gcc
A : 100000 x 100000, 100000000 non-zeros 800000000 bytes for non-zeros
repeat : 5 times
1000000000 flops
repeat_spmv : warm up + error check
repeat_spmv : start
2002500000 flops in 6.514223060e+00 sec (3.074042724e-01 GFLOPS)
lambda = 5.006385423e+02
```

help
=================

Just run

```
$ ./spmv.gcc -h
usage:

./spmv.gcc [options ...]

options:
  --help             show this help
  --M N              set the number of rows to N [100000]
  --N N              set the number of colums to N [0]
  -z,--nnz N         set the number of non-zero elements to N [0]
  -r,--repeat N      repeat N times [5]
  -f,--format F      set sparse matrix format to F [coo]
  -t,--matrix-type M set matrix type to T [random]
  -a,--algo A        set algorithm to A [serial]
  --coo-file F       read matrix from F [mat.txt]
  --rmat a,b,c,d     set rmat probability [4,1,2,3]
  --dump F           dump matrix to image (gnuplot) file []
  --img-M M          number of rows in the dumped image [512]
  --img-N N          number of columns in the dumped image [512]
  -s,--seed S        set random seed to S [4567890123]
```

Learn how it works
=================

Compile it with -O0 -g options and run it with small parameters inside a debugger.

```
$ ... modify Makefile and set -O0 and -g to cflags ...

$ make -B
g++  -Wall -Wextra -O0 -g -fopenmp   -c -o spmv.o spmv.cc
g++ -o spmv spmv.o  -Wall -Wextra -O0 -g -fopenmp

$ gdb ./spmv

   ...
Reading symbols from ./spmv...done.

(gdb) b main
Breakpoint 1 at 0x402dc8: file spmv.cc, line 790.

(gdb) run --M 10
Starting program: /home/tau/parallel-distributed-handson/00spmv/spmv --M 10
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Breakpoint 1, main (argc=3, argv=0x7fffffffe828) at spmv.cc:790
790     int main(int argc, char ** argv) {

```

If you use Emacs, it is much better to use it from within Emacs.

```
M-x gud-gdb
```

and continue as before.

Notable features
=================

Verifying your results
-----------------


How to modify the file
=================

