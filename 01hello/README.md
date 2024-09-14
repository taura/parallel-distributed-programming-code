
This directory explains how to run OpenMP programs.


Build
================
```
$ make
gcc -fopenmp -o hello hello.c 
```

Run (submit)
================

You should be able to run OpenMP programs with the right number of threads, allocating the right number of cores.

A good formula is this:


```
$ srun -p <partition> -t 0:01:00 -n 1 -c <max_nthreads> bash -c "OMP_PROC_BIND=TRUE OMP_NUM_THREADS=<nthreads> ./hello"
```
where max_nthreads should at least be nthreads for a reasonable result.

Or,
```
$ srun -p <partition> -t 0:01:00 -n 1 --exclusive bash -c "OMP_PROC_BIND=TRUE OMP_NUM_THREADS=<nthreads> ./hello"
```

to exclusively allocate an entire node to this job.

 * <max_nthreads> is the number of processors you request to the job manager
 * <nthreads> is the actual number of threads your OpenMP programs will launch when it encounters a parallel pragma
 * Never forget "-n 1", which specifies the number of instances that will run.  When omitting this, it may create multiple instances of the specified command line, which is in most cases not what you want and is very confusing.

--exclusive option allocates an entire node for you.  For the best result, you may consider using this option to fully control threads-processor mapping.  -c <max_nthreads> may result in suboptimal threads-processor mapping.  For example, it may allocate two processors on the same physical core or choose processors from sockets in an unbalanced fashion.

--exclusive option will be useful when you run multiple processes (presumaly from a shell script or a makefile) with various number of threads.

Examples
------------------

* Let the job manager chooses a processor and run a thread there.

```
$ srun -p big -t 0:01:00 -n 1 -c 1 bash -c "OMP_PROC_BIND=TRUE OMP_NUM_THREADS=1 ./hello" 
hello before omp parallel
hello from thread 0 of 1 on 13
hello after omp parallel
```

* Let the job manager chooses two processors and run two threads there.

```
$ srun -p big -t 0:01:00 -n 1 -c 2 bash -c "OMP_PROC_BIND=TRUE OMP_NUM_THREADS=2 ./hello" 
hello before omp parallel
hello from thread 1 of 2 on 77
hello from thread 0 of 2 on 13
hello after omp parallel
```

* Just to demonstrate that you can run any number of threads no matter how many/small processors you requested.

```
$ srun -p big -t 0:01:00 -n 1 -c 2 bash -c "OMP_PROC_BIND=TRUE OMP_NUM_THREADS=5 ./hello" 
hello before omp parallel
hello from thread 2 of 5 on 77
hello from thread 0 of 5 on 13
hello from thread 4 of 5 on 13
hello from thread 3 of 5 on 77
hello from thread 1 of 5 on 13
hello after omp parallel
```

Note that 5 threads run on two threads, which in most cases will not be what you want.

* Conversely, you may request many processors but run a few threads.

```
$ srun -p big -t 0:01:00 -n 1 -c 5 bash -c "OMP_PROC_BIND=TRUE OMP_NUM_THREADS=2 ./hello" 
hello before omp parallel
hello from thread 0 of 2 on 13
hello from thread 1 of 2 on 14
hello after omp parallel
```

* you may consider writing complex command lines in a shell script or a makefile.  For example, write the following shell script.
```
#!/bin/bash
for t in 1 2 4 6 8 ; do
    echo "======= ${t} threads ======="
    OMP_PROC_BIND=TRUE OMP_NUM_THREADS=${t} ./hello
done
```
and run it as follows, requesting eight processors.
```
$ srun -p big -t 0:01:00 -n 1 -c 8 ./hello.sh 
======= 1 threads =======
hello before omp parallel
hello from thread 0 of 1 on 44
hello after omp parallel
======= 2 threads =======
hello before omp parallel
hello from thread 0 of 2 on 44
hello from thread 1 of 2 on 45
hello after omp parallel
======= 4 threads =======
hello before omp parallel
hello from thread 0 of 4 on 44
hello from thread 2 of 4 on 46
hello from thread 1 of 4 on 45
hello from thread 3 of 4 on 47
hello after omp parallel
======= 6 threads =======
hello before omp parallel
hello from thread 3 of 6 on 47
hello from thread 2 of 6 on 46
hello from thread 0 of 6 on 44
hello from thread 1 of 6 on 45
hello from thread 4 of 6 on 108
hello from thread 5 of 6 on 109
hello after omp parallel
======= 8 threads =======
hello before omp parallel
hello from thread 3 of 8 on 47
hello from thread 7 of 8 on 111
hello from thread 1 of 8 on 45
hello from thread 5 of 8 on 109
hello from thread 2 of 8 on 46
hello from thread 6 of 8 on 110
hello from thread 0 of 8 on 44
hello from thread 4 of 8 on 108
hello after omp parallel
```

