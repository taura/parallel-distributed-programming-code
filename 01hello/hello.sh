#!/bin/bash
for t in 1 2 4 6 8 ; do
    echo "======= ${t} threads ======="
    OMP_PROC_BIND=TRUE OMP_NUM_THREADS=${t} ./hello
done
