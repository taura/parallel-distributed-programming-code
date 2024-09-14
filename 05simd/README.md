
This directory contains many small programs that test vectorization ability of compilers and how to manually vectorize them.

 * 00bb : a simplest loop
 * 01if : the loop body contains a simple if statement
 * 02loop_c : the loop body contains a nested loop whose trip count is 15 (a small compile-time constant)
 * 03loop_m : the loop body contains a nested loop whose trip count is invariant (across all iterations) but known only at runtime
 * 04loop_i : the loop body contains a nested loop whose trip count varies across iterations 
 * 05fun : the loop body calls a function declared with omp declare simd
 * 06stride : the loop body reads an array with a non-unit stride (a[i * 2])
 * 07random : the loop body reads an array with a random expression a[i * i]
 * 08indirect : the loop body reads an array with an indirect indexing array a[idx[i]]
 * 09indirect_store : similar to 08, but this one stores 
 
