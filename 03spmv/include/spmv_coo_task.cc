/**
   @file spmv_coo_task.cc
   @brief y = A * x for coo with tasks
 */

/** 
    @brief y = A * x for coo with tasks
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
static int spmv_coo_task(sparse_t A, vec_t vx, vec_t vy) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:spmv_coo_task: not implemented\n"
          "you can leave it unimplemented\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);
  (void)A;
  (void)vx;
  (void)vy;
  /* it is possible, but does not make much sense to work on this.
     you can leave it as it is.
     instead work on spmv_coo_sorted_task and
     spmv_csr_task */
  return 0;
}

