/**
   @file spmv_coo_sorted_cuda.cc
   @brief y = A * x for coo_sorted with cuda
 */

/** 
    @brief y = A * x for coo_sorted with cuda
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
static int spmv_coo_sorted_cuda(sparse_t A, vec_t vx, vec_t vy) {
  /* the same no matter whether elements are sorted or not.
     call spmv_coo_cuda and we are done */
  return spmv_coo_cuda(A, vx, vy);
}
