/** 
    @file spmv_csr_udr.cc
    @brief y = A * x for csr with parallel for + user-defined functions
*/

/** 
    @brief y = A * x for csr with parallel for + user-defined functions
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
static int spmv_csr_udr(sparse_t A, vec_t vx, vec_t vy) {
  /* you don't need UDR for CSR format.
     just call spmv_csr_parallel and you are done */
  return spmv_csr_parallel(A, vx, vy);
}

