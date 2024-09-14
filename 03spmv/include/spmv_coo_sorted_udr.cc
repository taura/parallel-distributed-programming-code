/**
   @file spmv_coo_sorted_udr.cc
   @brief y = A * x for coo_sorted with parallel for + user-defined reductions
 */

/** 
    @brief y = A * x for coo_sorted with parallel for + user-defined reductions
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
static int spmv_coo_sorted_udr(sparse_t A, vec_t vx, vec_t vy) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:spmv_coo_sorted_udr:\n"
          "write a code that performs SPMV for SORTED COO format in parallel\n"
          "using parallel for directives + user-defined reductions.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);

  /* this is a serial code for your reference */
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    y[i] += ax;
  }
  return 1;
}

