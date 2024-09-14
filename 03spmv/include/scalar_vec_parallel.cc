/** 
    @file scalar_vec_parallel.cc
    @brief scalar x vector multiply with parallel for
 */

/** 
    @brief k x v in parallel
    @param (k) a scalar
    @param (v) a vector
    @returns 1 
    @details multiply each element of v by k
*/
static int scalar_vec_parallel(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems;
#pragma omp parallel for
  for (idx_t i = 0; i < n; i++) {
    x[i] *= k;
  }
  return 1;
}

