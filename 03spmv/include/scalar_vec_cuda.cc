/** 
    @file scalar_vec_cuda.cc
    @brief scalar x vector multiply with cuda
 */

/** 
    @brief the device procedure for k x v in parallel with cuda
    @param (k) a scalar
    @param (v) a vector
    @details multiply each element of v by k
*/
__global__ void scalar_vec_dev(real k, vec_t v) {
  idx_t n = v.n;
  real * x = v.elems_dev;
  idx_t i = get_thread_id_x();
  if (i < n) {
    x[i] *= k;
  }
}

/** 
    @brief k x v in parallel with cuda
    @param (k) a scalar
    @param (v) a vector
    @details multiply each element of v by k
*/
static int scalar_vec_cuda(real k, vec_t v) {
  idx_t n = v.n;
  int scalar_vec_block_sz = 1024;
  int n_scalar_vec_blocks = (n + scalar_vec_block_sz - 1) / scalar_vec_block_sz;
  check_launch_error((scalar_vec_dev<<<n_scalar_vec_blocks,scalar_vec_block_sz>>>(k, v)));
  return 1;
}
