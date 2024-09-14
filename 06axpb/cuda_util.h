/**
   @file cuda_util.h
   @brief small utility functions for cuda
   @author Kenjiro Taura
   @date Oct. 14, 2018
 */

/**
   @brief do not use this function directly. use check_api_error macro
   @sa check_api_error
 */

static void check_api_error_(cudaError_t e,
                              const char * msg, const char * file, int line) {
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

/**
   @brief check if a CUDA API invocation succeeded and show the error msg if any
   @details usage:  check_api_error(cuda_api_call()). for example,
   check_api_error(cudaMalloc(&p, size));
 */

#define check_api_error(e) check_api_error_(e, #e, __FILE__, __LINE__)

/**
   @brief do not use this function directly. use check_launch_error macro
   @sa check_launch_error
 */

static void check_launch_error_(const char * msg, const char * file, int line) {
  cudaError_t e = cudaGetLastError();
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

/**
   @brief check kernel launch error
   @details usage: check_launch_error((kernel-launch-expression)). for example,
   check_launch_error((your_gpu_kernel<<<n_blocks,block_sz>>>(a,b,c))). 
   note that you need to put parens around the expression.
 */

#define check_launch_error(exp) do { exp; check_launch_error_(#exp, __FILE__, __LINE__); } while (0)

/**
   @brief get SM executing the caller
 */
__device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

/**
   @brief get device frequency
 */
static int get_freq() {
  struct cudaDeviceProp prop[1];
  check_api_error(cudaGetDeviceProperties(prop, 0));
  return prop->clockRate;
}

/**
   @brief wrap cudaMalloc.  cudaMalloc + error check + more ordinary malloc-like interface (return pointer)
 */
static void * dev_malloc(size_t sz) {
  void * a = 0;
  cudaError_t e = cudaMalloc(&a, sz);
  if (!a) {
    fprintf(stderr, "error: %s\n", cudaGetErrorString(e));
    exit(1);
  }
  return a;
}

/**
   @brief wrap cudaFree
 */
static void dev_free(void * a) {
  cudaFree(a);
}

/**
   @brief wrap cudaMemcpy to copy from device to host (and check an error if any)
 */
void to_host(void * dst, void * src, size_t sz) {
  check_api_error(cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost));
}

/**
   @brief wrap cudaMemcpy to copy from host to device (and check an error if any)
 */
static void to_dev(void * dst, void * src, size_t sz) {
  check_api_error(cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice));
}

/**
   @brief thread ID along x-dimension
 */
__device__ inline int get_thread_id_x() {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

/**
   @brief thread ID along y-dimension
 */
__device__ inline int get_thread_id_y() {
  return blockDim.y * blockIdx.y + threadIdx.y;
}

/**
   @brief thread ID along z-dimension
 */
__device__ inline int get_thread_id_z() {
  return blockDim.z * blockIdx.z + threadIdx.z;
}

/**
   @brief number of threads along x-dimension
 */
__device__ inline int get_nthreads_x() {
  return gridDim.x * blockDim.x;
}

/**
   @brief number of threads along y-dimension
 */
__device__ inline int get_nthreads_y() {
  return gridDim.y * blockDim.y;
}

/**
   @brief number of threads along z-dimension
 */
__device__ inline int get_nthreads_z() {
  return gridDim.z * blockDim.z;
}

/**
   @brief global (x,y,z combined into an integer) thread ID
 */
__device__ inline int get_thread_id() {
  int x = get_thread_id_x();
  int y = get_thread_id_y();
  int z = get_thread_id_z();
  int nx = get_nthreads_x();
  int ny = get_nthreads_y();
  return nx * ny * z + nx * y + x;
}

/**
   @brief total number of threads
 */
__device__ inline int get_nthreads() {
  int nx = get_nthreads_x();
  int ny = get_nthreads_y();
  int nz = get_nthreads_z();
  return nx * ny * nz;
}


