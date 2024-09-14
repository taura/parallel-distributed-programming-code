/**
   @file hello_gpu.cu
  */
#include <assert.h>
#include <stdio.h>

/*
  you'd better spend time on making sure you always check errors ...
*/

void check_api_error_(cudaError_t e,
                      const char * msg, const char * file, int line) {
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

#define check_api_error(e) check_api_error_(e, #e, __FILE__, __LINE__)

void check_launch_error_(const char * msg, const char * file, int line) {
  cudaError_t e = cudaGetLastError();
  if (e) {
    fprintf(stderr, "%s:%d:error: %s %s\n",
            file, line, msg, cudaGetErrorString(e));
    exit(1);
  }
}

#define check_launch_error(exp) do { exp; check_launch_error_(#exp, __FILE__, __LINE__); } while (0)




__global__ void worker(double * a, long n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    a[i] += i;
  }
}

int main(int argc, char ** argv) {
  int n               = (argc > 1 ? atoi(argv[1]) : 100);
  int thread_block_sz = (argc > 2 ? atoi(argv[2]) : 64);
  int n_thread_blocks = (n + thread_block_sz - 1) / thread_block_sz;

  /* prepare data and copy it to the device */
  printf("allocate data on device\n"); fflush(stdout);
  size_t sz = sizeof(double) * n;
  double * a_host = (double *)malloc(sz);
  memset(a_host, 0, sz);
  double * a_dev;
  check_api_error(cudaMalloc((void **)&a_dev, sz));
  check_api_error(cudaMemcpy(a_dev, a_host, sz, cudaMemcpyHostToDevice));
  
  /* launch the kernel */
  printf("launch the kernel\n"); fflush(stdout);
  check_launch_error((worker<<<n_thread_blocks,thread_block_sz>>>(a_dev, n)));
  check_api_error(cudaThreadSynchronize());

  /* get the result back */
  printf("get the result back\n"); fflush(stdout);
  check_api_error(cudaMemcpy(a_host, a_dev, sz, cudaMemcpyDeviceToHost));

  printf("check the result\n"); fflush(stdout);
  for (int i = 0; i < n; i++) {
    assert(a_host[i] == i);
  }
  printf("OK\n"); fflush(stdout);
  return 0;
}
