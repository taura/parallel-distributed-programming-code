/**
   @file coo_to_dev.cc
   @brief make a deivce copy of a sparse matrix in coo format
*/

/** 
    @brief make a deivce copy of a sparse matrix in coo format.
    @param (A) the reference to a matrix whose elems_dev has not 
    been set (i.e., = NULL)
    @returns 1 if succeed. 0 if failed.
    @details this function allocates memory on the device and
    transfers A's non-zero elements in the allocated memory.
    it also should set A's elems_dev to the address of the allocated 
    memory, so that if you pass A as an argument of a kernel launch,
    the device code can obtain all necessary information of A from
    the parameter.
    @sa sparse_to_dev
*/
static int coo_to_dev(sparse_t& A) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:coo_to_dev:\n"
          "write a code that copies the elements of csr to the device.\n"
          "use dev_malloc and to_dev utility functions in cuda_util.h\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);
  return 1;
}

