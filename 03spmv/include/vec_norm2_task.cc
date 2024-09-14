/**
   @file vec_norm2_task.cc
   @brief square norm of a vector in serial
*/

/** 
    @brief square norm of a vector in serial
    @param (v) a vector
    @returns the square norm of v (v[0]^2 + ... + v[n-1]^2)
*/
static real vec_norm2_task(vec_t v) {
  /* you don't need to do anything specific for this.
     just call the parallel version and you are done */
  return vec_norm2_parallel(v);
}

