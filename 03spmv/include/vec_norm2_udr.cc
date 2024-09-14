/**
   @file vec_norm2_udr.cc
   @brief square norm of a vector in parallel using user-defined reduction
*/

/** 
    @brief square norm of a vector in parallel using user-defined reduction
    @param (v) a vector
    @returns the square norm of v (v[0]^2 + ... + v[n-1]^2)
*/
static real vec_norm2_udr(vec_t v) {
  /* you don't need to do anything specific for this.
     just call the parallel version and you are done */
  return vec_norm2_parallel(v);
}

