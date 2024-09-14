/** 
    @file scalar_vec_task.cc
    @brief scalar x vector multiply with tasks
 */

/** 
    @brief k x v in parallel with tasks
    @param (k) a scalar
    @param (v) a vector
    @returns 1 
    @details multiply each element of v by k
*/
static int scalar_vec_task(real k, vec_t v) {
  /* you don't need to do anything specific for this.
     just call the parallel version and you are done */
  return scalar_vec_parallel(k, v);
}
