/**
   @file get_pfn_info.h
 */
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
   @brief 64-bit data structure describing the status
   of a page obtained from /proc/PID/pagemap. see
   https://www.kernel.org/doc/Documentation/vm/pagemap.txt
*/

typedef union {
  struct {
    uint64_t pfn : 55;          /**< physical page frame number */
    unsigned int soft_dirty : 1; /**< pte is soft-dirty (see Documentation/vm/soft-dirty.txt) */
    unsigned int exclusively_mapped : 1; /**< page exclusively mapped (since 4.2) */
    unsigned int zero : 4;               /**< zero  */
    unsigned int file_page_or_shared_anon : 1; /**< page is file-page or shared-anon (since 3.5)  */
    unsigned int swapped : 1;                  /**< page swapped */
    unsigned int present : 1;                  /**< page present */
  } p;
  struct {
    unsigned int swap_type : 5; /**< swap type if swapped */
    uint64_t swap_offset : 50;  /**< swap offset if swapped */
    unsigned int soft_dirty : 1; /**< pte is soft-dirty (see Documentation/vm/soft-dirty.txt) */
    unsigned int exclusively_mapped : 1; /**< page exclusively mapped (since 4.2) */
    unsigned int zero : 4;               /**< zero  */
    unsigned int file_page_or_shared_anon : 1;  /**< page is file-page or shared-anon (since 3.5)  */
    unsigned int swapped : 1;                  /**< page swapped */
    unsigned int present : 1;                  /**< page present */
  } s;
} __attribute__((packed)) pfn_info_t;

/**
   @brief a structure containg an array of pfn_info_t 
   and its number of elements
*/

typedef struct {
  long n;                       /**< the number of elements in pfn  */
  pfn_info_t * pfn;             /**< the pointer to elements */
} pfn_info_array_t;

/**
   @brief return the maximum multiple of sz that is <= q
  */
static long make_a_multiple(long q, long sz) {
  return q - q % sz;
}

/**
   @brief return the number of pages containing [begin:end-1]
*/
static long get_num_pages(void * begin, void * end) {
  off_t page_size = 4096;
  /* the virtual page number containing begin */
  off_t begin_vpn =  (off_t)begin / page_size;
  /* the virtual page number containing end */
  off_t   end_vpn = ((off_t)end + page_size - 1) / page_size;
  return end_vpn - begin_vpn;
}

/**
   @brief return page frame information of virtual pages
   covering [begin:end-1] into pfn[0:n]

   @param (begin) the start address of the range
   @param (end)   the end address of the range + 1
   @param (pfn)   the array to return the result to
   @param (n)     the number of elements pfn can accomodate
   @return the number of pages containing [begin:end-1]

   @details get page frame number information using /proc/PID/pagemap
   of all pages that have at least share a byte with 
   the address range [begin,end-1], and put them in pfn[0:n].
   if the number of pages exceed n, only the information about 
   the first n pages will be returned. 
   return the number of pages that have at least share a byte
   with [begin,end-1]. 

   @sa make_pfn_info_array */

static long get_pfn_info(void * begin, void * end, pfn_info_t * pfn, long n) {
  pid_t pid = getpid();
  char proc_pid_pagemap[100];
  off_t page_size = 4096;
  /* the virtual page number containing begin */
  off_t begin_vpn =  (off_t)begin / page_size;
  /* the virtual page number containing end */
  off_t   end_vpn = ((off_t)end + page_size - 1) / page_size;
  /* we must read a multiple of 128KB */
  off_t block_sz = 128 * 1024;
  /* the number of pages per 128KB */
  off_t n_pages_per_block = block_sz / sizeof(pfn_info_t);
  /* the page number suitable for read */
  off_t begin_aligned_vpn = make_a_multiple(begin_vpn, n_pages_per_block);
  off_t   end_aligned_vpn = make_a_multiple(end_vpn + n_pages_per_block - 1,
                                            n_pages_per_block);
  off_t n_aligned_pages = end_aligned_vpn - begin_aligned_vpn;
  off_t read_sz = sizeof(pfn_info_t) * n_aligned_pages;
  assert(read_sz % block_sz == 0);
  pfn_info_t * buf = (pfn_info_t *)malloc(read_sz);
  sprintf(proc_pid_pagemap, "/proc/%d/pagemap", pid);
  int fd = open(proc_pid_pagemap, O_RDONLY);
  off_t o = lseek(fd, begin_aligned_vpn * sizeof(pfn_info_t), SEEK_SET);
  if (o == -1) { perror("lseek"); exit(1); }
  ssize_t rd = read(fd, buf, read_sz);
  if (rd == -1) { perror("read"); exit(1); }
  assert(rd == read_sz);
  long i = 0;
  for (off_t v = begin_vpn; v < end_vpn; v++) {
    pfn_info_t pi = buf[v - begin_aligned_vpn];
    if (i < n) {
      pfn[i] = pi;
    }
    i++;
  }
  free(buf);
  close(fd);
  return i;
}

/**
   @brief return page frame information of virtual pages
   covering [begin:end-1] and return it in a new array.

   @param (begin) the start address of the range
   @param (end)   the end address of the range + 1
   @return pfn_info_array_t structure containing the 
   requested information

   @details see get_pfn_info
   @sa get_pfn_info
*/

static pfn_info_array_t make_pfn_info_array(void * begin, void * end) {
  long n = get_num_pages(begin, end);
  pfn_info_t * pi = (pfn_info_t *)malloc(sizeof(pfn_info_t) * n);
  long np = get_pfn_info(begin, end, pi, n);
  assert(np == n);
  pfn_info_array_t pa = { n, pi };
  return pa;
}

static int is_power_of_two(unsigned long x) {
  return !(x & (x - 1));
}

static double show_cache_set_info(void * a, void * b) {
  size_t line_size = 64;
  long page_size = 4096;
  size_t cache_size = 1024 * 1024;
  size_t associativity = 16;
  assert(cache_size % (associativity * line_size) == 0);
  size_t n_sets = cache_size / (associativity * line_size);
  assert(is_power_of_two(n_sets));
  /* line_counts[i] = the number of lines mapped to set i */
  size_t * line_counts = (size_t *)calloc(sizeof(size_t), n_sets);
  
  pfn_info_array_t pa = make_pfn_info_array(a, b);
  for (long i = 0; i < pa.n; i++) {
    pfn_info_t pi = pa.pfn[i];
    uint64_t paddr = (pi.p.present ? pi.p.pfn * page_size : 0);
    uint64_t set = (paddr / line_size) & (n_sets - 1);
    assert(set < n_sets);
    for (unsigned int i = 0; i < page_size / line_size; i++) {
      line_counts[(set + i) % n_sets]++;
    }
  }

  int overflow_lines = 0;
  int non_overflow_lines = 0;
  for (size_t i = 0; i < n_sets; i++) {
    if (line_counts[i] > associativity) {
      overflow_lines += line_counts[i];
    } else {
      non_overflow_lines += line_counts[i];
    }
  }
  free(line_counts);
  return overflow_lines / (double)(overflow_lines + non_overflow_lines);
}

#if 0
/* usage 
   
   ## BE SURE TO RUN IT AS ROOT
   ## otherwise it won't get physical address

   $ sudo ./a.out N 

   allocate N bytes, touch every page and prints

      virtual_address_0 physical_address_0
      virtual_address_1 physical_address_1
      virtual_address_2 physical_address_2
             ...

   for all pages allocated

*/

int main(int argc, char ** argv) {
  long page_size = 4096;
  size_t n = (argc > 1 ? atol(argv[1]) : page_size * 10);
  char * a = calloc(n, 1);
  memset(a, 1, n);
  pfn_info_array_t pa = make_pfn_info_array(a, a + n);
  char * vaddr = (char *)make_a_multiple((off_t)a, page_size);
  for (long i = 0; i < pa.n; i++) {
    pfn_info_t pi = pa.pfn[i];
    void * paddr = (void *)(pi.p.present ? pi.p.pfn * page_size : 0);
    printf("%p %p\n", vaddr, paddr);
    vaddr += page_size;
  }
  return 0;
}

#endif
