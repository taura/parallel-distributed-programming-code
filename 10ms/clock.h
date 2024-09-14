/**
   @file clock.h
   @brief a small procedure to get CPU/reference cycle
 */

/* these two are Linux-specific.
   make them zero on other OSes */
#if __linux__
#define HAVE_PERF_EVENT 1
#define HAVE_CLOCK_GETTIME 1
#else
#define HAVE_PERF_EVENT 0
#define HAVE_CLOCK_GETTIME 0
#endif


#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <pthread.h>

/**
   @brief read reference clock
  */
static inline long long rdtsc() {
  long long u;
  asm volatile ("rdtsc;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"(u)::"%rdx");
  return u;
}

#if HAVE_PERF_EVENT
#include <linux/perf_event.h>
#include <asm/unistd.h>

/**
   this is a wrapper to Linux system call perf_event_open
 */
static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  int ret;
  ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                group_fd, flags);
  return ret;
}
#endif

/**
   @brief a structure encapsulating a performance counter
 */
typedef struct {
  pthread_t tid;                /**< thread ID this is valid for */
#if HAVE_PERF_EVENT
  int fd;                       /**< what perf_event_open returned  */
#endif
} cpu_clock_counter_t;

/**
   @brief make a cpu_clock_counter
   @details 
   cpu_clock_counter_t t = mk_cpu_clock_counter();
   long c0 = cpu_clock_counter_get(t);
      ... do something ...
   long c1 = cpu_clock_counter_get(t);
   long dc = c1 - c0; <- the number of CPU clocks between c0 and c1
  */
static cpu_clock_counter_t mk_cpu_clock_counter() {
  pthread_t tid = pthread_self();
#if HAVE_PERF_EVENT
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(struct perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(struct perf_event_attr);
  //pe.config = PERF_COUNT_HW_INSTRUCTIONS;
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  
  int fd = perf_event_open(&pe, 0, -1, -1, 0);
  if (fd == -1) {
    perror("perf_event_open");
  }
  if (fd != -1 && ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1) {
    perror("ioctl");
    close(fd);
    fd = -1;
  }
  if (fd != -1 && ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
    perror("ioctl");
    close(fd);
    fd = -1;
  }
  if (fd == -1) {
    fprintf(stderr,
            "%s:%d:warning: the environment does not support perf_event."
            " CPU clock cannot be obtained\n", __FILE__, __LINE__);
  }
  cpu_clock_counter_t cc = { tid, fd };
#else
  fprintf(stderr, "%s:%d:warning: OS does not support perf_event. CPU clock cannot be obtained\n", __FILE__, __LINE__);
  cpu_clock_counter_t cc = { tid };
#endif
  return cc;
}

/**
   @brief destroy a cpu clock counter
  */
static void cpu_clock_counter_destroy(cpu_clock_counter_t cc) {
#if HAVE_PERF_EVENT
  if (cc.fd != -1) {
    close(cc.fd);
  }
#else
  (void)cc;
#endif
}

/**
   @brief get CPU clock
  */
static long long cpu_clock_counter_get(cpu_clock_counter_t cc) {
  pthread_t tid = pthread_self();
  if (tid != cc.tid) {
    fprintf(stderr,
            "%s:%d:cpu_clock_counter_get: the caller thread (%ld)"
            " is invalid (!= %ld)\n", 
            __FILE__, __LINE__, (long)tid, (long)cc.tid);
    return -1;
  } else {
    long long c;
#if HAVE_PERF_EVENT
    if (cc.fd == -1) {
      c = 0; // rdtsc();
    } else {
      ssize_t rd = read(cc.fd, &c, sizeof(long long));
      if (rd == -1) {
        perror("read"); 
        exit(EXIT_FAILURE);
      }
      assert(rd == sizeof(long long));
    }
#else
    c = rdtsc();
#endif
    return c;
  }
}

/**
   @brief get ns
  */
static inline long long cur_time_ns() {
#if HAVE_CLOCK_GETTIME
  struct timespec ts[1];
  if (clock_gettime(CLOCK_REALTIME, ts) == -1) {
    perror("clock_gettime"); exit(1);
  }
  return ts->tv_sec * 1000000000L + ts->tv_nsec;
#else
  /* resort to us-level timer */
  struct timeval tv[1];
  if (gettimeofday(tv, 0) == -1) {
    perror("gettimeofday"); exit(1);
  }
  return tv->tv_sec * 1000000000L + tv->tv_usec * 1000L;
#endif
}

typedef struct {
  cpu_clock_counter_t cc;
} clock_counters_t;

typedef struct {
  long long ref_clock;
  long long cpu_clock;
  long long wall_clock;
} clocks_t;

static inline clock_counters_t mk_clock_counters() {
  clock_counters_t cc;
  cc.cc = mk_cpu_clock_counter();
  return cc;
}

static inline clocks_t clock_counters_get(clock_counters_t cc) {
  clocks_t c;
  c.ref_clock = rdtsc();
  c.wall_clock = cur_time_ns();
  c.cpu_clock = cpu_clock_counter_get(cc.cc);
  return c;
}

static inline void clock_counters_destroy(clock_counters_t cc) {
  cpu_clock_counter_destroy(cc.cc);
}
