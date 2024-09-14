/**
   @file hello_tbb.c
  */
#include <stdio.h>
#include <sched.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>
#include <time.h>

double cur_time() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec + ts->tv_nsec * 1.0e-9;
}

void waste_a_sec() {
  double t0 = cur_time();
  while (cur_time() < t0 + 1) {
  }
}

void make_tasks(long a, long b, long n_tasks) {
  if (b - a == 1) {
    int cpu = sched_getcpu();
    waste_a_sec();
    printf("hello from task %ld of %ld on %d\n", a, n_tasks, cpu);
  } else {
    tbb::task_group tg;
    long c = (a + b) / 2;
    tg.run([&] { make_tasks(a, c, n_tasks); });
    make_tasks(c, b, n_tasks);
    tg.wait();
  }
}

int main(int argc, char ** argv) {
  int i = 1;
  long n_tasks = (argc > i ? atol(argv[1]) : 10); i++;
  long n_workers = (argc > i ? atol(argv[1]) : 10); i++;
  tbb::task_scheduler_init * tsi = new tbb::task_scheduler_init(n_workers);
  make_tasks(0, n_tasks, n_tasks);
  delete tsi;
  return 0;
}
