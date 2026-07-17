/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_NUM_THREADS 128
#define gettid() syscall(SYS_gettid)

static int getProcessorID(cpu_set_t *cpu_set)
{
  int processorId;

  for (processorId = 0; processorId < MAX_NUM_THREADS; processorId++) {
    if (CPU_ISSET(processorId, cpu_set)) {
      break;
    }
  }
  return processorId;
}

#ifdef _OPENMP
int affinity_getProcessorId()
{
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  sched_getaffinity(gettid(), sizeof(cpu_set_t), &cpu_set);

  return getProcessorID(&cpu_set);
}

void affinity_pinThread(int processorId)
{
  cpu_set_t cpuset;
  pthread_t thread;

  thread = pthread_self();
  CPU_ZERO(&cpuset);
  CPU_SET(processorId, &cpuset);
  pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
}

void affinity_pinProcess(int processorId)
{
  cpu_set_t cpuset;

  CPU_ZERO(&cpuset);
  CPU_SET(processorId, &cpuset);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}
#endif /*_OPENMP*/

void affinity_getmask(void)
{
  int i = 0;
  cpu_set_t my_set;
  int nproc = sysconf(_SC_NPROCESSORS_ONLN);
  CPU_ZERO(&my_set);
  sched_getaffinity(gettid(), sizeof(cpu_set_t), &my_set);
  for (i = 0; i < nproc; i++) {
    printf("%02d ", i);
  }
  printf("\n");
  for (i = 0; i < nproc; i++) {
    if (CPU_ISSET(i, &my_set)) {
      printf(" * ");
    } else {
      printf(" - ");
    }
  }
  printf("\n");
}

#endif /*__linux__*/
