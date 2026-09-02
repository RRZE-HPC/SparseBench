/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * CPU section timer (section_timer.h): getTimeStamp() pairs. The whole
 * body compiles out on GPU toolchains, where the event twin
 * (cuda_section_timer.cu) provides the symbols.
 */
#if !defined(RUNTIME_BACKEND_IS_CUDA) && !defined(RUNTIME_BACKEND_IS_HIP)

#include <stdlib.h>

#include "section_timer.h"
#include "timing.h"

struct SectionTimer {
  int nsections;
  double *start; /* last Start timestamp per section */
  double *total; /* accumulated seconds per section */
  unsigned long long *counts;
};

static int sectionOk(const SectionTimer *t, int section)
{
  return t != NULL && section >= 0 && section < t->nsections;
}

SectionTimer *sectionTimerCreate(int nsections)
{
  if (nsections <= 0) {
    return NULL;
  }
  SectionTimer *t = (SectionTimer *)calloc(1, sizeof(SectionTimer));
  if (t == NULL) {
    return NULL;
  }
  t->nsections = nsections;
  t->start     = (double *)calloc((size_t)nsections, sizeof(double));
  t->total     = (double *)calloc((size_t)nsections, sizeof(double));
  t->counts = (unsigned long long *)calloc((size_t)nsections, sizeof(unsigned long long));
  if (t->start == NULL || t->total == NULL || t->counts == NULL) {
    sectionTimerFree(t);
    return NULL;
  }
  return t;
}

void sectionTimerFree(SectionTimer *t)
{
  if (t == NULL) {
    return;
  }
  free(t->start);
  free(t->total);
  free(t->counts);
  free(t);
}

void sectionTimerStart(SectionTimer *t, int section)
{
  if (!sectionOk(t, section)) {
    return;
  }
  t->start[section] = getTimeStamp();
}

void sectionTimerStop(SectionTimer *t, int section)
{
  if (!sectionOk(t, section)) {
    return;
  }
  t->total[section] += getTimeStamp() - t->start[section];
  t->counts[section]++;
}

void sectionTimerSync(SectionTimer *t)
{
  (void)t; /* host timestamps are already folded in on Stop */
}

void sectionTimerSetStream(SectionTimer *t, int section, SectionTimerStream stream)
{
  (void)t;
  (void)section;
  (void)stream; /* stream placement is a GPU-only concept */
}

double sectionTimerGetSec(SectionTimer *t, int section)
{
  if (!sectionOk(t, section)) {
    return 0.0;
  }
  return t->total[section];
}

double sectionTimerGetWallSec(SectionTimer *t, int section)
{
  /* Host timestamps are the only clock here, so wall == event total. */
  return sectionTimerGetSec(t, section);
}

unsigned long long sectionTimerGetCount(SectionTimer *t, int section)
{
  if (!sectionOk(t, section)) {
    return 0ull;
  }
  return t->counts[section];
}

#endif /* !RUNTIME_BACKEND_IS_CUDA && !RUNTIME_BACKEND_IS_HIP */
