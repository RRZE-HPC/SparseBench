/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */

/*
 * GPU section timer (section_timer.h): event pairs recorded on each
 * section's stream — the legacy default stream by default, which brackets
 * all work issued in program order because every other stream here is
 * blocking (cuda_matrix_stream.cu); non-blocking streams need
 * sectionTimerSetStream. Records never synchronize; drainSection folds the
 * intervals in. Both events of a pair share a stream in order, so waiting
 * for the stop implies the start finished — what EventElapsedTime
 * requires.
 */
#include "section_timer.h"

#include <stdio.h>
#include <stdlib.h>

#include "gpu_backend.h"

typedef struct {
  gpuEvent_t start;
  gpuEvent_t stop;
  gpuStream_t stream; /* queue both records go on; NULL = legacy default */
  int pending;        /* a pair is recorded but not yet folded into totalSec */
  double totalSec;
  unsigned long long counts;
} SectionTimerPart;

struct SectionTimer {
  int nsections;
  SectionTimerPart *sec;
};

static int sectionOk(const SectionTimer *t, int section)
{
  return t != NULL && section >= 0 && section < t->nsections;
}

static void drainSection(SectionTimer *t, int section)
{
  SectionTimerPart *s = &t->sec[section];
  if (!s->pending) {
    return;
  }
  GPU_SAFE_CALL(gpuEventSynchronize(s->stop));
  float ms = 0.0f;
  GPU_SAFE_CALL(gpuEventElapsedTime(&ms, s->start, s->stop));
  s->totalSec += (double)ms * 1.0e-3;
  s->counts++;
  s->pending = 0;
}

extern "C" SectionTimer *sectionTimerCreate(int nsections)
{
  if (nsections <= 0) {
    return NULL;
  }
  SectionTimer *t = (SectionTimer *)calloc(1, sizeof(SectionTimer));
  if (t == NULL) {
    return NULL;
  }
  t->nsections = nsections;
  t->sec       = (SectionTimerPart *)calloc((size_t)nsections, sizeof(SectionTimerPart));
  if (t->sec == NULL) {
    free(t);
    return NULL;
  }
  for (int i = 0; i < nsections; i++) {
    GPU_SAFE_CALL(gpuEventCreate(&t->sec[i].start));
    GPU_SAFE_CALL(gpuEventCreate(&t->sec[i].stop));
    t->sec[i].stream = (gpuStream_t)0;
  }
  return t;
}

extern "C" void sectionTimerFree(SectionTimer *t)
{
  if (t == NULL) {
    return;
  }
  for (int i = 0; i < t->nsections; i++) {
    GPU_SAFE_CALL(gpuEventDestroy(t->sec[i].start));
    GPU_SAFE_CALL(gpuEventDestroy(t->sec[i].stop));
  }
  free(t->sec);
  free(t);
}

extern "C" void sectionTimerStart(SectionTimer *t, int section)
{
  if (!sectionOk(t, section)) {
    return;
  }
  drainSection(t, section); /* else the previous interval is lost */
  GPU_SAFE_CALL(gpuEventRecord(t->sec[section].start, t->sec[section].stream));
}

extern "C" void sectionTimerStop(SectionTimer *t, int section)
{
  if (!sectionOk(t, section)) {
    return;
  }
  GPU_SAFE_CALL(gpuEventRecord(t->sec[section].stop, t->sec[section].stream));
  t->sec[section].pending = 1;
}

extern "C" void sectionTimerSetStream(
    SectionTimer *t, int section, SectionTimerStream stream)
{
  if (!sectionOk(t, section)) {
    return;
  }
  /* Fold in pending pairs on the old stream first. */
  drainSection(t, section);
  t->sec[section].stream = (gpuStream_t)stream;
}

extern "C" void sectionTimerSync(SectionTimer *t)
{
  if (t == NULL) {
    return;
  }
  for (int i = 0; i < t->nsections; i++) {
    drainSection(t, i);
  }
}

extern "C" double sectionTimerGetSec(SectionTimer *t, int section)
{
  if (!sectionOk(t, section)) {
    return 0.0;
  }
  drainSection(t, section);
  return t->sec[section].totalSec;
}

extern "C" unsigned long long sectionTimerGetCount(SectionTimer *t, int section)
{
  if (!sectionOk(t, section)) {
    return 0ull;
  }
  drainSection(t, section);
  return t->sec[section].counts;
}
