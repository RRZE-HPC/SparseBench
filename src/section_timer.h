/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __SECTION_TIMER_H_
#define __SECTION_TIMER_H_

/*
 * Section timer: one start/stop pair per caller-defined section,
 * accumulated over many intervals; accessors return seconds.
 *
 * GPU builds (cuda_section_timer.cu): CUDA/HIP events. Start/Stop only
 * enqueue records — totals are folded in later (Sync, accessors, or the
 * next Start). Records default to the legacy default stream, which
 * brackets all work issued in program order because every other stream
 * here is blocking (cuda_matrix_stream.cu). With non-blocking streams the
 * legacy bracket breaks silently — pin such sections to their work stream
 * via sectionTimerSetStream.
 *
 * CPU builds (section_timer.c): getTimeStamp() pairs.
 *
 * ENABLE_SECTIMER=false compiles every SECTION_TIMER_* macro to dead code
 * (the functions always exist, so tests can call them directly). gcc-safe:
 * no gpu_backend.h; the struct is opaque.
 */

typedef struct SectionTimer SectionTimer;

/* Stream handle for SetStream: GPU callers pass a gpuStream_t (implicit
 * conversion); CPU builds ignore it. */
typedef void *SectionTimerStream;

#ifdef __cplusplus
extern "C" {
#endif

/* nsections pairs; NULL only on allocation failure. */
SectionTimer *sectionTimerCreate(int nsections);

void sectionTimerFree(SectionTimer *t); /* NULL-tolerant */

/* Open / close one interval. GPU: async event record. CPU: timestamp. */
void sectionTimerStart(SectionTimer *t, int section);
void sectionTimerStop(SectionTimer *t, int section);

/* Fold pending intervals into the totals (no-op on CPU). */
void sectionTimerSync(SectionTimer *t);

/* Pin a section's records to one stream (GPU; default NULL = legacy
 * stream). Needed for non-blocking streams. Drains pending pairs first;
 * don't call between a Start and its Stop. Ignored on CPU. */
void sectionTimerSetStream(SectionTimer *t, int section, SectionTimerStream stream);

/* Accumulated seconds / interval count (drain pending first). */
double sectionTimerGetSec(SectionTimer *t, int section);
unsigned long long sectionTimerGetCount(SectionTimer *t, int section);

#ifdef __cplusplus
}
#endif

#if defined(USE_SECTION_TIMER)

#define SECTION_TIMER_ON 1
#define SECTION_TIMER_CREATE(nsections) sectionTimerCreate(nsections)
#define SECTION_TIMER_FREE(t) sectionTimerFree(t)
#define SECTION_TIMER_START(t, section) sectionTimerStart((t), (section))
#define SECTION_TIMER_STOP(t, section) sectionTimerStop((t), (section))
#define SECTION_TIMER_SYNC(t) sectionTimerSync(t)
#define SECTION_TIMER_SET_STREAM(t, section, stream)                                     \
  sectionTimerSetStream((t), (section), (stream))
#define SECTION_TIMER_SEC(t, section) sectionTimerGetSec((t), (section))
#define SECTION_TIMER_COUNT(t, section) sectionTimerGetCount((t), (section))

#else /* disabled: (void)-cast the args so locals stay "used" under -Wall */

#define SECTION_TIMER_ON 0
#define SECTION_TIMER_CREATE(nsections) ((SectionTimer *)0)
#define SECTION_TIMER_FREE(t) ((void)(t))
#define SECTION_TIMER_START(t, section) ((void)(t), (void)(section))
#define SECTION_TIMER_STOP(t, section) ((void)(t), (void)(section))
#define SECTION_TIMER_SYNC(t) ((void)(t))
#define SECTION_TIMER_SET_STREAM(t, section, stream)                                     \
  ((void)(t), (void)(section), (void)(stream))
#define SECTION_TIMER_SEC(t, section) (0.0)
#define SECTION_TIMER_COUNT(t, section) (0ull)

#endif /* USE_SECTION_TIMER */

#endif /* __SECTION_TIMER_H_ */
