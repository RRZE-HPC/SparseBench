/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __ALLOCATE_H_
#define __ALLOCATE_H_
#include <stdlib.h>

// Where buffers handed out by allocate() live on GPU builds.
typedef enum {
  ALLOC_PAGEABLE = 0, // malloc
  ALLOC_EXPLICIT,     // allocate: cudaMallocHost | allocateDevice: cudaMalloc
  ALLOC_MANAGED       // cudaMallocManaged for both
} AllocType;

#ifdef __cplusplus
extern "C" {
#endif

const char *allocTypeName(AllocType type);

AllocType allocTypeFromName(const char *name);

extern void *allocate(size_t alignment, size_t bytesize);

extern void deallocate(void *ptr);

// Device-resident variant of allocate()/deallocate()
extern void *allocateDevice(size_t bytesize);

extern void deallocateDevice(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
