/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocate.h"
#ifdef _GPU
#include "cuda_kernels.h"
#endif

const char *allocTypeName(AllocType type)
{
  switch (type) {
  case ALLOC_PAGEABLE:
    return "pageable";
  case ALLOC_EXPLICIT:
    return "explicit";
  case ALLOC_MANAGED:
    return "managed";
  default:
    return "unknown";
  }
}

AllocType allocTypeFromName(const char *name)
{
  if (strcmp(name, "pageable") == 0)
    return ALLOC_PAGEABLE;
  if (strcmp(name, "explicit") == 0)
    return ALLOC_EXPLICIT;
  if (strcmp(name, "managed") == 0)
    return ALLOC_MANAGED;
  fprintf(stderr,
      "Error: unknown allocation type '%s' "
      "(expected pageable, explicit or managed)\n",
      name);
  exit(EXIT_FAILURE);
}

void *allocate(size_t alignment, size_t bytesize)
{
#ifdef _GPU
  (void)alignment;
  return gpu_allocate(bytesize);
#else
  int errorCode;
  void *ptr;

  errorCode = posix_memalign(&ptr, alignment, bytesize);

  if (errorCode) {
    if (errorCode == EINVAL) {
      fprintf(stderr, "Error: Alignment parameter is not a power of two\n");
      exit(EXIT_FAILURE);
    }
    if (errorCode == ENOMEM) {
      fprintf(stderr, "Error: Insufficient memory to fulfill the request\n");
      exit(EXIT_FAILURE);
    }
  }

  if (ptr == NULL) {
    fprintf(stderr, "Error: posix_memalign failed!\n");
    exit(EXIT_FAILURE);
  }

  return ptr;
#endif
}

extern void deallocate(void *ptr)
{
#ifdef _GPU
  gpu_free(ptr);
#else
  free(ptr);
#endif
}

extern void *allocateDevice(size_t bytesize)
{
#ifdef _GPU
  return gpu_allocate_device(bytesize);
#else
  return allocate(ARRAY_ALIGNMENT, bytesize);
#endif
}

extern void deallocateDevice(void *ptr)
{
#ifdef _GPU
  gpu_free_device(ptr);
#else
  free(ptr);
#endif
}

extern void *allocateHost(size_t bytesize)
{
#ifdef _GPU
  return gpu_allocate_host(bytesize);
#else
  return allocate(ARRAY_ALIGNMENT, bytesize);
#endif
}

extern void deallocateHost(void *ptr)
{
#ifdef _GPU
  gpu_free_host(ptr);
#else
  free(ptr);
#endif
}
