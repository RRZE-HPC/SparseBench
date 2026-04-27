/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "allocate.h"
#ifdef _GPU
#include "cuda/cuda_kernels.h"
#endif

void *allocate(size_t alignment, size_t bytesize)
{
#ifdef _GPU
  (void)alignment;
  return gpu_allocate_managed(bytesize);
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
  gpu_free_managed(ptr);
#else
  free(ptr);
#endif
}
