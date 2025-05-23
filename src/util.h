/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __UTIL_H_
#define __UTIL_H_

#include <string.h>

#define HLINE                                                                  \
  "----------------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

#ifndef IS_EQUAL
#define IS_EQUAL(a, b) (strcmp((a), (b)) == 0)
#endif

#define DEBUG_MESSAGE debug_printf

#ifndef MAXLINE
#define MAXLINE 4096
#endif

#if UINT_TYPE == 1
#define CG_UINT      unsigned int
#define MPI_INT_TYPE MPI_UNSIGNED
#define UINT_STRING  "unsigned int"
#else
#define CG_UINT      unsigned long long int
#define MPI_INT_TYPE MPI_UNSIGNED_LONG_LONG
#define UINT_STRING  "unsigned long long int"
#endif

#if PRECISION == 1
#define CG_FLOAT         float
#define MPI_FLOAT_TYPE   MPI_FLOAT
#define PRECISION_STRING "single"
#else
#define CG_FLOAT         double
#define MPI_FLOAT_TYPE   MPI_DOUBLE
#define PRECISION_STRING "double"
#endif

extern char* changeFileEnding(char* filename, char* newEnding);

#endif
