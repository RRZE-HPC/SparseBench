/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef CLI_H
#define CLI_H

#include <stdbool.h>
#include <stddef.h>

#include "comm.h"
#include "parameter.h"

typedef enum { CG = 0, SPMV, SPMMV, GMRES, CHEBFD, NUMTYPES } BenchEnumType;
extern int BenchType;

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define HELPTEXT_BASE                                                                    \
  "Usage: sparseBench [options]\n\n"                                                     \
  "Options:\n"                                                                           \
  "  -h         Show this help text\n"                                                   \
  "  -c <file name>   Convert MM matrix to binary matrix file.\n"                        \
  "  -f <parameter file>   Load options from a parameter file\n"                         \
  "  -m <MM matrix>   Load a matrix market file\n"                                       \
  "  -t <bench type>   Benchmark type, can be cg, spmv, or gmres. Default "              \
  "cg.\n"                                                                                \
  "  -x <int>   Size in x for generated matrix, ignored if MM file is "                  \
  "loaded. Default 100.\n"                                                               \
  "  -y <int>   Size in y for generated matrix, ignored if MM file is "                  \
  "loaded. Default 100.\n"                                                               \
  "  -z <int>   Size in z for generated matrix, ignored if MM file is "                  \
  "loaded. Default 100.\n"                                                               \
  "  -i <int>   Number of solver iterations. Default 150.\n"                             \
  "  -w <int>   Width of block vector for SpMMV\n"                                       \
  "  -v         Enable verbose output\n"                                                 \
  "  -e <float>  Convergence criteria epsilon. Default 0.0.\n"                           \
  "  -d <int>   GPU device index (multi-GPU nodes). Default 0.\n"

#define BASE_ARGS_COMMON "hc:t:f:m:x:y:z:i:e:w:v:d:"

#ifdef SCS

#define BASE_ARGS BASE_ARGS_COMMON "k:s:"
// clang-format off
#define HELPTEXT                                                                         \
  HELPTEXT_BASE                                                                          \
  "  -k <int>  Chunk size value for SELL-c-sigma. Default " STR(SELL_CHUNK) ".\n"        \
  "  -s <int>  Sigma size value for SELL-c-sigma. Default " STR(SELL_SIGMA) ".\n"

#else

#define BASE_ARGS BASE_ARGS_COMMON
#define HELPTEXT HELPTEXT_BASE

#endif

// clang-format on

extern void parseArguments(CommType *, Parameter *, int, char **);

#endif /*CLI_H*/
