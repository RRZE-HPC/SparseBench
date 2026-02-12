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

typedef enum { CG = 0, SPMV, GMRES, CHEBFD, NUMTYPES } BenchEnumType;
extern int BenchType;

#define HELPTEXT                                                                         \
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
  "  -e <float>  Convergence criteria epsilon. Default 0.0.\n"

extern void parseArguments(CommType *, Parameter *, int, char **);

#endif /*CLI_H*/
