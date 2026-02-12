/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __PARAMETER_H_
#define __PARAMETER_H_

typedef struct {
  char *filename;
  int nx, ny, nz;
  int itermax;
  double eps;
} Parameter;

void initParameter(Parameter *);
void readParameter(Parameter *, const char *);
void printParameter(Parameter *);

#ifdef CRS
#define FMT "CRS"
#endif
#ifdef SCS
#define FMT "SCS"
#endif
#ifdef CCRS
#define FMT "CCRS"
#endif

#endif
