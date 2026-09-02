/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __PARAMETER_H_
#define __PARAMETER_H_

#include "allocate.h" /* AllocType */

// NTS : CHEB_FD params
typedef struct {
  double a;        // spectrum lower bound  (valid iff have_bounds)
  double b;        // spectrum upper bound  (valid iff have_bounds)
  double lam_lo;   // target interval lower bound
  double lam_hi;   // target interval upper bound
  int Np;          // filter polynomial degree
  int NS;          // number of search vectors
  int kernel;      // 0=none 1=Fejer 2=Jackson 3=Lanczos
  int mu;          // Lanczos kernel exponent (use 2)
  int have_bounds; // 1 => both cheb_a/cheb_b user-supplied, else Gershgorin
  int have_target; // 1 => both cheb_lam_lo/cheb_lam_hi user-supplied
} ChebFDParam;

typedef struct {
  char *filename;
  int nx, ny, nz;
  int itermax;
  double eps;
#ifdef SCS
  int C;
  int Sigma;
#endif
  int blockwidth;
  int verbose;
  int device;
  AllocType allocType; /* GPU buffer placement; ignored by the CPU build */
  ChebFDParam cheb;    // NTS : compostion to keep struct clean
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
