/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "comm.h"
#include "solver.h"
#include "util.h"

void waxpby(const CG_UINT n,
    const CG_FLOAT alpha,
    const CG_FLOAT *restrict x,
    const CG_FLOAT beta,
    const CG_FLOAT *restrict y,
    CG_FLOAT *const w)
{
  if (alpha == 1.0) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      w[i] = x[i] + beta * y[i];
    }
  } else if (beta == 1.0) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }
}

void ddot(const CG_UINT n,
    const CG_FLOAT *restrict x,
    const CG_FLOAT *restrict y,
    CG_FLOAT *restrict result)
{
  CG_FLOAT sum = 0.0;

  if (y == x) {
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < n; i++) {
      sum += x[i] * x[i];
    }
  } else {
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < n; i++) {
      sum += x[i] * y[i];
    }
  }

  commReduction(&sum, SUM);
  *result = sum;
}
