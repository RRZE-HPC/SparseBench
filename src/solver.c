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
#include "vtype.h"

void waxpby(const CG_UINT n,
    const V_ELE alpha,
    const V_ELE *restrict x,
    const V_ELE beta,
    const V_ELE *restrict y,
    V_ELE *const w)
{
  if (alpha == 1.0) {
#pragma omp parallel for schedule(OMP_SCHEDULE)
    for (int i = 0; i < n; i++) {
      w[i] = x[i] + beta * y[i];
    }
  } else if (beta == 1.0) {
#pragma omp parallel for schedule(OMP_SCHEDULE)
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  } else {
#pragma omp parallel for schedule(OMP_SCHEDULE)
    for (int i = 0; i < n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }
}

void ddot(const CG_UINT n,
    const V_ELE *restrict x,
    const V_ELE *restrict y,
    V_ELE *restrict result)
{
  V_ELE sum = 0.0;

#ifdef USE_COMPLEX
  if (y == x) {
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
    for (int i = 0; i < n; i++) {
      sum += VCONJ(x[i]) * x[i];
    }
  } else {
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
    for (int i = 0; i < n; i++) {
      sum += VCONJ(x[i]) * y[i];
    }
  }
#else
  if (y == x) {
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
    for (int i = 0; i < n; i++) {
      sum += x[i] * x[i];
    }
  } else {
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
    for (int i = 0; i < n; i++) {
      sum += x[i] * y[i];
    }
  }
#endif

  commReductionV(&sum, SUM);
  *result = sum;
}
