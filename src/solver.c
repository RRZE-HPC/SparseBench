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

/* Not restrict: in-place callers alias w with x (CGSolver `x += alpha*p`) or
 * with y (`p = r + beta*p`), which restrict would make undefined. `omp simd`
 * asserts the elementwise pattern instead, so the loops still vectorize fully
 * rather than falling back to the scalar path of a runtime alias check. */
void waxpby(const CG_UINT n,
    const V_ELE alpha,
    const V_ELE *x,
    const V_ELE beta,
    const V_ELE *y,
    V_ELE *const w)
{
  if (alpha == 1.0) {
#pragma omp parallel for simd schedule(OMP_SCHEDULE)
    for (CG_UINT i = 0; i < n; i++) {
      w[i] = x[i] + beta * y[i];
    }
  } else if (beta == 1.0) {
#pragma omp parallel for simd schedule(OMP_SCHEDULE)
    for (CG_UINT i = 0; i < n; i++) {
      w[i] = alpha * x[i] + y[i];
    }
  } else {
#pragma omp parallel for simd schedule(OMP_SCHEDULE)
    for (CG_UINT i = 0; i < n; i++) {
      w[i] = alpha * x[i] + beta * y[i];
    }
  }
}

/* x/y are not restrict: the function branches on `y == x`, which restrict would
 * make undefined. `result` must not alias either and keeps its restrict. */
void ddot(const CG_UINT n, const V_ELE *x, const V_ELE *y, V_ELE *restrict result)
{
  V_ELE sum = 0.0;

#ifdef USE_COMPLEX
  if (y == x) {
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
    for (CG_UINT i = 0; i < n; i++) {
      sum += VCONJ(x[i]) * x[i];
    }
  } else {
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
    for (CG_UINT i = 0; i < n; i++) {
      sum += VCONJ(x[i]) * y[i];
    }
  }
#else
  if (y == x) {
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
    for (CG_UINT i = 0; i < n; i++) {
      sum += x[i] * x[i];
    }
  } else {
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
    for (CG_UINT i = 0; i < n; i++) {
      sum += x[i] * y[i];
    }
  }
#endif

  commReductionV(&sum, SUM);
  *result = sum;
}

/* Not restrict: callers alias w with x for in-place updates (MGS
 * `e[:,k] -= coef*e[:,j]`). PRECONDITION: aliased vectors must share a stride;
 * otherwise w[i*incw] can clobber a source element no iteration has read yet. */
void waxpby_stride(const CG_UINT n,
    const V_ELE alpha,
    const V_ELE *x,
    const CG_UINT incx,
    const V_ELE beta,
    const V_ELE *y,
    const CG_UINT incy,
    V_ELE *const w,
    const CG_UINT incw)
{
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (CG_UINT i = 0; i < n; i++) {
    w[i * incw] = alpha * x[i * incx] + beta * y[i * incy];
  }
}

/* x/y are not restrict: MGS passes x == y to form a column norm. */
void ddot_stride(const CG_UINT n,
    const V_ELE *x,
    const CG_UINT incx,
    const V_ELE *y,
    const CG_UINT incy,
    V_ELE *restrict result)
{
  V_ELE sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(OMP_SCHEDULE)
  for (CG_UINT i = 0; i < n; i++) {
    sum += x[i * incx] * y[i * incy];
  }
  commReductionV(&sum, SUM);
  *result = sum;
}
