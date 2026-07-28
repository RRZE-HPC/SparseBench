/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __SOLVER_H_
#define __SOLVER_H_
#include "comm.h"
#include "parameter.h"
#include "util.h"
#include "vtype.h"

extern int solveCG(CommType *comm, Parameter *param, Matrix *m);

typedef struct {
  V_ELE *r;
  V_ELE *p;
  V_ELE *ap;
  V_ELE *x;
  V_ELE *b;
  V_ELE *xexact;
  V_ELE *permTmp;
} CGData;

// centralized methods to allocate and deallocate
extern void allocCGData(CGData *d, Matrix *m, bool useXexact);
extern void freeCGData(CGData *d);

// extern void solverCheckResidual(Solver* s, Comm* c);
extern void spMVM(Matrix *m, const V_ELE *restrict x, V_ELE *restrict y);
extern void spMMVM(Matrix *m, const DMatrix *x, DMatrix *y);

extern void waxpby(const CG_UINT n,
    const V_ELE alpha,
    const V_ELE *x,
    const V_ELE beta,
    const V_ELE *y,
    V_ELE *const w);

/* x/y may alias (ddot branches on y == x); `result` must not alias either. */
extern void ddot(const CG_UINT n, const V_ELE *e, const V_ELE *y, V_ELE *restrict result);

/* Strided variants for dense-block ops (e.g. ChebFD columns of row-major DMatrix
 * blocks); the stride-1 versions above stay the hot path for contiguous vectors.
 * waxpby_stride: w may alias x/y only if the aliased vectors share a stride. */
extern void waxpby_stride(const CG_UINT n,
    const V_ELE alpha,
    const V_ELE *x,
    const CG_UINT incx,
    const V_ELE beta,
    const V_ELE *y,
    const CG_UINT incy,
    V_ELE *const w,
    const CG_UINT incw);

extern void ddot_stride(const CG_UINT n,
    const V_ELE *x,
    const CG_UINT incx,
    const V_ELE *y,
    const CG_UINT incy,
    V_ELE *restrict result);
#endif // __SOLVER_H_
