/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of CG-Bench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __SOLVER_H_
#define __SOLVER_H_
#include "comm.h"
#include "matrix.h"
#include "parameter.h"
#include "util.h"

typedef struct {
  Matrix A;
  CG_FLOAT* x;
  CG_FLOAT* b;
  CG_FLOAT* xexact;
} Solver;

void initSolver(Solver* s, Comm* c, Parameter*);
void spMVM(Matrix* m, const CG_FLOAT* restrict x, CG_FLOAT* restrict y);

void waxpby(const CG_UINT n,
    const CG_FLOAT alpha,
    const CG_FLOAT* restrict x,
    const CG_FLOAT beta,
    const CG_FLOAT* restrict y,
    double* restrict w);

void ddot(const CG_UINT n,
    const CG_FLOAT* restrict e,
    const CG_FLOAT* restrict y,
    CG_FLOAT* restrict result);
#endif // __SOLVER_H_
