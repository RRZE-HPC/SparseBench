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
// extern void solverCheckResidual(Solver* s, Comm* c);
extern void spMVM(Matrix *m, const V_ELE *restrict x, V_ELE *restrict y);
extern void spMMVM(Matrix *m, const DMatrix *x, DMatrix *y);

extern void waxpby(const CG_UINT n,
    const V_ELE alpha,
    const V_ELE *restrict x,
    const V_ELE beta,
    const V_ELE *restrict y,
    V_ELE *restrict w);

extern void ddot(const CG_UINT n,
    const V_ELE *restrict e,
    const V_ELE *restrict y,
    V_ELE *restrict result);
#endif // __SOLVER_H_
