/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CHEBFDSOLVER_H_
#define __CHEBFDSOLVER_H_

#include "matrix.h"
#include "comm.h"
#include "parameter.h"

extern int solveChebFD(CommType *comm, Parameter *param, Matrix *A);

// Gershgorin circle-theorem estimate of the spectrum bounds [a,b] of A.
extern void gershgorinBounds(CommType *comm, Matrix *A, double *a_out, double *b_out);

// Centralized allocation for ChebFD work buffers so nothing is allocated
// during the iterations. 
typedef struct {
  DMatrix Y;   // search vectors block (vecRows x NS)
  DMatrix AY;  // A*Y block (vecRows x NS)
  DMatrix u;   // block scratch
  DMatrix w;   // block scratch
  DMatrix tmp; // block scratch
  V_ELE *vbuf;  // single Ritz vector (nr)
  V_ELE *avbuf; // single A*Ritz vector (nr)
  double *H;
  double *eval;
  double *evec;
  int NS;
} ChebData;

extern void allocChebData(ChebData *d, Matrix *m, int NS);
extern void freeChebData(ChebData *d);

#endif // __CHEBFDSOLVER_H_
