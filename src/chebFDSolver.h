/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CHEBFDSOLVER_H_
#define __CHEBFDSOLVER_H_

#include "chebFilter.h"
#include "comm.h"
#include "matrix.h"
#include "parameter.h"

extern int solveChebFD(CommType *comm, Parameter *param, Matrix *A);

// Gershgorin estimate of the spectrum bounds [a,b] of A.
extern void gershgorinBounds(CommType *comm, Matrix *A, double *a_out, double *b_out);

// Internal ChebFD steps (Alg. 3.1 in the paper), exposed from chebFDSolver.c
// for unit tests only; not part of the stable API.

// Step 5 (Fig. 6): Y <- p(H)Y via the Chebyshev recurrence; U/W are scratch
// blocks shaped like Y on entry (see allocChebData).
extern void applyFilter(Matrix *A, ChebFilter *f, DMatrix *X, DMatrix *U, DMatrix *W);

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
// GPU only: install the matrix-streaming context + subblock width used by
// applyFilter and rayleighRitz (see cuda_matrix_stream.cu). NULL disables
// streaming; nb=0 means full-width kernels. solveChebFD manages this itself
// from the gpu_stream_mb / cheb_nb parameters; the setter exists so the
// streaming unit tests can drive applyFilter directly.
struct GpuMatrixStream;
extern void chebFDSetMatrixStream(struct GpuMatrixStream *s, int nb);
#endif

// Step 6: rank-revealing CGS2 of the nr x nc row-major block e; returns the
// accepted rank m <= nc and compacts/repacks columns in place.
extern int orthoMGS(CG_UINT nr, V_ELE *e, int nc, double tol);

// Step 7 (inner): H = Y^T AY, written exactly symmetric; shared by the
// host and device paths (see kernel_dispatch.h).
extern void gramYtAY(CG_UINT nr, int m, const V_ELE *Ye, const V_ELE *AYe, double *H);

// Step 7: Rayleigh-Ritz projection H = Y^T A Y (m x m) and its eigenpairs.
extern void rayleighRitz(Matrix *A,
    DMatrix *Y,
    DMatrix *AY,
    int m,
    CG_UINT nr,
    double *H,
    double *eval,
    double *evec);

// Step 8: residual avbuf = AY*evec[:,k] - evalk*(Y*evec[:,k]); evk is
// length-m scratch.
extern void computeRitzResidual(DMatrix *Y,
    DMatrix *AY,
    int m,
    CG_UINT nr,
    double evalk,
    double *evec,
    int k,
    double *evk,
    V_ELE *avbuf);

// Step 8: 2-norm of a length-nr residual vector.
extern double residualNorm(CG_UINT nr, V_ELE *avbuf);

// Centralized work buffers so nothing is allocated during the iterations.
typedef struct {
  DMatrix Y;    // search vectors block (vecRows x NS)
  DMatrix AY;   // A*Y block (vecRows x NS)
  DMatrix u;    // block scratch
  DMatrix w;    // block scratch
  V_ELE *avbuf; // residual of a single Ritz pair (nr)
  double *evk;  // contiguous gather of one eigenvector column of evec (NS)
  double *H;
  double *eval;
  double *evec;
  double *accEval; // accepted (converged) in-interval eigenvalues (NS)
  int NS;
} ChebData;

extern void allocChebData(ChebData *d, Matrix *m, int NS);
extern void freeChebData(ChebData *d);

#endif // __CHEBFDSOLVER_H_
