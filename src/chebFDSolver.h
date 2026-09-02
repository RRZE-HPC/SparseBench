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
// GPU only, Step 6 on a streamed search space: rank-revealing Cholesky-QR
// (Loewdin form: Y <- Y V diag(lambda^-1/2) from the eigenpairs of Y^T Y,
// directions with sqrt(lambda) < tol dropped), `passes` times (2 = CGS2-level
// orthogonality). Y is the pinned host block at stride nc; on return it is
// repacked to stride m. G / eval / evec are nc*nc / nc / nc*nc host scratch.
struct GpuVectorStream;
extern int chebOrthoCholQR2(struct GpuVectorStream *vs,
    V_ELE *Y,
    int nc,
    double tol,
    double *G,
    double *eval,
    double *evec,
    int passes);
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
// Y and AY are pinned host blocks (allocateHost): on GPU builds they are
// streamed through the device (cuda_vector_stream.cu), the CPU build works
// on them directly. u / w / avbuf / evk serve the host recurrence and the
// unit tests only; the GPU solver keeps its recurrence scratch on the device.
typedef struct {
  DMatrix Y;    // search vectors block (vecRows x NS)
  DMatrix AY;   // A*Y block (vecRows x NS)
  DMatrix u;    // block scratch (host recurrence)
  DMatrix w;    // block scratch (host recurrence)
  V_ELE *avbuf; // residual of a single Ritz pair (nr)
  double *evk;  // contiguous gather of one eigenvector column of evec (NS)
  double *H;
  double *eval;
  double *evec;
  double *accEval; // accepted (converged) in-interval eigenvalues (NS)
  int *sel;        // in-interval Ritz pair indices of the current iteration (NS)
  double *res2;    // their squared residual norms (NS)
  int NS;
} ChebData;

extern void allocChebData(ChebData *d, Matrix *m, int NS);
extern void freeChebData(ChebData *d);

#endif // __CHEBFDSOLVER_H_
