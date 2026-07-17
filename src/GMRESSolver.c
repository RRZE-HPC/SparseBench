/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "allocate.h"
#include "comm.h"
#include "profiler.h"
#include "solver.h"
#include "timing.h"
#include "vtype.h"

#ifdef USE_COMPLEX
#define CAST(v) VREAL((v))
#else
#define CAST(v) v
#endif

static void initVectors(Matrix *m, V_ELE *x, V_ELE *b)
{
  CG_UINT numRows = m->nr;

#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int rowID = 0; rowID < (int)numRows; rowID++) {
    x[rowID] = 0.0;
    b[rowID] = 1.0;
  }
}

/* Restarted GMRES(m).
 *
 * param->restart controls the Krylov subspace size per restart cycle.
 * param->itermax is the maximum total number of Arnoldi steps across all
 * restarts, consistent with CG's itermax semantics.
 *
 * Note: the Hessenberg matrix H is stored as CG_FLOAT (real). Dot products
 * are cast to real via CAST(). Full complex GMRES (complex Givens rotations)
 * is not implemented; USE_COMPLEX builds produce incorrect results. */
int solveGMRES(CommType *comm, Parameter *param, Matrix *A)
{
  CG_FLOAT eps = (CG_FLOAT)param->eps;
  int itermax  = param->itermax;
  int m        = param->restart;

  CG_UINT nrow = A->nr;
  CG_UINT ncol = A->nc;

  /* Krylov basis: (m+1) vectors each of size ncol so commExchange can fill
   * the halo region [nrow..ncol-1] in-place before each SpMV call. */
  V_ELE **V = (V_ELE **)malloc((m + 1) * sizeof(V_ELE *));
  for (int i = 0; i <= m; i++) {
    V[i] = (V_ELE *)allocate(ARRAY_ALIGNMENT, ncol * sizeof(V_ELE));
    memset(V[i], 0, ncol * sizeof(V_ELE));
  }

  /* Upper Hessenberg (m+1) x m, column-major: H[i + j*(m+1)] = H_{i,j} */
  CG_FLOAT *H  = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, (m + 1) * m * sizeof(CG_FLOAT));
  CG_FLOAT *cs = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, m * sizeof(CG_FLOAT));
  CG_FLOAT *sn = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, m * sizeof(CG_FLOAT));
  CG_FLOAT *g  = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, (m + 1) * sizeof(CG_FLOAT));
  CG_FLOAT *y  = (CG_FLOAT *)allocate(ARRAY_ALIGNMENT, m * sizeof(CG_FLOAT));

  V_ELE *x     = (V_ELE *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(V_ELE));
  V_ELE *bvec  = (V_ELE *)allocate(ARRAY_ALIGNMENT, nrow * sizeof(V_ELE));

  initVectors(A, x, bvec);

  int printFreq = itermax / 10;
  if (printFreq > 50)
    printFreq = 50;
  if (printFreq < 1)
    printFreq = 1;

  double timeStart, timeStop, ts;
  V_ELE rtmp;
  CG_FLOAT normr = 0.0;

  /* Compute initial residual before timer for display only */
  waxpby(nrow, 1.0, x, 0.0, x, V[0]);
  commExchange(comm, nrow, V[0]);
  spMVM(A, V[0], V[1]);
  waxpby(nrow, 1.0, bvec, -1.0, V[1], V[0]);
  ddot(nrow, V[0], V[0], &rtmp);
  normr = sqrt(CAST(rtmp));

  if (commIsMaster(comm)) {
    printf("Initial Residual = %E\n", normr);
  }

  int k     = 0;
  timeStart = getTimeStamp();

  while (k < itermax && normr > eps) {

    /* ---- Compute r = b - A*x and normalise into V[0] ---- */
    PROFILE(WAXPBY, waxpby(nrow, 1.0, x, 0.0, x, V[0]));
    PROFILE(COMM, commExchange(comm, nrow, V[0]));
    PROFILE(SPMVM, spMVM(A, V[0], V[1]));
    PROFILE(WAXPBY, waxpby(nrow, 1.0, bvec, -1.0, V[1], V[0]));
    PROFILE(DDOT, ddot(nrow, V[0], V[0], &rtmp));
    CG_FLOAT beta = sqrt(CAST(rtmp));
    normr         = beta;
    if (normr <= eps)
      break;
    PROFILE(WAXPBY, waxpby(nrow, (V_ELE)(1.0 / beta), V[0], (V_ELE)0.0, V[0], V[0]));

    memset(g, 0, (m + 1) * sizeof(CG_FLOAT));
    memset(cs, 0, m * sizeof(CG_FLOAT));
    memset(sn, 0, m * sizeof(CG_FLOAT));
    g[0] = beta;

    /* Steps remaining before hitting itermax */
    int jlimit = (k + m <= itermax) ? m : (itermax - k);
    int jj     = 0;

    /* ---- Arnoldi inner loop ---- */
    for (int j = 0; j < jlimit; j++) {
      jj = j;

      /* w = A * V[j]; result written into V[j+1][0..nrow-1] */
      PROFILE(COMM, commExchange(comm, nrow, V[j]));
      PROFILE(SPMVM, spMVM(A, V[j], V[j + 1]));

      /* Modified Gram-Schmidt orthogonalisation against V[0..j] */
      for (int i = 0; i <= j; i++) {
        PROFILE(DDOT, ddot(nrow, V[i], V[j + 1], &rtmp));
        CG_FLOAT hij       = CAST(rtmp);
        H[i + j * (m + 1)] = hij;
        PROFILE(WAXPBY, waxpby(nrow, 1.0, V[j + 1], (V_ELE)(-hij), V[i], V[j + 1]));
      }

      /* Normalise new basis vector */
      PROFILE(DDOT, ddot(nrow, V[j + 1], V[j + 1], &rtmp));
      CG_FLOAT h_next          = sqrt(CAST(rtmp));
      H[(j + 1) + j * (m + 1)] = h_next;
      if (h_next < 1e-14) {
        /* Exact solution found (lucky breakdown) */
        normr = 0.0;
        break;
      }
      PROFILE(WAXPBY,
          waxpby(nrow, (V_ELE)(1.0 / h_next), V[j + 1], (V_ELE)0.0, V[j + 1], V[j + 1]));

      /* Apply previous Givens rotations to column j of H */
      for (int i = 0; i < j; i++) {
        CG_FLOAT h_ij            = H[i + j * (m + 1)];
        CG_FLOAT h_i1j           = H[(i + 1) + j * (m + 1)];
        H[i + j * (m + 1)]       = cs[i] * h_ij + sn[i] * h_i1j;
        H[(i + 1) + j * (m + 1)] = -sn[i] * h_ij + cs[i] * h_i1j;
      }

      /* Compute new Givens rotation to zero H[j+1][j] */
      CG_FLOAT h_jj            = H[j + j * (m + 1)];
      CG_FLOAT h_jp1j          = H[(j + 1) + j * (m + 1)];
      CG_FLOAT rlen            = (CG_FLOAT)sqrt((double)(h_jj * h_jj + h_jp1j * h_jp1j));
      cs[j]                    = h_jj / rlen;
      sn[j]                    = h_jp1j / rlen;
      H[j + j * (m + 1)]       = rlen;
      H[(j + 1) + j * (m + 1)] = 0.0;

      /* Update g and track residual norm estimate */
      CG_FLOAT g_jp1 = -sn[j] * g[j];
      g[j]           = cs[j] * g[j];
      g[j + 1]       = g_jp1;
      normr          = (CG_FLOAT)fabs((double)g_jp1);

      if (commIsMaster(comm) && ((k + j + 1) % printFreq == 0)) {
        printf("Iteration = %d   Residual = %E\n", k + j + 1, normr);
      }

      if (normr <= eps)
        break;
    }

    /* ---- Back substitution: solve H[0..jj][0..jj] * y = g[0..jj] ---- */
    for (int i = jj; i >= 0; i--) {
      y[i] = g[i];
      for (int kk = i + 1; kk <= jj; kk++) {
        y[i] -= H[i + kk * (m + 1)] * y[kk];
      }
      y[i] /= H[i + i * (m + 1)];
    }

    /* ---- Solution update: x += sum_i y[i] * V[i] ---- */
    for (int i = 0; i <= jj; i++) {
      PROFILE(WAXPBY, waxpby(nrow, 1.0, x, (V_ELE)y[i], V[i], x));
    }

    k += jj + 1;
  }

  timeStop = getTimeStamp();

  if (commIsMaster(comm)) {
    printf("Solution performed %d iterations and took %.2fs\n", k, timeStop - timeStart);
  }

  for (int i = 0; i <= m; i++) {
    deallocate(V[i]);
  }
  free(V);
  deallocate(H);
  deallocate(cs);
  deallocate(sn);
  deallocate(g);
  deallocate(y);
  deallocate(x);
  deallocate(bvec);

  return k;
}
