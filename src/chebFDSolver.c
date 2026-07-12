/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "chebFDSolver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "allocate.h"
#include "chebFilter.h"
#include "denseJacobi.h"
#include "matrix.h"
#include "profiler.h"
#include "solver.h"
#include "timing.h"
#include "vtype.h"

/* Map the integer kernel selector to the ChebFilter enum. */
static KernelType mapKernel(int k)
{
  switch (k) {
  case 0:
    return KERNEL_NONE;
  case 1:
    return KERNEL_FEJER;
  case 2:
    return KERNEL_JACKSON;
  default:
    return KERNEL_LANCZOS;
  }
}

// https://en.wikipedia.org/wiki/Gershgorin_circle_theorem
// to find the lower and upper bounds quickly with relatively acceptable(to me) accuracy
void gershgorinBounds(CommType *comm, Matrix *A, double *a_out, double *b_out)
{

  CG_FLOAT lo = (CG_FLOAT)1e30;
  CG_FLOAT hi = (CG_FLOAT)-1e30;

#ifdef SCS
  CG_UINT *colInd    = A->colInd;
  V_ELE *val         = A->val;
  CG_UINT *chunkPtr  = A->chunkPtr;
  CG_UINT *chunkLens = A->chunkLens;
  CG_UINT C          = A->C;
  CG_UINT nChunks    = A->nChunks;

  /* SCS stores colInd in permuted (new) ordering, and the row index i*C+k is
   * also in new ordering, so the diagonal test compares them directly. No
   * permutation lookup is needed (it would also read out of bounds on padded
   * rows where i*C+k >= nr); gershgorin bounds are permutation-invariant
   * anyway. Loop k (row-in-chunk) outer so each row accumulates into scalars,
   * mirroring the CRS branch. */
#pragma omp parallel for schedule(OMP_SCHEDULE) reduction(min : lo) reduction(max : hi)
  for (CG_UINT i = 0; i < nChunks; ++i) {
    CG_UINT chunkOffset = chunkPtr[i];
    CG_UINT len         = chunkLens[i];
    for (CG_UINT k = 0; k < C; ++k) {
      CG_UINT newRow = i * C + k;
      if (newRow >= A->nr)
        break; /* skip C-padding rows (not part of A) */
      double diag = 0.0;
      double off  = 0.0;
      for (CG_UINT j = 0; j < len; ++j) {
        CG_UINT idx = chunkOffset + j * C + k;
        if (colInd[idx] == newRow) {
          diag += (double)val[idx];
        } else {
          off += fabs((double)val[idx]);
        }
      }
      double rowLo = diag - off;
      double rowHi = diag + off;
      if (rowLo < lo)
        lo = (CG_FLOAT)rowLo;
      if (rowHi > hi)
        hi = (CG_FLOAT)rowHi;
    }
  }

#else
  CG_UINT *rowPtr = A->rowPtr;
  CG_UINT *colInd = A->colInd;
  V_ELE *val      = A->val;
  CG_UINT nr      = A->nr;

#pragma omp parallel for schedule(OMP_SCHEDULE) reduction(min : lo) reduction(max : hi)
  for (CG_UINT i = 0; i < nr; i++) {
    double diag = 0.0;
    double off  = 0.0;
    for (CG_UINT j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
      if (colInd[j] == i) {
        diag += (double)val[j];
      } else {
        off += fabs((double)val[j]);
      }
    }
    double rowLo = diag - off;
    double rowHi = diag + off;
    if (rowLo < lo)
      lo = (CG_FLOAT)rowLo;
    if (rowHi > hi)
      hi = (CG_FLOAT)rowHi;
  }
#endif
  /* commReduction supports MAX only; obtain the min via negation. */
  CG_FLOAT neglo = -lo;
  commReduction(&neglo, MAX);
  commReduction(&hi, MAX);
  *a_out = (double)(-neglo);
  *b_out = (double)hi;
}

/* Deterministic pseudo-random fill of the search block. Each column k is
 * seeded by k (and the rank) so the result is reproducible. Rows [0,nr) are
 * filled; padding rows [nr,vecRows) (SCS) are zeroed. Serial xorshift32 — the
 * state carries across rows, so no omp (the old per-vector loop was racy). */
static void randomInitBlock(CommType *comm, CG_UINT nr, CG_UINT vecRows, V_ELE *e, int nv)
{
  for (int k = 0; k < nv; k++) {
    unsigned int state =
        12345u + (unsigned int)k * 7919u + (unsigned int)comm->rank * 1000003u;
    for (CG_UINT r = 0; r < nr; r++) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      double rv     = (double)state / 4294967295.0 * 2.0 - 1.0;
      e[r * nv + k] = (V_ELE)rv;
    }
  }
  for (CG_UINT r = nr; r < vecRows; r++) {
    for (int k = 0; k < nv; k++) {
      e[r * nv + k] = (V_ELE)0.0;
    }
  }
}

/* Step 5 (paper Fig. 6): replace each column of X by p(H) x via the
 * Clenshaw/Chebyshev recurrence, applied to the whole subspace at once.
 * X/U/W/TMP are DMatrix blocks of width X->nc; elementwise steps run over the
 * contiguous vecRows*nc prefix (block is row-major packed at stride nc). No
 * halo exchange — ChebFD is single-process only. */
static void applyFilter(
    Matrix *A, ChebFilter *f, DMatrix *X, DMatrix *U, DMatrix *W, DMatrix *TMP)
{
  V_ELE alpha = (V_ELE)f->alpha;
  V_ELE beta  = (V_ELE)f->beta;
  double *gc  = f->gc;
  int Np      = f->Np;

  int nv      = (int)X->nc;
  CG_UINT n   = X->nr * (CG_UINT)nv;
  U->nc = W->nc = TMP->nc = nv;

  /* u = (alpha H + beta) x = T_1(H) x */
  spMMVM(A, X, U);
  waxpby(n, alpha, U->entries, beta, X->entries, U->entries);

  /* tmp = (alpha H + beta) u ; w = 2 tmp - x = T_2(H) x */
  spMMVM(A, U, TMP);
  waxpby(n, alpha, TMP->entries, beta, U->entries, TMP->entries);
  waxpby(n, (V_ELE)2.0, TMP->entries, (V_ELE)(-1.0), X->entries, W->entries);

  /* x = gc0 x + gc1 u + gc2 w */
  waxpby(n, (V_ELE)gc[0], X->entries, (V_ELE)0.0, X->entries, X->entries);
  waxpby(n, (V_ELE)1.0, X->entries, (V_ELE)gc[1], U->entries, X->entries);
  waxpby(n, (V_ELE)1.0, X->entries, (V_ELE)gc[2], W->entries, X->entries);

  /* Remaining recurrence steps. Invariant: U = T_{n-2}, W = T_{n-1}. */
  for (int nn = 3; nn <= Np; nn++) {
    spMMVM(A, W, TMP); /* tmp = H w            */
    waxpby(n, alpha, TMP->entries, beta, W->entries, TMP->entries); /* tmp = (aH+b) w */
    waxpby(n,
        (V_ELE)2.0,
        TMP->entries,
        (V_ELE)(-1.0),
        U->entries,
        U->entries); /* U = 2 tmp - U = T_n */
    DMatrix *t = U;  /* U <- T_{n-1}, W <- T_n */
    U          = W;
    W          = t;
    waxpby(n,
        (V_ELE)1.0,
        X->entries,
        (V_ELE)gc[nn],
        W->entries,
        X->entries); /* x += gc[nn] T_n */
  }
}

/* Step 6: rank-revealing Modified Gram-Schmidt with reorthogonalization,
 * operating on the columns of a row-major block `e` (nr rows x nc cols, stride
 * nc). Accepts columns whose remaining norm exceeds tol, orthogonalizes in
 * place, compacts the accepted orthonormal columns to the front, and repacks
 * them to stride m so the result is a tight nr x m block. Returns m. */
static int orthoMGS(CG_UINT nr, V_ELE *e, int nc, double tol)
{
  int m = 0;
  for (int k = 0; k < nc; k++) {
    for (int pass = 0; pass < 2; pass++) {
      for (int j = 0; j < m; j++) {
        V_ELE coef = (V_ELE)0.0;
        for (CG_UINT r = 0; r < nr; r++) {
          coef += e[r * nc + k] * e[r * nc + j];
        }
        for (CG_UINT r = 0; r < nr; r++) {
          e[r * nc + k] -= coef * e[r * nc + j];
        }
      }
    }
    V_ELE nrm2 = (V_ELE)0.0;
    for (CG_UINT r = 0; r < nr; r++) {
      nrm2 += e[r * nc + k] * e[r * nc + k];
    }
    double nrm = sqrt((double)nrm2);
    if (nrm < tol) {
      continue; /* linearly dependent -> drop */
    }
    V_ELE inv = (V_ELE)(1.0 / nrm);
    for (CG_UINT r = 0; r < nr; r++) {
      e[r * nc + k] *= inv;
    }
    if (m != k) {
      for (CG_UINT r = 0; r < nr; r++) {
        e[r * nc + m] = e[r * nc + k];
      }
    }
    m++;
  }
  /* Repack accepted columns from stride nc to stride m (forward in place:
   * dest index r*m+i <= src index r*nc+i, and all earlier writes stay below
   * the current source, so no source is clobbered before being read). */
  for (int i = 0; i < m; i++) {
    for (CG_UINT r = 0; r < nr; r++) {
      e[r * m + i] = e[r * nc + i];
    }
  }
  return m;
}

/* Norm of the residual for the k-th Ritz pair is computed inline in
 * solveChebFD (Step 8) so the Ritz vector does not need to be materialized. */

int solveChebFD(CommType *comm, Parameter *param, Matrix *A)
{
#ifdef USE_COMPLEX
  if (commIsMaster(comm)) {
    printf("ChebFD (v1) supports real-symmetric matrices only "
           "(rebuild without USE_COMPLEX).\n");
  }
  return -1;
#endif

  if (!param->cheb.have_target || !(param->cheb.lam_lo < param->cheb.lam_hi)) {
    if (commIsMaster(comm)) {
      printf("ChebFD: set a valid target interval (cheb_lam_lo < cheb_lam_hi) "
             "in the parameter file.\n");
    }
    return -1;
  }
  if (param->cheb.Np < 2) {
    if (commIsMaster(comm)) {
      printf("ChebFD: set filter polynomial degree cheb_Np>=2 in the parameter file.\n");
    }
    return -1;
  }
  if (param->cheb.NS < 2) {
    if (commIsMaster(comm)) {
      printf("ChebFD: set number of search vectors cheb_NS>=2 in the parameter file.\n");
    }
    return -1;
  }

  // Alg. 3.1, Step 1: spectrum bounds [a,b] containing eigen values of A.
  double a, b;
  if (param->cheb.have_bounds) {
    a = param->cheb.a;
    b = param->cheb.b;
  } else {
    gershgorinBounds(comm, A, &a, &b);
  }
  if (commIsMaster(comm)) {
    printf("ChebFD spectrum bounds [a,b] = [%.6g, %.6g]\n", a, b);
  }
  // Alg. 3.1, Step 2 estimate Nt directly from par file

  // Alg. 3.1, Step 3 (§2.1): construct the filter polynomial
  // p(H) = Σ gₙ cₙ Tₙ(αH+βI) of degree Np.
  ChebFilter f;
  if (chebFilterInit(&f,
          a,
          b,
          param->cheb.lam_lo,
          param->cheb.lam_hi,
          param->cheb.Np,
          mapKernel(param->cheb.kernel),
          param->cheb.mu) != 0) {
    return -1;
  }
  if (commIsMaster(comm)) {
    chebFilterPrint(&f);
  }

  // work-space setup
  double tol    = param->eps > 0.0 ? param->eps : 1e-8;
  int maxiter   = param->itermax > 0 ? param->itermax : 50;
  CG_UINT nr    = A->nr;
  int NS        = param->cheb.NS;
  double lam_lo = param->cheb.lam_lo;
  double lam_hi = param->cheb.lam_hi;

  // allocate data
  ChebData d;
  allocChebData(&d, A, NS);

  DMatrix *Y   = &d.Y;
  DMatrix *AY  = &d.AY;
  DMatrix *u   = &d.u;
  DMatrix *w   = &d.w;
  DMatrix *tmp = &d.tmp;
  V_ELE *vbuf  = d.vbuf;
  V_ELE *avbuf = d.avbuf;
  double *H    = d.H;
  double *eval = d.eval;
  double *evec = d.evec;

  // Alg. 3.1, Step 4: construct NS random search vectors.
  randomInitBlock(comm, nr, Y->nr, Y->entries, NS);

  int NT_found     = 0;
  int mLast        = NS;
  double timeStart = getTimeStamp();
  int iter;
  for (iter = 1; iter <= maxiter; iter++) {
    int curNS = NS;
    // Alg. 3.1, Step 5 (Fig. 6): apply the polynomial filter to the whole subspace.
    Y->nc = curNS;
    applyFilter(A, &f, Y, u, w, tmp);

    // Alg. 3.1, Step 6: orthogonalize the filtered search vectors (rank-revealing MGS).
    int m = orthoMGS(nr, Y->entries, curNS, 1e-8);
    if (m == 0) {
      if (commIsMaster(comm)) {
        printf("iter %d: search space collapsed to rank 0.\n", iter);
      }
      break;
    }
    mLast = m;
    Y->nc = m;

    // Alg. 3.1, Step 7: Rayleigh-Ritz — H = YᵀAY, Ritz pairs via Jacobi. AY = A*Y (block).
    AY->nc = m;
    spMMVM(A, Y, AY);
    V_ELE *Ye  = Y->entries;
    V_ELE *AYe = AY->entries;
    for (int i = 0; i < m; i++) {
      for (int j = i; j < m; j++) {
        V_ELE h = (V_ELE)0.0;
        for (CG_UINT r = 0; r < nr; r++) {
          h += Ye[r * m + i] * AYe[r * m + j];
        }
        double hv    = (double)h;
        H[i * m + j] = hv;
        H[j * m + i] = hv;
      }
    }
    jacobiEigen(H, m, eval, evec);

    if (param->verbose && commIsMaster(comm)) {
      int inint = 0;
      for (int k = 0; k < m; k++) {
        if (eval[k] >= lam_lo && eval[k] <= lam_hi) {
          inint++;
        }
      }
      printf("  [dbg] eval range [%.4f, %.4f], %d Ritz values in interval\n",
          eval[0],
          eval[m - 1],
          inint);
    }

    // Alg. 3.1, Step 8 : convergence check.
    NT_found         = 0;
    double maxres    = 0.0;
    double minres_in = 1e30;
    for (int k = 0; k < m; k++) {
      if (eval[k] < lam_lo || eval[k] > lam_hi) {
        continue;
      }
      for (CG_UINT r = 0; r < nr; r++) {
        CG_UINT row = r * (CG_UINT)m;
        V_ELE vk    = (V_ELE)0.0;
        V_ELE avk   = (V_ELE)0.0;
        for (int j = 0; j < m; j++) {
          V_ELE ej = (V_ELE)evec[j * m + k];
          vk += ej * Ye[row + j];
          avk += ej * AYe[row + j];
        }
        vbuf[r]  = vk;
        avbuf[r] = avk;
      }
      /* r = Av_k - eval_k v_k ; ||r|| */
      waxpby(nr, (V_ELE)1.0, avbuf, (V_ELE)(-eval[k]), vbuf, avbuf);
      V_ELE res2;
      ddot(nr, avbuf, avbuf, &res2);
      double res = sqrt((double)res2);
      if (res < minres_in) {
        minres_in = res;
      }
      if (res <= sqrt(tol)) { /* accept (discard ghosts with large residual) */
        NT_found++;
        if (res > maxres) {
          maxres = res;
        }
      }
    }
    if (commIsMaster(comm) && param->verbose) {
      printf("  [dbg] in-interval residuals: min=%.3e max(seen)=%.3e "
             "(accept thr=%.3e)\n",
          minres_in,
          maxres,
          sqrt(tol));
    }

    if (commIsMaster(comm)) {
      printf("iter %d: search rank m=%d, target pairs found=%d, "
             "max residual=%.3e\n",
          iter,
          m,
          NT_found,
          maxres);
    }

    if (NT_found > 0 && maxres <= tol) {
      if (commIsMaster(comm)) {
        printf("ChebFD converged after %d iterations.\n", iter);
      }
      break;
    }

    /* Alg. 3.1, Step 8b: restart from the m orthonormal filtered search vectors.
     * Y is already a tight nr x m block. */
    NS = m;
  }
  double timeStop = getTimeStamp();

  if (commIsMaster(comm)) {
    printf("ChebFD finished after %d iterations in %.2fs\n", iter, timeStop - timeStart);
    printf("Found %d eigenpairs in target interval [%.6g, %.6g]:\n",
        NT_found,
        lam_lo,
        lam_hi);
    int shown = 0;
    for (int k = 0; k < mLast && shown < NT_found; k++) {
      if (eval[k] >= lam_lo && eval[k] <= lam_hi) {
        printf("  lambda = %.10f\n", eval[k]);
        shown++;
      }
    }
  }

  freeChebData(&d);
  chebFilterFree(&f);

  return NT_found;
}

void allocChebData(ChebData *d, Matrix *m, int NS)
{
  CG_UINT nr = m->nr;
#ifdef SCS
  CG_UINT vecRows = m->nrPadded;
#else
  CG_UINT vecRows = m->nr;
#endif

  d->NS = NS;

  /* All DMatrix blocks are stored row-major (vecRows x NS). The active width
   * (.nc) is shrunk during iterations; storage stays NS-wide. */
  d->Y.nr        = vecRows;
  d->Y.nc        = NS;
  d->Y.entries   = (V_ELE *)allocate(ARRAY_ALIGNMENT, vecRows * NS * sizeof(V_ELE));

  d->AY.nr       = vecRows;
  d->AY.nc       = NS;
  d->AY.entries  = (V_ELE *)allocate(ARRAY_ALIGNMENT, vecRows * NS * sizeof(V_ELE));

  d->u.nr        = vecRows;
  d->u.nc        = NS;
  d->u.entries   = (V_ELE *)allocate(ARRAY_ALIGNMENT, vecRows * NS * sizeof(V_ELE));

  d->w.nr        = vecRows;
  d->w.nc        = NS;
  d->w.entries   = (V_ELE *)allocate(ARRAY_ALIGNMENT, vecRows * NS * sizeof(V_ELE));

  d->tmp.nr      = vecRows;
  d->tmp.nc      = NS;
  d->tmp.entries = (V_ELE *)allocate(ARRAY_ALIGNMENT, vecRows * NS * sizeof(V_ELE));

  d->vbuf        = (V_ELE *)allocate(ARRAY_ALIGNMENT, nr * sizeof(V_ELE));
  d->avbuf       = (V_ELE *)allocate(ARRAY_ALIGNMENT, nr * sizeof(V_ELE));

  d->H           = (double *)allocate(ARRAY_ALIGNMENT, NS * NS * sizeof(double));
  d->eval        = (double *)allocate(ARRAY_ALIGNMENT, NS * sizeof(double));
  d->evec        = (double *)allocate(ARRAY_ALIGNMENT, NS * NS * sizeof(double));
}

void freeChebData(ChebData *d)
{
  deallocate(d->Y.entries);
  deallocate(d->AY.entries);
  deallocate(d->u.entries);
  deallocate(d->w.entries);
  deallocate(d->tmp.entries);
  deallocate(d->vbuf);
  deallocate(d->avbuf);
  deallocate(d->H);
  deallocate(d->eval);
  deallocate(d->evec);
}
