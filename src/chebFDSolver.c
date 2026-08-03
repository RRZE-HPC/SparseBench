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

// Quick spectrum bounds, accurate enough (to me):
// https://en.wikipedia.org/wiki/Gershgorin_circle_theorem
void gershgorinBounds(CommType *comm, Matrix *A, double *a_out, double *b_out)
{

  /* Finite sentinels, not +/-INFINITY: -ffast-math lets the compiler assume Inf
   * away. CG_FLOAT_MAX is also OpenMP's identity for the reductions below. */
  CG_FLOAT lo = CG_FLOAT_MAX;
  CG_FLOAT hi = -CG_FLOAT_MAX;

#ifdef SCS
  CG_UINT *colInd    = A->colInd;
  V_ELE *val         = A->val;
  CG_UINT *chunkPtr  = A->chunkPtr;
  CG_UINT *chunkLens = A->chunkLens;
  CG_UINT C          = A->C;
  CG_UINT nChunks    = A->nChunks;

  /* colInd and the row index i*C+k are both in permuted (new) ordering, so the
   * diagonal test compares them directly — no permutation lookup, which would
   * also read out of bounds on padded rows. Bounds are permutation-invariant.
   * Loop k (row-in-chunk) outer so each row accumulates into scalars, as in CRS. */
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

/* Stateless splitmix64 mix of a 64-bit key -> uniformly-distributed bits. */
static inline unsigned long long splitmix64(unsigned long long z)
{
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/* Deterministic pseudo-random fill of the search block. Hashing each entry from
 * (row, column, rank) makes it order-independent, so the loop runs over rows for
 * NUMA first-touch. Rows [0,nr) get [-1,1); SCS padding rows are zeroed. */
static void randomInitBlock(CommType *comm, CG_UINT nr, CG_UINT vecRows, V_ELE *e, int nv)
{
  unsigned long long rankKey = (unsigned long long)comm->rank * 0xD1B54A32D192ED03ull;
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (CG_UINT r = 0; r < vecRows; r++) {
    if (r < nr) {
      for (int k = 0; k < nv; k++) {
        unsigned long long key = rankKey + (unsigned long long)r * 0x9E3779B97F4A7C15ull +
                                 (unsigned long long)k * 0xC2B2AE3D27D4EB4Full;
        unsigned long long h = splitmix64(key);
        double rv            = (double)(h >> 11) / (double)(1ull << 53) * 2.0 - 1.0;
        e[r * (CG_UINT)nv + (CG_UINT)k] = (V_ELE)rv;
      }
    } else {
      for (int k = 0; k < nv; k++) {
        e[r * (CG_UINT)nv + (CG_UINT)k] = (V_ELE)0.0;
      }
    }
  }
}

/* Step 5 (paper Fig. 6): replace each column of X by p(H) x via the Chebyshev
 * recurrence, applied to the whole subspace at once. X/U/W/TMP are row-major
 * blocks of width X->nc; no halo exchange — ChebFD is single-process only. */
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

  /* x = gc0 x + gc1 u + gc2 w  (fused into two sweeps) */
  waxpby(n, (V_ELE)gc[0], X->entries, (V_ELE)gc[1], U->entries, X->entries);
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

/* Step 6: rank-revealing Modified Gram-Schmidt with reorthogonalization over the
 * columns of the row-major block `e` (nr x nc, stride nc). Keeps the columns
 * with norm > tol, compacts them to the front, repacks to stride m, returns m. */
static int orthoMGS(CG_UINT nr, V_ELE *e, int nc, double tol)
{
  int m = 0;
  for (int k = 0; k < nc; k++) {
    for (int pass = 0; pass < 2; pass++) {
      for (int j = 0; j < m; j++) {
        /* coef = e[:,k]·e[:,j] ; e[:,k] -= coef*e[:,j]  (columns at stride nc) */
        V_ELE coef;
        ddot_stride(nr, &e[k], nc, &e[j], nc, &coef);
        waxpby_stride(nr, (V_ELE)1.0, &e[k], nc, (V_ELE)(-coef), &e[j], nc, &e[k], nc);
      }
    }
    V_ELE nrm2;
    ddot_stride(nr, &e[k], nc, &e[k], nc, &nrm2);
    double nrm = sqrt((double)nrm2);
    if (nrm < tol) {
      continue; /* linearly dependent -> drop */
    }
    V_ELE inv = (V_ELE)(1.0 / nrm);
    /* e[:,k] *= inv ; if compacting, e[:,m] = e[:,k] */
    waxpby_stride(nr, inv, &e[k], nc, (V_ELE)0.0, &e[k], nc, &e[k], nc);
    if (m != k) {
      waxpby_stride(nr, (V_ELE)1.0, &e[k], nc, (V_ELE)0.0, &e[k], nc, &e[m], nc);
    }
    m++;
  }
  /* Repack accepted columns from stride nc to stride m. Explicit row-wise loop:
   * the strided primitives would alias (narrowing stride in place). MUST stay
   * serial and ascending in r — row r overwrites the window that the earlier
   * row floor(r*m/nc) still reads. */
  if (m != nc) {
    for (CG_UINT r = 0; r < nr; r++) {
      for (int i = 0; i < m; i++) {
        e[r * m + i] = e[r * nc + i];
      }
    }
  }
  return m;
}

/* Step 7: Rayleigh-Ritz on the filtered subspace. First form the block product
 * AY = A·Y, then the m x m symmetric projection matrix H = YᵀAY (one parallel
 * region over the upper-triangular column pairs with a serial inner dot, instead
 * of a ddot_stride fork-join per pair), then solve the projected eigenproblem
 * H z = θ z via cyclic Jacobi for the Ritz values (eval, ascending) and Ritz
 * vectors (evec, column k pairs with eval[k]). AY is written for reuse by the
 * residual step. */
static void rayleighRitz(
    Matrix *A, DMatrix *Y, DMatrix *AY, int m, CG_UINT nr, double *H, double *eval,
    double *evec)
{
  AY->nc = m;
  spMMVM(A, Y, AY);

  V_ELE *Ye  = Y->entries;
  V_ELE *AYe = AY->entries;
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < m; i++) {
    for (int j = i; j < m; j++) {
      /* H[i,j] = Y[:,i]·AY[:,j]  (columns at stride m) */
      double hv = 0.0;
      for (CG_UINT r = 0; r < nr; r++) {
        hv += (double)Ye[r * (CG_UINT)m + i] * (double)AYe[r * (CG_UINT)m + j];
      }
      H[i * m + j] = hv;
      H[j * m + i] = hv;
    }
  }
  jacobiEigen(H, m, eval, evec);
}

/* Step 8: residual of the k-th Ritz pair, avbuf = AY·evec[:,k] - evalk·(Y·evec[:,k]).
 * Gathers the k-th Ritz vector's coefficients into evk first, then one parallel
 * region over rows with a serial dot along each contiguous Y/AY row, so the Ritz
 * vector stays in a register. */
static void computeRitzResidual(
    DMatrix *Y, DMatrix *AY, int m, CG_UINT nr, double evalk, double *evec, int k,
    double *evk, V_ELE *avbuf)
{
  V_ELE *Ye  = Y->entries;
  V_ELE *AYe = AY->entries;
  for (int j = 0; j < m; j++) {
    evk[j] = evec[(CG_UINT)j * m + k];
  }
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (CG_UINT r = 0; r < nr; r++) {
    double vv = 0.0, av = 0.0;
    for (int j = 0; j < m; j++) {
      vv += evk[j] * (double)Ye[r * (CG_UINT)m + j];
      av += evk[j] * (double)AYe[r * (CG_UINT)m + j];
    }
    avbuf[r] = (V_ELE)(av - evalk * vv);
  }
}

/* Step 8: 2-norm of a residual vector, ||avbuf||₂ = sqrt(avbufᵀ avbuf). */
static double residualNorm(CG_UINT nr, V_ELE *avbuf)
{
  V_ELE res2;
  ddot(nr, avbuf, avbuf, &res2);
  return sqrt((double)res2);
}

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
  if (param->cheb.kernel < 0 || param->cheb.kernel > 3) {
    if (commIsMaster(comm)) {
      printf("ChebFD: cheb_kernel must be 0=none, 1=Fejer, 2=Jackson or 3=Lanczos "
             "(got %d).\n",
          param->cheb.kernel);
    }
    return -1;
  }
#ifdef SCS
  /* spMMVM (SCS) puts a per-thread V_ELE tmp[C * NS] VLA on the worker stack;
   * reject a width that would overflow it. main.c guards SPMMV the same way. */
  if (!spMMVMBlockWidthOk(A->C, param->cheb.NS)) {
    if (commIsMaster(comm)) {
      printf("ChebFD: cheb_NS=%d too large for the SCS spMMVM stack scratch "
             "(C=%llu, limit ~%u bytes/thread); reduce cheb_NS or raise "
             "OMP_STACKSIZE.\n",
          param->cheb.NS,
          (unsigned long long)A->C,
          (unsigned)SCS_MAX_SPMMVM_VLA_BYTES);
    }
    return -1;
  }
#endif

  /* H = YᵀAY is rank-local and Rayleigh-Ritz is replicated, so this solver is
   * single-process only. main.c also rejects _MPI at compile time. */
  if (comm->size > 1) {
    if (commIsMaster(comm)) {
      printf("ChebFD: MPI is not supported (comm size %d); run with one rank.\n",
          comm->size);
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

  DMatrix *Y      = &d.Y;
  DMatrix *AY     = &d.AY;
  DMatrix *u      = &d.u;
  DMatrix *w      = &d.w;
  DMatrix *tmp    = &d.tmp;
  V_ELE *avbuf    = d.avbuf;
  double *evk     = d.evk;
  double *H       = d.H;
  double *eval    = d.eval;
  double *evec    = d.evec;
  double *accEval = d.accEval;

  // Alg. 3.1, Step 4: construct NS random search vectors.
  randomInitBlock(comm, nr, Y->nr, Y->entries, NS);

  int NT_found     = 0;
  double timeStart = getTimeStamp();
  int iter;
  for (iter = 1; iter <= maxiter; iter++) {
    // Alg. 3.1, Step 5 (Fig. 6): apply the polynomial filter to the whole subspace.
    Y->nc = NS;
    applyFilter(A, &f, Y, u, w, tmp);

    // Alg. 3.1, Step 6: orthogonalize the filtered search vectors (rank-revealing MGS).
    int m = orthoMGS(nr, Y->entries, NS, 1e-8);
    if (m == 0) {
      if (commIsMaster(comm)) {
        printf("iter %d: search space collapsed to rank 0.\n", iter);
      }
      break;
    }
    Y->nc = m;

    /* orthoMGS repacks only rows [0,nr); re-zero the SCS padding at the new
     * stride m, or applyFilter's Np-step recurrence blows it up to Inf/NaN. */
    for (CG_UINT r = nr; r < Y->nr; r++) {
      for (int i = 0; i < m; i++) {
        Y->entries[r * (CG_UINT)m + i] = (V_ELE)0.0;
      }
    }

    // Alg. 3.1, Step 7: Rayleigh-Ritz — project H = YᵀAY and solve for Ritz pairs.
    rayleighRitz(A, Y, AY, m, nr, H, eval, evec);

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
    double minres_in = DBL_MAX;
    double accThresh = sqrt(tol);
    for (int k = 0; k < m; k++) {
      if (eval[k] < lam_lo || eval[k] > lam_hi) {
        continue;
      }
      /* Residual of the k-th Ritz pair: avbuf = AY·evec[:,k] - evalk·(Y·evec[:,k]). */
      double evalk = eval[k];
      computeRitzResidual(Y, AY, m, nr, evalk, evec, k, evk, avbuf);
      double res = residualNorm(nr, avbuf);
      if (res < minres_in) {
        minres_in = res;
      }
      if (res <= accThresh) { /* accept (discard ghosts with large residual) */
        accEval[NT_found] = eval[k];
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
          accThresh);
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

  /* `iter` is maxiter+1 when the loop ran to exhaustion; report what ran. */
  int itersRun = (iter > maxiter) ? maxiter : iter;

  if (commIsMaster(comm)) {
    printf(
        "ChebFD finished after %d iterations in %.2fs\n", itersRun, timeStop - timeStart);
    printf("Found %d eigenpairs in target interval [%.6g, %.6g]:\n",
        NT_found,
        lam_lo,
        lam_hi);
    /* The accepted eigenvalues from the last iteration — not the first NT_found
     * in-interval Ritz values, which may include an unconverged ghost. */
    for (int i = 0; i < NT_found; i++) {
      printf("  lambda = %.10f\n", accEval[i]);
    }
  }

  freeChebData(&d);
  chebFilterFree(&f);

  return NT_found;
}

/* Allocate one row-major (vecRows x NS) block; size_t math so the element count
 * cannot wrap in 32-bit CG_UINT arithmetic. */
static void allocDMat(DMatrix *M, CG_UINT vecRows, int NS)
{
  M->nr = vecRows;
  M->nc = NS;
  M->entries =
      (V_ELE *)allocate(ARRAY_ALIGNMENT, (size_t)vecRows * (size_t)NS * sizeof(V_ELE));
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
  allocDMat(&d->Y, vecRows, NS);
  allocDMat(&d->AY, vecRows, NS);
  allocDMat(&d->u, vecRows, NS);
  allocDMat(&d->w, vecRows, NS);
  allocDMat(&d->tmp, vecRows, NS);

  d->avbuf   = (V_ELE *)allocate(ARRAY_ALIGNMENT, (size_t)nr * sizeof(V_ELE));
  d->evk     = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));

  d->H       = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
  d->eval    = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));
  d->evec    = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
  d->accEval = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));
}

void freeChebData(ChebData *d)
{
  deallocate(d->Y.entries);
  deallocate(d->AY.entries);
  deallocate(d->u.entries);
  deallocate(d->w.entries);
  deallocate(d->tmp.entries);
  deallocate(d->avbuf);
  deallocate(d->evk);
  deallocate(d->H);
  deallocate(d->eval);
  deallocate(d->evec);
  deallocate(d->accEval);
}
