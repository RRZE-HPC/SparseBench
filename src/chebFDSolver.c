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
#include "kernel_dispatch.h"
#include "matrix.h"
#include "nvtx_marker.h"
#include "profiler.h"
#include "section_timer.h"
#include "solver.h"
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

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#define CHEB_GPU 1
/* Default columns per streamed search-space sub-block (cheb_nb 0). */
#define CHEB_DEFAULT_NB 16
#else
#define CHEB_GPU 0
#endif

// Spectrum bounds via Gershgorin's circle theorem.
void gershgorinBounds(CommType *comm, Matrix *A, double *a_out, double *b_out)
{

  /* Finite sentinels, not +/-INFINITY (-ffast-math assumes Inf away);
   * CG_FLOAT_MAX is also OpenMP's reduction identity below. */
  CG_FLOAT lo = CG_FLOAT_MAX;
  CG_FLOAT hi = -CG_FLOAT_MAX;

#ifdef SCS
  CG_UINT *colInd    = A->colInd;
  V_ELE *val         = A->val;
  CG_UINT *chunkPtr  = A->chunkPtr;
  CG_UINT *chunkLens = A->chunkLens;
  CG_UINT C          = A->C;
  CG_UINT nChunks    = A->nChunks;

  /* colInd and the row index i*C+k are both permuted, so the diagonal test
   * compares them directly (a permutation lookup would read OOB on padded
   * rows). k outer so each row accumulates into scalars, as in CRS. */
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

/* Deterministic pseudo-random fill of the search block, hashed from
 * (row, column, rank) so it is order-independent; loop over rows for NUMA
 * first-touch. Rows >= nr are SCS padding and are zeroed. The block is
 * host memory on every build (pinned on GPU builds), so this runs on the
 * host everywhere; gpu_randomInitBlock keeps the same key layout. */
static void randomInitBlock(CommType *comm, CG_UINT nr, CG_UINT vecRows, V_ELE *e, int nv)
{
  unsigned long long rankKey = (unsigned long long)comm->rank * 0xD1B54A32D192ED03ull;
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (CG_UINT r = 0; r < vecRows; r++) {
    if (r < nr) {
      for (int k = 0; k < nv; k++) {
        unsigned long long key = rankKey + (unsigned long long)r * 0x9E3779B97F4A7C15ull +
                                 (unsigned long long)k * 0xC2B2AE3D27D4EB4Full;
        unsigned long long h   = splitmix64(key);
        double rv              = (double)(h >> 11) / (double)(1ull << 53) * 2.0 - 1.0;
        e[r * (CG_UINT)nv + (CG_UINT)k] = (V_ELE)rv;
      }
    } else {
      for (int k = 0; k < nv; k++) {
        e[r * (CG_UINT)nv + (CG_UINT)k] = (V_ELE)0.0;
      }
    }
  }
}

/* Roofline-style accounting for one whole-block matrix pass (one SpMMV over
 * nv columns), used to report GFlop/s and GB/s so runs with different
 * gpu_alloc / gpu_stream_mb / cheb_nb settings are directly comparable:
 *   flops = 2*nnz*nv                       (multiply-adds; axpy epilogue
 *                                           flops are neglected)
 *   bytes = nElems*(val+colInd+x-gather)   (matrix re-read + worst-case
 *                                           scattered gathers)
 *         + 3*rows*nv*sizeof(V_ELE)        (block streams: read operand,
 *                                           write y, one fused aux term) */
static void accountMatvec(
    const Matrix *A, int nv, int npasses, double *flops, double *bytes)
{
#ifdef SCS
  const double stor = (double)A->nElems;
  const double rows = (double)A->nrPadded;
#else
  const double stor = (double)A->nnz;
  const double rows = (double)A->nr;
#endif
  *flops += 2.0 * stor * (double)nv * (double)npasses;
  *bytes += (stor * (2.0 * sizeof(V_ELE) + sizeof(CG_UINT)) +
                3.0 * rows * (double)nv * sizeof(V_ELE)) *
            (double)npasses;
}

/* Block kernels that touch the matrix, through the dispatched (host or
 * device) kernel on whole resident blocks. The GPU solver itself does not
 * take this path — it streams the search space (gpu_vstream_*) — but the
 * CPU build and the unit tests do. */
static void chebFusedOp(Matrix *A,
    const DMatrix *x,
    V_ELE cA,
    const DMatrix *p,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y)
{
  SPMMVMFUSEDFUNC(A, x, cA, p, cP, q, cQ, y);
}

static void chebRecurrenceOp(Matrix *A,
    const DMatrix *w,
    V_ELE cA,
    V_ELE cP,
    const DMatrix *q,
    V_ELE cQ,
    DMatrix *y,
    V_ELE gc,
    DMatrix *x)
{
  CHEBFDOPFUNC(A, w, cA, cP, q, cQ, y, gc, x);
}

static void chebBlockMatvec(Matrix *A, const DMatrix *x, DMatrix *y)
{
  SPMMVMFUNC(A, x, y);
}

/* Step 5 (paper Fig. 6): replace each column of X by p(H)x via the Chebyshev
 * recurrence on the whole subspace at once. X/U/W are row-major blocks of
 * width X->nc; ChebFD is single-process only (no halo exchange). Each term
 * uses a fused matvec+axpy kernel (spMMVMFused, then chebfdOp from n=3) so
 * T_n is never written out and re-read by a separate axpy pass. Degree-outer
 * order over resident blocks; the GPU solver runs the same recurrence
 * block-outer on streamed column sub-blocks (gpu_vstream_filter). */
void applyFilter(Matrix *A, ChebFilter *f, DMatrix *X, DMatrix *U, DMatrix *W)
{
  V_ELE alpha = (V_ELE)f->alpha;
  V_ELE beta  = (V_ELE)f->beta;
  double *gc  = f->gc;
  int Np      = f->Np;

  int nv      = (int)X->nc;
  CG_UINT n   = X->nr * (CG_UINT)nv;
  U->nc = W->nc = nv;

  /* u = (alpha H + beta) x = T_1(H) x */
  chebFusedOp(A, X, alpha, X, beta, NULL, (V_ELE)0.0, U);

  /* w = 2*(alpha H + beta) u - x = T_2(H) x  (x is still T_0 here) */
  chebFusedOp(A, U, (V_ELE)2.0 * alpha, U, (V_ELE)2.0 * beta, X, (V_ELE)(-1.0), W);

  /* x = gc0*x + gc1*u + gc2*w; overwrites T_0 now that T_2 no longer needs it. */
  WAXPBY3FUNC(n,
      (V_ELE)gc[0],
      X->entries,
      (V_ELE)gc[1],
      U->entries,
      (V_ELE)gc[2],
      W->entries,
      X->entries);

  /* Remaining recurrence steps. Invariant: U = T_{n-2}, W = T_{n-1}. */
  NVTX_RANGE_PUSH_C("ChebFD.filter.recurrence", NVTX_C_FILTER);
  for (int nn = 3; nn <= Np; nn++) {
    /* U <- T_n in place; aliasing the q=U read with the y=U write is
     * row-local safe. Fused with x += gc[nn]*T_n. */
    chebRecurrenceOp(A,
        W,
        (V_ELE)2.0 * alpha,
        (V_ELE)2.0 * beta,
        U,
        (V_ELE)(-1.0),
        U,
        (V_ELE)gc[nn],
        X);
    DMatrix *t = U; /* U <- T_{n-1}, W <- T_n */
    U          = W;
    W          = t;
  }
  NVTX_RANGE_POP();
}

/* Step 6: rank-revealing CGS2 over the columns of the row-major block e
 * (nr x nc, stride nc). Keeps columns with norm > tol, compacts them to the
 * front, repacks to stride m, returns m. All m projections run as one
 * unit-stride pass into coefs[0:m] instead of a ddot_stride/waxpby_stride
 * pair per column pair: per-pair primitives fork a parallel region each and
 * walk stride-nc columns, which is O(nc^2) fork/joins with a cache-hostile
 * access pattern for wide blocks. Two reorthogonalization passes make this
 * CGS2, numerically as stable as MGS in practice. */
int orthoMGS(CG_UINT nr, V_ELE *e, int nc, double tol)
{
  int m        = 0;
  V_ELE *coefs = (V_ELE *)allocate(ARRAY_ALIGNMENT, (size_t)nc * sizeof(V_ELE));

  for (int k = 0; k < nc; k++) {
    for (int pass = 0; pass < 2 && m > 0; pass++) {
      for (int j = 0; j < m; j++) {
        coefs[j] = (V_ELE)0.0;
      }
#pragma omp parallel for schedule(OMP_SCHEDULE) reduction(+ : coefs[0 : m])
      for (CG_UINT i = 0; i < nr; i++) {
        V_ELE ek = e[i * (CG_UINT)nc + (CG_UINT)k];
        for (int j = 0; j < m; j++) {
          coefs[j] += ek * e[i * (CG_UINT)nc + (CG_UINT)j];
        }
      }
#pragma omp parallel for schedule(OMP_SCHEDULE)
      for (CG_UINT i = 0; i < nr; i++) {
        V_ELE s = e[i * (CG_UINT)nc + (CG_UINT)k];
        for (int j = 0; j < m; j++) {
          s -= coefs[j] * e[i * (CG_UINT)nc + (CG_UINT)j];
        }
        e[i * (CG_UINT)nc + (CG_UINT)k] = s;
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
  deallocate(coefs);
  /* Repack accepted columns from stride nc to stride m. Must stay serial and
   * ascending in r: row r overwrites the window an earlier row still reads. */
  if (m != nc) {
    for (CG_UINT r = 0; r < nr; r++) {
      for (int i = 0; i < m; i++) {
        e[r * m + i] = e[r * nc + i];
      }
    }
  }
  return m;
}

#if CHEB_GPU
/* Step 6 on a streamed search space: rank-revealing Cholesky-QR in Loewdin
 * form. Per pass: G = Y^T Y (one row-chunk pass), eigenpairs of the m x m G
 * on the host, drop directions with sqrt(lambda) < tol, then one block
 * update Y <- Y V diag(lambda^-1/2) written at the new stride. Two passes
 * give CGS2-level orthogonality; ~6 block transfers instead of the ~4*nc
 * full-block passes of column-wise CGS2. */
int chebOrthoCholQR2(GpuVectorStream *vs,
    V_ELE *Y,
    int nc,
    double tol,
    double *G,
    double *eval,
    double *evec,
    int passes)
{
  int m = nc;
  for (int pass = 0; pass < passes && m > 0; pass++) {
    gpu_vstream_gram(vs, Y, NULL, m, G);
    jacobiEigen(G, m, eval, evec);

    /* B (m x mNew, row-major) = kept eigenvectors scaled by lambda^-1/2.
     * G is no longer needed and is at least m*m: reuse it for B. */
    int mNew = 0;
    for (int j = 0; j < m; j++) {
      double lam = eval[j] > 0.0 ? eval[j] : 0.0;
      if (sqrt(lam) < tol) {
        continue; /* linearly dependent direction -> drop */
      }
      mNew++;
    }
    if (mNew == 0) {
      return 0;
    }
    double *B = G;
    int t     = 0;
    for (int j = 0; j < m; j++) {
      double lam = eval[j] > 0.0 ? eval[j] : 0.0;
      if (sqrt(lam) < tol) {
        continue;
      }
      double inv = 1.0 / sqrt(lam);
      for (int i = 0; i < m; i++) {
        B[(size_t)i * mNew + t] = evec[(size_t)i * m + j] * inv;
      }
      t++;
    }
    gpu_vstream_update(vs, Y, m, B, mNew);
    m = mNew;
  }
  return m;
}
#endif

/* Step 7: Rayleigh-Ritz: AY = A*Y, then H = Y^T AY (m x m), then cyclic
 * Jacobi for the Ritz values (eval, ascending) and vectors (evec[:,k] pairs
 * with eval[k]). AY is kept for the residual step. */
void rayleighRitz(Matrix *A,
    DMatrix *Y,
    DMatrix *AY,
    int m,
    CG_UINT nr,
    double *H,
    double *eval,
    double *evec)
{
  AY->nc = m;
  chebBlockMatvec(A, Y, AY);
  GRAMFUNC(nr, m, Y->entries, AY->entries, H);
  jacobiEigen(H, m, eval, evec);
}

/* H = Y^T AY for two nr x m row-major blocks; both triangles written from
 * one accumulator so H is exactly symmetric, as jacobiEigen assumes. */
void gramYtAY(CG_UINT nr, int m, const V_ELE *Ye, const V_ELE *AYe, double *H)
{
#pragma omp parallel for schedule(OMP_SCHEDULE)
  for (int i = 0; i < m; i++) {
    for (int j = i; j < m; j++) {
      double hv = 0.0;
      for (CG_UINT r = 0; r < nr; r++) {
        hv += (double)Ye[r * (CG_UINT)m + i] * (double)AYe[r * (CG_UINT)m + j];
      }
      H[i * m + j] = hv;
      H[j * m + i] = hv;
    }
  }
}

/* Step 8: residual of the k-th Ritz pair,
 * avbuf = AY*evec[:,k] - evalk*(Y*evec[:,k]); evk gathers evec[:,k]. */
void computeRitzResidual(DMatrix *Y,
    DMatrix *AY,
    int m,
    CG_UINT nr,
    double evalk,
    double *evec,
    int k,
    double *evk,
    V_ELE *avbuf)
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
double residualNorm(CG_UINT nr, V_ELE *avbuf)
{
  V_ELE res2;
  /* Dispatched: a host dot would fault avbuf back per Ritz pair. */
  DDOTFUNC(nr, avbuf, avbuf, &res2);
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
#if defined(SCS) && !CHEB_GPU
  /* SCS spMMVM puts a per-thread tmp[C * NS] VLA on the worker stack; reject
   * widths that would overflow it (main.c guards SPMMV the same way). The
   * GPU solver never runs the host spMMVM. */
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

  if (param->chebNb < 0) {
    if (commIsMaster(comm)) {
      printf("ChebFD: cheb_nb must be >= 0 (got %d).\n", param->chebNb);
    }
    return -1;
  }
#if !CHEB_GPU
  if (param->chebNb > 0 && commIsMaster(comm)) {
    printf("ChebFD: cheb_nb is GPU-only; ignoring.\n");
  }
#endif

  NVTX_RANGE_PUSH_C("ChebFD.solve", NVTX_C_SETUP);

  // Alg. 3.1, Step 1: spectrum bounds [a,b] containing eigen values of A.
  double a, b;
  if (param->cheb.have_bounds) {
    a = param->cheb.a;
    b = param->cheb.b;
  } else {
    NVTX_RANGE_PUSH_C("ChebFD.setup.gershgorin", NVTX_C_SETUP);
    gershgorinBounds(comm, A, &a, &b);
    NVTX_RANGE_POP();
  }
  if (commIsMaster(comm)) {
    printf("ChebFD spectrum bounds [a,b] = [%.6g, %.6g]\n", a, b);
  }
  // Alg. 3.1, Step 2 estimate Nt directly from par file

  // Alg. 3.1, Step 3: build the degree-Np filter polynomial p(H).
  ChebFilter f;
  if (chebFilterInit(&f,
          a,
          b,
          param->cheb.lam_lo,
          param->cheb.lam_hi,
          param->cheb.Np,
          mapKernel(param->cheb.kernel),
          param->cheb.mu) != 0) {
    NVTX_RANGE_POP();
    return -1;
  }
  if (commIsMaster(comm)) {
    chebFilterPrint(&f);
  }

  double tol    = param->eps > 0.0 ? param->eps : 1e-8;
  int maxiter   = param->itermax > 0 ? param->itermax : 50;
  CG_UINT nr    = A->nr;
  int NS        = param->cheb.NS;
  double lam_lo = param->cheb.lam_lo;
  double lam_hi = param->cheb.lam_hi;

  ChebData d;
  NVTX_RANGE_PUSH_C("ChebFD.setup.alloc", NVTX_C_SETUP);
  allocChebData(&d, A, NS);
  NVTX_RANGE_POP();

  DMatrix *Y      = &d.Y;
  DMatrix *AY     = &d.AY;
  DMatrix *u      = &d.u;
  DMatrix *w      = &d.w;
  V_ELE *avbuf    = d.avbuf;
  double *evk     = d.evk;
  double *H       = d.H;
  double *eval    = d.eval;
  double *evec    = d.evec;
  double *accEval = d.accEval;
  int *sel        = d.sel;
  double *res2    = d.res2;
#if CHEB_GPU
  (void)u; /* host-recurrence scratch; the GPU solver streams Y instead */
  (void)w;
  (void)avbuf;
  (void)evk;
#endif

  // Alg. 3.1, Step 4: construct NS random search vectors (host block).
  NVTX_RANGE_PUSH_C("ChebFD.setup.randomInit", NVTX_C_SETUP);
  randomInitBlock(comm, nr, Y->nr, Y->entries, NS);
  NVTX_RANGE_POP();

#if CHEB_GPU
  /* One-time setup outside the timed region: pull the managed matrix to
   * the device (no first-touch migration inside the loop) and create the
   * search-space streaming context. */
  int nb = (param->chebNb > 0) ? param->chebNb : CHEB_DEFAULT_NB;
  if (nb > NS) {
    if (param->chebNb > NS && commIsMaster(comm)) {
      printf("ChebFD: cheb_nb=%d > NS=%d; clamping to NS.\n", param->chebNb, NS);
    }
    nb = NS;
  }
  NVTX_RANGE_PUSH_C("ChebFD.setup.prefetch", NVTX_C_SETUP);
  gpu_matrix_prefetch(A);
  NVTX_RANGE_POP();
  GpuVectorStream *vs = gpu_vstream_init(A, NS, nb, 0, param->verbose);
  if (vs == NULL) {
    if (commIsMaster(comm)) {
      printf("ChebFD: gpu_vstream_init failed (bad arguments?).\n");
    }
    freeChebData(&d);
    chebFilterFree(&f);
    NVTX_RANGE_POP();
    return -1;
  }
  if (commIsMaster(comm)) {
    printf("ChebFD: matrix device-resident (managed, prefetched); search space "
           "streamed from pinned host memory in %d-column sub-blocks\n",
        nb);
  }
#endif

  int NT_found = 0;
  /* Throughput accounting over all matrix passes (filter + Rayleigh-Ritz),
   * reported at the end so config comparisons need only one run each. */
  double filterFlops = 0.0, filterBytes = 0.0, rrFlops = 0.0, rrBytes = 0.0;
  unsigned long long nFilterPasses = 0, nRRPasses = 0;
  int iter;
  /* All timing flows through the section timer (section_timer.h): wall
   * seconds per section, and on GPU builds also device (event) seconds —
   * the difference is host overhead. CHEBT_SPAN spans the whole loop. */
  enum {
    CHEBT_FILTER,
    CHEBT_ORTHO,
    CHEBT_RR,
    CHEBT_RESID,
    CHEBT_SPAN,
    CHEBT_NUM,
  };
  SectionTimer *chebTimer = SECTION_TIMER_CREATE(CHEBT_NUM);
  SECTION_TIMER_START(chebTimer, CHEBT_SPAN);
  for (iter = 1; iter <= maxiter; iter++) {
    /* Fold in the previous iteration's event pairs. */
    SECTION_TIMER_SYNC(chebTimer);
    NVTX_RANGE_PUSHF(NVTX_C_FILTER, "ChebFD.iter=%d", iter);
    // Alg. 3.1, Step 5 (Fig. 6): apply the polynomial filter to the whole subspace.
    Y->nc = NS;
    SECTION_TIMER_START(chebTimer, CHEBT_FILTER);
    NVTX_RANGE_PUSH_C("ChebFD.filter", NVTX_C_FILTER);
#if CHEB_GPU
    gpu_vstream_filter(vs, Y->entries, NS, f.alpha, f.beta, f.gc, f.Np);
#else
    applyFilter(A, &f, Y, u, w);
#endif
    NVTX_RANGE_POP();
    SECTION_TIMER_STOP(chebTimer, CHEBT_FILTER);
    accountMatvec(A, NS, param->cheb.Np, &filterFlops, &filterBytes);
    nFilterPasses += param->cheb.Np;

    // Alg. 3.1, Step 6: orthogonalize the filtered search vectors (rank revealing).
    SECTION_TIMER_START(chebTimer, CHEBT_ORTHO);
    NVTX_RANGE_PUSH_C("ChebFD.ortho", NVTX_C_ORTHO);
#if CHEB_GPU
    int m = chebOrthoCholQR2(vs, Y->entries, NS, 1e-8, H, eval, evec, 2);
#else
    int m = ORTHOMGSFUNC(nr, Y->entries, NS, 1e-8);
#endif
    NVTX_RANGE_POP();
    SECTION_TIMER_STOP(chebTimer, CHEBT_ORTHO);
    if (m == 0) {
      if (commIsMaster(comm)) {
        printf("iter %d: search space collapsed to rank 0.\n", iter);
      }
      NVTX_RANGE_POP();
      break;
    }
    Y->nc = m;

#if !CHEB_GPU
    /* orthoMGS repacks only rows [0,nr); re-zero SCS padding at the new
     * stride m, or applyFilter's recurrence blows it up to Inf/NaN. (The
     * streamed block update runs over all vecRows rows, so the GPU path's
     * padding rows stay zero by construction.) */
    NVTX_RANGE_PUSH_C("ChebFD.repad", NVTX_C_ORTHO);
    for (CG_UINT r = nr; r < Y->nr; r++) {
      for (int i = 0; i < m; i++) {
        Y->entries[r * (CG_UINT)m + i] = (V_ELE)0.0;
      }
    }
    NVTX_RANGE_POP();
#endif

    // Alg. 3.1, Step 7: Rayleigh-Ritz — project H = YᵀAY and solve for Ritz pairs.
    SECTION_TIMER_START(chebTimer, CHEBT_RR);
    NVTX_RANGE_PUSH_C("ChebFD.rr", NVTX_C_RR);
#if CHEB_GPU
    AY->nc = m;
    gpu_vstream_spmmv(vs, Y->entries, AY->entries, m);
    gpu_vstream_gram(vs, Y->entries, AY->entries, m, H);
    jacobiEigen(H, m, eval, evec);
#else
    rayleighRitz(A, Y, AY, m, nr, H, eval, evec);
#endif
    NVTX_RANGE_POP();
    SECTION_TIMER_STOP(chebTimer, CHEBT_RR);
    accountMatvec(A, m, 1, &rrFlops, &rrBytes); /* one A*Y pass */
    nRRPasses += 1;

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
    double maxres    = 0.0; /* max over the ACCEPTED pairs; drives convergence */
    double minres_in = DBL_MAX;
    double maxres_in = 0.0; /* max over ALL in-interval pairs; for reporting */
    int nInInterval  = 0;
    double accThresh = sqrt(tol);
    SECTION_TIMER_START(chebTimer, CHEBT_RESID);
    NVTX_RANGE_PUSH_C("ChebFD.residual", NVTX_C_RESID);
    /* Residuals of every in-interval pair: one streamed pass over Y / AY on
     * the GPU, per-pair host kernels otherwise. */
    int nsel = 0;
    for (int k = 0; k < m; k++) {
      if (eval[k] >= lam_lo && eval[k] <= lam_hi) {
        sel[nsel++] = k;
      }
    }
#if CHEB_GPU
    gpu_vstream_ritzResiduals(vs, Y->entries, AY->entries, m, eval, evec, sel, nsel, res2);
#else
    for (int t = 0; t < nsel; t++) {
      RITZRESIDUALFUNC(Y, AY, m, nr, eval[sel[t]], evec, sel[t], evk, avbuf);
      double r = residualNorm(nr, avbuf);
      res2[t]  = r * r;
    }
#endif
    for (int t = 0; t < nsel; t++) {
      int k      = sel[t];
      double res = sqrt(res2[t]);
      nInInterval++;
      if (res < minres_in) {
        minres_in = res;
      }
      if (res > maxres_in) {
        maxres_in = res;
      }
      if (res <= accThresh) { /* accept (discard ghosts with large residual) */
        accEval[NT_found] = eval[k];
        NT_found++;
        if (res > maxres) {
          maxres = res;
        }
      }
    }
    NVTX_RANGE_POP();
    SECTION_TIMER_STOP(chebTimer, CHEBT_RESID);
    if (commIsMaster(comm) && param->verbose) {
      printf("  [dbg] in-interval residuals over %d pairs: min=%.3e max=%.3e "
             "(accept thr=%.3e)\n",
          nInInterval,
          nInInterval ? minres_in : 0.0,
          maxres_in,
          accThresh);
    }

    if (commIsMaster(comm)) {
      /* Max over ALL in-interval pairs: maxres covers only accepted ones and
       * would print a bogus 0.000e+00 when nothing is accepted yet. */
      printf("iter %d: search rank m=%d, target pairs found=%d, "
             "max residual=%.3e (accepted max=%.3e)\n",
          iter,
          m,
          NT_found,
          maxres_in,
          maxres);
    }

    if (NT_found > 0 && maxres <= tol) {
      if (commIsMaster(comm)) {
        printf("ChebFD converged after %d iterations.\n", iter);
      }
      NVTX_RANGE_POP();
      break;
    }

    /* Alg. 3.1, Step 8b: restart from the m orthonormal filtered vectors. */
    NS = m;
    NVTX_RANGE_POP();
  }
  SECTION_TIMER_STOP(chebTimer, CHEBT_SPAN);

#if CHEB_GPU
  {
    size_t h2d = 0, d2h = 0;
    unsigned long long colPasses = 0, rowPasses = 0;
    gpu_vstream_stats(vs, &h2d, &d2h, &colPasses, &rowPasses);
    if (commIsMaster(comm)) {
      double gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
      printf("  search-space streaming: %.2f GiB H2D, %.2f GiB D2H over %llu "
             "column passes and %llu row-chunk passes\n",
          (double)h2d * gib,
          (double)d2h * gib,
          colPasses,
          rowPasses);
    }
    gpu_vstream_free(vs);
  }
#endif

  /* `iter` is maxiter+1 when the loop ran to exhaustion; report what ran. */
  int itersRun = (iter > maxiter) ? maxiter : iter;

  SECTION_TIMER_SYNC(chebTimer); /* last iteration's pairs + the span */

  if (commIsMaster(comm)) {
#if SECTION_TIMER_ON
    printf("ChebFD finished after %d iterations in %.2fs\n",
        itersRun,
        SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_SPAN));
    printf("  step breakdown: filter %.2fs, ortho %.2fs, rayleigh-ritz %.2fs, "
           "residual %.2fs\n",
        SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_FILTER),
        SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_ORTHO),
        SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_RR),
        SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_RESID));
    /* Must stay below the "step breakdown" line: runBench.sh's sed keeps
     * the first regex match, and this wording matches it too. */
    printf("  section time: filter %.2fs, ortho %.2fs, rayleigh-ritz %.2fs, "
           "residual %.2fs\n",
        SECTION_TIMER_SEC(chebTimer, CHEBT_FILTER),
        SECTION_TIMER_SEC(chebTimer, CHEBT_ORTHO),
        SECTION_TIMER_SEC(chebTimer, CHEBT_RR),
        SECTION_TIMER_SEC(chebTimer, CHEBT_RESID));
#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
    double devSpan  = SECTION_TIMER_SEC(chebTimer, CHEBT_SPAN);
    double wallSpan = SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_SPAN);
    /* The gap is host bookkeeping plus unsectioned kernels (e.g. repad). */
    printf("  device vs host: %.2fs of the %.2fs solve span spent on device "
           "(%.1f%%), host/other %.2fs\n",
        devSpan,
        wallSpan,
        wallSpan > 0.0 ? 100.0 * devSpan / wallSpan : 0.0,
        wallSpan - devSpan);
#endif
#else
    printf("ChebFD finished after %d iterations\n", itersRun);
#endif
    /* Same cost model as accountMatvec. Filter passes dominate (Np per
     * iteration vs one Rayleigh-Ritz pass), hence the separate figure. */
    double matvecFlops         = filterFlops + rrFlops;
    double matvecBytes         = filterBytes + rrBytes;
    double tMatvec             = SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_FILTER) +
                                 SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_RR);
    unsigned long long nPasses = nFilterPasses + nRRPasses;
    if (tMatvec > 0.0 && nPasses > 0) {
      printf("  matvec throughput: %.2f GFlop/s, %.1f GB/s "
             "(%llu passes, %.2f ms/pass; filter alone: %.2f ms/pass)\n",
          1.0e-9 * matvecFlops / tMatvec,
          1.0e-9 * matvecBytes / tMatvec,
          nPasses,
          1.0e3 * tMatvec / (double)nPasses,
          nFilterPasses > 0
              ? 1.0e3 * SECTION_TIMER_WALL_SEC(chebTimer, CHEBT_FILTER) /
                (double)nFilterPasses
              : 0.0);
    }
    printf("Found %d eigenpairs in target interval [%.6g, %.6g]:\n",
        NT_found,
        lam_lo,
        lam_hi);
    /* Accepted eigenvalues from the last iteration; earlier in-interval
     * Ritz values may include unconverged ghosts. */
    for (int i = 0; i < NT_found; i++) {
      printf("  lambda = %.10f\n", accEval[i]);
    }
  }

  SECTION_TIMER_FREE(chebTimer);
  freeChebData(&d);
  chebFilterFree(&f);

  NVTX_RANGE_POP();
  return NT_found;
}

/* Allocate one row-major (vecRows x NS) block; size_t math vs 32-bit overflow.
 * Y / AY are pinned host memory (allocateHost): the GPU solver streams them
 * through the device, the CPU build reads them directly. The u / w scratch
 * blocks only serve the host recurrence (CPU build, unit tests) and stay on
 * the general allocator, which on GPU builds is managed memory that costs
 * nothing until touched. */
static void allocDMat(DMatrix *M, CG_UINT vecRows, int NS, int pinned)
{
  M->nr      = vecRows;
  M->nc      = NS;
  size_t sz  = (size_t)vecRows * (size_t)NS * sizeof(V_ELE);
  M->entries = pinned ? (V_ELE *)allocateHost(sz) : (V_ELE *)allocate(ARRAY_ALIGNMENT, sz);
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

  /* Row-major blocks; .nc shrinks during iterations, storage stays NS-wide. */
  allocDMat(&d->Y, vecRows, NS, 1);
  allocDMat(&d->AY, vecRows, NS, 1);
  allocDMat(&d->u, vecRows, NS, 0);
  allocDMat(&d->w, vecRows, NS, 0);

  /* avbuf is written by the Ritz residual kernel and consumed by ddot on
   * the host path; the dense arrays below are host-read (jacobiEigen). */
  d->avbuf   = (V_ELE *)allocateDevice((size_t)nr * sizeof(V_ELE));
  d->evk     = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));

  d->H       = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
  d->eval    = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));
  d->evec    = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
  d->accEval = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));
  d->sel     = (int *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(int));
  d->res2    = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));
}

void freeChebData(ChebData *d)
{
  deallocateHost(d->Y.entries);
  deallocateHost(d->AY.entries);
  deallocate(d->u.entries);
  deallocate(d->w.entries);
  deallocateDevice(d->avbuf);
  deallocate(d->evk);
  deallocate(d->H);
  deallocate(d->eval);
  deallocate(d->evec);
  deallocate(d->accEval);
  deallocate(d->sel);
  deallocate(d->res2);
}
