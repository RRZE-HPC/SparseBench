/* Unit tests for the GPU search-space streaming (cuda_vector_stream.cu):
 * the matrix is device-resident, the dense blocks live in pinned host
 * memory and are pulled through the device in column sub-blocks (filter,
 * A*Y) and row chunks (Gram, block update, Ritz residuals). Every streamed
 * result is checked against the resident full-width GPU kernels on the
 * same inputs; sub-block widths and chunk sizes are chosen so that partial
 * last sub-blocks and multi-chunk pipelines are exercised on a tiny
 * matrix. GPU builds only; trivially passes elsewhere. */
#include "chebFDStreamTests.h"

#include <stdio.h>

#if !defined(RUNTIME_BACKEND_IS_CUDA) && !defined(RUNTIME_BACKEND_IS_HIP)

int chebFDStreamTests(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  printf("Skipping ChebFD streaming tests (CPU build; GPU-only feature).\n");
  return 0;
}

#else

#ifdef USE_COMPLEX

int chebFDStreamTests(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  printf("Skipping ChebFD streaming tests (USE_COMPLEX build; ChebFD v1 is "
         "real-symmetric only).\n");
  return 0;
}

#else

#include "../../src/allocate.h"
#include "../../src/chebFDSolver.h"
#include "../../src/chebFilter.h"
#include "../../src/comm.h"
#include "../../src/cuda_kernels.h"
#include "../../src/matrix.h"
#include "../../src/parameter.h"
#include "../../src/solver.h"
#include "../common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- shared helpers (conventions from chebFDUnitTests.c) --------------- */

/* 1-D Laplacian tridiagonal, converted with an explicit SCS chunk height /
 * sigma so multi-lane chunks, padding rows and the sigma permutation are
 * exercised. */
static void buildTridiagMatrixCS(Matrix *A, GMatrix *gm, int n, int C, int sigma)
{
  memset(gm, 0, sizeof(*gm));
  gm->nr       = (CG_UINT)n;
  gm->nc       = (CG_UINT)n;
  gm->nnz      = (CG_UINT)(3 * n - 2);
  gm->totalNr  = (CG_UINT)n;
  gm->totalNnz = gm->nnz;
  gm->startRow = 0;
  gm->stopRow  = (CG_UINT)n;
  gm->rowPtr   = (CG_UINT *)allocate(ARRAY_ALIGNMENT, (size_t)(n + 1) * sizeof(CG_UINT));
  gm->entries  = (Entry *)allocate(ARRAY_ALIGNMENT, (size_t)gm->nnz * sizeof(Entry));

  CG_UINT idx = 0;
  for (int i = 0; i < n; i++) {
    gm->rowPtr[i] = idx;
    if (i > 0) {
      gm->entries[idx].col = (CG_UINT)(i - 1);
      gm->entries[idx].val = (V_ELE)(-1.0);
      idx++;
    }
    gm->entries[idx].col = (CG_UINT)i;
    gm->entries[idx].val = (V_ELE)2.0;
    idx++;
    if (i < n - 1) {
      gm->entries[idx].col = (CG_UINT)(i + 1);
      gm->entries[idx].val = (V_ELE)(-1.0);
      idx++;
    }
  }
  gm->rowPtr[n] = idx;

  memset(A, 0, sizeof(*A));
#ifdef SCS
  A->C     = (CG_UINT)C;
  A->sigma = (CG_UINT)sigma;
#else
  (void)C;
  (void)sigma;
#endif
  convertMatrix(A, gm);
  gpu_matrix_prefetch(A);
}

static CG_UINT vecRowsOf(Matrix *A)
{
#ifdef SCS
  return A->nrPadded;
#else
  return A->nr;
#endif
}

static void tridiagEigenvalue(int n, int k, double *lambda)
{
  *lambda = 2.0 - 2.0 * cos((double)k * M_PI / (double)(n + 1));
}

static unsigned long long splitmix64Local(unsigned long long z)
{
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/* Random fill of rows [0, nr), zero padding rows [nr, vecRows). */
static void fillRandomBlock(
    V_ELE *e, CG_UINT vecRows, CG_UINT nr, int nc, unsigned long long seed)
{
  for (CG_UINT r = 0; r < vecRows; r++) {
    for (int c = 0; c < nc; c++) {
      double rv = 0.0;
      if (r < nr) {
        unsigned long long h = splitmix64Local(seed + r * 0x9E3779B97F4A7C15ull +
                                               (unsigned long long)c * 0xff51afd7ed558ccdull);
        rv                   = (double)(h >> 11) / (double)(1ull << 53) * 2.0 - 1.0;
      }
      e[r * (CG_UINT)nc + (CG_UINT)c] = (V_ELE)rv;
    }
  }
}

static double maxAbsDiff(const V_ELE *a, const V_ELE *b, size_t sz)
{
  double maxd = 0.0;
  for (size_t i = 0; i < sz; i++) {
    maxd = fmax(maxd, fabs((double)(a[i] - b[i])));
  }
  return maxd;
}

static double maxAbsDiffD(const double *a, const double *b, size_t sz)
{
  double maxd = 0.0;
  for (size_t i = 0; i < sz; i++) {
    maxd = fmax(maxd, fabs(a[i] - b[i]));
  }
  return maxd;
}

/* Managed (device-visible) block for the resident reference kernels. */
static DMatrix makeBlock(CG_UINT rows, int nc)
{
  DMatrix m;
  m.nr      = rows;
  m.nc      = (CG_UINT)nc;
  m.entries = (V_ELE *)allocate(ARRAY_ALIGNMENT, (size_t)rows * (size_t)nc * sizeof(V_ELE));
  return m;
}

/* Pinned host block, as the solver streams it. */
static DMatrix makeHostBlock(CG_UINT rows, int nc)
{
  DMatrix m;
  m.nr      = rows;
  m.nc      = (CG_UINT)nc;
  m.entries = (V_ELE *)allocateHost((size_t)rows * (size_t)nc * sizeof(V_ELE));
  return m;
}

#define CHECK(cond, msg, ...)                                                            \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      printf("    FAIL: " msg "\n", ##__VA_ARGS__);                                      \
      ok = 0;                                                                            \
    }                                                                                    \
  } while (0)

/* Tiny row chunks (16 rows at NS*8 B each) so every row-chunk pass on the
 * 44-row test block runs three chunks and hits the write-back tail. */
static size_t tinyChunkBytes(int NS)
{
  return (size_t)16 * (size_t)NS * sizeof(V_ELE);
}

/* ---- Section 1: filter, block-outer streamed vs resident degree-outer -- */

static int testFilterParity(void)
{
  int ok = 1;
  printf("  gpu_vstream_filter vs applyFilter (resident):\n");

  const int n = 42, C = 4, sigma = 4; /* nChunks=11, 2 padding rows */
  Matrix A;
  GMatrix gm;
  buildTridiagMatrixCS(&A, &gm, n, C, sigma);
  CG_UINT vecRows = vecRowsOf(&A);

  const int NS = 7;
  ChebFilter f;
  double lamLo, lamHi;
  tridiagEigenvalue(n, n / 2, &lamLo);
  lamHi = lamLo + 0.3;
  lamLo -= 0.1;
  CHECK(chebFilterInit(&f, 0.0, 4.0, lamLo, lamHi, 60, KERNEL_LANCZOS, 2) == 0,
      "chebFilterInit should succeed");

  DMatrix y0   = makeBlock(vecRows, NS);
  DMatrix yRef = makeBlock(vecRows, NS), uRef = makeBlock(vecRows, NS),
          wRef = makeBlock(vecRows, NS);
  DMatrix yStr = makeHostBlock(vecRows, NS);
  size_t sz    = (size_t)vecRows * (size_t)NS;
  fillRandomBlock(y0.entries, vecRows, A.nr, NS, 0xbeef);

  memcpy(yRef.entries, y0.entries, sz * sizeof(V_ELE));
  applyFilter(&A, &f, &yRef, &uRef, &wRef);

  /* nb 2 -> column-pair kernel, nb 4 -> column-quad kernel (+ a 3-wide
   * scalar tail), nb 3 -> scalar, nb NS -> exact fit, one sub-block. */
  const int nbs[4] = { 2, 4, 3, NS };
  for (int i = 0; i < 4; i++) {
    GpuVectorStream *vs = gpu_vstream_init(&A, NS, nbs[i], tinyChunkBytes(NS), 0);
    CHECK(vs != NULL, "gpu_vstream_init(nb=%d) failed", nbs[i]);
    if (vs == NULL) {
      continue;
    }
    memcpy(yStr.entries, y0.entries, sz * sizeof(V_ELE));
    gpu_vstream_filter(vs, yStr.entries, NS, f.alpha, f.beta, f.gc, f.Np);
    double maxd = maxAbsDiff(yRef.entries, yStr.entries, sz);
    CHECK(maxd < 1e-10, "filter nb=%d max|diff|=%.3e", nbs[i], maxd);
    /* Narrower column count than the allocation width (NS shrinks to m). */
    int nc = NS - 2;
    for (CG_UINT r = 0; r < vecRows; r++) {
      for (int c = 0; c < nc; c++) {
        yStr.entries[r * nc + c] = y0.entries[r * NS + c]; /* pack at stride nc */
      }
    }
    gpu_vstream_filter(vs, yStr.entries, nc, f.alpha, f.beta, f.gc, f.Np);
    double maxdn = 0.0;
    for (CG_UINT r = 0; r < vecRows; r++) {
      for (int c = 0; c < nc; c++) {
        maxdn = fmax(maxdn,
            fabs((double)(yRef.entries[r * NS + c] - yStr.entries[r * nc + c])));
      }
    }
    CHECK(maxdn < 1e-10, "filter nb=%d nc=%d max|diff|=%.3e", nbs[i], nc, maxdn);
    gpu_vstream_free(vs);
  }

  deallocate(y0.entries);
  deallocate(yRef.entries);
  deallocate(uRef.entries);
  deallocate(wRef.entries);
  deallocateHost(yStr.entries);
  chebFilterFree(&f);
  freeMatrix(&A);
  freeGMatrix(&gm);

  printf(ok ? "    PASS\n" : "");
  return ok;
}

/* ---- Section 2: A*Y, Gram and residuals vs resident kernels ------------ */

static int testDensePassParity(void)
{
  int ok = 1;
  printf("  gpu_vstream_spmmv / gram / ritzResiduals vs resident kernels:\n");

  const int n = 42, C = 4, sigma = 4;
  Matrix A;
  GMatrix gm;
  buildTridiagMatrixCS(&A, &gm, n, C, sigma);
  CG_UINT vecRows = vecRowsOf(&A);
  const int NS    = 6;
  size_t sz       = (size_t)vecRows * (size_t)NS;

  DMatrix y = makeBlock(vecRows, NS), ayRef = makeBlock(vecRows, NS);
  DMatrix yH = makeHostBlock(vecRows, NS), ayH = makeHostBlock(vecRows, NS);
  fillRandomBlock(y.entries, vecRows, A.nr, NS, 0x1234);
  memcpy(yH.entries, y.entries, sz * sizeof(V_ELE));

  GpuVectorStream *vs = gpu_vstream_init(&A, NS, 4, tinyChunkBytes(NS), 0);
  CHECK(vs != NULL, "gpu_vstream_init failed");
  if (vs != NULL) {
    /* AY */
    gpu_spMMVM(&A, &y, &ayRef);
    gpu_vstream_spmmv(vs, yH.entries, ayH.entries, NS);
    double maxd = maxAbsDiff(ayRef.entries, ayH.entries, sz);
    CHECK(maxd < 1e-12, "spmmv max|diff|=%.3e", maxd);

    /* Gram Y^T AY and Y^T Y */
    double *Href = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
    double *Hstr = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
    gpu_gramYtAY(A.nr, NS, y.entries, ayRef.entries, Href);
    gpu_vstream_gram(vs, yH.entries, ayH.entries, NS, Hstr);
    maxd = maxAbsDiffD(Href, Hstr, (size_t)NS * NS);
    CHECK(maxd < 1e-10, "gram Y^T AY max|diff|=%.3e", maxd);
    gpu_gramYtAY(A.nr, NS, y.entries, y.entries, Href);
    gpu_vstream_gram(vs, yH.entries, NULL, NS, Hstr);
    maxd = maxAbsDiffD(Href, Hstr, (size_t)NS * NS);
    CHECK(maxd < 1e-10, "gram Y^T Y max|diff|=%.3e", maxd);
    int sym = 1;
    for (int i = 0; i < NS; i++) {
      for (int j = 0; j < NS; j++) {
        sym = sym && (Hstr[i * NS + j] == Hstr[j * NS + i]);
      }
    }
    CHECK(sym, "streamed Gram must be exactly symmetric");

    /* Ritz residuals for a subset of pairs, arbitrary evec / eval. */
    double *evec = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
    double *eval = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));
    double *evk  = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));
    V_ELE *avbuf = (V_ELE *)allocateDevice((size_t)A.nr * sizeof(V_ELE));
    for (int i = 0; i < NS * NS; i++) {
      evec[i] = (double)((i * 7919) % 23) / 23.0 - 0.5;
    }
    for (int k = 0; k < NS; k++) {
      eval[k] = 0.3 * k + 0.1;
    }
    int sel[3]     = { 0, 2, 5 };
    double res2[3] = { 0, 0, 0 };
    gpu_vstream_ritzResiduals(vs, yH.entries, ayH.entries, NS, eval, evec, sel, 3, res2);
    for (int t = 0; t < 3; t++) {
      int k = sel[t];
      gpu_computeRitzResidual(&y, &ayRef, NS, A.nr, eval[k], evec, k, evk, avbuf);
      V_ELE r2;
      gpu_ddot_sync(A.nr, avbuf, avbuf, &r2);
      double d = fabs((double)r2 - res2[t]);
      CHECK(d < 1e-10 * fmax(1.0, fabs((double)r2)),
          "ritz residual k=%d: resident %.12e streamed %.12e",
          k,
          (double)r2,
          res2[t]);
    }
    deallocate(evec);
    deallocate(eval);
    deallocate(evk);
    deallocateDevice(avbuf);
    deallocate(Href);
    deallocate(Hstr);
    gpu_vstream_free(vs);
  }

  deallocate(y.entries);
  deallocate(ayRef.entries);
  deallocateHost(yH.entries);
  deallocateHost(ayH.entries);
  freeMatrix(&A);
  freeGMatrix(&gm);

  printf(ok ? "    PASS\n" : "");
  return ok;
}

/* ---- Section 3: block update and Cholesky-QR ---------------------------- */

static int testUpdateAndOrtho(void)
{
  int ok = 1;
  printf("  gpu_vstream_update / chebOrthoCholQR2:\n");

  const int n = 42, C = 4, sigma = 4;
  Matrix A;
  GMatrix gm;
  buildTridiagMatrixCS(&A, &gm, n, C, sigma);
  CG_UINT vecRows = vecRowsOf(&A);
  const int NS    = 6;
  size_t sz       = (size_t)vecRows * (size_t)NS;

  DMatrix y0 = makeBlock(vecRows, NS);
  DMatrix yH = makeHostBlock(vecRows, NS);
  fillRandomBlock(y0.entries, vecRows, A.nr, NS, 0x4321);

  GpuVectorStream *vs = gpu_vstream_init(&A, NS, 3, tinyChunkBytes(NS), 0);
  CHECK(vs != NULL, "gpu_vstream_init failed");
  if (vs != NULL) {
    /* Y <- Y B with B = 2 * [I_4 ; 0] (m=6 -> mOut=4): compaction + scale. */
    const int mOut = 4;
    double *B      = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * mOut * sizeof(double));
    for (int i = 0; i < NS; i++) {
      for (int j = 0; j < mOut; j++) {
        B[i * mOut + j] = (i == j) ? 2.0 : 0.0;
      }
    }
    memcpy(yH.entries, y0.entries, sz * sizeof(V_ELE));
    gpu_vstream_update(vs, yH.entries, NS, B, mOut);
    double maxd = 0.0;
    for (CG_UINT r = 0; r < vecRows; r++) {
      for (int j = 0; j < mOut; j++) {
        maxd = fmax(maxd,
            fabs((double)yH.entries[r * mOut + j] - 2.0 * (double)y0.entries[r * NS + j]));
      }
    }
    CHECK(maxd < 1e-12, "update (compact to %d, x2) max|diff|=%.3e", mOut, maxd);
    deallocate(B);

    /* Cholesky-QR2: full rank -> m = NS and Y^T Y = I. */
    double *G    = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
    double *eval = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * sizeof(double));
    double *evec = (double *)allocate(ARRAY_ALIGNMENT, (size_t)NS * NS * sizeof(double));
    memcpy(yH.entries, y0.entries, sz * sizeof(V_ELE));
    int m = chebOrthoCholQR2(vs, yH.entries, NS, 1e-8, G, eval, evec, 2);
    CHECK(m == NS, "full-rank block: m=%d, expected %d", m, NS);
    if (m > 0) {
      gpu_vstream_gram(vs, yH.entries, NULL, m, G);
      double offd = 0.0;
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          offd = fmax(offd, fabs(G[i * m + j] - (i == j ? 1.0 : 0.0)));
        }
      }
      CHECK(offd < 1e-12, "Y^T Y - I max|.|=%.3e after CholQR2", offd);
      /* padding rows must still be zero */
      double pad = 0.0;
      for (CG_UINT r = A.nr; r < vecRows; r++) {
        for (int j = 0; j < m; j++) {
          pad = fmax(pad, fabs((double)yH.entries[r * m + j]));
        }
      }
      CHECK(pad == 0.0, "padding rows not zero after update (%.3e)", pad);
    }

    /* Rank deficient: column 3 := column 1 -> exactly one direction dropped. */
    memcpy(yH.entries, y0.entries, sz * sizeof(V_ELE));
    for (CG_UINT r = 0; r < vecRows; r++) {
      yH.entries[r * NS + 3] = yH.entries[r * NS + 1];
    }
    m = chebOrthoCholQR2(vs, yH.entries, NS, 1e-8, G, eval, evec, 2);
    CHECK(m == NS - 1, "rank-deficient block: m=%d, expected %d", m, NS - 1);
    if (m > 0) {
      gpu_vstream_gram(vs, yH.entries, NULL, m, G);
      double offd = 0.0;
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          offd = fmax(offd, fabs(G[i * m + j] - (i == j ? 1.0 : 0.0)));
        }
      }
      CHECK(offd < 1e-12, "Y^T Y - I max|.|=%.3e after rank-deficient CholQR2", offd);
    }
    deallocate(G);
    deallocate(eval);
    deallocate(evec);
    gpu_vstream_free(vs);
  }

  deallocate(y0.entries);
  deallocateHost(yH.entries);
  freeMatrix(&A);
  freeGMatrix(&gm);

  printf(ok ? "    PASS\n" : "");
  return ok;
}

/* ---- Section 4: solveChebFD end-to-end --------------------------------- */

static int testSolveChebFD(void)
{
  int ok = 1;
  printf("  solveChebFD (end-to-end, streamed search space):\n");

  const int n = 42, C = 4, sigma = 4;
  Matrix A;
  GMatrix gm;
  buildTridiagMatrixCS(&A, &gm, n, C, sigma);

  /* Window around three consecutive interior eigenvalues. */
  int kMid = n / 2;
  double lamLo, lamHi;
  tridiagEigenvalue(n, kMid - 1, &lamLo);
  tridiagEigenvalue(n, kMid + 1, &lamHi);
  lamLo -= 0.05;
  lamHi += 0.05;

  CommType comm;
  memset(&comm, 0, sizeof(comm));
  comm.rank = 0;
  comm.size = 1;

  Parameter param;
  memset(&param, 0, sizeof(param));
  param.eps              = 1e-10;
  param.itermax          = 60;
  param.verbose          = 0;
  param.allocType        = ALLOC_MANAGED;
  param.cheb.lam_lo      = lamLo;
  param.cheb.lam_hi      = lamHi;
  param.cheb.Np          = 80;
  param.cheb.NS          = 10;
  param.cheb.kernel      = 3; /* Lanczos */
  param.cheb.mu          = 2;
  param.cheb.have_bounds = 0;
  param.cheb.have_target = 1;

  param.chebNb = 0; /* default width (clamped to NS) */
  int found1   = solveChebFD(&comm, &param, &A);
  CHECK(found1 == 3, "cheb_nb=0: found %d eigenpairs, expected 3", found1);

  param.chebNb = 4; /* three sub-blocks, partial last one */
  int found2   = solveChebFD(&comm, &param, &A);
  CHECK(found2 == 3, "cheb_nb=4: found %d eigenpairs, expected 3", found2);

  freeMatrix(&A);
  freeGMatrix(&gm);
  printf(ok ? "    PASS\n" : "");
  return ok;
}

int chebFDStreamTests(int argc, char **argv)
{
  (void)argc;
  (void)argv;

  printf("Running ChebFD search-space streaming tests:\n");

  int results[4];
  int i = 0;
  results[i++] = testFilterParity();
  results[i++] = testDensePassParity();
  results[i++] = testUpdateAndOrtho();
  results[i++] = testSolveChebFD();

  int passed = 0;
  for (int j = 0; j < i; j++) {
    passed += results[j] ? 1 : 0;
  }
  printf("\nSummary: %d/%d ChebFD streaming test sections passed.\n", passed, i);
  return (passed == i) ? 0 : 1;
}

#endif /* USE_COMPLEX */

#endif /* GPU backend */
