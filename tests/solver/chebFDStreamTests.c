/* Unit tests for GPU matrix streaming (host-resident matrix streamed in
 * double-buffered parts — cuda_matrix_stream.cu and the gpu_stream_* /
 * *_nb kernels in cuda_spmv_{scs,crs}.cu) plus the applyFilter /
 * rayleighRitz / solveChebFD wiring around them.
 *
 * Everything is checked against the resident full-width GPU kernels with
 * the same inputs. Parting, subblock tiling and strided views do not
 * change the per-row accumulation order, so results should match to the
 * last bit; the comparisons still use small tolerances to stay robust to
 * harmless reassociation. GPU builds only; trivially passes elsewhere
 * (matching the USE_COMPLEX handling in chebFDUnitTests.c, since ChebFD
 * v1 is real-symmetric only). */
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

/* The standard 1-D Laplacian tridiagonal matrix, converted with an
 * explicit SCS chunk height / sigma so multi-lane chunks, padding rows and
 * the sigma row permutation are all exercised (C=1, sigma=1 disables all
 * three and is used as the degenerate case). */
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

static void fillRandomBlock(V_ELE *e, CG_UINT nr, int nc, unsigned long long seed)
{
  for (CG_UINT r = 0; r < nr; r++) {
    for (int c = 0; c < nc; c++) {
      unsigned long long h = splitmix64Local(
          seed + r * 0x9E3779B97F4A7C15ull + (unsigned long long)c * 0xff51afd7ed558ccdull);
      double rv                     = (double)(h >> 11) / (double)(1ull << 53) * 2.0 - 1.0;
      e[r * (CG_UINT)nc + (CG_UINT)c] = (V_ELE)rv;
    }
  }
}

/* SCS padding rows stay zero: the filter recurrence would otherwise blow
 * them up (NaN by iteration's end), matching the contract in
 * chebFDSolver.c. */
static void zeroPadRows(V_ELE *e, CG_UINT vecRows, CG_UINT nr, int nc)
{
  for (CG_UINT r = nr; r < vecRows; r++) {
    for (int c = 0; c < nc; c++) {
      e[r * (CG_UINT)nc + (CG_UINT)c] = (V_ELE)0.0;
    }
  }
}

static double maxAbsDiff(const V_ELE *a, const V_ELE *b, CG_UINT sz)
{
  double maxd = 0.0;
  for (CG_UINT i = 0; i < sz; i++) {
    maxd = fmax(maxd, fabs((double)(a[i] - b[i])));
  }
  return maxd;
}

static DMatrix makeBlock(CG_UINT rows, int nc)
{
  DMatrix m;
  m.nr      = rows;
  m.nc      = (CG_UINT)nc;
  m.entries = (V_ELE *)allocate(
      ARRAY_ALIGNMENT, (size_t)rows * (size_t)nc * sizeof(V_ELE));
  return m;
}

#define CHECK(cond, msg, ...)                                                            \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      printf("    FAIL: " msg "\n", ##__VA_ARGS__);                                      \
      ok = 0;                                                                            \
    }                                                                                    \
  } while (0)

/* ---- Section 1: streamed part sweeps vs resident full-width kernels ---- */

static int testPartSweepParity(void)
{
  int ok = 1;
  printf("  gpu_stream_* part sweeps (multi-part and single-part):\n");

  const int n = 42, C = 4, sigma = 4; /* nChunks=11, 2 padding rows */
  Matrix A;
  GMatrix gm;
  buildTridiagMatrixCS(&A, &gm, n, C, sigma);
  CG_UINT vecRows = vecRowsOf(&A);

  const int nc = 5;
  CG_UINT sz   = vecRows * (CG_UINT)nc;
  DMatrix w = makeBlock(vecRows, nc), q = makeBlock(vecRows, nc);
  DMatrix yRef = makeBlock(vecRows, nc), yStr = makeBlock(vecRows, nc);
  DMatrix xRef = makeBlock(vecRows, nc), xStr = makeBlock(vecRows, nc);

  fillRandomBlock(w.entries, vecRows, nc, 0x1234);
  fillRandomBlock(q.entries, vecRows, nc, 0x5678);
  fillRandomBlock(xRef.entries, vecRows, nc, 0x9abc);
  zeroPadRows(w.entries, vecRows, A.nr, nc);
  zeroPadRows(q.entries, vecRows, A.nr, nc);
  zeroPadRows(xRef.entries, vecRows, A.nr, nc);

  V_ELE cA = (V_ELE)0.7, cP = (V_ELE)(-1.3), cQ = (V_ELE)2.1, gc = (V_ELE)0.37;

  const size_t partBytes[2] = { 192, ((size_t)1) << 30 };
  for (int ip = 0; ip < 2; ip++) {
    GpuMatrixStream *s = gpu_matrix_stream_init(&A, partBytes[ip], 0);
    CHECK(s != NULL, "gpu_matrix_stream_init(%zu) failed", partBytes[ip]);
    if (s == NULL) {
      continue;
    }
    int nParts = 0;
    gpu_matrix_stream_stats(s, NULL, &nParts, NULL, NULL);
    printf("    partBytes=%zu -> %d parts\n", partBytes[ip], nParts);
    CHECK(nParts >= 1, "nParts=%d", nParts);

    /* spMMVM */
    gpu_spMMVM(&A, &w, &yRef);
    gpu_stream_spMMVM(s, &w, &yStr, 0);
    CHECK(maxAbsDiff(yRef.entries, yStr.entries, sz) < 1e-12,
        "spMMVM partBytes=%zu max|diff|=%.3e",
        partBytes[ip],
        maxAbsDiff(yRef.entries, yStr.entries, sz));

    /* spMMVMFused, 3-term and 2-term */
    gpu_spMMVMFused(&A, &w, cA, &w, cP, &q, cQ, &yRef);
    gpu_stream_spMMVMFused(s, &w, cA, &w, cP, &q, cQ, &yStr, 0);
    CHECK(maxAbsDiff(yRef.entries, yStr.entries, sz) < 1e-12,
        "spMMVMFused (3-term) partBytes=%zu max|diff|=%.3e",
        partBytes[ip],
        maxAbsDiff(yRef.entries, yStr.entries, sz));
    gpu_spMMVMFused(&A, &w, cA, &w, cP, NULL, (V_ELE)0.0, &yRef);
    gpu_stream_spMMVMFused(s, &w, cA, &w, cP, NULL, (V_ELE)0.0, &yStr, 0);
    CHECK(maxAbsDiff(yRef.entries, yStr.entries, sz) < 1e-12,
        "spMMVMFused (2-term) partBytes=%zu max|diff|=%.3e",
        partBytes[ip],
        maxAbsDiff(yRef.entries, yStr.entries, sz));

    /* chebfdOp with the y==q in-place aliasing applyFilter relies on */
    memcpy(xStr.entries, xRef.entries, (size_t)sz * sizeof(V_ELE));
    gpu_chebfdOp(&A, &w, cA, cP, &yRef, cQ, &yRef, gc, &xRef);
    gpu_stream_chebfdOp(s, &w, cA, cP, &yStr, cQ, &yStr, gc, &xStr, 0);
    CHECK(maxAbsDiff(yRef.entries, yStr.entries, sz) < 1e-12,
        "chebfdOp (y==q) y partBytes=%zu max|diff|=%.3e",
        partBytes[ip],
        maxAbsDiff(yRef.entries, yStr.entries, sz));
    CHECK(maxAbsDiff(xRef.entries, xStr.entries, sz) < 1e-12,
        "chebfdOp (y==q) x partBytes=%zu max|diff|=%.3e",
        partBytes[ip],
        maxAbsDiff(xRef.entries, xStr.entries, sz));

    gpu_matrix_stream_free(s);
  }

  /* Degenerate SCS layout: C=1, sigma=1 (one row per chunk, no permutation,
   * no padding). */
  Matrix A1;
  GMatrix gm1;
  buildTridiagMatrixCS(&A1, &gm1, n, 1, 1);
  GpuMatrixStream *s1 = gpu_matrix_stream_init(&A1, 192, 0);
  CHECK(s1 != NULL, "gpu_matrix_stream_init (C=1) failed");
  if (s1 != NULL) {
    gpu_spMMVM(&A1, &w, &yRef);
    gpu_stream_spMMVM(s1, &w, &yStr, 0);
    CHECK(maxAbsDiff(yRef.entries, yStr.entries, sz) < 1e-12,
        "spMMVM (C=1) max|diff|=%.3e",
        maxAbsDiff(yRef.entries, yStr.entries, sz));
    gpu_matrix_stream_free(s1);
  }

  deallocate(w.entries);
  deallocate(q.entries);
  deallocate(yRef.entries);
  deallocate(yStr.entries);
  deallocate(xRef.entries);
  deallocate(xStr.entries);
  freeMatrix(&A);
  freeGMatrix(&gm);
  freeMatrix(&A1);
  freeGMatrix(&gm1);

  printf(ok ? "    PASS\n" : "");
  return ok;
}

/* ---- Section 2: resident subblock kernels (cheb_nb) -------------------- */

static int testResidentSubblock(void)
{
  int ok = 1;
  printf("  gpu_*_nb resident subblock kernels:\n");

  const int n = 42, C = 4, sigma = 4;
  Matrix A;
  GMatrix gm;
  buildTridiagMatrixCS(&A, &gm, n, C, sigma);
  CG_UINT vecRows = vecRowsOf(&A);

  const int nc = 5;
  CG_UINT sz   = vecRows * (CG_UINT)nc;
  DMatrix w = makeBlock(vecRows, nc), q = makeBlock(vecRows, nc);
  DMatrix yRef = makeBlock(vecRows, nc), yNb = makeBlock(vecRows, nc);
  DMatrix xRef = makeBlock(vecRows, nc), xNb = makeBlock(vecRows, nc);

  fillRandomBlock(w.entries, vecRows, nc, 0x4321);
  fillRandomBlock(q.entries, vecRows, nc, 0x8765);
  fillRandomBlock(xRef.entries, vecRows, nc, 0xcba9);
  zeroPadRows(w.entries, vecRows, A.nr, nc);
  zeroPadRows(q.entries, vecRows, A.nr, nc);
  zeroPadRows(xRef.entries, vecRows, A.nr, nc);

  V_ELE cA = (V_ELE)0.7, cP = (V_ELE)(-1.3), cQ = (V_ELE)2.1, gc = (V_ELE)0.37;

  const int nbs[4] = { 1, 3, 5, 7 }; /* includes == nc and > nc (clamped) */
  for (int i = 0; i < 4; i++) {
    int nb = nbs[i];

    gpu_spMMVM(&A, &w, &yRef);
    gpu_spMMVM_nb(&A, &w, &yNb, nb);
    CHECK(maxAbsDiff(yRef.entries, yNb.entries, sz) < 1e-12,
        "spMMVM nb=%d max|diff|=%.3e",
        nb,
        maxAbsDiff(yRef.entries, yNb.entries, sz));

    gpu_spMMVMFused(&A, &w, cA, &w, cP, &q, cQ, &yRef);
    gpu_spMMVMFused_nb(&A, &w, cA, &w, cP, &q, cQ, &yNb, nb);
    CHECK(maxAbsDiff(yRef.entries, yNb.entries, sz) < 1e-12,
        "spMMVMFused nb=%d max|diff|=%.3e",
        nb,
        maxAbsDiff(yRef.entries, yNb.entries, sz));

    memcpy(xNb.entries, xRef.entries, (size_t)sz * sizeof(V_ELE));
    gpu_chebfdOp(&A, &w, cA, cP, &yRef, cQ, &yRef, gc, &xRef);
    gpu_chebfdOp_nb(&A, &w, cA, cP, &yNb, cQ, &yNb, gc, &xNb, nb);
    CHECK(maxAbsDiff(yRef.entries, yNb.entries, sz) < 1e-12,
        "chebfdOp (y==q) nb=%d y max|diff|=%.3e",
        nb,
        maxAbsDiff(yRef.entries, yNb.entries, sz));
    CHECK(maxAbsDiff(xRef.entries, xNb.entries, sz) < 1e-12,
        "chebfdOp (y==q) nb=%d x max|diff|=%.3e",
        nb,
        maxAbsDiff(xRef.entries, xNb.entries, sz));
  }

  deallocate(w.entries);
  deallocate(q.entries);
  deallocate(yRef.entries);
  deallocate(yNb.entries);
  deallocate(xRef.entries);
  deallocate(xNb.entries);
  freeMatrix(&A);
  freeGMatrix(&gm);

  printf(ok ? "    PASS\n" : "");
  return ok;
}

/* ---- Section 3: streaming + subblock combined --------------------------- */

static int testStreamSubblock(void)
{
  int ok = 1;
  printf("  gpu_stream_* with cheb_nb subblock tiling:\n");

  const int n = 42, C = 4, sigma = 4;
  Matrix A;
  GMatrix gm;
  buildTridiagMatrixCS(&A, &gm, n, C, sigma);
  CG_UINT vecRows = vecRowsOf(&A);

  const int nc = 5;
  CG_UINT sz   = vecRows * (CG_UINT)nc;
  DMatrix w = makeBlock(vecRows, nc), q = makeBlock(vecRows, nc);
  DMatrix yRef = makeBlock(vecRows, nc), yStr = makeBlock(vecRows, nc);
  DMatrix xRef = makeBlock(vecRows, nc), xStr = makeBlock(vecRows, nc);

  fillRandomBlock(w.entries, vecRows, nc, 0x3141);
  fillRandomBlock(q.entries, vecRows, nc, 0x5926);
  fillRandomBlock(xRef.entries, vecRows, nc, 0x5358);
  zeroPadRows(w.entries, vecRows, A.nr, nc);
  zeroPadRows(q.entries, vecRows, A.nr, nc);
  zeroPadRows(xRef.entries, vecRows, A.nr, nc);

  V_ELE cA = (V_ELE)0.7, cP = (V_ELE)(-1.3), cQ = (V_ELE)2.1, gc = (V_ELE)0.37;

  GpuMatrixStream *s = gpu_matrix_stream_init(&A, 192, 0);
  CHECK(s != NULL, "gpu_matrix_stream_init failed");
  if (s != NULL) {
    gpu_spMMVMFused(&A, &w, cA, &w, cP, &q, cQ, &yRef);
    gpu_stream_spMMVMFused(s, &w, cA, &w, cP, &q, cQ, &yStr, 2);
    CHECK(maxAbsDiff(yRef.entries, yStr.entries, sz) < 1e-12,
        "spMMVMFused (nb=2) max|diff|=%.3e",
        maxAbsDiff(yRef.entries, yStr.entries, sz));

    memcpy(xStr.entries, xRef.entries, (size_t)sz * sizeof(V_ELE));
    gpu_chebfdOp(&A, &w, cA, cP, &yRef, cQ, &yRef, gc, &xRef);
    gpu_stream_chebfdOp(s, &w, cA, cP, &yStr, cQ, &yStr, gc, &xStr, 2);
    CHECK(maxAbsDiff(yRef.entries, yStr.entries, sz) < 1e-12,
        "chebfdOp (nb=2, y==q) y max|diff|=%.3e",
        maxAbsDiff(yRef.entries, yStr.entries, sz));
    CHECK(maxAbsDiff(xRef.entries, xStr.entries, sz) < 1e-12,
        "chebfdOp (nb=2, y==q) x max|diff|=%.3e",
        maxAbsDiff(xRef.entries, xStr.entries, sz));

    gpu_matrix_stream_free(s);
  }

  deallocate(w.entries);
  deallocate(q.entries);
  deallocate(yRef.entries);
  deallocate(yStr.entries);
  deallocate(xRef.entries);
  deallocate(xStr.entries);
  freeMatrix(&A);
  freeGMatrix(&gm);

  printf(ok ? "    PASS\n" : "");
  return ok;
}

/* ---- Section 4: applyFilter via streaming vs resident ------------------- */

static int testApplyFilterParity(void)
{
  int ok = 1;
  printf("  applyFilter (streaming vs resident):\n");

  const int n = 42, C = 4, sigma = 4;
  Matrix A;
  GMatrix gm;
  buildTridiagMatrixCS(&A, &gm, n, C, sigma);

  const int NS = 8;
  ChebFilter f;
  double a = 0.0, b = 4.0; /* tridiag spectrum lies in (0,4) */
  double lamLo, lamHi;
  tridiagEigenvalue(n, n / 2, &lamLo);
  lamHi = lamLo + 0.3;
  lamLo -= 0.1;
  CHECK(chebFilterInit(&f, a, b, lamLo, lamHi, 60, KERNEL_LANCZOS, 2) == 0,
      "chebFilterInit should succeed");

  ChebData dRef, dStr;
  allocChebData(&dRef, &A, NS);
  allocChebData(&dStr, &A, NS);
  DMatrix y0 = makeBlock(dRef.Y.nr, NS); /* pristine input, refilled each run */

  fillRandomBlock(y0.entries, dRef.Y.nr, NS, 0xbeef);
  zeroPadRows(y0.entries, dRef.Y.nr, A.nr, NS);

  chebFDSetMatrixStream(NULL, 0); /* resident baseline */
  memcpy(dRef.Y.entries, y0.entries, (size_t)dRef.Y.nr * (size_t)NS * sizeof(V_ELE));
  dRef.Y.nc = NS;
  applyFilter(&A, &f, &dRef.Y, &dRef.u, &dRef.w);

  GpuMatrixStream *s = gpu_matrix_stream_init(&A, 192, 0);
  CHECK(s != NULL, "gpu_matrix_stream_init failed");
  if (s != NULL) {
    const int nbs[2] = { 0, 2 };
    for (int i = 0; i < 2; i++) {
      memcpy(dStr.Y.entries, y0.entries, (size_t)dStr.Y.nr * (size_t)NS * sizeof(V_ELE));
      dStr.Y.nc = NS;
      chebFDSetMatrixStream(s, nbs[i]);
      applyFilter(&A, &f, &dStr.Y, &dStr.u, &dStr.w);
      double maxd = maxAbsDiff(
          dRef.Y.entries, dStr.Y.entries, (CG_UINT)(dRef.Y.nr * (CG_UINT)NS));
      CHECK(maxd < 1e-10, "applyFilter nb=%d max|diff|=%.3e", nbs[i], maxd);
    }
    chebFDSetMatrixStream(NULL, 0);
    gpu_matrix_stream_free(s);
  }

  deallocate(y0.entries);
  freeChebData(&dRef);
  freeChebData(&dStr);
  chebFilterFree(&f);
  freeMatrix(&A);
  freeGMatrix(&gm);

  printf(ok ? "    PASS\n" : "");
  return ok;
}

/* ---- Section 5: solveChebFD end-to-end, streaming vs resident ----------- */

static int testSolveChebFDStreaming(void)
{
  int ok = 1;
  printf("  solveChebFD (end-to-end, streaming vs resident):\n");

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
  param.eps             = 1e-10;
  param.itermax         = 60;
  param.verbose         = 0;
  param.allocType       = ALLOC_MANAGED; /* tests allocate with this default */
  param.cheb.lam_lo     = lamLo;
  param.cheb.lam_hi     = lamHi;
  param.cheb.Np         = 80;
  param.cheb.NS         = 10;
  param.cheb.kernel     = 3; /* Lanczos */
  param.cheb.mu         = 2;
  param.cheb.have_bounds = 0;
  param.cheb.have_target = 1;
  param.streamMb         = 0;
  param.chebNb           = 0;

  int found1 = solveChebFD(&comm, &param, &A);
  CHECK(found1 == 3, "resident: found %d eigenpairs, expected 3", found1);

  /* Streaming with a 1 MiB part target (the tiny test matrix lands in one
   * part; multi-part behavior is covered by sections 1/3/4) and a subblock
   * width, exercising the full parameter -> context -> sweep -> free path. */
  param.streamMb = 1;
  param.chebNb   = 4;
  int found2     = solveChebFD(&comm, &param, &A);
  CHECK(found2 == 3, "streaming: found %d eigenpairs, expected 3", found2);
  CHECK(found1 == found2, "streaming must find the same pairs as resident");

  freeMatrix(&A);
  freeGMatrix(&gm);
  printf(ok ? "    PASS\n" : "");
  return ok;
}

int chebFDStreamTests(int argc, char **argv)
{
  (void)argc;
  (void)argv;

  printf("Running ChebFD streaming tests:\n");

  int results[5];
  int i = 0;
  results[i++] = testPartSweepParity();
  results[i++] = testResidentSubblock();
  results[i++] = testStreamSubblock();
  results[i++] = testApplyFilterParity();
  results[i++] = testSolveChebFDStreaming();

  int passed = 0;
  for (int j = 0; j < i; j++) {
    passed += results[j] ? 1 : 0;
  }
  printf("\nSummary: %d/%d ChebFD streaming test sections passed.\n", passed, i);
  return (passed == i) ? 0 : 1;
}

#endif /* USE_COMPLEX */

#endif /* GPU backend */
