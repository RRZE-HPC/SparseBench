
#include "../../src/allocate.h"
#include "../../src/chebFDSolver.h"
#include "../../src/comm.h"
#include "../../src/matrix.h"
#include "../common.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Reference Gershgorin bounds computed from the GMatrix produced by matrixConvertfromMM 
static void gershgorinRef(const GMatrix *g, double *a_out, double *b_out)
{
  double lo = 1e30; // TODO : cant find DBL_MAX AND MIN so work around for now 
  double hi = -1e30;
  for (CG_UINT i = 0; i < g->nr; i++) {
    double diag = 0.0;
    double off  = 0.0;
    for (CG_UINT j = g->rowPtr[i]; j < g->rowPtr[i + 1]; j++) {
      double v = (double)g->entries[j].val;
      if ((CG_UINT)g->entries[j].col == i) {
        diag += v;
      } else {
        off += fabs(v);
      }
    }
    if (diag - off < lo)
      lo = diag - off;
    if (diag + off > hi)
      hi = diag + off;
  }
  *a_out = lo;
  *b_out = hi;
}

typedef struct {
  CG_UINT C;
  CG_UINT sigma;
  char label[32];
} GershConfig;

static int buildConfigs(GershConfig *cfgs, int max)
{
  int n = 0;
#ifdef SCS
  static const CG_UINT cVals[]     = { 1, 2, 4 };
  static const CG_UINT sigmaVals[] = { 1, 4 };
  for (int ci = 0; ci < (int)(sizeof(cVals) / sizeof(cVals[0])) && n < max; ci++) {
    for (int si = 0; si < (int)(sizeof(sigmaVals) / sizeof(sigmaVals[0])) && n < max;
         si++) {
      cfgs[n].C     = cVals[ci];
      cfgs[n].sigma = sigmaVals[si];
      snprintf(cfgs[n].label,
          sizeof(cfgs[n].label),
          "C=%llu sigma=%llu",
          (unsigned long long)cVals[ci],
          (unsigned long long)sigmaVals[si]);
      n++;
    }
  }
#else
  // CRS nothing to do
  if (max > 0) {
    cfgs[0].C     = 0;
    cfgs[0].sigma = 0;
    snprintf(cfgs[0].label, sizeof(cfgs[0].label), "row");
    n              = 1;
  }
#endif
  return n;
}

static int checkMatrix(const GMatrix *gm,
    const char *name,
    CommType *comm,
    const GershConfig *cfgs,
    int nConfigs)
{
  double refA, refB;
  gershgorinRef(gm, &refA, &refB);

  /* Relative tolerance: ample for double, safe for float builds too. */
  const double tolA = 1e-5 * (1.0 + fabs(refA));
  const double tolB = 1e-5 * (1.0 + fabs(refB));

  int fails = 0;
  for (int ci = 0; ci < nConfigs; ci++) {
    Matrix A;
    memset(&A, 0, sizeof(A));
#ifdef SCS
    A.C     = cfgs[ci].C;
    A.sigma = cfgs[ci].sigma;
#endif
    convertMatrix(&A, (GMatrix *)gm);

    double a, b;
    gershgorinBounds(comm, &A, &a, &b);

    int ok = (fabs(a - refA) <= tolA) && (fabs(b - refB) <= tolB) && (a <= b);
    if (!ok) {
      if (fails == 0) {
        printf("  %s (%llux%llu, nnz=%llu) ref=[%.4g,%.4g]\n",
            name,
            (unsigned long long)gm->nr,
            (unsigned long long)gm->nr,
            (unsigned long long)gm->nnz,
            refA,
            refB);
      }
      printf("    %-14s gersh=[%.4g,%.4g]  FAIL\n", cfgs[ci].label, a, b);
      fails++;
    }
    freeMatrix(&A);
  }
  return fails;
}

int chebFDGershgorinTests(int argc, char **argv)
{
  (void)argc;
  (void)argv;

  const char *dirPath = "./data/testMatrices";
  DIR *dir            = opendir(dirPath);
  if (dir == NULL) {
    fprintf(stderr, "gershgorinBounds tests: cannot open %s\n", dirPath);
    return 1;
  }

  CommType comm;
  memset(&comm, 0, sizeof(comm));
  comm.rank = 0;
  comm.size = 1;

  GershConfig cfgs[32];
  int nConfigs = buildConfigs(cfgs, (int)(sizeof(cfgs) / sizeof(cfgs[0])));

#ifdef SCS
  const char *fmtName = "SCS";
#elif defined(CRS)
  const char *fmtName = "CRS";
#else
  const char *fmtName = "row";
#endif

  printf("Running gershgorinBounds (%s) tests over %s:\n", fmtName, dirPath);

  int totalChecks    = 0;
  int passedChecks   = 0;
  int matrices       = 0;
  int failedMatrices = 0;

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (strstr(entry->d_name, ".mtx") == NULL) {
      continue;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dirPath, entry->d_name);

    MMMatrix mm;
    MMMatrixRead(&mm, path);

    GMatrix gm;
    matrixConvertfromMM(&mm, &gm);

    int fails = checkMatrix(&gm, entry->d_name, &comm, cfgs, nConfigs);
    matrices++;
    totalChecks += nConfigs;
    passedChecks += nConfigs - fails;
    if (fails == 0) {
      printf("  %-14s (%llux%llu)  all %d configs PASS\n",
          entry->d_name,
          (unsigned long long)gm.nr,
          (unsigned long long)gm.nr,
          nConfigs);
    } else {
      failedMatrices++;
    }

    freeGMatrix(&gm);
    freeMMMatrix(&mm);
  }
  closedir(dir);

  printf("\nSummary: %d/%d matrices fully passed, %d/%d configs passed.\n",
      matrices - failedMatrices,
      matrices,
      passedChecks,
      totalChecks);
  return (passedChecks == totalChecks) ? 0 : 1;
}
