/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "chebFilter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Damping kernel factors g_n (paper Table 1, cf. KPM Rev. Mod. Phys. 78, 275).
 * All kernels use the convention g_0 = 1 and vanish for n > Np. */
static double kernelFactor(KernelType k, int n, int Np, int mu)
{
  switch (k) {
  case KERNEL_NONE:
    return 1.0;
  case KERNEL_FEJER:
    return 1.0 - (double)n / (double)(Np + 1);
  case KERNEL_JACKSON: {
    double n1   = (double)(Np + 1);
    double pn   = M_PI * (double)n / n1;
    double cotp = cos(M_PI / n1) / sin(M_PI / n1);
    return ((double)(Np - n + 1) * cos(pn) + sin(pn) * cotp) / n1;
  }
  case KERNEL_LANCZOS: {
    if (n == 0) {
      return 1.0;
    }
    double arg = M_PI * (double)n / (double)(Np + 1);
    double s   = sin(arg) / arg;
    return pow(s, (double)mu);
  }
  default:
    return 1.0;
  }
}

static double clamp(double x, double lo, double hi)
{
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

int chebFilterInit(ChebFilter *f,
    double a,
    double b,
    double lam_lo,
    double lam_hi,
    int Np,
    KernelType kernel,
    int mu)
{
  if (f == NULL) {
    return -1;
  }
  if (a >= b) {
    fprintf(stderr, "chebFilterInit: need a < b (got a=%g, b=%g)\n", a, b);
    return -1;
  }
  if (!(lam_lo < lam_hi) || lam_lo < a || lam_hi > b) {
    fprintf(stderr,
        "chebFilterInit: need a <= lam_lo < lam_hi <= b "
        "(got [%g, %g] in [%g, %g])\n",
        lam_lo,
        lam_hi,
        a,
        b);
    return -1;
  }
  if (Np < 2) {
    fprintf(stderr, "chebFilterInit: need Np >= 2 (got %d)\n", Np);
    return -1;
  }
  if (kernel == KERNEL_LANCZOS && mu < 1) {
    fprintf(stderr, "chebFilterInit: need mu >= 1 (got %d)\n", mu);
    return -1;
  }

  f->a      = a;
  f->b      = b;
  f->alpha  = 2.0 / (b - a);
  f->beta   = -(a + b) / (b - a);
  f->lam_lo = lam_lo;
  f->lam_hi = lam_hi;
  f->Np     = Np;
  f->kernel = kernel;
  f->mu     = mu;

  /* mapped target bounds xi = alpha * x + beta in [-1, 1];
   * theta = acos(xi); since xi_lo < xi_hi, theta_lo > theta_hi */
  double xi_lo    = clamp(f->alpha * lam_lo + f->beta, -1.0, 1.0);
  double xi_hi    = clamp(f->alpha * lam_hi + f->beta, -1.0, 1.0);
  double theta_lo = acos(xi_lo);
  double theta_hi = acos(xi_hi);

  f->gc           = (double *)malloc((size_t)(Np + 1) * sizeof(double));
  if (f->gc == NULL) {
    fprintf(stderr, "chebFilterInit: allocation of %d coeffs failed\n", Np + 1);
    return -1;
  }

  /* Window Chebyshev moments c_n of the indicator on [xi_lo, xi_hi]:
   *   c_0 = (theta_lo - theta_hi) / pi
   *   c_n = 2 (sin(n theta_lo) - sin(n theta_hi)) / (pi n),  n >= 1
   * Combined with the kernel damping: gc[n] = g_n c_n. */
  f->gc[0] = kernelFactor(kernel, 0, Np, mu) * (theta_lo - theta_hi) / M_PI;
  for (int n = 1; n <= Np; n++) {
    double cn = 2.0 * (sin((double)n * theta_lo) - sin((double)n * theta_hi)) /
                (M_PI * (double)n);
    f->gc[n]  = kernelFactor(kernel, n, Np, mu) * cn;
  }

  return 0;
}

void chebFilterFree(ChebFilter *f)
{
  if (f != NULL && f->gc != NULL) {
    free(f->gc);
    f->gc = NULL;
  }
}

void chebFilterPrint(const ChebFilter *f)
{
  if (f == NULL) {
    return;
  }
  const char *kname = "unknown";
  switch (f->kernel) {
  case KERNEL_NONE:
    kname = "none";
    break;
  case KERNEL_FEJER:
    kname = "Fejer";
    break;
  case KERNEL_JACKSON:
    kname = "Jackson";
    break;
  case KERNEL_LANCZOS:
    kname = "Lanczos";
    break;
  default:
    break;
  }
  printf("Chebyshev filter polynomial:\n");
  printf("  spectrum   [a, b]    = [%.6g, %.6g]\n", f->a, f->b);
  printf("  affine map alpha,beta= %.6g, %.6g\n", f->alpha, f->beta);
  printf("  target     [lo, hi]  = [%.6g, %.6g]\n", f->lam_lo, f->lam_hi);
  printf("  degree Np            = %d\n", f->Np);
  printf("  kernel               = %s", kname);
  if (f->kernel == KERNEL_LANCZOS) {
    printf(" (mu=%d)", f->mu);
  }
  printf("\n");
  printf("  g_0 c_0              = %.6g\n", f->gc[0]);
}
