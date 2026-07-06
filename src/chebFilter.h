/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __CHEBFILTER_H_
#define __CHEBFILTER_H_

/* Construction of Chebyshev filter polynomials for ChebFD (paper Sec. 2.1).
 *
 * The filter polynomial approximates the rectangular window
 *     W(x) = 1  for x in the target interval [lam_lo, lam_hi]
 *          = 0  otherwise
 * on the spectral interval [a, b] via the damped Chebyshev expansion
 *     p(x) = sum_{n=0}^{Np} g_n c_n T_n(alpha x + beta)
 * with alpha x + beta in [-1, 1] for x in [a, b]. */

typedef enum {
  KERNEL_NONE = 0, /* g_n = 1 (no damping) */
  KERNEL_FEJER,
  KERNEL_JACKSON,
  KERNEL_LANCZOS /* recommended; uses mu (paper uses mu = 2) */
} KernelType;

typedef struct {
  /* spectral interval [a, b] that contains the full spectrum of H */
  double a, b;
  /* affine map xi = alpha * x + beta  (x in [a,b] -> xi in [-1,1]) */
  double alpha, beta;
  /* target interval [lam_lo, lam_hi] */
  double lam_lo, lam_hi;
  /* polynomial degree and kernel settings */
  int Np;
  KernelType kernel;
  int mu;
  /* combined filter coefficients g_n * c_n, length (Np + 1) */
  double *gc;
} ChebFilter;

/* Construct the filter polynomial coefficients.
 *
 *   a, b      : spectral interval [a, b] (spectrum of H contained here)
 *   lam_lo/hi : target interval (must satisfy a <= lam_lo < lam_hi <= b)
 *   Np        : polynomial degree (>= 2)
 *   kernel    : damping kernel (KERNEL_LANCZOS recommended)
 *   mu        : Lanczos kernel exponent (use 2; ignored otherwise)
 *
 * On success allocates f->gc (free with chebFilterFree) and returns 0.
 * Returns -1 on invalid input. */
int chebFilterInit(ChebFilter *f,
    double a,
    double b,
    double lam_lo,
    double lam_hi,
    int Np,
    KernelType kernel,
    int mu);

/* Release the coefficient buffer allocated by chebFilterInit. */
void chebFilterFree(ChebFilter *f);

/* Print the filter configuration and a coefficient summary to stdout. */
void chebFilterPrint(const ChebFilter *f);

#endif // __CHEBFILTER_H_
