/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#include "denseJacobi.h"

#include <math.h>
#include <string.h>

/* Classic cyclic Jacobi rotations until the off-diagonal norm is negligible.
 * Threshold strategy (Schur): skip tiny rotations in the first sweeps. */
void jacobiEigen(double *a, int n, double *eval, double *evec)
{
  if (n <= 0) {
    return;
  }
  if (n == 1) {
    eval[0] = a[0];
    evec[0] = 1.0;
    return;
  }

  /* eigenvectors start as identity */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      evec[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
  }

  const int maxsweeps = 100;
  for (int sweep = 0; sweep < maxsweeps; sweep++) {
    /* Sum of |a_pq| over the strict upper triangle. It must be the same measure
     * `thresh` below is compared against (|a_pq|), and it reaches exactly 0.0
     * once every off-diagonal has been rotated/flushed to zero -- an absolute
     * epsilon here never trips for a matrix of O(1..100) scale (round-off keeps
     * the sum near eps*||A||), which would burn all `maxsweeps` sweeps. */
    double off = 0.0;
    for (int p = 0; p < n - 1; p++) {
      for (int q = p + 1; q < n; q++) {
        off += fabs(a[p * n + q]);
      }
    }
    if (off == 0.0) {
      break;
    }
    /* In the first three sweeps rotate only on entries above a threshold. */
    double thresh = (sweep < 3) ? 0.2 * off / ((double)n * n) : 0.0;

    for (int p = 0; p < n - 1; p++) {
      for (int q = p + 1; q < n; q++) {
        double apq = a[p * n + q];
        double g   = 100.0 * fabs(apq);
        double app = a[p * n + p];
        double aqq = a[q * n + q];

        if (sweep > 3 && fabs(app) + g == fabs(app) && fabs(aqq) + g == fabs(aqq)) {
          a[p * n + q] = 0.0; /* element negligible */
        } else if (fabs(apq) > thresh) {
          double h = aqq - app;
          double t;
          if (fabs(h) + g == fabs(h)) {
            t = apq / h;
          } else {
            double theta = 0.5 * h / apq;
            t            = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
            if (theta < 0.0)
              t = -t;
          }
          double c     = 1.0 / sqrt(1.0 + t * t);
          double s     = t * c;
          double tau   = s / (1.0 + c);

          double h_rot = t * apq;
          a[p * n + p] = app - h_rot;
          a[q * n + q] = aqq + h_rot;
          a[p * n + q] = 0.0;
          a[q * n + p] = 0.0;

          for (int k = 0; k < n; k++) {
            if (k != p && k != q) {
              double akp   = a[k * n + p];
              double akq   = a[k * n + q];
              a[k * n + p] = akp - s * (akq + tau * akp);
              a[p * n + k] = a[k * n + p];
              a[k * n + q] = akq + s * (akp - tau * akq);
              a[q * n + k] = a[k * n + q];
            }
            double vkp      = evec[k * n + p];
            double vkq      = evec[k * n + q];
            evec[k * n + p] = vkp - s * (vkq + tau * vkp);
            evec[k * n + q] = vkq + s * (vkp - tau * vkq);
          }
        }
      }
    }
  }

  for (int i = 0; i < n; i++) {
    eval[i] = a[i * n + i];
  }

  /* selection sort of eigenpairs by ascending eigenvalue (swaps full columns) */
  for (int i = 0; i < n - 1; i++) {
    int best = i;
    for (int j = i + 1; j < n; j++) {
      if (eval[j] < eval[best]) {
        best = j;
      }
    }
    if (best != i) {
      double tmp = eval[i];
      eval[i]    = eval[best];
      eval[best] = tmp;
      for (int k = 0; k < n; k++) {
        double vk          = evec[k * n + i];
        evec[k * n + i]    = evec[k * n + best];
        evec[k * n + best] = vk;
      }
    }
  }
}
