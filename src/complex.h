/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __COMPLEX_H_
#define __COMPLEX_H_

#include "util.h"

typedef struct {
  union {
    struct {
      CG_FLOAT real;
      CG_FLOAT imag;
    };
    CG_FLOAT vals[2];
  };
} complex;

extern complex make_complex(const CG_FLOAT real, const CG_FLOAT imag);
extern CG_FLOAT Creal(const complex *c1);
extern CG_FLOAT Cimag(const complex *c1);
extern complex Cadd(const complex *c1, const complex *c2);
extern complex Csub(const complex *c1, const complex *c2);
extern complex Cmul(const complex *c1, const complex *c2);
extern complex Cdiv(const complex *c1, const complex *c2);
extern complex Cinv(const complex *c1);
extern complex Cconj(const complex *c1);
extern CG_FLOAT Cabs2(const complex *c1);
extern CG_FLOAT Cabs(const complex *c1);
extern CG_FLOAT Carg(const complex *c1);

#ifdef USE_COMPLEX
  #define V_ELE complex
#else
  #define V_ELE CG_FLOAT
#endif

#endif