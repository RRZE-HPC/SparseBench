/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __VTYPE_H_
#define __VTYPE_H_

#include "util.h"

#ifdef USE_COMPLEX
#include <complex.h>
#undef I

#if PRECISION == 1
  #define V_ELE float _Complex
  #define VCONST(r, i) CMPLXF((r), (i))
#else
  #define V_ELE double _Complex
  #define VCONST(r, i) CMPLX((r), (i))
#endif

#else
  #define V_ELE CG_FLOAT
#endif

#endif
