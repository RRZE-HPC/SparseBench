/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __VTYPE_H_
#define __VTYPE_H_

#include "util.h"

#ifdef USE_COMPLEX

#ifdef __NVCC__
/* CUDA C++ compilation context (.cu files compiled by nvcc).
 * thrust::complex<T> has the same memory layout as C99 _Complex
 * and supports the same arithmetic operators in device code. */
#include <thrust/complex.h>
#if PRECISION == 1
  #define V_ELE thrust::complex<float>
  #define VCONST(r, i) thrust::complex<float>((r), (i))
#else
  #define V_ELE thrust::complex<double>
  #define VCONST(r, i) thrust::complex<double>((r), (i))
#endif

#else
/* C compilation context (.c files compiled by the host compiler). */
#include <complex.h>
#undef I

#if PRECISION == 1
  #define V_ELE float _Complex
  #define VCONST(r, i) CMPLXF((r), (i))
#else
  #define V_ELE double _Complex
  #define VCONST(r, i) CMPLX((r), (i))
#endif

#endif /* __NVCC__ */

#else
  #define V_ELE CG_FLOAT
#endif /* USE_COMPLEX */

#endif /* __VTYPE_H_ */
