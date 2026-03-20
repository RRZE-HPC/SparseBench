/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __VTYPE_H_
#define __VTYPE_H_

#include "util.h"

#ifdef USE_COMPLEX

#if defined(__NVCC__) || defined(__HIPCC__)
/* CUDA C++ compilation context (.cu files compiled by nvcc).
 * thrust::complex<T> has the same memory layout as C99 _Complex
 * and supports the same arithmetic operators in device code. */
#include <thrust/complex.h>
#if PRECISION == 1
#define V_ELE thrust::complex<float>
#define VCONST(r, i) thrust::complex<float>((r), (i))
#define PRECISION_STRING "complex float"
#else
#define V_ELE thrust::complex<double>
#define PRECISION_STRING "complex double"
#define VCONST(r, i) thrust::complex<double>((r), (i))
#endif
#define VCONJ(z) thrust::conj(z)
#define VREAL(z) (z).real()
#define VIMAG(z) (z).imag()
#define VABS(z) thrust::abs(z)

#else
/* C compilation context (.c files compiled by the host compiler). */
#include <complex.h>
#undef I

#if PRECISION == 1
#define V_ELE float _Complex
#define PRECISION_STRING "complex float"
#define VCONST(r, i) CMPLXF((r), (i))
#define VCONJ(z) conjf(z)
#define VREAL(z) crealf(z)
#define VIMAG(z) cimagf(z)
#define VABS(z) cabsf(z)
#else
#define V_ELE double _Complex
#define PRECISION_STRING "complex double"
#define VCONST(r, i) CMPLX((r), (i))
#define VCONJ(z) conj(z)
#define VREAL(z) creal(z)
#define VIMAG(z) cimag(z)
#define VABS(z) cabs(z)
#endif

#endif /* __NVCC__ || __HIPCC__ */

#else
#define V_ELE CG_FLOAT
#define VCONST(r, i) (r)
#define VCONJ(z) (z)
#if PRECISION == 1
#define PRECISION_STRING "float"
#else
#define PRECISION_STRING "double"
#endif
#endif /* USE_COMPLEX */

#endif /* __VTYPE_H_ */
