/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __KERNEL_DISPATCH_H_
#define __KERNEL_DISPATCH_H_

/*
 * Compile-time selection between the host (OpenMP) kernels in solver.h /
 * matrix-*.c and the device kernels in cuda_kernels.h.
 *
 * The GPU build routes every allocate() through managed memory (allocate.c),
 * so a solver that has only been partially ported still runs correctly — the
 * un-ported steps just fault their pages back to the host. That makes each
 * kernel independently switchable, which is why this is a set of per-kernel
 * macros rather than one all-or-nothing branch.
 *
 * The *FUNC names keep the exact signatures of their host counterparts, so a
 * call site is switched by renaming the call and nothing else.
 */

#include "solver.h"

#if defined(RUNTIME_BACKEND_IS_CUDA) || defined(RUNTIME_BACKEND_IS_HIP)
#include "cuda_kernels.h"

#define SPMVMFUNC gpu_spMVM
#define SPMMVMFUNC gpu_spMMVM
#define SPMMVMFUSEDFUNC gpu_spMMVMFused
#define CHEBFDOPFUNC gpu_chebfdOp
#define WAXBYFUNC gpu_waxpby_sync
#define WAXPBY3FUNC gpu_waxpby3_sync
#define DDOTFUNC gpu_ddot_sync
#define ORTHOMGSFUNC gpu_orthoMGS
#define GRAMFUNC gpu_gramYtAY
#define RITZRESIDUALFUNC gpu_computeRitzResidual

#else

#define SPMVMFUNC spMVM
#define SPMMVMFUNC spMMVM
#define SPMMVMFUSEDFUNC spMMVMFused
#define CHEBFDOPFUNC chebfdOp
#define WAXBYFUNC waxpby
#define WAXPBY3FUNC waxpby3
#define DDOTFUNC ddot
#define ORTHOMGSFUNC orthoMGS
#define GRAMFUNC gramYtAY
#define RITZRESIDUALFUNC computeRitzResidual

#endif

#endif // __KERNEL_DISPATCH_H_
