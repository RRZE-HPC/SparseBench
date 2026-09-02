/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __GPU_CUB_H_
#define __GPU_CUB_H_

/*
 * CUB (CUDA) / hipCUB (HIP) device-wide primitives behind one namespace.
 *
 * Only include this from .cu TUs that actually launch reduction primitives:
 * cub.cuh is heavy and would slow every TU that pulls gpu_backend.h.
 */

#if defined(RUNTIME_BACKEND_IS_CUDA)
#include <cub/cub.cuh>
namespace gpucub = cub;
#elif defined(RUNTIME_BACKEND_IS_HIP)
#include <hipcub/hipcub.hpp>
namespace gpucub = hipcub;
#else
#error "gpu_cub.h needs RUNTIME_BACKEND_IS_CUDA or RUNTIME_BACKEND_IS_HIP."
#endif

#endif /* __GPU_CUB_H_ */
