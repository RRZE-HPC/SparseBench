# Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved. This file is part of SparseBench.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.

# NVTX range markers for nsys / ncu timelines (src/nvtx_marker.h).
#
# Header-only NVTX3 C API shipped with the CUDA toolkit (>= 10):
# nvToolsExt.h defines everything inline behind weak symbols and binds the
# injection library with dlopen, so there is no libnvToolsExt to link
# against. nvcc finds <nvtx3/...> on its own; the host .c files need
# -I$(CUDA_HOME)/include.
#
# NVCC only: the HIP analogue would be rocTX, which is out of scope here
# (nvtx_marker.h silently no-ops when USE_NVTX is undefined).
NVTX_DEFINES ?= -DUSE_NVTX

ifeq ($(strip $(ENABLE_NVTX)),true)
  ifneq ($(filter $(TOOLCHAIN),NVCC),)
    ifeq ($(strip $(CUDA_HOME)),)
      $(error ENABLE_NVTX=true needs CUDA_HOME - load the cuda module first)
    endif
INCLUDES += -I$(CUDA_HOME)/include
DEFINES  += $(NVTX_DEFINES)
LIBS     += -ldl
  else
    $(info NVTX: ENABLE_NVTX=true is only wired up for TOOLCHAIN=NVCC; ignoring.)
  endif
endif
