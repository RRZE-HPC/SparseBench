# Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved. This file is part of SparseBench.
# Use of this source code is governed by a MIT style
# license that can be found in the LICENSE file.

# Section timing (src/section_timer.{h,c}, src/cuda_section_timer.cu):
# CUDA/HIP events on GPU builds, getTimeStamp on CPU builds. The -D
# switches the SECTION_TIMER_* macros from no-ops to real calls; no extra
# libs (the implementations build unconditionally).
SECTIMER_DEFINES ?= -DUSE_SECTION_TIMER

ifeq ($(strip $(ENABLE_SECTIMER)),true)
DEFINES += $(SECTIMER_DEFINES)
endif
