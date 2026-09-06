# Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved. This file is part of SparseBench.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.

# ScaMaC matrix generation (src/matrixScamac.c).
#
# ScaMaC (Scalable Matrix Collection, https://bitbucket.org/essex/matrixcollection,
# modified BSD license) generates application matrices of scalable size from
# parameter strings, e.g.  -m scamac:Anderson,Lx=100,Ly=100,Lz=100,ranpot=2.5
#
# It is an external dependency, linked like LIKWID/NVTX: opt-in via
# ENABLE_SCAMAC in config.mk. 
SCAMAC_INSTALL ?= $(HOME)/.local/scamac

ifeq ($(strip $(ENABLE_SCAMAC)),true)
  ifeq ($(wildcard $(SCAMAC_INSTALL)/include/scamac.h),)
    $(error ENABLE_SCAMAC=true needs a ScaMaC installation - no scamac.h in $(SCAMAC_INSTALL)/include (set SCAMAC_INSTALL))
  endif
INCLUDES += -I$(SCAMAC_INSTALL)/include
DEFINES += -D_SCAMAC
LIBS += -lscamac -lm
LFLAGS += -L$(SCAMAC_INSTALL)/lib
endif
