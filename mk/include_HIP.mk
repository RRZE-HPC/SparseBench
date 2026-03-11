# Host C compiler (used for .c files)
ifeq ($(strip $(ENABLE_MPI)),true)
CC = mpicc
DEFINES += -D_MPI
else
CC = gcc
endif

# HIP compiler
NVCC = hipcc

# Linker: use hipcc so it can resolve HIP runtime symbols
LD = $(NVCC)

ifeq ($(strip $(ENABLE_OPENMP)),true)
OPENMP   = -fopenmp
endif

VERSION  = --version
CFLAGS   = -O3 -g -std=c99 $(OPENMP)
LFLAGS   = $(OPENMP) -lm

# HIP flags
HIP_ARCH ?= gfx906
NVCCFLAGS = -O3 -g --offload-arch=$(HIP_ARCH)

DEFINES  += -D_GNU_SOURCE -DRUNTIME_BACKEND_IS_HIP
INCLUDES  =
LIBS      = -lm
