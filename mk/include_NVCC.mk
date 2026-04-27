# Host C compiler (used for .c files)
ifeq ($(strip $(ENABLE_MPI)),true)
CC = mpicc
DEFINES += -D_MPI
else
CC = gcc
endif

# CUDA compiler
NVCC = nvcc

# Linker: use nvcc so it can resolve CUDA runtime symbols
LD = $(NVCC)

ifeq ($(strip $(ENABLE_OPENMP)),true)
OPENMP   = -fopenmp
endif

VERSION  = --version
CFLAGS   = -O3 -g -std=c99 $(OPENMP)
LFLAGS   = -Xcompiler "$(OPENMP)" -lm $(CUDA_ARCH)

# NVCC flags — host compiler options passed via -Xcompiler
# CUDA_ARCH ?= sm_80
NVCCFLAGS = -O3 -g $(CUDA_ARCH) -Xcompiler "$(OPENMP)"

DEFINES  += -D_GNU_SOURCE -DRUNTIME_BACKEND_IS_CUDA
INCLUDES  =
LIBS      = -lcudart -lm
