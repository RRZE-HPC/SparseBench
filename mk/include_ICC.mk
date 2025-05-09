ifeq ($(strip $(ENABLE_MPI)),true)
CC = mpiicx
DEFINES += -D_MPI
else
CC = icx
endif

LD = $(CC)

ifeq ($(strip $(ENABLE_OPENMP)),true)
OPENMP   = -qopenmp
endif

VERSION  = --version
CFLAGS   =  -O3 -ffast-math -xHost -std=c23 $(OPENMP)
# CFLAGS   = -O0 -g -std=c99 $(OPENMP)
LFLAGS   = $(OPENMP)
DEFINES  += -D_GNU_SOURCE # -DVERBOSE
INCLUDES =
