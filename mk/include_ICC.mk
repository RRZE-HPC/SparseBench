ifeq ($(strip $(ENABLE_MPI)),true)
CC = mpiicx
DEFINES = -D_MPI
else
CC = icx
endif

LD = $(CC)

ifeq ($(strip $(ENABLE_OPENMP)),true)
OPENMP   = -qopenmp
endif

VERSION  = --version
CFLAGS   =  -fast -xHost -std=c99 $(OPENMP)
LFLAGS   = $(OPENMP)
DEFINES  = -D_GNU_SOURCE
INCLUDES =
LIBS     =
