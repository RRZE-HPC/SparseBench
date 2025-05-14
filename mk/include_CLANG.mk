ifeq ($(strip $(ENABLE_MPI)),true)
CC = mpicc
DEFINES += -D_MPI
else
CC = clang
endif

LD = $(CC)

ifeq ($(strip $(ENABLE_OPENMP)),true)
OPENMP   = -fopenmp
#OPENMP   = -Xpreprocessor -fopenmp #required on Macos with homebrew libomp
endif

VERSION  = --version
CFLAGS   = -O3 -ffast-math -std=c23 $(OPENMP)
# CFLAGS   = -O0 -g -std=c99 $(OPENMP)
LFLAGS   = $(OPENMP)
DEFINES  += -D_GNU_SOURCE
INCLUDES = -I/opt/homebrew/include
LIBS     = -lm
