# Supported: GCC, CLANG, ICC
TOOLCHAIN ?= ICC
# Supported CRS, SCS, CCRS
MTX_FMT ?= CRS
ENABLE_MPI ?= true
ENABLE_OPENMP ?= false
FLOAT_TYPE ?= DP # SP for float, DP for double
UINT_TYPE ?= U # U for unsigned int, ULL for unsigned long long int

#Feature options
OPTIONS +=  -DARRAY_ALIGNMENT=64
OPTIONS +=  -DOMP_SCHEDULE=static
#OPTIONS +=  -DVERBOSE
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER


################################################################
# DO NOT EDIT BELOW !!!
################################################################
DEFINES =

DEFINES += -D$(MTX_FMT)

ifeq ($(strip $(FLOAT_TYPE)),SP)
    DEFINES += -DPRECISION=1
else
    DEFINES += -DPRECISION=2
endif

ifeq ($(strip $(UINT_TYPE)),U)
    DEFINES += -DUINT_TYPE=1
else
    DEFINES += -DUINT_TYPE=2
endif
