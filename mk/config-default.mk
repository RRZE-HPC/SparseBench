# Supported: GCC, CLANG, ICX
TOOLCHAIN ?= CLANG
# Supported CRS, SCS, CCRS
MTX_FMT ?= SCS
ENABLE_MPI ?= false
ENABLE_OPENMP ?= true
FLOAT_TYPE ?= DP # SP for float, DP for double
UINT_TYPE ?= U # U for unsigned int, ULL for unsigned long long int
SELL_CHUNK_VALUE ?= 32
SELL_SIGMA_VALUE ?= 64
NUM_VEC ?= 10
USE_COMPLEX_ELEMENTS ?= false

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
DEFINES += -DNUMVEC=$(NUM_VEC)

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

ifeq ($(strip $(MTX_FMT)),SCS)
    DEFINES += -DSELL_CHUNK=$(SELL_CHUNK_VALUE)
    DEFINES += -DSELL_SIGMA=$(SELL_SIGMA_VALUE)
endif

ifeq ($(strip $(USE_COMPLEX_ELEMENTS)),true)
    DEFINES += -DUSE_COMPLEX
endif
