# Supported: GCC, CLANG, ICX, NVCC, HIP
TOOLCHAIN ?= CLANG
# Supported CRS, SCS, CCRS
MTX_FMT ?= SCS
ENABLE_MPI ?= false
ENABLE_OPENMP ?= true
FLOAT_TYPE ?= DP # SP for float, DP for double
UINT_TYPE ?= U # U for unsigned int, ULL for unsigned long long int
SELL_CHUNK_VALUE ?= 32
SELL_SIGMA_VALUE ?= 1
NUM_VEC ?= 10
# GPU architecture (only used when TOOLCHAIN=NVCC or HIP)
# NVCC: sm_70 (V100), sm_80 (A100), sm_90 (H100)
# HIP:  gfx906 (MI50), gfx908 (MI100), gfx90a (MI250X)
CUDA_ARCH ?= sm_70
HIP_ARCH  ?= gfx906

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
