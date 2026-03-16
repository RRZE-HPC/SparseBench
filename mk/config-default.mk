# Supported: GCC, CLANG, ICX, NVCC, HIP
TOOLCHAIN ?= NVCC
# Supported CRS, SCS, CCRS
MTX_FMT ?= SCS
ENABLE_MPI ?= false
ENABLE_OPENMP ?= false
FLOAT_TYPE ?= DP # SP for float, DP for double
UINT_TYPE ?= U # U for unsigned int, ULL for unsigned long long int
USE_COMPLEX_ELEMENTS ?= false
SELL_CHUNK_VALUE ?= 64
SELL_SIGMA_VALUE ?= 64
NUM_VEC ?= 10
# GPU architecture (only used when TOOLCHAIN=NVCC or HIP)
# # NVCC: 
#       -gencode=arch=compute_80,code=sm_80 # for A100
#       -gencode=arch=compute_86,code=sm_86 # for A40
#       -gencode=arch=compute_90,code=sm_90 # for GH200
CUDA_ARCH ?= -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_90,code=sm_90
# # HIP:  
#       gfx908 # for MI100
#       gfx90a # for MI210A
#       gfx1030 # for RX 6900 XT
#       gfx942 # for MI300X & MI300A
HIP_ARCH  ?= gfx1030,gfx942
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