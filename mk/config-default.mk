# Supported: GCC, CLANG, ICX, NVCC, HIP
TOOLCHAIN ?= CLANG
# Supported CRS, SCS, CCRS
MTX_FMT ?= CRS
# Enable LIKWID instrumentation (true or false)
ENABLE_LIKWID ?= false
# Enable NVTX range markers for nsys/ncu timelines (TOOLCHAIN=NVCC only)
ENABLE_NVTX ?= false
# Enable section timing (events + host wall on GPU, getTimeStamp on CPU).
# solveChebFD's step/throughput reporting flows through it — disable only
# to measure the timer's own (negligible) overhead.
ENABLE_SECTIMER ?= true
# Enable MPI parallelization (true or false)
ENABLE_MPI ?= false
# Enable OpenMP shared memory parallelization (true or false)
# Can be combined with MPI
ENABLE_OPENMP ?= false
FLOAT_TYPE ?= DP # SP for float, DP for double
UINT_TYPE ?= U # U for unsigned int, ULL for unsigned long long int
USE_COMPLEX_ELEMENTS ?= false
SELL_CHUNK_VALUE ?= 64
SELL_SIGMA_VALUE ?= 64
NUM_VEC ?= 10
# Schedule used by every #pragma omp parallel for in src/. Override at build
# time with `make OMP_SCHEDULE=...`.  Keep `static` for NUMA first-touch
# correctness — dynamic/guided break the requirement that init loops touch
# the same pages the kernels later read.
OMP_SCHEDULE ?= static
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
OPTIONS +=  -DOMP_SCHEDULE=$(OMP_SCHEDULE)
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
