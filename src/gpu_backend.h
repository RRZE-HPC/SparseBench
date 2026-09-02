/* Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of SparseBench.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file. */
#ifndef __GPU_BACKEND_H_
#define __GPU_BACKEND_H_

/*
 * Backend-agnostic GPU abstraction.
 *
 * The Makefile defines either -DRUNTIME_BACKEND=cuda or -DRUNTIME_BACKEND=hip.
 * All GPU API calls go through GCXX_RUNTIME_BACKEND(Name) which expands to
 * cudaName or hipName via token pasting.
 *
 * Usage:
 *   GCXX_RUNTIME_BACKEND(Malloc)(&ptr, size)   → cudaMalloc / hipMalloc
 *   GCXX_RUNTIME_BACKEND(Free)(ptr)            → cudaFree   / hipFree
 *   GCXX_RUNTIME_BACKEND(DeviceProp) prop;     → cudaDeviceProp / hipDeviceProp
 */

/* --- Token-pasting helpers ---------------------------------------- */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define STRINGIFY_AND_APPEND(a, b) a##b
#define APPEND_NAME(a, b) STRINGIFY_AND_APPEND(a, b)

#define GCXX_RUNTIME_BACKEND(name) APPEND_NAME(RUNTIME_BACKEND, name)
#define GCXX_ATTRIBUTE_BACKEND(name) APPEND_NAME(ATTRIBUTE_BACKEND, name)

/* --- Include the right runtime header ----------------------------- */
#if defined(RUNTIME_BACKEND_IS_CUDA)
#include <cuda_runtime.h>
#define RUNTIME_BACKEND cuda
#define ATTRIBUTE_BACKEND cuda
#elif defined(RUNTIME_BACKEND_IS_HIP)
#include <hip/hip_runtime.h>
#define RUNTIME_BACKEND hip
#define ATTRIBUTE_BACKEND hip
#else
#error                                                                                   \
    "No GPU backend selected. Define RUNTIME_BACKEND_IS_CUDA or RUNTIME_BACKEND_IS_HIP."
#endif

/* Backend name as a string for printing */
#define GPU_BACKEND_STR TOSTRING(RUNTIME_BACKEND)

/* --- Portable success value --------------------------------------- */
#if defined(RUNTIME_BACKEND_IS_CUDA)
#define GPU_SUCCESS cudaSuccess
#elif defined(RUNTIME_BACKEND_IS_HIP)
#define GPU_SUCCESS hipSuccess
#endif

/* --- Portable memcpy direction enums ------------------------------ */
#define gpuMemcpyHostToDevice GCXX_RUNTIME_BACKEND(MemcpyHostToDevice)
#define gpuMemcpyDeviceToHost GCXX_RUNTIME_BACKEND(MemcpyDeviceToHost)
#define gpuMemcpyDeviceToDevice GCXX_RUNTIME_BACKEND(MemcpyDeviceToDevice)

/* --- Portable convenience wrappers -------------------------------- */
#define gpuMalloc(ptr, size) GCXX_RUNTIME_BACKEND(Malloc)((ptr), (size))
#define gpuMallocManaged(ptr, size) GCXX_RUNTIME_BACKEND(MallocManaged)((ptr), (size))
#define gpuMallocHost(ptr, size) GCXX_RUNTIME_BACKEND(MallocHost)((ptr), (size))
#define gpuFree(ptr) GCXX_RUNTIME_BACKEND(Free)((ptr))
#define gpuFreeHost(ptr) GCXX_RUNTIME_BACKEND(FreeHost)((ptr))
#define gpuMemcpy(dst, src, sz, k) GCXX_RUNTIME_BACKEND(Memcpy)((dst), (src), (sz), (k))
#define gpuMemset(ptr, val, sz) GCXX_RUNTIME_BACKEND(Memset)((ptr), (val), (sz))
#define gpuMemsetAsync(ptr, val, sz, str)                                               \
  GCXX_RUNTIME_BACKEND(MemsetAsync)((ptr), (val), (sz), (str))
#define gpuDeviceSynchronize() GCXX_RUNTIME_BACKEND(DeviceSynchronize)()

/* Streams / events / async copies (host-resident matrix streaming). Every
 * name has an identical-signature hip* twin, except MemPrefetchAsync which
 * takes the extra flags argument on both runtimes. */
#define gpuStreamCreate(str) GCXX_RUNTIME_BACKEND(StreamCreate)(str)
#define gpuStreamDestroy(str) GCXX_RUNTIME_BACKEND(StreamDestroy)(str)
#define gpuStreamSynchronize(str) GCXX_RUNTIME_BACKEND(StreamSynchronize)(str)
#define gpuStreamWaitEvent(str, ev, flags)                                               \
  GCXX_RUNTIME_BACKEND(StreamWaitEvent)((str), (ev), (flags))
#define gpuEventCreate(ev) GCXX_RUNTIME_BACKEND(EventCreate)(ev)
#define gpuEventDestroy(ev) GCXX_RUNTIME_BACKEND(EventDestroy)(ev)
#define gpuEventRecord(ev, str) GCXX_RUNTIME_BACKEND(EventRecord)((ev), (str))
#define gpuEventSynchronize(ev) GCXX_RUNTIME_BACKEND(EventSynchronize)(ev)
#define gpuEventElapsedTime(ms, a, b)                                                    \
  GCXX_RUNTIME_BACKEND(EventElapsedTime)((ms), (a), (b))
#define gpuMemcpyAsync(dst, src, sz, k, str)                                             \
  GCXX_RUNTIME_BACKEND(MemcpyAsync)((dst), (src), (sz), (k), (str))
#define gpuGetDevice(dev) GCXX_RUNTIME_BACKEND(GetDevice)(dev)

/* Prefetch a managed range to a device. CUDA 13 replaced the int-device
 * overload with a cudaMemLocation struct; HIP keeps the plain form. */
#if defined(RUNTIME_BACKEND_IS_CUDA)
#if CUDART_VERSION >= 13000
static inline GCXX_RUNTIME_BACKEND(Error_t)
    gpuMemPrefetch(const void *ptr, size_t bytes, int dev)
{
  cudaMemLocation loc;
  loc.type = cudaMemLocationTypeDevice;
  loc.id   = dev;
  return cudaMemPrefetchAsync(ptr, bytes, loc, 0, 0);
}
#else
static inline GCXX_RUNTIME_BACKEND(Error_t)
    gpuMemPrefetch(const void *ptr, size_t bytes, int dev)
{
  return cudaMemPrefetchAsync(ptr, bytes, dev, 0);
}
#endif
#else /* HIP */
static inline GCXX_RUNTIME_BACKEND(Error_t)
    gpuMemPrefetch(const void *ptr, size_t bytes, int dev)
{
  return hipMemPrefetchAsync(ptr, bytes, dev, 0, 0);
}
#endif

/* --- Safe call macro with error checking -------------------------- */
#define GPU_SAFE_CALL(call)                                                              \
  do {                                                                                   \
    auto err = (call);                                                                   \
    if (err != GPU_SUCCESS) {                                                            \
      fprintf(stderr,                                                                    \
          "GPU error at %s:%d — %s\n",                                                   \
          __FILE__,                                                                      \
          __LINE__,                                                                      \
          GCXX_RUNTIME_BACKEND(GetErrorString)(err));                                    \
      exit(EXIT_FAILURE);                                                                \
    }                                                                                    \
  } while (0)

/* --- Portable type aliases ---------------------------------------- */
typedef GCXX_RUNTIME_BACKEND(Error_t) gpuError_t;
typedef GCXX_RUNTIME_BACKEND(Stream_t) gpuStream_t;
typedef GCXX_RUNTIME_BACKEND(Event_t) gpuEvent_t;

/* DeviceProp: CUDA uses cudaDeviceProp, HIP uses hipDeviceProp_t */
#if defined(RUNTIME_BACKEND_IS_CUDA)
typedef struct cudaDeviceProp gpuDeviceProp_t;
#elif defined(RUNTIME_BACKEND_IS_HIP)
typedef hipDeviceProp_t gpuDeviceProp_t;
#endif

#endif /* __GPU_BACKEND_H_ */
