/* Unit tests for the GPU vector kernels in cuda_vector_ops.cu, focused on
 * gpu_ddot: the kernel reduces each block with CUB and folds the block sums
 * into the result scalar with atomicAddV — which under USE_COMPLEX
 * accumulates the real and imaginary halves of a thrust::complex value
 * through separate scalar atomics (there is no complex atomicAdd). Unlike
 * the ChebFD suites these tests also run on complex builds, so that
 * component path is covered; runTests has already called gpu_init. */
#include "vectorOpsTests.h"

#include <stdio.h>

#if !defined(RUNTIME_BACKEND_IS_CUDA) && !defined(RUNTIME_BACKEND_IS_HIP)

int vectorOpsTests(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  printf("Skipping GPU vector-op tests (CPU build; GPU-only feature).\n");
  return 0;
}

#else

#include "../common.h"

#include <math.h>
#include <stdlib.h>

#include "../../src/allocate.h"
#include "../../src/cuda_kernels.h"

#define CHECK(cond, msg, ...)                                                            \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      printf("    FAIL: " msg "\n", ##__VA_ARGS__);                                      \
      ok = 0;                                                                            \
    }                                                                                    \
  } while (0)

/* kernel_ddot reduces with DDOT_THREADS=256 threads per block; sizes around
 * the block boundary and a many-block size exercise the tail guard, the
 * single-block path and the cross-block atomicAddV folds. */
#define DDOT_THREADS_REF 256

/* Relative slack for the GPU-vs-CPU comparison. The kernel sums in tree
 * order (block) + arrival order (atomics), the reference below in long
 * double, so the difference is pure floating-point noise: ~1e-13 observed
 * for double; float gets a wider band for its 1e-7 eps. */
#if PRECISION == 1
#define DDOT_TOL 1e-3
#else
#define DDOT_TOL 1e-12
#endif

/* Deterministic splitmix64 fill (same generator as chebFDTestUtil.h, kept
 * local so this suite does not pull in the ChebFD helpers). */
static unsigned long long sm64(unsigned long long z)
{
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static double rand01(unsigned long long h)
{
  return (double)(h >> 11) / (double)(1ull << 53) * 2.0 - 1.0;
}

/* x[j] = (rx, ix), y[j] = (ry, iy), each component from its own hash stream
 * so real and imaginary halves carry independent data. */
static void fillDdotVectors(V_ELE *x, V_ELE *y, CG_UINT n, unsigned long long seed)
{
  for (CG_UINT j = 0; j < n; j++) {
    unsigned long long h = sm64(seed + j * 0x9E3779B97F4A7C15ull);
    x[j]                 = VCONST(rand01(h), rand01(sm64(h)));
    h                    = sm64(h ^ 0xff51afd7ed558ccdull);
    y[j]                 = VCONST(rand01(h), rand01(sm64(h)));
  }
}

/* x^H y accumulated componentwise in long double. */
static void ddotReference(
    const V_ELE *x, const V_ELE *y, CG_UINT n, long double *re, long double *im)
{
  *re = 0.0L;
  *im = 0.0L;
  for (CG_UINT j = 0; j < n; j++) {
#ifdef USE_COMPLEX
    V_ELE p = VCONJ(x[j]) * y[j];
    *re += (long double)VREAL(p);
    *im += (long double)VIMAG(p);
#else
    *re += (long double)(VCONJ(x[j]) * y[j]);
#endif
  }
}

/* |got - want| <= tol * max(1, |want|) componentwise. */
static int componentNear(long double got, long double want)
{
  long double d = fabsl(got - want);
  long double w = fabsl(want) > 1.0L ? fabsl(want) : 1.0L;
  return d <= (long double)DDOT_TOL * w;
}

/* ---- Section 1: gpu_ddot vs long-double CPU reference ------------------- */

static int testDdotReference(void)
{
  int ok = 1;
  printf("  gpu_ddot vs CPU reference:\n");

  const CG_UINT sizes[] = {
    1, DDOT_THREADS_REF - 1, DDOT_THREADS_REF, DDOT_THREADS_REF + 1, 5000, 100003
  };
  const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

  for (int s = 0; s < nsizes; s++) {
    CG_UINT n    = sizes[s];
    size_t bytes = (size_t)n * sizeof(V_ELE);
    V_ELE *x     = (V_ELE *)allocate(ARRAY_ALIGNMENT, bytes);
    V_ELE *y     = (V_ELE *)allocate(ARRAY_ALIGNMENT, bytes);
    V_ELE r      = VCONST(0, 0);
    long double re, im;

    fillDdotVectors(x, y, n, 0xdd07 + (unsigned long long)s);
    ddotReference(x, y, n, &re, &im);
    gpu_ddot(n, x, y, &r);

    int near;
#ifdef USE_COMPLEX
    near = componentNear((long double)VREAL(r), re) &&
           componentNear((long double)VIMAG(r), im);
    CHECK(near,
        "n=%u: got (%.12Le,%.12Le) want (%.12Le,%.12Le)",
        (unsigned)n,
        (long double)VREAL(r),
        (long double)VIMAG(r),
        re,
        im);
#else
    near = componentNear((long double)r, re);
    CHECK(near, "n=%u: got %.12Le want %.12Le", (unsigned)n, (long double)r, re);
#endif

    deallocate(x);
    deallocate(y);
  }

  printf(ok ? "    PASS\n" : "");
  return ok;
}

/* ---- Section 2: conjugation and component placement (complex only) ------ */

#ifdef USE_COMPLEX
static int testDdotConjugation(void)
{
  int ok = 1;
  printf("  gpu_ddot conjugation / component placement:\n");

  const CG_UINT n = 1000;
  size_t bytes    = (size_t)n * sizeof(V_ELE);
  V_ELE *x        = (V_ELE *)allocate(ARRAY_ALIGNMENT, bytes);
  V_ELE *y        = (V_ELE *)allocate(ARRAY_ALIGNMENT, bytes);

  /* x purely imaginary, y real: x^H y = -i * sum(ix*yr). A dropped
   * conjugate flips the sign of the imaginary part; atomicAddV hitting the
   * wrong half (or the halves swapped) moves the sum into the real part. */
  long double sum = 0.0L;
  for (CG_UINT j = 0; j < n; j++) {
    double a = rand01(sm64(0xc0ffee11ull + j));
    double b = rand01(sm64(0x1234567ull + j));
    x[j]     = VCONST(0.0, a);
    y[j]     = VCONST(b, 0.0);
    sum += (long double)(a * b);
  }
  V_ELE r = VCONST(0, 0);
  gpu_ddot(n, x, y, &r);
  CHECK(componentNear((long double)VREAL(r), 0.0L),
      "x=iy: real part %.12Le, want 0",
      (long double)VREAL(r));
  CHECK(componentNear((long double)VIMAG(r), -sum),
      "x=iy: imag %.12Le, want %.12Le",
      (long double)VIMAG(r),
      -sum);

  /* x^H x must be real and non-negative regardless of x. */
  fillDdotVectors(x, x, n, 0x51de5);
  gpu_ddot(n, x, x, &r);
  CHECK(componentNear((long double)VIMAG(r), 0.0L),
      "x^H x: imaginary part %.12Le, want 0",
      (long double)VIMAG(r));
  CHECK((long double)VREAL(r) >= 0.0L,
      "x^H x: real part %.12Le, want >= 0",
      (long double)VREAL(r));

  deallocate(x);
  deallocate(y);
  printf(ok ? "    PASS\n" : "");
  return ok;
}
#endif

/* ---- Section 3: per-call zeroing of the result accumulator -------------- */

static int testDdotRepeatCalls(void)
{
  int ok = 1;
  printf("  gpu_ddot repeat calls (accumulator zeroing):\n");

  const CG_UINT n = 1000;
  size_t bytes    = (size_t)n * sizeof(V_ELE);
  V_ELE *x        = (V_ELE *)allocate(ARRAY_ALIGNMENT, bytes);
  V_ELE *y        = (V_ELE *)allocate(ARRAY_ALIGNMENT, bytes);
  long double re, im;

  /* Three alternating calls: if the internal result scalar were not reset
   * between calls, results would stack onto the previous one from call two
   * on. gpu_ddot_sync shares the same device scalar, so it joins the loop. */
  for (int call = 0; call < 3; call++) {
    fillDdotVectors(x, y, n, 0x2eea7 + (unsigned long long)call);
    ddotReference(x, y, n, &re, &im);

    V_ELE r = VCONST(0, 0);
    if (call == 1) {
      gpu_ddot_sync(n, x, y, &r);
    } else {
      gpu_ddot(n, x, y, &r);
    }

    int near;
#ifdef USE_COMPLEX
    near = componentNear((long double)VREAL(r), re) &&
           componentNear((long double)VIMAG(r), im);
    CHECK(near,
        "call %d: got (%.12Le,%.12Le) want (%.12Le,%.12Le)",
        call,
        (long double)VREAL(r),
        (long double)VIMAG(r),
        re,
        im);
#else
    near = componentNear((long double)r, re);
    CHECK(near, "call %d: got %.12Le want %.12Le", call, (long double)r, re);
#endif
  }

  deallocate(x);
  deallocate(y);
  printf(ok ? "    PASS\n" : "");
  return ok;
}

int vectorOpsTests(int argc, char **argv)
{
  (void)argc;
  (void)argv;

  printf("Running GPU vector-op (ddot) tests:\n");

  int results[3];
  int i        = 0;
  results[i++] = testDdotReference();
#ifdef USE_COMPLEX
  results[i++] = testDdotConjugation();
#endif
  results[i++] = testDdotRepeatCalls();

  int passed   = 0;
  for (int j = 0; j < i; j++) {
    passed += results[j] ? 1 : 0;
  }
  printf("\nSummary: %d/%d ddot test sections passed.\n", passed, i);
  return (passed == i) ? 0 : 1;
}

#endif /* GPU backend */
