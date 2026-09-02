/* Unit tests for the section timer (section_timer.h): accumulation,
 * isolation, idempotent queries, stream re-pinning and index robustness.
 * Calls the functions directly so it runs regardless of ENABLE_SECTIMER
 * (GPU builds need the context runTests creates via gpu_init). */
#include "sectionTimerTests.h"

#include <stdio.h>

#include "../../src/section_timer.h"

#define CHECK(cond, msg, ...)                                                            \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      printf("    FAIL: " msg "\n", ##__VA_ARGS__);                                      \
      ok = 0;                                                                            \
    }                                                                                    \
  } while (0)

int sectionTimerTests(int argc, char **argv)
{
  (void)argc;
  (void)argv;

  printf("Running section timer tests:\n");

  int ok = 1;
  enum { T_A, T_B, T_N };

  SectionTimer *t = sectionTimerCreate(T_N);
  CHECK(t != NULL, "sectionTimerCreate(%d) failed", T_N);
  if (t != NULL) {
    /* Two intervals on section A: must accumulate, not overwrite. */
    for (int r = 0; r < 2; r++) {
      sectionTimerStart(t, T_A);
      sectionTimerStop(t, T_A);
    }
    sectionTimerSync(t);
    double sec             = sectionTimerGetSec(t, T_A);
    unsigned long long cnt = sectionTimerGetCount(t, T_A);
    CHECK(cnt == 2, "count=%llu after two rounds, expected 2", cnt);
    CHECK(sec >= 0.0, "elapsed sec=%f is negative", sec);

    /* Untouched section stays at zero. */
    CHECK(sectionTimerGetCount(t, T_B) == 0, "untouched section count != 0");
    CHECK(sectionTimerGetSec(t, T_B) == 0.0, "untouched section sec != 0");

    /* Re-query is idempotent (no double count, no drift). */
    (void)sectionTimerGetSec(t, T_A);
    CHECK(sectionTimerGetCount(t, T_A) == 2, "query changed the count");

    /* Stream re-pinning (NULL = legacy stream): must keep accumulating. */
    sectionTimerSetStream(t, T_A, NULL);
    sectionTimerStart(t, T_A);
    sectionTimerStop(t, T_A);
    CHECK(sectionTimerGetCount(t, T_A) == 3,
        "count=%llu after re-pinning, expected 3",
        sectionTimerGetCount(t, T_A));

    /* Defensive handling: out-of-range sections ignored, no crash. */
    sectionTimerStart(t, -1);
    sectionTimerStop(t, T_N);
    sectionTimerSync(NULL);
    CHECK(sectionTimerGetCount(t, T_A) == 3, "bogus section changed the count");

    sectionTimerFree(t);
  }
  sectionTimerFree(NULL); /* NULL-tolerant */

  printf(ok ? "    PASS\n" : "");
  return ok ? 0 : 1;
}
