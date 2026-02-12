#ifndef __DEBUGGER_H_
#define __DEBUGGER_H_

#define VALIDATE_MATRIX_FORMAT(fmt)                                                      \
  {                                                                                      \
    if (strcmp((fmt), "CRS") != 0 && strcmp((fmt), "SCS") != 0) {                        \
      fprintf(stderr, "ERROR: matrixFormat not recognized.\n");                          \
      free((fmt));                                                                       \
      exit(EXIT_FAILURE);                                                                \
    }                                                                                    \
  }

#endif // __DEBUGGER_H_