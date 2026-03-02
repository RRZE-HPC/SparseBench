#include <math.h>

#include "complex.h"

complex make_complex(const CG_FLOAT real, const CG_FLOAT imag)
{
  complex c = { .real = real, .imag = imag };
  return c;
}

CG_FLOAT Creal(const complex *c1)
{
  return c1->real;
}

CG_FLOAT Cimag(const complex *c1)
{
  return c1->imag;
}

complex Cadd(const complex *c1, const complex *c2)
{
  return make_complex(c1->real + c2->real, c1->imag + c2->imag);
}

complex Csub(const complex *c1, const complex *c2)
{
  return make_complex(c1->real - c2->real, c1->imag - c2->imag);
}

complex Cmul(const complex *c1, const complex *c2)
{
  CG_FLOAT a = c1->real;
  CG_FLOAT b = c1->imag;
  CG_FLOAT c = c2->real;
  CG_FLOAT d = c2->imag;
  return make_complex((a * c) - (b * d), (a * d) + (b * c));
}

complex Cdiv(const complex *c1, const complex *c2)
{
  CG_FLOAT u       = c1->real;
  CG_FLOAT v       = c1->imag;
  CG_FLOAT x       = c2->real;
  CG_FLOAT y       = c2->imag;
  CG_FLOAT c2_abs2 = Cabs2(c2);
  return make_complex((u * x + v * y) / c2_abs2, (v * x + u * y) / c2_abs2);
}

complex Cinv(const complex *c1)
{
  CG_FLOAT x       = c1->real;
  CG_FLOAT y       = c1->imag;
  CG_FLOAT c1_abs2 = Cabs2(c1);
  return make_complex((x) / c1_abs2, (y) / c1_abs2);
}

complex Cconj(const complex *c1)
{
  return make_complex(c1->real, -(c1->imag));
}

CG_FLOAT Cabs(const complex *c1)
{
  const CG_FLOAT abs2 = Cabs2(c1);
#if PRECISION == 1
  return sqrtf(abs2);
#else
  return sqrt(abs2);
#endif
}
CG_FLOAT Cabs2(const complex *c1)
{
  CG_FLOAT x = c1->real;
  CG_FLOAT y = c1->real;
  return (x * x) + (y * y);
}

CG_FLOAT Carg(const complex *c1)
{
  const CG_FLOAT ratio = c1->imag / c1->imag;
#if PRECISION == 1
  return tanf(ratio)
#else
  return tan(ratio);
#endif
}
