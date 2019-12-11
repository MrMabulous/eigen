// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPECIAL_FUNCTIONS_H
#define EIGEN_SPECIAL_FUNCTIONS_H

namespace Eigen {
namespace internal {

//  Parts of this code are based on the Cephes Math Library.
//
//  Cephes Math Library Release 2.8:  June, 2000
//  Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
//
//  Permission has been kindly provided by the original author
//  to incorporate the Cephes software into the Eigen codebase:
//
//    From: Stephen Moshier
//    To: Eugene Brevdo
//    Subject: Re: Permission to wrap several cephes functions in Eigen
//
//    Hello Eugene,
//
//    Thank you for writing.
//
//    If your licensing is similar to BSD, the formal way that has been
//    handled is simply to add a statement to the effect that you are incorporating
//    the Cephes software by permission of the author.
//
//    Good luck with your project,
//    Steve


/****************************************************************************
 * Implementation of lgamma, requires C++11/C99                             *
 ****************************************************************************/

template <typename Scalar>
struct lgamma_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <typename Scalar>
struct lgamma_retval {
  typedef Scalar type;
};

#if EIGEN_HAS_C99_MATH
template <>
struct lgamma_impl<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(float x) {
#if !defined(EIGEN_GPU_COMPILE_PHASE) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
    int dummy;
    return ::lgammaf_r(x, &dummy);
#elif defined(SYCL_DEVICE_ONLY)
    return cl::sycl::lgamma(x);
#else
    return ::lgammaf(x);
#endif
  }
};

template <>
struct lgamma_impl<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(double x) {
#if !defined(EIGEN_GPU_COMPILE_PHASE) && (defined(_BSD_SOURCE) || defined(_SVID_SOURCE)) && !defined(__APPLE__)
    int dummy;
    return ::lgamma_r(x, &dummy);
#elif defined(SYCL_DEVICE_ONLY)
    return cl::sycl::lgamma(x);
#else
    return ::lgamma(x);
#endif
  }
};
#endif

/****************************************************************************
 * Implementation of digamma (psi), based on Cephes                         *
 ****************************************************************************/

template <typename Scalar>
struct digamma_retval {
  typedef Scalar type;
};

/*
 *
 * Polynomial evaluation helper for the Psi (digamma) function.
 *
 * digamma_impl_maybe_poly::run(s) evaluates the asymptotic Psi expansion for
 * input Scalar s, assuming s is above 10.0.
 *
 * If s is above a certain threshold for the given Scalar type, zero
 * is returned.  Otherwise the polynomial is evaluated with enough
 * coefficients for results matching Scalar machine precision.
 *
 *
 */
template <typename Scalar>
struct digamma_impl_maybe_poly {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};


template <>
struct digamma_impl_maybe_poly<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(const float s) {
    const float A[] = {
      -4.16666666666666666667E-3f,
      3.96825396825396825397E-3f,
      -8.33333333333333333333E-3f,
      8.33333333333333333333E-2f
    };

    float z;
    if (s < 1.0e8f) {
      z = 1.0f / (s * s);
      return z * internal::ppolevl<float, 3>::run(z, A);
    } else return 0.0f;
  }
};

template <>
struct digamma_impl_maybe_poly<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(const double s) {
    const double A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2
    };

    double z;
    if (s < 1.0e17) {
      z = 1.0 / (s * s);
      return z * internal::ppolevl<double, 6>::run(z, A);
    }
    else return 0.0;
  }
};

template <typename Scalar>
struct digamma_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar x) {
    /*
     *
     *     Psi (digamma) function (modified for Eigen)
     *
     *
     * SYNOPSIS:
     *
     * double x, y, psi();
     *
     * y = psi( x );
     *
     *
     * DESCRIPTION:
     *
     *              d      -
     *   psi(x)  =  -- ln | (x)
     *              dx
     *
     * is the logarithmic derivative of the gamma function.
     * For integer x,
     *                   n-1
     *                    -
     * psi(n) = -EUL  +   >  1/k.
     *                    -
     *                   k=1
     *
     * If x is negative, it is transformed to a positive argument by the
     * reflection formula  psi(1-x) = psi(x) + pi cot(pi x).
     * For general positive x, the argument is made greater than 10
     * using the recurrence  psi(x+1) = psi(x) + 1/x.
     * Then the following asymptotic expansion is applied:
     *
     *                           inf.   B
     *                            -      2k
     * psi(x) = log(x) - 1/2x -   >   -------
     *                            -        2k
     *                           k=1   2k x
     *
     * where the B2k are Bernoulli numbers.
     *
     * ACCURACY (float):
     *    Relative error (except absolute when |psi| < 1):
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        30000       1.3e-15     1.4e-16
     *    IEEE      -30,0       40000       1.5e-15     2.2e-16
     *
     * ACCURACY (double):
     *    Absolute error,  relative when |psi| > 1 :
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      -33,0        30000      8.2e-7      1.2e-7
     *    IEEE      0,33        100000      7.3e-7      7.7e-8
     *
     * ERROR MESSAGES:
     *     message         condition      value returned
     * psi singularity    x integer <=0      INFINITY
     */

    Scalar p, q, nz, s, w, y;
    bool negative = false;

    const Scalar maxnum = NumTraits<Scalar>::infinity();
    const Scalar m_pi = Scalar(EIGEN_PI);

    const Scalar zero = Scalar(0);
    const Scalar one = Scalar(1);
    const Scalar half = Scalar(0.5);
    nz = zero;

    if (x <= zero) {
      negative = true;
      q = x;
      p = numext::floor(q);
      if (p == q) {
        return maxnum;
      }
      /* Remove the zeros of tan(m_pi x)
       * by subtracting the nearest integer from x
       */
      nz = q - p;
      if (nz != half) {
        if (nz > half) {
          p += one;
          nz = q - p;
        }
        nz = m_pi / numext::tan(m_pi * nz);
      }
      else {
        nz = zero;
      }
      x = one - x;
    }

    /* use the recurrence psi(x+1) = psi(x) + 1/x. */
    s = x;
    w = zero;
    while (s < Scalar(10)) {
      w += one / s;
      s += one;
    }

    y = digamma_impl_maybe_poly<Scalar>::run(s);

    y = numext::log(s) - (half / s) - y - w;

    return (negative) ? y - nz : y;
  }
};

/****************************************************************************
 * Implementation of erf, requires C++11/C99                                *
 ****************************************************************************/

/** \internal \returns the error function of \a a (coeff-wise)
    Doesn't do anything fancy, just a 13/8-degree rational interpolant which
    is accurate up to a couple of ulp in the range [-4, 4], outside of which
    fl(erf(x)) = +/-1.

    This implementation works on both scalars and Ts.
*/
template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_fast_erf_float(const T& a_x) {
  // Clamp the inputs to the range [-4, 4] since anything outside
  // this range is +/-1.0f in single-precision.
  const T plus_4 = pset1<T>(4.f);
  const T minus_4 = pset1<T>(-4.f);
  const T x = pmax(pmin(a_x, plus_4), minus_4);
  // The monomial coefficients of the numerator polynomial (odd).
  const T alpha_1 = pset1<T>(-1.60960333262415e-02f);
  const T alpha_3 = pset1<T>(-2.95459980854025e-03f);
  const T alpha_5 = pset1<T>(-7.34990630326855e-04f);
  const T alpha_7 = pset1<T>(-5.69250639462346e-05f);
  const T alpha_9 = pset1<T>(-2.10102402082508e-06f);
  const T alpha_11 = pset1<T>(2.77068142495902e-08f);
  const T alpha_13 = pset1<T>(-2.72614225801306e-10f);

  // The monomial coefficients of the denominator polynomial (even).
  const T beta_0 = pset1<T>(-1.42647390514189e-02f);
  const T beta_2 = pset1<T>(-7.37332916720468e-03f);
  const T beta_4 = pset1<T>(-1.68282697438203e-03f);
  const T beta_6 = pset1<T>(-2.13374055278905e-04f);
  const T beta_8 = pset1<T>(-1.45660718464996e-05f);

  // Since the polynomials are odd/even, we need x^2.
  const T x2 = pmul(x, x);

  // Evaluate the numerator polynomial p.
  T p = pmadd(x2, alpha_13, alpha_11);
  p = pmadd(x2, p, alpha_9);
  p = pmadd(x2, p, alpha_7);
  p = pmadd(x2, p, alpha_5);
  p = pmadd(x2, p, alpha_3);
  p = pmadd(x2, p, alpha_1);
  p = pmul(x, p);

  // Evaluate the denominator polynomial p.
  T q = pmadd(x2, beta_8, beta_6);
  q = pmadd(x2, q, beta_4);
  q = pmadd(x2, q, beta_2);
  q = pmadd(x2, q, beta_0);

  // Divide the numerator by the denominator.
  return pdiv(p, q);
}

template <typename T>
struct erf_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE T run(const T x) {
    return generic_fast_erf_float(x);
  }
};

template <typename Scalar>
struct erf_retval {
  typedef Scalar type;
};

#if EIGEN_HAS_C99_MATH
template <>
struct erf_impl<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(float x) {
#if defined(SYCL_DEVICE_ONLY)
    return cl::sycl::erf(x);
#else
    return generic_fast_erf_float(x);
#endif
  }
};

template <>
struct erf_impl<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(double x) {
#if defined(SYCL_DEVICE_ONLY)
    return cl::sycl::erf(x);
#else
    return ::erf(x);
#endif
  }
};
#endif  // EIGEN_HAS_C99_MATH

/***************************************************************************
* Implementation of erfc, requires C++11/C99                               *
****************************************************************************/

template <typename Scalar>
struct erfc_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <typename Scalar>
struct erfc_retval {
  typedef Scalar type;
};

#if EIGEN_HAS_C99_MATH
template <>
struct erfc_impl<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(const float x) {
#if defined(SYCL_DEVICE_ONLY)
    return cl::sycl::erfc(x);
#else
    return ::erfcf(x);
#endif
  }
};

template <>
struct erfc_impl<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(const double x) {
#if defined(SYCL_DEVICE_ONLY)
    return cl::sycl::erfc(x);
#else
    return ::erfc(x);
#endif
  }
};
#endif  // EIGEN_HAS_C99_MATH


/***************************************************************************
* Implementation of ndtri.                                                 *
****************************************************************************/

/* Inverse of Normal distribution function (modified for Eigen).
 *
 *
 * SYNOPSIS:
 *
 * double x, y, ndtri();
 *
 * x = ndtri( y );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the argument, x, for which the area under the
 * Gaussian probability density function (integrated from
 * minus infinity to x) is equal to y.
 *
 *
 * For small arguments 0 < y < exp(-2), the program computes
 * z = sqrt( -2.0 * log(y) );  then the approximation is
 * x = z - log(z)/z  - (1/z) P(1/z) / Q(1/z).
 * There are two rational functions P/Q, one for 0 < y < exp(-32)
 * and the other for y up to exp(-2).  For larger arguments,
 * w = y - 0.5, and  x/sqrt(2pi) = w + w**3 R(w**2)/S(w**2)).
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain        # trials      peak         rms
 *    DEC      0.125, 1         5500       9.5e-17     2.1e-17
 *    DEC      6e-39, 0.135     3500       5.7e-17     1.3e-17
 *    IEEE     0.125, 1        20000       7.2e-16     1.3e-16
 *    IEEE     3e-308, 0.135   50000       4.6e-16     9.8e-17
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition    value returned
 * ndtri domain       x <= 0        -MAXNUM
 * ndtri domain       x >= 1         MAXNUM
 *
 */
 /*
   Cephes Math Library Release 2.2: June, 1992
   Copyright 1985, 1987, 1992 by Stephen L. Moshier
   Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */


// TODO: Add a cheaper approximation for float.


template<typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T flipsign(
    const T& should_flipsign, const T& x) {
  const T sign_mask = pset1<T>(-0.0);
  T sign_bit = pand<T>(should_flipsign, sign_mask);
  return pxor<T>(sign_bit, x);
}

template<>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double flipsign<double>(
    const double& should_flipsign, const double& x) {
  return should_flipsign == 0 ? x : -x;
}

template<>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float flipsign<float>(
    const float& should_flipsign, const float& x) {
  return should_flipsign == 0 ? x : -x;
}

// We split this computation in to two so that in the scalar path
// only one branch is evaluated (due to our template specialization of pselect
// being an if statement.)

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_ndtri_gt_exp_neg_two(const T& b) {
  const ScalarType p0[] = {
    ScalarType(-5.99633501014107895267e1),
    ScalarType(9.80010754185999661536e1),
    ScalarType(-5.66762857469070293439e1),
    ScalarType(1.39312609387279679503e1),
    ScalarType(-1.23916583867381258016e0)
  };
  const ScalarType q0[] = {
    ScalarType(1.0),
    ScalarType(1.95448858338141759834e0),
    ScalarType(4.67627912898881538453e0),
    ScalarType(8.63602421390890590575e1),
    ScalarType(-2.25462687854119370527e2),
    ScalarType(2.00260212380060660359e2),
    ScalarType(-8.20372256168333339912e1),
    ScalarType(1.59056225126211695515e1),
    ScalarType(-1.18331621121330003142e0)
  };
  const T sqrt2pi = pset1<T>(ScalarType(2.50662827463100050242e0));
  const T half = pset1<T>(ScalarType(0.5));
  T c, c2, ndtri_gt_exp_neg_two;

  c = psub(b, half);
  c2 = pmul(c, c);
  ndtri_gt_exp_neg_two = pmadd(c, pmul(
      c2, pdiv(
          internal::ppolevl<T, 4>::run(c2, p0),
          internal::ppolevl<T, 8>::run(c2, q0))), c);
  return pmul(ndtri_gt_exp_neg_two, sqrt2pi);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_ndtri_lt_exp_neg_two(
    const T& b, const T& should_flipsign) {
  /* Approximation for interval z = sqrt(-2 log a ) between 2 and 8
   * i.e., a between exp(-2) = .135 and exp(-32) = 1.27e-14.
   */
  const ScalarType p1[] = {
    ScalarType(4.05544892305962419923e0),
    ScalarType(3.15251094599893866154e1),
    ScalarType(5.71628192246421288162e1),
    ScalarType(4.40805073893200834700e1),
    ScalarType(1.46849561928858024014e1),
    ScalarType(2.18663306850790267539e0),
    ScalarType(-1.40256079171354495875e-1),
    ScalarType(-3.50424626827848203418e-2),
    ScalarType(-8.57456785154685413611e-4)
  };
  const ScalarType q1[] = {
    ScalarType(1.0),
    ScalarType(1.57799883256466749731e1),
    ScalarType(4.53907635128879210584e1),
    ScalarType(4.13172038254672030440e1),
    ScalarType(1.50425385692907503408e1),
    ScalarType(2.50464946208309415979e0),
    ScalarType(-1.42182922854787788574e-1),
    ScalarType(-3.80806407691578277194e-2),
    ScalarType(-9.33259480895457427372e-4)
  };
  /* Approximation for interval z = sqrt(-2 log a ) between 8 and 64
   * i.e., a between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
   */
  const ScalarType p2[] = {
    ScalarType(3.23774891776946035970e0),
    ScalarType(6.91522889068984211695e0),
    ScalarType(3.93881025292474443415e0),
    ScalarType(1.33303460815807542389e0),
    ScalarType(2.01485389549179081538e-1),
    ScalarType(1.23716634817820021358e-2),
    ScalarType(3.01581553508235416007e-4),
    ScalarType(2.65806974686737550832e-6),
    ScalarType(6.23974539184983293730e-9)
  };
  const ScalarType q2[] = {
    ScalarType(1.0),
    ScalarType(6.02427039364742014255e0),
    ScalarType(3.67983563856160859403e0),
    ScalarType(1.37702099489081330271e0),
    ScalarType(2.16236993594496635890e-1),
    ScalarType(1.34204006088543189037e-2),
    ScalarType(3.28014464682127739104e-4),
    ScalarType(2.89247864745380683936e-6),
    ScalarType(6.79019408009981274425e-9)
  };
  const T eight = pset1<T>(ScalarType(8.0));
  const T one = pset1<T>(ScalarType(1));
  const T neg_two = pset1<T>(ScalarType(-2));
  T x, x0, x1, z;

  x = psqrt(pmul(neg_two, plog(b)));
  x0 = psub(x, pdiv(plog(x), x));
  z = pdiv(one, x);
  x1 = pmul(
      z, pselect(
          pcmp_lt(x, eight),
          pdiv(internal::ppolevl<T, 8>::run(z, p1),
               internal::ppolevl<T, 8>::run(z, q1)),
          pdiv(internal::ppolevl<T, 8>::run(z, p2),
               internal::ppolevl<T, 8>::run(z, q2))));
  return flipsign(should_flipsign, psub(x0, x1));
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T generic_ndtri(const T& a) {
  const T maxnum = pset1<T>(NumTraits<ScalarType>::infinity());
  const T neg_maxnum = pset1<T>(-NumTraits<ScalarType>::infinity());

  const T zero = pset1<T>(ScalarType(0));
  const T one = pset1<T>(ScalarType(1));
  // exp(-2)
  const T exp_neg_two = pset1<T>(ScalarType(0.13533528323661269189));
  T b, ndtri, should_flipsign;

  should_flipsign = pcmp_le(a, psub(one, exp_neg_two));
  b = pselect(should_flipsign, a, psub(one, a));

  ndtri = pselect(
      pcmp_lt(exp_neg_two, b),
      generic_ndtri_gt_exp_neg_two<T, ScalarType>(b),
      generic_ndtri_lt_exp_neg_two<T, ScalarType>(b, should_flipsign));

  return pselect(
      pcmp_le(a, zero), neg_maxnum,
      pselect(pcmp_le(one, a), maxnum, ndtri));
}

template <typename Scalar>
struct ndtri_retval {
  typedef Scalar type;
};

#if !EIGEN_HAS_C99_MATH

template <typename Scalar>
struct ndtri_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

# else

template <typename Scalar>
struct ndtri_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar x) {
    return generic_ndtri<Scalar, Scalar>(x);
  }
};

#endif  // EIGEN_HAS_C99_MATH


/**************************************************************************************************************
 * Implementation of igammac (complemented incomplete gamma integral), based on Cephes but requires C++11/C99 *
 **************************************************************************************************************/

template <typename Scalar>
struct igammac_retval {
  typedef Scalar type;
};

// NOTE: cephes_helper is also used to implement zeta
template <typename Scalar>
struct cephes_helper {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar machep() { assert(false && "machep not supported for this type"); return 0.0; }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar big() { assert(false && "big not supported for this type"); return 0.0; }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar biginv() { assert(false && "biginv not supported for this type"); return 0.0; }
};

template <>
struct cephes_helper<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float machep() {
    return NumTraits<float>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float big() {
    // use epsneg (1.0 - epsneg == 1.0)
    return 1.0f / (NumTraits<float>::epsilon() / 2);
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float biginv() {
    // epsneg
    return machep();
  }
};

template <>
struct cephes_helper<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double machep() {
    return NumTraits<double>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double big() {
    return 1.0 / NumTraits<double>::epsilon();
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double biginv() {
    // inverse of eps
    return NumTraits<double>::epsilon();
  }
};

enum IgammaComputationMode { VALUE, DERIVATIVE, SAMPLE_DERIVATIVE };

template <typename Scalar, IgammaComputationMode mode>
EIGEN_DEVICE_FUNC
int igamma_num_iterations() {
  /* Returns the maximum number of internal iterations for igamma computation.
   */
  if (mode == VALUE) {
    return 2000;
  }

  if (internal::is_same<Scalar, float>::value) {
    return 200;
  } else if (internal::is_same<Scalar, double>::value) {
    return 500;
  } else {
    return 2000;
  }
}

template <typename Scalar, IgammaComputationMode mode>
struct igammac_cf_impl {
  /* Computes igamc(a, x) or derivative (depending on the mode)
   * using the continued fraction expansion of the complementary
   * incomplete Gamma function.
   *
   * Preconditions:
   *   a > 0
   *   x >= 1
   *   x >= a
   */
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar two = 2;
    const Scalar machep = cephes_helper<Scalar>::machep();
    const Scalar big = cephes_helper<Scalar>::big();
    const Scalar biginv = cephes_helper<Scalar>::biginv();

    if ((numext::isinf)(x)) {
      return zero;
    }

    // continued fraction
    Scalar y = one - a;
    Scalar z = x + y + one;
    Scalar c = zero;
    Scalar pkm2 = one;
    Scalar qkm2 = x;
    Scalar pkm1 = x + one;
    Scalar qkm1 = z * x;
    Scalar ans = pkm1 / qkm1;

    Scalar dpkm2_da = zero;
    Scalar dqkm2_da = zero;
    Scalar dpkm1_da = zero;
    Scalar dqkm1_da = -x;
    Scalar dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;

    for (int i = 0; i < igamma_num_iterations<Scalar, mode>(); i++) {
      c += one;
      y += one;
      z += two;

      Scalar yc = y * c;
      Scalar pk = pkm1 * z - pkm2 * yc;
      Scalar qk = qkm1 * z - qkm2 * yc;

      Scalar dpk_da = dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c;
      Scalar dqk_da = dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c;

      if (qk != zero) {
        Scalar ans_prev = ans;
        ans = pk / qk;

        Scalar dans_da_prev = dans_da;
        dans_da = (dpk_da - ans * dqk_da) / qk;

        if (mode == VALUE) {
          if (numext::abs(ans_prev - ans) <= machep * numext::abs(ans)) {
            break;
          }
        } else {
          if (numext::abs(dans_da - dans_da_prev) <= machep) {
            break;
          }
        }
      }

      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      dpkm2_da = dpkm1_da;
      dpkm1_da = dpk_da;
      dqkm2_da = dqkm1_da;
      dqkm1_da = dqk_da;

      if (numext::abs(pk) > big) {
        pkm2 *= biginv;
        pkm1 *= biginv;
        qkm2 *= biginv;
        qkm1 *= biginv;

        dpkm2_da *= biginv;
        dpkm1_da *= biginv;
        dqkm2_da *= biginv;
        dqkm1_da *= biginv;
      }
    }

    /* Compute  x**a * exp(-x) / gamma(a)  */
    Scalar logax = a * numext::log(x) - x - lgamma_impl<Scalar>::run(a);
    Scalar dlogax_da = numext::log(x) - digamma_impl<Scalar>::run(a);
    Scalar ax = numext::exp(logax);
    Scalar dax_da = ax * dlogax_da;

    switch (mode) {
      case VALUE:
        return ans * ax;
      case DERIVATIVE:
        return ans * dax_da + dans_da * ax;
      case SAMPLE_DERIVATIVE:
      default: // this is needed to suppress clang warning
        return -(dans_da + ans * dlogax_da) * x;
    }
  }
};

template <typename Scalar, IgammaComputationMode mode>
struct igamma_series_impl {
  /* Computes igam(a, x) or its derivative (depending on the mode)
   * using the series expansion of the incomplete Gamma function.
   *
   * Preconditions:
   *   x > 0
   *   a > 0
   *   !(x > 1 && x > a)
   */
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar machep = cephes_helper<Scalar>::machep();

    /* power series */
    Scalar r = a;
    Scalar c = one;
    Scalar ans = one;

    Scalar dc_da = zero;
    Scalar dans_da = zero;

    for (int i = 0; i < igamma_num_iterations<Scalar, mode>(); i++) {
      r += one;
      Scalar term = x / r;
      Scalar dterm_da = -x / (r * r);
      dc_da = term * dc_da + dterm_da * c;
      dans_da += dc_da;
      c *= term;
      ans += c;

      if (mode == VALUE) {
        if (c <= machep * ans) {
          break;
        }
      } else {
        if (numext::abs(dc_da) <= machep * numext::abs(dans_da)) {
          break;
        }
      }
    }

    /* Compute  x**a * exp(-x) / gamma(a + 1)  */
    Scalar logax = a * numext::log(x) - x - lgamma_impl<Scalar>::run(a + one);
    Scalar dlogax_da = numext::log(x) - digamma_impl<Scalar>::run(a + one);
    Scalar ax = numext::exp(logax);
    Scalar dax_da = ax * dlogax_da;

    switch (mode) {
      case VALUE:
        return ans * ax;
      case DERIVATIVE:
        return ans * dax_da + dans_da * ax;
      case SAMPLE_DERIVATIVE:
      default: // this is needed to suppress clang warning
        return -(dans_da + ans * dlogax_da) * x / a;
    }
  }
};

#if !EIGEN_HAS_C99_MATH

template <typename Scalar>
struct igammac_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

#else

template <typename Scalar>
struct igammac_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    /*  igamc()
     *
     *	Incomplete gamma integral (modified for Eigen)
     *
     *
     *
     * SYNOPSIS:
     *
     * double a, x, y, igamc();
     *
     * y = igamc( a, x );
     *
     * DESCRIPTION:
     *
     * The function is defined by
     *
     *
     *  igamc(a,x)   =   1 - igam(a,x)
     *
     *                            inf.
     *                              -
     *                     1       | |  -t  a-1
     *               =   -----     |   e   t   dt.
     *                    -      | |
     *                   | (a)    -
     *                             x
     *
     *
     * In this implementation both arguments must be positive.
     * The integral is evaluated by either a power series or
     * continued fraction expansion, depending on the relative
     * values of a and x.
     *
     * ACCURACY (float):
     *
     *                      Relative error:
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,30        30000       7.8e-6      5.9e-7
     *
     *
     * ACCURACY (double):
     *
     * Tested at random a, x.
     *                a         x                      Relative error:
     * arithmetic   domain   domain     # trials      peak         rms
     *    IEEE     0.5,100   0,100      200000       1.9e-14     1.7e-15
     *    IEEE     0.01,0.5  0,100      200000       1.4e-13     1.6e-15
     *
     */
    /*
      Cephes Math Library Release 2.2: June, 1992
      Copyright 1985, 1987, 1992 by Stephen L. Moshier
      Direct inquiries to 30 Frost Street, Cambridge, MA 02140
    */
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    if ((x < zero) || (a <= zero)) {
      // domain error
      return nan;
    }

    if ((numext::isnan)(a) || (numext::isnan)(x)) {  // propagate nans
      return nan;
    }

    if ((x < one) || (x < a)) {
      return (one - igamma_series_impl<Scalar, VALUE>::run(a, x));
    }

    return igammac_cf_impl<Scalar, VALUE>::run(a, x);
  }
};

#endif  // EIGEN_HAS_C99_MATH

/************************************************************************************************
 * Implementation of igamma (incomplete gamma integral), based on Cephes but requires C++11/C99 *
 ************************************************************************************************/

#if !EIGEN_HAS_C99_MATH

template <typename Scalar, IgammaComputationMode mode>
struct igamma_generic_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(Scalar a, Scalar x) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

#else

template <typename Scalar, IgammaComputationMode mode>
struct igamma_generic_impl {
  EIGEN_DEVICE_FUNC
  static Scalar run(Scalar a, Scalar x) {
    /* Depending on the mode, returns
     * - VALUE: incomplete Gamma function igamma(a, x)
     * - DERIVATIVE: derivative of incomplete Gamma function d/da igamma(a, x)
     * - SAMPLE_DERIVATIVE: implicit derivative of a Gamma random variable
     * x ~ Gamma(x | a, 1), dx/da = -1 / Gamma(x | a, 1) * d igamma(a, x) / dx
     *
     * Derivatives are implemented by forward-mode differentiation.
     */
    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar nan = NumTraits<Scalar>::quiet_NaN();

    if (x == zero) return zero;

    if ((x < zero) || (a <= zero)) {  // domain error
      return nan;
    }

    if ((numext::isnan)(a) || (numext::isnan)(x)) {  // propagate nans
      return nan;
    }

    if ((x > one) && (x > a)) {
      Scalar ret = igammac_cf_impl<Scalar, mode>::run(a, x);
      if (mode == VALUE) {
        return one - ret;
      } else {
        return -ret;
      }
    }

    return igamma_series_impl<Scalar, mode>::run(a, x);
  }
};

#endif  // EIGEN_HAS_C99_MATH

template <typename Scalar>
struct igamma_retval {
  typedef Scalar type;
};

template <typename Scalar>
struct igamma_impl : igamma_generic_impl<Scalar, VALUE> {
  /* igam()
   * Incomplete gamma integral.
   *
   * The CDF of Gamma(a, 1) random variable at the point x.
   *
   * Accuracy estimation. For each a in [10^-2, 10^-1...10^3] we sample
   * 50 Gamma random variables x ~ Gamma(x | a, 1), a total of 300 points.
   * The ground truth is computed by mpmath. Mean absolute error:
   * float: 1.26713e-05
   * double: 2.33606e-12
   *
   * Cephes documentation below.
   *
   * SYNOPSIS:
   *
   * double a, x, y, igam();
   *
   * y = igam( a, x );
   *
   * DESCRIPTION:
   *
   * The function is defined by
   *
   *                           x
   *                            -
   *                   1       | |  -t  a-1
   *  igam(a,x)  =   -----     |   e   t   dt.
   *                  -      | |
   *                 | (a)    -
   *                           0
   *
   *
   * In this implementation both arguments must be positive.
   * The integral is evaluated by either a power series or
   * continued fraction expansion, depending on the relative
   * values of a and x.
   *
   * ACCURACY (double):
   *
   *                      Relative error:
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      0,30       200000       3.6e-14     2.9e-15
   *    IEEE      0,100      300000       9.9e-14     1.5e-14
   *
   *
   * ACCURACY (float):
   *
   *                      Relative error:
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      0,30        20000       7.8e-6      5.9e-7
   *
   */
  /*
    Cephes Math Library Release 2.2: June, 1992
    Copyright 1985, 1987, 1992 by Stephen L. Moshier
    Direct inquiries to 30 Frost Street, Cambridge, MA 02140
  */

  /* left tail of incomplete gamma function:
   *
   *          inf.      k
   *   a  -x   -       x
   *  x  e     >   ----------
   *           -     -
   *          k=0   | (a+k+1)
   *
   */
};

template <typename Scalar>
struct igamma_der_a_retval : igamma_retval<Scalar> {};

template <typename Scalar>
struct igamma_der_a_impl : igamma_generic_impl<Scalar, DERIVATIVE> {
  /* Derivative of the incomplete Gamma function with respect to a.
   *
   * Computes d/da igamma(a, x) by forward differentiation of the igamma code.
   *
   * Accuracy estimation. For each a in [10^-2, 10^-1...10^3] we sample
   * 50 Gamma random variables x ~ Gamma(x | a, 1), a total of 300 points.
   * The ground truth is computed by mpmath. Mean absolute error:
   * float: 6.17992e-07
   * double: 4.60453e-12
   *
   * Reference:
   * R. Moore. "Algorithm AS 187: Derivatives of the incomplete gamma
   * integral". Journal of the Royal Statistical Society. 1982
   */
};

template <typename Scalar>
struct gamma_sample_der_alpha_retval : igamma_retval<Scalar> {};

template <typename Scalar>
struct gamma_sample_der_alpha_impl
    : igamma_generic_impl<Scalar, SAMPLE_DERIVATIVE> {
  /* Derivative of a Gamma random variable sample with respect to alpha.
   *
   * Consider a sample of a Gamma random variable with the concentration
   * parameter alpha: sample ~ Gamma(alpha, 1). The reparameterization
   * derivative that we want to compute is dsample / dalpha =
   * d igammainv(alpha, u) / dalpha, where u = igamma(alpha, sample).
   * However, this formula is numerically unstable and expensive, so instead
   * we use implicit differentiation:
   *
   * igamma(alpha, sample) = u, where u ~ Uniform(0, 1).
   * Apply d / dalpha to both sides:
   * d igamma(alpha, sample) / dalpha
   *     + d igamma(alpha, sample) / dsample * dsample/dalpha  = 0
   * d igamma(alpha, sample) / dalpha
   *     + Gamma(sample | alpha, 1) dsample / dalpha = 0
   * dsample/dalpha = - (d igamma(alpha, sample) / dalpha)
   *                   / Gamma(sample | alpha, 1)
   *
   * Here Gamma(sample | alpha, 1) is the PDF of the Gamma distribution
   * (note that the derivative of the CDF w.r.t. sample is the PDF).
   * See the reference below for more details.
   *
   * The derivative of igamma(alpha, sample) is computed by forward
   * differentiation of the igamma code. Division by the Gamma PDF is performed
   * in the same code, increasing the accuracy and speed due to cancellation
   * of some terms.
   *
   * Accuracy estimation. For each alpha in [10^-2, 10^-1...10^3] we sample
   * 50 Gamma random variables sample ~ Gamma(sample | alpha, 1), a total of 300
   * points. The ground truth is computed by mpmath. Mean absolute error:
   * float: 2.1686e-06
   * double: 1.4774e-12
   *
   * Reference:
   * M. Figurnov, S. Mohamed, A. Mnih "Implicit Reparameterization Gradients".
   * 2018
   */
};

/*****************************************************************************
 * Implementation of Riemann zeta function of two arguments, based on Cephes *
 *****************************************************************************/

template <typename Scalar>
struct zeta_retval {
    typedef Scalar type;
};

template <typename Scalar>
struct zeta_impl_series {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <>
struct zeta_impl_series<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE bool run(float& a, float& b, float& s, const float x, const float machep) {
    int i = 0;
    while(i < 9)
    {
        i += 1;
        a += 1.0f;
        b = numext::pow( a, -x );
        s += b;
        if( numext::abs(b/s) < machep )
            return true;
    }

    //Return whether we are done
    return false;
  }
};

template <>
struct zeta_impl_series<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE bool run(double& a, double& b, double& s, const double x, const double machep) {
    int i = 0;
    while( (i < 9) || (a <= 9.0) )
    {
        i += 1;
        a += 1.0;
        b = numext::pow( a, -x );
        s += b;
        if( numext::abs(b/s) < machep )
            return true;
    }

    //Return whether we are done
    return false;
  }
};

template <typename Scalar>
struct zeta_impl {
    EIGEN_DEVICE_FUNC
    static Scalar run(Scalar x, Scalar q) {
        /*							zeta.c
         *
         *	Riemann zeta function of two arguments
         *
         *
         *
         * SYNOPSIS:
         *
         * double x, q, y, zeta();
         *
         * y = zeta( x, q );
         *
         *
         *
         * DESCRIPTION:
         *
         *
         *
         *                 inf.
         *                  -        -x
         *   zeta(x,q)  =   >   (k+q)
         *                  -
         *                 k=0
         *
         * where x > 1 and q is not a negative integer or zero.
         * The Euler-Maclaurin summation formula is used to obtain
         * the expansion
         *
         *                n
         *                -       -x
         * zeta(x,q)  =   >  (k+q)
         *                -
         *               k=1
         *
         *           1-x                 inf.  B   x(x+1)...(x+2j)
         *      (n+q)           1         -     2j
         *  +  ---------  -  -------  +   >    --------------------
         *        x-1              x      -                   x+2j+1
         *                   2(n+q)      j=1       (2j)! (n+q)
         *
         * where the B2j are Bernoulli numbers.  Note that (see zetac.c)
         * zeta(x,1) = zetac(x) + 1.
         *
         *
         *
         * ACCURACY:
         *
         * Relative error for single precision:
         * arithmetic   domain     # trials      peak         rms
         *    IEEE      0,25        10000       6.9e-7      1.0e-7
         *
         * Large arguments may produce underflow in powf(), in which
         * case the results are inaccurate.
         *
         * REFERENCE:
         *
         * Gradshteyn, I. S., and I. M. Ryzhik, Tables of Integrals,
         * Series, and Products, p. 1073; Academic Press, 1980.
         *
         */

        int i;
        Scalar p, r, a, b, k, s, t, w;

        const Scalar A[] = {
            Scalar(12.0),
            Scalar(-720.0),
            Scalar(30240.0),
            Scalar(-1209600.0),
            Scalar(47900160.0),
            Scalar(-1.8924375803183791606e9), /*1.307674368e12/691*/
            Scalar(7.47242496e10),
            Scalar(-2.950130727918164224e12), /*1.067062284288e16/3617*/
            Scalar(1.1646782814350067249e14), /*5.109094217170944e18/43867*/
            Scalar(-4.5979787224074726105e15), /*8.028576626982912e20/174611*/
            Scalar(1.8152105401943546773e17), /*1.5511210043330985984e23/854513*/
            Scalar(-7.1661652561756670113e18) /*1.6938241367317436694528e27/236364091*/
            };

        const Scalar maxnum = NumTraits<Scalar>::infinity();
        const Scalar zero = 0.0, half = 0.5, one = 1.0;
        const Scalar machep = cephes_helper<Scalar>::machep();
        const Scalar nan = NumTraits<Scalar>::quiet_NaN();

        if( x == one )
            return maxnum;

        if( x < one )
        {
            return nan;
        }

        if( q <= zero )
        {
            if(q == numext::floor(q))
            {
                return maxnum;
            }
            p = x;
            r = numext::floor(p);
            if (p != r)
                return nan;
        }

        /* Permit negative q but continue sum until n+q > +9 .
         * This case should be handled by a reflection formula.
         * If q<0 and x is an integer, there is a relation to
         * the polygamma function.
         */
        s = numext::pow( q, -x );
        a = q;
        b = zero;
        // Run the summation in a helper function that is specific to the floating precision
        if (zeta_impl_series<Scalar>::run(a, b, s, x, machep)) {
            return s;
        }

        w = a;
        s += b*w/(x-one);
        s -= half * b;
        a = one;
        k = zero;
        for( i=0; i<12; i++ )
        {
            a *= x + k;
            b /= w;
            t = a*b/A[i];
            s = s + t;
            t = numext::abs(t/s);
            if( t < machep ) {
              break;
            }
            k += one;
            a *= x + k;
            b /= w;
            k += one;
        }
        return s;
  }
};

/****************************************************************************
 * Implementation of polygamma function, requires C++11/C99                 *
 ****************************************************************************/

template <typename Scalar>
struct polygamma_retval {
    typedef Scalar type;
};

#if !EIGEN_HAS_C99_MATH

template <typename Scalar>
struct polygamma_impl {
    EIGEN_DEVICE_FUNC
    static EIGEN_STRONG_INLINE Scalar run(Scalar n, Scalar x) {
        EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                            THIS_TYPE_IS_NOT_SUPPORTED);
        return Scalar(0);
    }
};

#else

template <typename Scalar>
struct polygamma_impl {
    EIGEN_DEVICE_FUNC
    static Scalar run(Scalar n, Scalar x) {
        Scalar zero = 0.0, one = 1.0;
        Scalar nplus = n + one;
        const Scalar nan = NumTraits<Scalar>::quiet_NaN();

        // Check that n is an integer
        if (numext::floor(n) != n) {
            return nan;
        }
        // Just return the digamma function for n = 1
        else if (n == zero) {
            return digamma_impl<Scalar>::run(x);
        }
        // Use the same implementation as scipy
        else {
            Scalar factorial = numext::exp(lgamma_impl<Scalar>::run(nplus));
            return numext::pow(-one, nplus) * factorial * zeta_impl<Scalar>::run(nplus, x);
        }
  }
};

#endif  // EIGEN_HAS_C99_MATH

/************************************************************************************************
 * Implementation of betainc (incomplete beta integral), based on Cephes but requires C++11/C99 *
 ************************************************************************************************/

template <typename Scalar>
struct betainc_retval {
  typedef Scalar type;
};

#if !EIGEN_HAS_C99_MATH

template <typename Scalar>
struct betainc_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(Scalar a, Scalar b, Scalar x) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

#else

template <typename Scalar>
struct betainc_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(Scalar, Scalar, Scalar) {
    /*	betaincf.c
     *
     *	Incomplete beta integral
     *
     *
     * SYNOPSIS:
     *
     * float a, b, x, y, betaincf();
     *
     * y = betaincf( a, b, x );
     *
     *
     * DESCRIPTION:
     *
     * Returns incomplete beta integral of the arguments, evaluated
     * from zero to x.  The function is defined as
     *
     *                  x
     *     -            -
     *    | (a+b)      | |  a-1     b-1
     *  -----------    |   t   (1-t)   dt.
     *   -     -     | |
     *  | (a) | (b)   -
     *                 0
     *
     * The domain of definition is 0 <= x <= 1.  In this
     * implementation a and b are restricted to positive values.
     * The integral from x to 1 may be obtained by the symmetry
     * relation
     *
     *    1 - betainc( a, b, x )  =  betainc( b, a, 1-x ).
     *
     * The integral is evaluated by a continued fraction expansion.
     * If a < 1, the function calls itself recursively after a
     * transformation to increase a to a+1.
     *
     * ACCURACY (float):
     *
     * Tested at random points (a,b,x) with a and b in the indicated
     * interval and x between 0 and 1.
     *
     * arithmetic   domain     # trials      peak         rms
     * Relative error:
     *    IEEE       0,30       10000       3.7e-5      5.1e-6
     *    IEEE       0,100      10000       1.7e-4      2.5e-5
     * The useful domain for relative error is limited by underflow
     * of the single precision exponential function.
     * Absolute error:
     *    IEEE       0,30      100000       2.2e-5      9.6e-7
     *    IEEE       0,100      10000       6.5e-5      3.7e-6
     *
     * Larger errors may occur for extreme ratios of a and b.
     *
     * ACCURACY (double):
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,5         10000       6.9e-15     4.5e-16
     *    IEEE      0,85       250000       2.2e-13     1.7e-14
     *    IEEE      0,1000      30000       5.3e-12     6.3e-13
     *    IEEE      0,10000    250000       9.3e-11     7.1e-12
     *    IEEE      0,100000    10000       8.7e-10     4.8e-11
     * Outputs smaller than the IEEE gradual underflow threshold
     * were excluded from these statistics.
     *
     * ERROR MESSAGES:
     *   message         condition      value returned
     * incbet domain      x<0, x>1          nan
     * incbet underflow                     nan
     */

    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

/* Continued fraction expansion #1 for incomplete beta integral (small_branch = True)
 * Continued fraction expansion #2 for incomplete beta integral (small_branch = False)
 */
template <typename Scalar>
struct incbeta_cfe {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(Scalar a, Scalar b, Scalar x, bool small_branch) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, float>::value ||
                         internal::is_same<Scalar, double>::value),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    const Scalar big = cephes_helper<Scalar>::big();
    const Scalar machep = cephes_helper<Scalar>::machep();
    const Scalar biginv = cephes_helper<Scalar>::biginv();

    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar two = 2;

    Scalar xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    Scalar k1, k2, k3, k4, k5, k6, k7, k8, k26update;
    Scalar ans;
    int n;

    const int num_iters = (internal::is_same<Scalar, float>::value) ? 100 : 300;
    const Scalar thresh =
        (internal::is_same<Scalar, float>::value) ? machep : Scalar(3) * machep;
    Scalar r = (internal::is_same<Scalar, float>::value) ? zero : one;

    if (small_branch) {
      k1 = a;
      k2 = a + b;
      k3 = a;
      k4 = a + one;
      k5 = one;
      k6 = b - one;
      k7 = k4;
      k8 = a + two;
      k26update = one;
    } else {
      k1 = a;
      k2 = b - one;
      k3 = a;
      k4 = a + one;
      k5 = one;
      k6 = a + b;
      k7 = a + one;
      k8 = a + two;
      k26update = -one;
      x = x / (one - x);
    }

    pkm2 = zero;
    qkm2 = one;
    pkm1 = one;
    qkm1 = one;
    ans = one;
    n = 0;

    do {
      xk = -(x * k1 * k2) / (k3 * k4);
      pk = pkm1 + pkm2 * xk;
      qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = (x * k5 * k6) / (k7 * k8);
      pk = pkm1 + pkm2 * xk;
      qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if (qk != zero) {
        r = pk / qk;
        if (numext::abs(ans - r) < numext::abs(r) * thresh) {
          return r;
        }
        ans = r;
      }

      k1 += one;
      k2 += k26update;
      k3 += two;
      k4 += two;
      k5 += one;
      k6 -= k26update;
      k7 += two;
      k8 += two;

      if ((numext::abs(qk) + numext::abs(pk)) > big) {
        pkm2 *= biginv;
        pkm1 *= biginv;
        qkm2 *= biginv;
        qkm1 *= biginv;
      }
      if ((numext::abs(qk) < biginv) || (numext::abs(pk) < biginv)) {
        pkm2 *= big;
        pkm1 *= big;
        qkm2 *= big;
        qkm1 *= big;
      }
    } while (++n < num_iters);

    return ans;
  }
};

/* Helper functions depending on the Scalar type */
template <typename Scalar>
struct betainc_helper {};

template <>
struct betainc_helper<float> {
  /* Core implementation, assumes a large (> 1.0) */
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float incbsa(float aa, float bb,
                                                            float xx) {
    float ans, a, b, t, x, onemx;
    bool reversed_a_b = false;

    onemx = 1.0f - xx;

    /* see if x is greater than the mean */
    if (xx > (aa / (aa + bb))) {
      reversed_a_b = true;
      a = bb;
      b = aa;
      t = xx;
      x = onemx;
    } else {
      a = aa;
      b = bb;
      t = onemx;
      x = xx;
    }

    /* Choose expansion for optimal convergence */
    if (b > 10.0f) {
      if (numext::abs(b * x / a) < 0.3f) {
        t = betainc_helper<float>::incbps(a, b, x);
        if (reversed_a_b) t = 1.0f - t;
        return t;
      }
    }

    ans = x * (a + b - 2.0f) / (a - 1.0f);
    if (ans < 1.0f) {
      ans = incbeta_cfe<float>::run(a, b, x, true /* small_branch */);
      t = b * numext::log(t);
    } else {
      ans = incbeta_cfe<float>::run(a, b, x, false /* small_branch */);
      t = (b - 1.0f) * numext::log(t);
    }

    t += a * numext::log(x) + lgamma_impl<float>::run(a + b) -
         lgamma_impl<float>::run(a) - lgamma_impl<float>::run(b);
    t += numext::log(ans / a);
    t = numext::exp(t);

    if (reversed_a_b) t = 1.0f - t;
    return t;
  }

  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float incbps(float a, float b, float x) {
    float t, u, y, s;
    const float machep = cephes_helper<float>::machep();

    y = a * numext::log(x) + (b - 1.0f) * numext::log1p(-x) - numext::log(a);
    y -= lgamma_impl<float>::run(a) + lgamma_impl<float>::run(b);
    y += lgamma_impl<float>::run(a + b);

    t = x / (1.0f - x);
    s = 0.0f;
    u = 1.0f;
    do {
      b -= 1.0f;
      if (b == 0.0f) {
        break;
      }
      a += 1.0f;
      u *= t * b / a;
      s += u;
    } while (numext::abs(u) > machep);

    return numext::exp(y) * (1.0f + s);
  }
};

template <>
struct betainc_impl<float> {
  EIGEN_DEVICE_FUNC
  static float run(float a, float b, float x) {
    const float nan = NumTraits<float>::quiet_NaN();
    float ans, t;

    if (a <= 0.0f) return nan;
    if (b <= 0.0f) return nan;
    if ((x <= 0.0f) || (x >= 1.0f)) {
      if (x == 0.0f) return 0.0f;
      if (x == 1.0f) return 1.0f;
      // mtherr("betaincf", DOMAIN);
      return nan;
    }

    /* transformation for small aa */
    if (a <= 1.0f) {
      ans = betainc_helper<float>::incbsa(a + 1.0f, b, x);
      t = a * numext::log(x) + b * numext::log1p(-x) +
          lgamma_impl<float>::run(a + b) - lgamma_impl<float>::run(a + 1.0f) -
          lgamma_impl<float>::run(b);
      return (ans + numext::exp(t));
    } else {
      return betainc_helper<float>::incbsa(a, b, x);
    }
  }
};

template <>
struct betainc_helper<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double incbps(double a, double b, double x) {
    const double machep = cephes_helper<double>::machep();

    double s, t, u, v, n, t1, z, ai;

    ai = 1.0 / a;
    u = (1.0 - b) * x;
    v = u / (a + 1.0);
    t1 = v;
    t = u;
    n = 2.0;
    s = 0.0;
    z = machep * ai;
    while (numext::abs(v) > z) {
      u = (n - b) * x / n;
      t *= u;
      v = t / (a + n);
      s += v;
      n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * numext::log(x);
    // TODO: gamma() is not directly implemented in Eigen.
    /*
    if ((a + b) < maxgam && numext::abs(u) < maxlog) {
      t = gamma(a + b) / (gamma(a) * gamma(b));
      s = s * t * pow(x, a);
    }
    */
    t = lgamma_impl<double>::run(a + b) - lgamma_impl<double>::run(a) -
        lgamma_impl<double>::run(b) + u + numext::log(s);
    return s = numext::exp(t);
  }
};

template <>
struct betainc_impl<double> {
  EIGEN_DEVICE_FUNC
  static double run(double aa, double bb, double xx) {
    const double nan = NumTraits<double>::quiet_NaN();
    const double machep = cephes_helper<double>::machep();
    // const double maxgam = 171.624376956302725;

    double a, b, t, x, xc, w, y;
    bool reversed_a_b = false;

    if (aa <= 0.0 || bb <= 0.0) {
      return nan;  // goto domerr;
    }

    if ((xx <= 0.0) || (xx >= 1.0)) {
      if (xx == 0.0) return (0.0);
      if (xx == 1.0) return (1.0);
      // mtherr("incbet", DOMAIN);
      return nan;
    }

    if ((bb * xx) <= 1.0 && xx <= 0.95) {
      return betainc_helper<double>::incbps(aa, bb, xx);
    }

    w = 1.0 - xx;

    /* Reverse a and b if x is greater than the mean. */
    if (xx > (aa / (aa + bb))) {
      reversed_a_b = true;
      a = bb;
      b = aa;
      xc = xx;
      x = w;
    } else {
      a = aa;
      b = bb;
      xc = w;
      x = xx;
    }

    if (reversed_a_b && (b * x) <= 1.0 && x <= 0.95) {
      t = betainc_helper<double>::incbps(a, b, x);
      if (t <= machep) {
        t = 1.0 - machep;
      } else {
        t = 1.0 - t;
      }
      return t;
    }

    /* Choose expansion for better convergence. */
    y = x * (a + b - 2.0) - (a - 1.0);
    if (y < 0.0) {
      w = incbeta_cfe<double>::run(a, b, x, true /* small_branch */);
    } else {
      w = incbeta_cfe<double>::run(a, b, x, false /* small_branch */) / xc;
    }

    /* Multiply w by the factor
         a      b   _             _     _
        x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

    y = a * numext::log(x);
    t = b * numext::log(xc);
    // TODO: gamma is not directly implemented in Eigen.
    /*
    if ((a + b) < maxgam && numext::abs(y) < maxlog && numext::abs(t) < maxlog)
    {
      t = pow(xc, b);
      t *= pow(x, a);
      t /= a;
      t *= w;
      t *= gamma(a + b) / (gamma(a) * gamma(b));
    } else {
    */
    /* Resort to logarithms.  */
    y += t + lgamma_impl<double>::run(a + b) - lgamma_impl<double>::run(a) -
         lgamma_impl<double>::run(b);
    y += numext::log(w / a);
    t = numext::exp(y);

    /* } */
    // done:

    if (reversed_a_b) {
      if (t <= machep) {
        t = 1.0 - machep;
      } else {
        t = 1.0 - t;
      }
    }
    return t;
  }
};

#endif  // EIGEN_HAS_C99_MATH


/***************************************************************************
* Implementation of Dawson's Integral.                                     *
****************************************************************************/

/*							dawsn.c
 *
 *	Dawson's Integral
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, dawsn();
 *
 * y = dawsn( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Approximates the integral
 *
 *                             x
 *                             -
 *                      2     | |        2
 *  dawsn(x)  =  exp( -x  )   |    exp( t  ) dt
 *                          | |
 *                           -
 *                           0
 *
 * Three different rational approximations are employed, for
 * the intervals 0 to 3.25; 3.25 to 6.25; and 6.25 up.
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,10        10000       6.9e-16     1.0e-16
 *    DEC       0,10         6000       7.4e-17     1.4e-17
 *
 *
 */

 /*
   Cephes Math Library Release 2.2: June, 1992
   Copyright 1985, 1987, 1992 by Stephen L. Moshier
   Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */


// TODO: Add a cheaper approximation for float.


// We split this computation in to two so that in the scalar path
// only one branch is evaluated (due to our template specialization of pselect
// being an if statement.)

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_dawsn_interval_1(const T& x) {
  // Rational approximation on [0, 3.25)
  const ScalarType AN[] = {
    ScalarType(1.13681498971755972054E-11),
    ScalarType(8.49262267667473811108E-10),
    ScalarType(1.94434204175553054283E-8),
    ScalarType(9.53151741254484363489E-7),
    ScalarType(3.07828309874913200438E-6),
    ScalarType(3.52513368520288738649E-4),
    ScalarType(-8.50149846724410912031E-4),
    ScalarType(4.22618223005546594270E-2),
    ScalarType(-9.17480371773452345351E-2),
    ScalarType(9.99999999999999994612E-1),
  };
  const ScalarType AD[] = {
    ScalarType(2.40372073066762605484E-11),
    ScalarType(1.48864681368493396752E-9),
    ScalarType(5.21265281010541664570E-8),
    ScalarType(1.27258478273186970203E-6),
    ScalarType(2.32490249820789513991E-5),
    ScalarType(3.25524741826057911661E-4),
    ScalarType(3.48805814657162590916E-3),
    ScalarType(2.79448531198828973716E-2),
    ScalarType(1.58874241960120565368E-1),
    ScalarType(5.74918629489320327824E-1),
    ScalarType(1.00000000000000000539E0),
  };
  const T x2 = pmul(x, x);
  T y = pmul(x, pdiv(internal::ppolevl<T, 9>::run(x2, AN),
                      internal::ppolevl<T, 10>::run(x2, AD)));
  return y;
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_dawsn_interval_2(const T& x) {
  // Rational approximation on [3.25, 6.25)
  const ScalarType BN[] = {
    ScalarType(5.08955156417900903354E-1),
    ScalarType(-2.44754418142697847934E-1),
    ScalarType(9.41512335303534411857E-2),
    ScalarType(-2.18711255142039025206E-2),
    ScalarType(3.66207612329569181322E-3),
    ScalarType(-4.23209114460388756528E-4),
    ScalarType(3.59641304793896631888E-5),
    ScalarType(-2.14640351719968974225E-6),
    ScalarType(9.10010780076391431042E-8),
    ScalarType(-2.40274520828250956942E-9),
    ScalarType(3.59233385440928410398E-11),
  };
  const ScalarType BD[] = {
    ScalarType(1.0),
    ScalarType(-6.31839869873368190192E-1),
    ScalarType(2.36706788228248691528E-1),
    ScalarType(-5.31806367003223277662E-2),
    ScalarType(8.48041718586295374409E-3),
    ScalarType(-9.47996768486665330168E-4),
    ScalarType(7.81025592944552338085E-5),
    ScalarType(-4.55875153252442634831E-6),
    ScalarType(1.89100358111421846170E-7),
    ScalarType(-4.91324691331920606875E-9),
    ScalarType(7.18466403235734541950E-11),
  };
  const T one = pset1<T>(ScalarType(1));
  const T half = pset1<T>(ScalarType(0.5));

  const T inverse_x = pdiv(one, x);
  const T inverse_x2 = pmul(inverse_x, inverse_x);
  T z = pdiv(internal::ppolevl<T, 10>::run(inverse_x2, BN),
             pmul(x, internal::ppolevl<T, 10>::run(inverse_x2, BD)));
  T y = pmadd(inverse_x2, z, inverse_x);
  return pmul(half, y);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_dawsn_interval_3(const T& x) {
  // Rational approximation on [6.25, 1.0e9)
  const ScalarType CN[] = {
    ScalarType(-5.90592860534773254987E-1),
    ScalarType(6.29235242724368800674E-1),
    ScalarType(-1.72858975380388136411E-1),
    ScalarType(1.64837047825189632310E-2),
    ScalarType(-4.86827613020462700845E-4),
  };
  const ScalarType CD[] = {
    ScalarType(1.0),
    ScalarType(-2.69820057197544900361E0),
    ScalarType( 1.73270799045947845857E0),
    ScalarType(-3.93708582281939493482E-1),
    ScalarType( 3.44278924041233391079E-2),
    ScalarType(-9.73655226040941223894E-4),
  };
  const T one = pset1<T>(1);
  const T half = pset1<T>(0.5);

  const T inverse_x = pdiv(one, x);
  const T inverse_x2 = pmul(inverse_x, inverse_x);
  T z = pdiv(internal::ppolevl<T, 4>::run(inverse_x2, CN),
             pmul(x, internal::ppolevl<T, 5>::run(inverse_x2, CD)));
  T y = pmadd(inverse_x2, z, inverse_x);
  return pmul(half, y);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T generic_dawsn(const T& x) {
  const T half = pset1<T>(0.5);
  const T a = pset1<T>(3.25);
  const T b = pset1<T>(6.25);
  const T c = pset1<T>(1.0e9);

  T abs_x = pabs(x);

  T dawsn_lt_b = pselect(
      pcmp_lt(abs_x, a),
      generic_dawsn_interval_1<T, ScalarType>(abs_x),
      generic_dawsn_interval_2<T, ScalarType>(abs_x));

  T dawsn_gt_b = pselect(
      pcmp_lt(abs_x, c),
      generic_dawsn_interval_3<T, ScalarType>(abs_x),
      pdiv(half, x));

  T dawsn = pselect(pcmp_lt(abs_x, b), dawsn_lt_b, dawsn_gt_b);

  return pselect(pcmp_lt(x, pset1<T>(0.0)), pnegate(dawsn), dawsn);
}

template <typename Scalar>
struct dawsn_retval {
  typedef Scalar type;
};

#if !EIGEN_HAS_C99_MATH

template <typename Scalar>
struct dawsn_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

# else

template <typename Scalar>
struct dawsn_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar x) {
    return generic_dawsn<Scalar, Scalar>(x);
  }
};

#endif  // EIGEN_HAS_C99_MATH


/***************************************************************************
* Implementation of Exponential Integral.                                  *
****************************************************************************/

/*							ei.c
 *
 *	Exponential integral
 *
 *
 * SYNOPSIS:
 *
 * double x, y, ei();
 *
 * y = ei( x );
 *
 *
 *
 * DESCRIPTION:
 *
 *               x
 *                -     t
 *               | |   e
 *    Ei(x) =   -|-   ---  dt .
 *             | |     t
 *              -
 *             -inf
 *
 * Not defined for x <= 0.
 * See also expn.c.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE       0,100       50000      8.6e-16     1.3e-16
 *
 */
 /*
   Cephes Math Library Release 2.2: June, 1992
   Copyright 1985, 1987, 1992 by Stephen L. Moshier
   Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */


// TODO: Add a cheaper approximation for float.


// We split this computation in to two so that in the scalar path
// only one branch is evaluated (due to our template specialization of pselect
// being an if statement.)
//
template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_expi_interval_1(const T& x) {
  /* 0 < x <= 2
   Ei(x) - EUL - ln(x) = x A(x)/B(x)
   Theoretical peak relative error 9.73e-18  */
  const ScalarType A[] = {
    ScalarType(-5.350447357812542947283E0),
    ScalarType(2.185049168816613393830E2),
    ScalarType(-4.176572384826693777058E3),
    ScalarType(5.541176756393557601232E4),
    ScalarType(-3.313381331178144034309E5),
    ScalarType(1.592627163384945414220E6),
  };
  const ScalarType B[] = {
    ScalarType(1.0),
    ScalarType(-5.250547959112862969197E1),
    ScalarType(1.259616186786790571525E3),
    ScalarType(-1.756549581973534652631E4),
    ScalarType(1.493062117002725991967E5),
    ScalarType(-7.294949239640527645655E5),
    ScalarType(1.592627163384945429726E6),
  };

  // Euler gamma.
  const T EUL = pset1<T>(0.5772156649015329);

  const T f = pdiv(
      internal::ppolevl<T, 5>::run(x, A),
      internal::ppolevl<T, 6>::run(x, B));
  return pmadd(x, f, padd(EUL, plog(x)));
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_expi_interval_2(const T& x) {
  /* 2 <= x <= 4
   x exp(-x) Ei(x) - 1  =  1/x A6(1/x) / B6(1/x)
   Theoretical absolute error = 4.89e-17  */
  const ScalarType A6[] = {
    ScalarType(1.981808503259689673238E-2),
    ScalarType(-1.271645625984917501326E0),
    ScalarType(-2.088160335681228318920E0),
    ScalarType(2.755544509187936721172E0),
    ScalarType(-4.409507048701600257171E-1),
    ScalarType(4.665623805935891391017E-2),
    ScalarType(-1.545042679673485262580E-3),
    ScalarType(7.059980605299617478514E-5),
  };
  const ScalarType B6[] = {
    ScalarType(1.0),
    ScalarType(1.476498670914921440652E0),
    ScalarType(5.629177174822436244827E-1),
    ScalarType(1.699017897879307263248E-1),
    ScalarType(2.291647179034212017463E-2),
    ScalarType(4.450150439728752875043E-3),
    ScalarType(1.727439612206521482874E-4),
    ScalarType(3.953167195549672482304E-5),
  };

  const T one = pset1<T>(1.0);
  const T w = pdiv(one, x);
  T f = pdiv(
      internal::ppolevl<T, 7>::run(w, A6),
      internal::ppolevl<T, 7>::run(w, B6));
  f = pmadd(w, f, one);
  return pmul(pmul(pexp(x), w), f);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_expi_interval_3(const T& x) {
  /* 4 <= x <= 8
     x exp(-x) Ei(x) - 1  =  1/x A5(1/x) / B5(1/x)
     Theoretical absolute error = 2.20e-17  */
  const ScalarType A5[] = {
    ScalarType(-1.373215375871208729803E0),
    ScalarType(-7.084559133740838761406E-1),
    ScalarType( 1.580806855547941010501E0),
    ScalarType(-2.601500427425622944234E-1),
    ScalarType( 2.994674694113713763365E-2),
    ScalarType(-1.038086040188744005513E-3),
    ScalarType( 4.371064420753005429514E-5),
    ScalarType( 2.141783679522602903795E-6),
  };
  const ScalarType B5[] = {
    ScalarType(1.0),
    ScalarType(8.585231423622028380768E-1),
    ScalarType(4.483285822873995129957E-1),
    ScalarType(7.687932158124475434091E-2),
    ScalarType(2.449868241021887685904E-2),
    ScalarType(8.832165941927796567926E-4),
    ScalarType(4.590952299511353531215E-4),
    ScalarType(-4.729848351866523044863E-6),
    ScalarType(2.665195537390710170105E-6),
  };

  const T one = pset1<T>(1.0);
  const T w = pdiv(one, x);
  T f = pdiv(
      internal::ppolevl<T, 7>::run(w, A5),
      internal::ppolevl<T, 8>::run(w, B5));
  f = pmadd(w, f, one);
  return pmul(pmul(pexp(x), w), f);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_expi_interval_4(const T& x) {
  /* 8 <= x <= 16
   x exp(-x) Ei(x) - 1 = 1/x R(1/x)
   Theoretical peak absolute error = 1.07e-17  */
  const ScalarType A2[] = {
    ScalarType(-2.106934601691916512584E0),
    ScalarType(1.732733869664688041885E0),
    ScalarType(-2.423619178935841904839E-1),
    ScalarType(2.322724180937565842585E-2),
    ScalarType(2.372880440493179832059E-4),
    ScalarType(-8.343219561192552752335E-5),
    ScalarType(1.363408795605250394881E-5),
    ScalarType(-3.655412321999253963714E-7),
    ScalarType(1.464941733975961318456E-8),
    ScalarType(6.176407863710360207074E-10),
  };
  const ScalarType B2[] = {
    ScalarType(1.0),
    ScalarType(-2.298062239901678075778E-1),
    ScalarType(1.105077041474037862347E-1),
    ScalarType(-1.566542966630792353556E-2),
    ScalarType(2.761106850817352773874E-3),
    ScalarType(-2.089148012284048449115E-4),
    ScalarType(1.708528938807675304186E-5),
    ScalarType(-4.459311796356686423199E-7),
    ScalarType(1.394634930353847498145E-8),
    ScalarType(6.150865933977338354138E-10),
  };

  const T one = pset1<T>(1.0);
  const T w = pdiv(one, x);
  T f = pdiv(
      internal::ppolevl<T, 9>::run(w, A2),
      internal::ppolevl<T, 9>::run(w, B2));
  f = pmadd(w, f, one);
  return pmul(pmul(pexp(x), w), f);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_expi_interval_5(const T& x) {
  /* 16 <= x <= 32
   x exp(-x) Ei(x) - 1  =  1/x A4(1/x) / B4(1/x)
   Theoretical absolute error = 1.22e-17  */
  const ScalarType A4[] = {
    ScalarType(-2.458119367674020323359E-1),
    ScalarType(-1.483382253322077687183E-1),
    ScalarType( 7.248291795735551591813E-2),
    ScalarType(-1.348315687380940523823E-2),
    ScalarType( 1.342775069788636972294E-3),
    ScalarType(-7.942465637159712264564E-5),
    ScalarType( 2.644179518984235952241E-6),
    ScalarType(-4.239473659313765177195E-8),
  };
  const ScalarType B4[] = {
    ScalarType(1.0),
    ScalarType(-1.044225908443871106315E-1),
    ScalarType(-2.676453128101402655055E-1),
    ScalarType( 9.695000254621984627876E-2),
    ScalarType(-1.601745692712991078208E-2),
    ScalarType( 1.496414899205908021882E-3),
    ScalarType(-8.462452563778485013756E-5),
    ScalarType( 2.728938403476726394024E-6),
    ScalarType(-4.239462431819542051337E-8),
  };

  const T one = pset1<T>(1.0);
  const T w = pdiv(one, x);
  T f = pdiv(
      internal::ppolevl<T, 7>::run(w, A4),
      internal::ppolevl<T, 8>::run(w, B4));
  f = pmadd(w, f, one);
  return pmul(pmul(pexp(x), w), f);

}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_expi_interval_6(const T& x) {
  /* 32 <= x <= 64
   x exp(-x) Ei(x) - 1  =  1/x A7(1/x) / B7(1/x)
   Theoretical absolute error = 7.71e-18  */
  const ScalarType A7[] = {
    ScalarType(1.212561118105456670844E-1),
    ScalarType(-5.823133179043894485122E-1),
    ScalarType(2.348887314557016779211E-1),
    ScalarType(-3.040034318113248237280E-2),
    ScalarType(1.510082146865190661777E-3),
    ScalarType(-2.523137095499571377122E-5),
  };
  const ScalarType B7[] = {
    ScalarType(1.0),
    ScalarType(-1.002252150365854016662E0),
    ScalarType(2.928709694872224144953E-1),
    ScalarType(-3.337004338674007801307E-2),
    ScalarType(1.560544881127388842819E-3),
    ScalarType(-2.523137093603234562648E-5),
  };

  const T one = pset1<T>(1.0);
  const T w = pdiv(one, x);
  T f = pdiv(
      internal::ppolevl<T, 5>::run(w, A7),
      internal::ppolevl<T, 5>::run(w, B7));
  f = pmadd(w, f, one);
  return pmul(pmul(pexp(x), w), f);

}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_expi_interval_7(const T& x) {
  /* x > 64
   x exp(-x) Ei(x) - 1  =  1/x A3(1/x)/B3(1/x)
   Theoretical absolute error = 6.15e-17  */
  const ScalarType A3[] = {
    ScalarType(-7.657847078286127362028E-1),
    ScalarType(6.886192415566705051750E-1),
    ScalarType(-2.132598113545206124553E-1),
    ScalarType(3.346107552384193813594E-2),
    ScalarType(-3.076541477344756050249E-3),
    ScalarType(1.747119316454907477380E-4),
    ScalarType(-6.103711682274170530369E-6),
    ScalarType(1.218032765428652199087E-7),
    ScalarType(-1.086076102793290233007E-9),
  };
  const ScalarType B3[] = {
    ScalarType(1.0),
    ScalarType(-1.888802868662308731041E0),
    ScalarType(1.066691687211408896850E0),
    ScalarType(-2.751915982306380647738E-1),
    ScalarType(3.930852688233823569726E-2),
    ScalarType(-3.414684558602365085394E-3),
    ScalarType(1.866844370703555398195E-4),
    ScalarType(-6.345146083130515357861E-6),
    ScalarType(1.239754287483206878024E-7),
    ScalarType(-1.086076102793126632978E-9),
  };

  const T one = pset1<T>(1.0);
  const T w = pdiv(one, x);
  T f = pdiv(
      internal::ppolevl<T, 8>::run(w, A3),
      internal::ppolevl<T, 9>::run(w, B3));
  f = pmadd(w, f, one);
  return pmul(pmul(pexp(x), w), f);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_expi(const T& x) {

  const T two = pset1<T>(2.0);
  const T four = pset1<T>(4.0);
  const T eight = pset1<T>(8.0);
  const T sixteen = pset1<T>(16.0);
  const T thirty_two = pset1<T>(32.0);
  const T sixty_four = pset1<T>(64.0);
  const T nan = pset1<T>(NumTraits<ScalarType>::quiet_NaN());

  const T expi_lt_16 =
      pselect(
          pcmp_lt(x, four),
          pselect(
              pcmp_lt(x, two),
              generic_expi_interval_1<T, ScalarType>(x),
              generic_expi_interval_2<T, ScalarType>(x)),
          pselect(
              pcmp_lt(x, eight),
              generic_expi_interval_3<T, ScalarType>(x),
              generic_expi_interval_4<T, ScalarType>(x)));

  const T expi_gt_16 =
      pselect(
          pcmp_lt(x, thirty_two),
          generic_expi_interval_5<T, ScalarType>(x),
          pselect(
              pcmp_lt(x, sixty_four),
              generic_expi_interval_6<T, ScalarType>(x),
              generic_expi_interval_7<T, ScalarType>(x)));
  T expi = pselect(pcmp_lt(x, sixteen), expi_lt_16, expi_gt_16);
  return pselect(pcmp_lt(x, pset1<T>(0.)), nan, expi);
}


template <typename Scalar>
struct expi_retval {
  typedef Scalar type;
};

#if !EIGEN_HAS_C99_MATH

template <typename Scalar>
struct expi_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

# else

template <typename Scalar>
struct expi_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar x) {
    return generic_expi<Scalar, Scalar>(x);
  }
};

#endif  // EIGEN_HAS_C99_MATH

/***************************************************************************
* Implementation of Fresnel Integrals.                                     *
****************************************************************************/

/*							fresnl.c
 *
 *	Fresnel integral
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, S, C;
 * void fresnl();
 *
 * fresnl( x, _&S, _&C );
 *
 *
 * DESCRIPTION:
 *
 * Evaluates the Fresnel integrals
 *
 *           x
 *           -
 *          | |
 * C(x) =   |   cos(pi/2 t**2) dt,
 *        | |
 *         -
 *          0
 *
 *           x
 *           -
 *          | |
 * S(x) =   |   sin(pi/2 t**2) dt.
 *        | |
 *         -
 *          0
 *
 *
 * The integrals are evaluated by a power series for x < 1.
 * For x >= 1 auxiliary functions f(x) and g(x) are employed
 * such that
 *
 * C(x) = 0.5 + f(x) sin( pi/2 x**2 ) - g(x) cos( pi/2 x**2 )
 * S(x) = 0.5 - f(x) cos( pi/2 x**2 ) - g(x) sin( pi/2 x**2 )
 *
 *
 *
 * ACCURACY:
 *
 *  Relative error.
 *
 * Arithmetic  function   domain     # trials      peak         rms
 *   IEEE       S(x)      0, 10       10000       2.0e-15     3.2e-16
 *   IEEE       C(x)      0, 10       10000       1.8e-15     3.3e-16
 *   DEC        S(x)      0, 10        6000       2.2e-16     3.9e-17
 *   DEC        C(x)      0, 10        5000       2.3e-16     3.9e-17
 */
 /*
   Cephes Math Library Release 2.2: June, 1992
   Copyright 1985, 1987, 1992 by Stephen L. Moshier
   Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */


// TODO: Add a cheaper approximation for float.


// We split this computation in to two so that in the scalar path
// only one branch is evaluated (due to our template specialization of pselect
// being an if statement.)

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_fresnel_cos_interval_1(const T& x) {
  const ScalarType CN[] = {
    ScalarType(-4.98843114573573548651E-8),
    ScalarType(9.50428062829859605134E-6),
    ScalarType(-6.45191435683965050962E-4),
    ScalarType(1.88843319396703850064E-2),
    ScalarType(-2.05525900955013891793E-1),
    ScalarType(9.99999999999999998822E-1),
  };
  const ScalarType CD[] = {
    ScalarType(3.99982968972495980367E-12),
    ScalarType(9.15439215774657478799E-10),
    ScalarType(1.25001862479598821474E-7),
    ScalarType(1.22262789024179030997E-5),
    ScalarType(8.68029542941784300606E-4),
    ScalarType(4.12142090722199792936E-2),
    ScalarType(1.00000000000000000118E0),
  };

  const T x2 = pmul(x, x);
  const T x4 = pmul(x2, x2);
  return pmul(x, pdiv(
      internal::ppolevl<T, 5>::run(x4, CN),
      internal::ppolevl<T, 6>::run(x4, CD)));
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_fresnel_sin_interval_1(const T& x) {
  const ScalarType SN[] = {
    ScalarType(-2.99181919401019853726E3),
    ScalarType(7.08840045257738576863E5),
    ScalarType(-6.29741486205862506537E7),
    ScalarType(2.54890880573376359104E9),
    ScalarType(-4.42979518059697779103E10),
    ScalarType(3.18016297876567817986E11),
  };
  const ScalarType SD[] = {
    ScalarType(1.0),
    ScalarType(2.81376268889994315696E2),
    ScalarType(4.55847810806532581675E4),
    ScalarType(5.17343888770096400730E6),
    ScalarType(4.19320245898111231129E8),
    ScalarType(2.24411795645340920940E10),
    ScalarType(6.07366389490084639049E11),
  };

  const T x2 = pmul(x, x);
  const T x4 = pmul(x2, x2);
  T z = pmul(x, x2);
  return pmul(z, pdiv(
      internal::ppolevl<T, 5>::run(x4, SN),
      internal::ppolevl<T, 6>::run(x4, SD)));
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_fresnel_asymp(
    const T& x, bool use_sin) {
  const ScalarType FN[] = {
    ScalarType(4.21543555043677546506E-1),
    ScalarType(1.43407919780758885261E-1),
    ScalarType(1.15220955073585758835E-2),
    ScalarType(3.45017939782574027900E-4),
    ScalarType(4.63613749287867322088E-6),
    ScalarType(3.05568983790257605827E-8),
    ScalarType(1.02304514164907233465E-10),
    ScalarType(1.72010743268161828879E-13),
    ScalarType(1.34283276233062758925E-16),
    ScalarType(3.76329711269987889006E-20),
  };
  const ScalarType FD[] = {
    ScalarType(1.0),
    ScalarType(7.51586398353378947175E-1),
    ScalarType(1.16888925859191382142E-1),
    ScalarType(6.44051526508858611005E-3),
    ScalarType(1.55934409164153020873E-4),
    ScalarType(1.84627567348930545870E-6),
    ScalarType(1.12699224763999035261E-8),
    ScalarType(3.60140029589371370404E-11),
    ScalarType(5.88754533621578410010E-14),
    ScalarType(4.52001434074129701496E-17),
    ScalarType(1.25443237090011264384E-20),
  };
  const ScalarType GN[] = {
    ScalarType(5.04442073643383265887E-1),
    ScalarType(1.97102833525523411709E-1),
    ScalarType(1.87648584092575249293E-2),
    ScalarType(6.84079380915393090172E-4),
    ScalarType(1.15138826111884280931E-5),
    ScalarType(9.82852443688422223854E-8),
    ScalarType(4.45344415861750144738E-10),
    ScalarType(1.08268041139020870318E-12),
    ScalarType(1.37555460633261799868E-15),
    ScalarType(8.36354435630677421531E-19),
    ScalarType(1.86958710162783235106E-22),
  };
  const ScalarType GD[] = {
    ScalarType(1.0),
    ScalarType(1.47495759925128324529E0),
    ScalarType(3.37748989120019970451E-1),
    ScalarType(2.53603741420338795122E-2),
    ScalarType(8.14679107184306179049E-4),
    ScalarType(1.27545075667729118702E-5),
    ScalarType(1.04314589657571990585E-7),
    ScalarType(4.60680728146520428211E-10),
    ScalarType(1.10273215066240270757E-12),
    ScalarType(1.38796531259578871258E-15),
    ScalarType(8.39158816283118707363E-19),
    ScalarType(1.86958710162783236342E-22),
  };

  const T HALF_PI = pset1<T>(1.5707963267948966);
  const T PI = pset1<T>(EIGEN_PI);
  const T one = pset1<T>(1);
  const T half = pset1<T>(0.5);

  const T x2 = pmul(x, x);
  const T t = pdiv(one, pmul(PI, x2));
  const T u = pmul(t, t);

  T f = pmadd(pnegate(u), pdiv(
      internal::ppolevl<T, 9>::run(u, FN),
      internal::ppolevl<T, 10>::run(u, FD)), one);
  T g = pmul(t, pdiv(
      internal::ppolevl<T, 10>::run(u, GN),
      internal::ppolevl<T, 11>::run(u, GD)));

  const T z = pmul(HALF_PI, x2);
  const T c = pcos(z);
  const T s = psin(z);
  const T y = pdiv(one, pmul(PI, x));
  if (use_sin) {
    T intermediate = pmul(f, c);
    intermediate = pmadd(g, s, intermediate);
    return pmadd(pnegate(intermediate), y, half);
  }
  T intermediate = pmul(f, s);
  intermediate = pmadd(pnegate(g), c, intermediate);
  return pmadd(intermediate, y, half);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T generic_fresnel_cos(const T& x) {

  const T half = pset1<T>(0.5);
  const T a = pset1<T>(2.5625);
  const T b = pset1<T>(36974.0);

  const T abs_x = pabs(x);
  const T x2 = pmul(x, x);

  T fresnel_cos = pselect(
      pcmp_lt(x2, a),
      generic_fresnel_cos_interval_1<T, ScalarType>(abs_x),
      generic_fresnel_asymp<T, ScalarType>(abs_x, false));

  fresnel_cos = pselect(pcmp_lt(abs_x, b), fresnel_cos, half);

  return pselect(pcmp_lt(x, pset1<T>(0.0)), pnegate(fresnel_cos), fresnel_cos);
}

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T generic_fresnel_sin(const T& x) {

  const T half = pset1<T>(0.5);
  const T a = pset1<T>(2.5625);
  const T b = pset1<T>(36974.0);

  const T abs_x = pabs(x);
  const T x2 = pmul(x, x);

  T fresnel_sin = pselect(
      pcmp_lt(x2, a),
      generic_fresnel_sin_interval_1<T, ScalarType>(abs_x),
      generic_fresnel_asymp<T, ScalarType>(abs_x, true));

  fresnel_sin = pselect(pcmp_lt(x, b), fresnel_sin, half);

  return pselect(pcmp_lt(x, pset1<T>(0.0)), pnegate(fresnel_sin), fresnel_sin);
}

template <typename Scalar>
struct fresnel_cos_retval {
  typedef Scalar type;
};

template <typename Scalar>
struct fresnel_sin_retval {
  typedef Scalar type;
};

#if !EIGEN_HAS_C99_MATH

template <typename Scalar>
struct fresnel_cos_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template <typename Scalar>
struct fresnel_sin_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};


# else

template <typename Scalar>
struct fresnel_cos_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar x) {
    return generic_fresnel_cos<Scalar, Scalar>(x);
  }
};

template <typename Scalar>
struct fresnel_sin_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar x) {
    return generic_fresnel_sin<Scalar, Scalar>(x);
  }
};

#endif  // EIGEN_HAS_C99_MATH


/***************************************************************************
* Implementation of Spence's Integral.                                     *
****************************************************************************/

/*							spence.c
 *
 *	Dilogarithm
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, spence();
 *
 * y = spence( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Computes the integral
 *
 *                    x
 *                    -
 *                   | | log t
 * spence(x)  =  -   |   ----- dt
 *                 | |   t - 1
 *                  -
 *                  1
 *
 * for x >= 0.  A rational approximation gives the integral in
 * the interval (0.5, 1.5).  Transformation formulas for 1/x
 * and 1-x are employed outside the basic expansion range.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,4         30000       3.9e-15     5.4e-16
 *    DEC       0,4          3000       2.5e-16     4.5e-17
 *
 *
 */
 /*
   Cephes Math Library Release 2.2: June, 1992
   Copyright 1985, 1987, 1992 by Stephen L. Moshier
   Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */


// TODO: Add a cheaper approximation for float.


// We split this computation in to two so that in the scalar path
// only one branch is evaluated (due to our template specialization of pselect
// being an if statement.)

template <typename T, typename ScalarType>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T generic_spence(const T& x) {
  const ScalarType A[] = {
    ScalarType(4.65128586073990045278E-5),
    ScalarType(7.31589045238094711071E-3),
    ScalarType(1.33847639578309018650E-1),
    ScalarType(8.79691311754530315341E-1),
    ScalarType(2.71149851196553469920E0),
    ScalarType(4.25697156008121755724E0),
    ScalarType(3.29771340985225106936E0),
    ScalarType(1.00000000000000000126E0),
  };
  const ScalarType B[] = {
    ScalarType(6.90990488912553276999E-4),
    ScalarType(2.54043763932544379113E-2),
    ScalarType(2.82974860602568089943E-1),
    ScalarType(1.41172597751831069617E0),
    ScalarType(3.63800533345137075418E0),
    ScalarType(5.03278880143316990390E0),
    ScalarType(3.54771340985225096217E0),
    ScalarType(9.99999999999999998740E-1),
  };
  const T zero = pset1<T>(0.0);
  const T one = pset1<T>(1.0);
  const T three_halves = pset1<T>(1.5);
  const T half = pset1<T>(0.5);
  const T nan = pset1<T>(NumTraits<ScalarType>::quiet_NaN());
  // pi**2 / 6.
  const T PI2O6 = pset1<T>(EIGEN_PI * EIGEN_PI / 6.0);
  T y = pselect(pcmp_lt(x, pset1<T>(2.0)), x, pdiv(one, x));
  T w = pselect(
      pcmp_lt(three_halves, y),
      psub(pdiv(one, y), one),
      pselect(
          pcmp_lt(y, half),
          pnegate(y),
          psub(y, one)));
  T spence = pmul(pnegate(w), pdiv(
      internal::ppolevl<T, 7>::run(w, A),
      internal::ppolevl<T, 7>::run(w, B)));
  spence = pselect(
      pcmp_lt(y, half),
      pmadd(pnegate(plog(y)), plog1p(-y), psub(PI2O6, spence)),
      spence);
  T z = plog(y);
  spence = pselect(
      pcmp_lt(three_halves, x),
      pmadd(pnegate(pmul(half, z)), z, pnegate(spence)),
      spence);

  spence = pselect(pcmp_eq(x, zero), PI2O6, spence);
  spence = pselect(pcmp_eq(x, one), zero, spence);
  spence = pselect(pcmp_lt(x, zero), nan, spence);
  return spence;
}

template <typename Scalar>
struct spence_retval {
  typedef Scalar type;
};

#if !EIGEN_HAS_C99_MATH

template <typename Scalar>
struct spence_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar) {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

# else

template <typename Scalar>
struct spence_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar x) {
    return generic_spence<Scalar, Scalar>(x);
  }
};

#endif  // EIGEN_HAS_C99_MATH



}  // end namespace internal

namespace numext {

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(lgamma, Scalar)
    lgamma(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(lgamma, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(digamma, Scalar)
    digamma(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(digamma, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(zeta, Scalar)
zeta(const Scalar& x, const Scalar& q) {
    return EIGEN_MATHFUNC_IMPL(zeta, Scalar)::run(x, q);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(polygamma, Scalar)
polygamma(const Scalar& n, const Scalar& x) {
    return EIGEN_MATHFUNC_IMPL(polygamma, Scalar)::run(n, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(erf, Scalar)
    erf(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(erf, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(erfc, Scalar)
    erfc(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(erfc, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(ndtri, Scalar)
    ndtri(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(ndtri, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(igamma, Scalar)
    igamma(const Scalar& a, const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(igamma, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(igamma_der_a, Scalar)
    igamma_der_a(const Scalar& a, const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(igamma_der_a, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(gamma_sample_der_alpha, Scalar)
    gamma_sample_der_alpha(const Scalar& a, const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(gamma_sample_der_alpha, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(igammac, Scalar)
    igammac(const Scalar& a, const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(igammac, Scalar)::run(a, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(betainc, Scalar)
    betainc(const Scalar& a, const Scalar& b, const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(betainc, Scalar)::run(a, b, x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(dawsn, Scalar)
    dawsn(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(dawsn, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(expi, Scalar)
    expi(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(expi, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(fresnel_cos, Scalar)
    fresnel_cos(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(fresnel_cos, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(fresnel_sin, Scalar)
    fresnel_sin(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(fresnel_sin, Scalar)::run(x);
}

template <typename Scalar>
EIGEN_DEVICE_FUNC inline EIGEN_MATHFUNC_RETVAL(spence, Scalar)
    spence(const Scalar& x) {
  return EIGEN_MATHFUNC_IMPL(spence, Scalar)::run(x);
}

}  // end namespace numext
}  // end namespace Eigen

#endif  // EIGEN_SPECIAL_FUNCTIONS_H
