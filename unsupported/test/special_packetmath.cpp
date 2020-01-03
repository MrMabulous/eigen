// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include "../Eigen/SpecialFunctions"
#include <typeinfo>

#if defined __GNUC__ && __GNUC__>=6
  #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
// using namespace Eigen;

#ifdef EIGEN_VECTORIZE_SSE
const bool g_vectorize_sse = true;
#else
const bool g_vectorize_sse = false;
#endif

bool g_first_pass = true;

namespace Eigen {
namespace internal {

template<typename T>
Map<const Array<unsigned char,sizeof(T),1> >
bits(const T& x) {
  return Map<const Array<unsigned char,sizeof(T),1> >(reinterpret_cast<const unsigned char *>(&x));
}

template<typename T>
bool biteq(T a, T b) {
  return (bits(a) == bits(b)).all();
}

}
}

// NOTE: we disable inlining for this function to workaround a GCC issue when using -O3 and the i387 FPU.
template<typename Scalar> EIGEN_DONT_INLINE
bool isApproxAbs(const Scalar& a, const Scalar& b, const typename NumTraits<Scalar>::Real& refvalue)
{
  return internal::isMuchSmallerThan(a-b, refvalue);
}

template<typename Scalar> bool areApproxAbs(const Scalar* a, const Scalar* b, int size, const typename NumTraits<Scalar>::Real& refvalue)
{
  for (int i=0; i<size; ++i)
  {
    if (!isApproxAbs(a[i],b[i],refvalue))
    {
      std::cout << "ref: [" << Map<const Matrix<Scalar,1,Dynamic> >(a,size) << "]" << " != vec: [" << Map<const Matrix<Scalar,1,Dynamic> >(b,size) << "]\n";
      return false;
    }
  }
  return true;
}

template<typename Scalar> bool areApprox(const Scalar* a, const Scalar* b, int size)
{
  for (int i=0; i<size; ++i)
  {
    if ((!internal::biteq(a[i],b[i])) && a[i]!=b[i] && !internal::isApprox(a[i],b[i]))
    {
      std::cout << "ref: [" << Map<const Matrix<Scalar,1,Dynamic> >(a,size) << "]" << " != vec: [" << Map<const Matrix<Scalar,1,Dynamic> >(b,size) << "]\n";
      return false;
    }
  }
  return true;
}

template<bool Cond,typename Packet>
struct packet_helper
{
  template<typename T>
  inline Packet load(const T* from) const { return internal::pload<Packet>(from); }

  template<typename T>
  inline Packet loadu(const T* from) const { return internal::ploadu<Packet>(from); }

  template<typename T>
  inline Packet load(const T* from, unsigned long long umask) const { return internal::ploadu<Packet>(from, umask); }

  template<typename T>
  inline void store(T* to, const Packet& x) const { internal::pstore(to,x); }

  template<typename T>
  inline void store(T* to, const Packet& x, unsigned long long umask) const { internal::pstoreu(to, x, umask); }
};

template<typename Packet>
struct packet_helper<false,Packet>
{
  template<typename T>
  inline T load(const T* from) const { return *from; }

  template<typename T>
  inline T loadu(const T* from) const { return *from; }

  template<typename T>
  inline T load(const T* from, unsigned long long) const { return *from; }

  template<typename T>
  inline void store(T* to, const T& x) const { *to = x; }

  template<typename T>
  inline void store(T* to, const T& x, unsigned long long) const { *to = x; }
};

#define CHECK_CWISE1_IF(COND, REFOP, POP) if(COND) { \
  packet_helper<COND,Packet> h; \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i]); \
  h.store(data2, POP(h.load(data1))); \
  VERIFY(areApprox(ref, data2, PacketSize) && #POP); \
}

#define CHECK_CWISE2_IF(COND, REFOP, POP) if(COND) { \
  packet_helper<COND,Packet> h; \
  for (int i=0; i<PacketSize; ++i) \
    ref[i] = REFOP(data1[i], data1[i+PacketSize]); \
  h.store(data2, POP(h.load(data1),h.load(data1+PacketSize))); \
  VERIFY(areApprox(ref, data2, PacketSize) && #POP); \
}


template<typename Scalar,typename Packet> void packetmath_real()
{
  using std::abs;
  typedef internal::packet_traits<Scalar> PacketTraits;
  const int PacketSize = internal::unpacket_traits<Packet>::size;

  const int size = PacketSize*4;
  EIGEN_ALIGN_MAX Scalar data1[PacketSize*4];
  EIGEN_ALIGN_MAX Scalar data2[PacketSize*4];
  EIGEN_ALIGN_MAX Scalar ref[PacketSize*4];

#if EIGEN_HAS_C99_MATH
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    packet_helper<internal::packet_traits<Scalar>::HasLGamma,Packet> h;
    h.store(data2, internal::plgamma(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
  if (internal::packet_traits<Scalar>::HasErf) {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    packet_helper<internal::packet_traits<Scalar>::HasErf,Packet> h;
    h.store(data2, internal::perf(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
  {
    data1[0] = std::numeric_limits<Scalar>::quiet_NaN();
    packet_helper<internal::packet_traits<Scalar>::HasErfc,Packet> h;
    h.store(data2, internal::perfc(h.load(data1)));
    VERIFY((numext::isnan)(data2[0]));
  }
  {
    for (int i=0; i<size; ++i) {
      data1[i] = internal::random<Scalar>(0,1);
    }
    CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasNdtri, numext::ndtri, internal::pndtri);
  }
#endif  // EIGEN_HAS_C99_MATH

  // For bessel_i*e and bessel_j*, the valid range is negative reals.
  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(-1,1) * std::pow(Scalar(10), internal::random<Scalar>(-6,6));
    data2[i] = internal::random<Scalar>(-1,1) * std::pow(Scalar(10), internal::random<Scalar>(-6,6));
  }

  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_i0e, internal::pbessel_i0e);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_i1e, internal::pbessel_i1e);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_j0, internal::pbessel_j0);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_j1, internal::pbessel_j1);

  // Use a smaller data range for the bessel_i* as these can become very large.
  // Following #1693, we also restrict this range further to avoid inf's due to
  // differences in pexp and exp.
  for (int i=0; i<size; ++i) {
      data1[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(9), internal::random<Scalar>(-1,2));
      data2[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(9), internal::random<Scalar>(-1,2));
  }
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_i0, internal::pbessel_i0);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_i1, internal::pbessel_i1);


  // y_i, and k_i are valid for x > 0.
  for (int i=0; i<size; ++i)
  {
    data1[i] = internal::random<Scalar>(0.01,1) * std::pow(Scalar(10), internal::random<Scalar>(-2,5));
    data2[i] = internal::random<Scalar>(0.01,1) * std::pow(Scalar(10), internal::random<Scalar>(-2,5));
  }

  // TODO(srvasude): Re-enable this test once properly investigated why the
  // scalar and vector paths differ.
  // CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_y0, internal::pbessel_y0);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_y1, internal::pbessel_y1);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_k0e, internal::pbessel_k0e);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_k1e, internal::pbessel_k1e);

  // Following #1693, we restrict the range for exp to avoid zeroing out too
  // fast.
  for (int i=0; i<size; ++i) {
      data1[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(9), internal::random<Scalar>(-1,2));
      data2[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(9), internal::random<Scalar>(-1,2));
  }
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_k0, internal::pbessel_k0);
  CHECK_CWISE1_IF(PacketTraits::HasBessel, numext::bessel_k1, internal::pbessel_k1);


  for (int i=0; i<size; ++i) {
      data1[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(10), internal::random<Scalar>(-1,2));
      data2[i] = internal::random<Scalar>(0.01,1) * std::pow(
          Scalar(10), internal::random<Scalar>(-1,2));
  }

#if EIGEN_HAS_C99_MATH && (__cplusplus > 199711L)
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasLGamma, std::lgamma, internal::plgamma);
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasErf, std::erf, internal::perf);
  CHECK_CWISE1_IF(internal::packet_traits<Scalar>::HasErfc, std::erfc, internal::perfc);
#endif

}

template<typename Scalar, typename PacketType >
struct runall {
  static void run() {
    packetmath_real<Scalar,PacketType>();
  }
};

template<
  typename Scalar,
  typename PacketType = typename internal::packet_traits<Scalar>::type,
  bool Vectorized = internal::packet_traits<Scalar>::Vectorizable,
  bool HasHalf = !internal::is_same<typename internal::unpacket_traits<PacketType>::half,PacketType>::value >
struct runner;

template<typename Scalar,typename PacketType>
struct runner<Scalar,PacketType,true,true>
{
  static void run() {
    runall<Scalar,PacketType>::run();
    runner<Scalar,typename internal::unpacket_traits<PacketType>::half>::run();
  }
};

template<typename Scalar,typename PacketType>
struct runner<Scalar,PacketType,true,false>
{
  static void run() {
    runall<Scalar,PacketType>::run();
    runall<Scalar,Scalar>::run();
  }
};

template<typename Scalar,typename PacketType>
struct runner<Scalar,PacketType,false,false>
{
  static void run() {
    runall<Scalar,PacketType>::run();
  }
};

EIGEN_DECLARE_TEST(packetmath)
{
  g_first_pass = true;
  for(int i = 0; i < g_repeat; i++) {

    CALL_SUBTEST_1( runner<float>::run() );
    CALL_SUBTEST_2( runner<double>::run() );
    g_first_pass = false;
  }
}
