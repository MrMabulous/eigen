// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_GPU_H
#define EIGEN_PACKET_MATH_GPU_H

namespace Eigen {

namespace internal {

// Make sure this is only available when targeting a GPU: we don't want to
// introduce conflicts between these packet_traits definitions and the ones
// we'll use on the host side (SSE, AVX, ...)
#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
template<> struct is_arithmetic<float4>  { enum { value = true }; };
template<> struct is_arithmetic<double2> { enum { value = true }; };

template<> struct packet_traits<float> : default_packet_traits
{
  typedef float4 type;
  typedef float4 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=4,
    HasHalfPacket = 0,

    HasDiv  = 1,
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasLGamma = 1,
    HasDiGamma = 1,
    HasZeta = 1,
    HasPolygamma = 1,
    HasErf = 1,
    HasErfc = 1,
    HasNdtri = 1,
    HasBessel = 1,
    HasIGamma = 1,
    HasIGammaDerA = 1,
    HasGammaSampleDerAlpha = 1,
    HasIGammac = 1,
    HasBetaInc = 1,

    HasBlend = 0,
    HasFloor = 1,
  };
};

template<> struct packet_traits<double> : default_packet_traits
{
  typedef double2 type;
  typedef double2 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=2,
    HasHalfPacket = 0,

    HasDiv  = 1,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 1,
    HasRsqrt = 1,
    HasLGamma = 1,
    HasDiGamma = 1,
    HasZeta = 1,
    HasPolygamma = 1,
    HasErf = 1,
    HasErfc = 1,
    HasNdtri = 1,
    HasBessel = 1,
    HasIGamma = 1,
    HasIGammaDerA = 1,
    HasGammaSampleDerAlpha = 1,
    HasIGammac = 1,
    HasBetaInc = 1,

    HasBlend = 0,
    HasFloor = 1,
  };
};


template<> struct unpacket_traits<float4>  { typedef float  type; enum {size=4, alignment=Aligned16, vectorizable=true, masked_load_available=false, masked_store_available=false}; typedef float4 half; };
template<> struct unpacket_traits<double2> { typedef double type; enum {size=2, alignment=Aligned16, vectorizable=true, masked_load_available=false, masked_store_available=false}; typedef double2 half; };

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pset1<float4>(const float&  from) {
  return make_float4(from, from, from, from);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pset1<double2>(const double& from) {
  return make_double2(from, from);
}

// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
// invoked by NVCC’ (e.g. on MacOS). The former needs to see both host and device implementation
// of the functions, while the latter can only deal with one of them.
#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
namespace {

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float bitwise_and(const float& a,
                                                        const float& b) {
  return __int_as_float(__float_as_int(a) & __float_as_int(b));
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double bitwise_and(const double& a,
                                                         const double& b) {
  return __longlong_as_double(__double_as_longlong(a) &
                              __double_as_longlong(b));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float bitwise_or(const float& a,
                                                       const float& b) {
  return __int_as_float(__float_as_int(a) | __float_as_int(b));
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double bitwise_or(const double& a,
                                                        const double& b) {
  return __longlong_as_double(__double_as_longlong(a) |
                              __double_as_longlong(b));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float bitwise_xor(const float& a,
                                                        const float& b) {
  return __int_as_float(__float_as_int(a) ^ __float_as_int(b));
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double bitwise_xor(const double& a,
                                                         const double& b) {
  return __longlong_as_double(__double_as_longlong(a) ^
                              __double_as_longlong(b));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float bitwise_andnot(const float& a,
                                                           const float& b) {
  return __int_as_float(__float_as_int(a) & ~__float_as_int(b));
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double bitwise_andnot(const double& a,
                                                            const double& b) {
  return __longlong_as_double(__double_as_longlong(a) &
                              ~__double_as_longlong(b));
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float eq_mask(const float& a,
                                                    const float& b) {
  return __int_as_float(a == b ? 0xffffffffu : 0u);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double eq_mask(const double& a,
                                                     const double& b) {
  return __longlong_as_double(a == b ? 0xffffffffffffffffull : 0ull);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float lt_mask(const float& a,
                                                    const float& b) {
  return __int_as_float(a < b ? 0xffffffffu : 0u);
}
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double lt_mask(const double& a,
                                                     const double& b) {
  return __longlong_as_double(a < b ? 0xffffffffffffffffull : 0ull);
}

}  // namespace

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pand<float4>(const float4& a,
                                                          const float4& b) {
  return make_float4(bitwise_and(a.x, b.x), bitwise_and(a.y, b.y),
                     bitwise_and(a.z, b.z), bitwise_and(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pand<double2>(const double2& a,
                                                            const double2& b) {
  return make_double2(bitwise_and(a.x, b.x), bitwise_and(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 por<float4>(const float4& a,
                                                         const float4& b) {
  return make_float4(bitwise_or(a.x, b.x), bitwise_or(a.y, b.y),
                     bitwise_or(a.z, b.z), bitwise_or(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 por<double2>(const double2& a,
                                                           const double2& b) {
  return make_double2(bitwise_or(a.x, b.x), bitwise_or(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pxor<float4>(const float4& a,
                                                          const float4& b) {
  return make_float4(bitwise_xor(a.x, b.x), bitwise_xor(a.y, b.y),
                     bitwise_xor(a.z, b.z), bitwise_xor(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pxor<double2>(const double2& a,
                                                            const double2& b) {
  return make_double2(bitwise_xor(a.x, b.x), bitwise_xor(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pandnot<float4>(const float4& a,
                                                             const float4& b) {
  return make_float4(bitwise_andnot(a.x, b.x), bitwise_andnot(a.y, b.y),
                     bitwise_andnot(a.z, b.z), bitwise_andnot(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2
pandnot<double2>(const double2& a, const double2& b) {
  return make_double2(bitwise_andnot(a.x, b.x), bitwise_andnot(a.y, b.y));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcmp_eq<float4>(const float4& a,
                                                             const float4& b) {
  return make_float4(eq_mask(a.x, b.x), eq_mask(a.y, b.y), eq_mask(a.z, b.z),
                     eq_mask(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pcmp_lt<float4>(const float4& a,
                                                             const float4& b) {
  return make_float4(lt_mask(a.x, b.x), lt_mask(a.y, b.y), lt_mask(a.z, b.z),
                     lt_mask(a.w, b.w));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2
pcmp_eq<double2>(const double2& a, const double2& b) {
  return make_double2(eq_mask(a.x, b.x), eq_mask(a.y, b.y));
}
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2
pcmp_lt<double2>(const double2& a, const double2& b) {
  return make_double2(lt_mask(a.x, b.x), lt_mask(a.y, b.y));
}
#endif  // EIGEN_CUDA_ARCH || defined(EIGEN_HIP_DEVICE_COMPILE)

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 plset<float4>(const float& a) {
  return make_float4(a, a+1, a+2, a+3);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 plset<double2>(const double& a) {
  return make_double2(a, a+1);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 padd<float4>(const float4& a, const float4& b) {
  return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 padd<double2>(const double2& a, const double2& b) {
  return make_double2(a.x+b.x, a.y+b.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 psub<float4>(const float4& a, const float4& b) {
  return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 psub<double2>(const double2& a, const double2& b) {
  return make_double2(a.x-b.x, a.y-b.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pnegate(const float4& a) {
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pnegate(const double2& a) {
  return make_double2(-a.x, -a.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pconj(const float4& a) { return a; }
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pconj(const double2& a) { return a; }

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmul<float4>(const float4& a, const float4& b) {
  return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmul<double2>(const double2& a, const double2& b) {
  return make_double2(a.x*b.x, a.y*b.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pdiv<float4>(const float4& a, const float4& b) {
  return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pdiv<double2>(const double2& a, const double2& b) {
  return make_double2(a.x/b.x, a.y/b.y);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmin<float4>(const float4& a, const float4& b) {
  return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmin<double2>(const double2& a, const double2& b) {
  return make_double2(fmin(a.x, b.x), fmin(a.y, b.y));
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pmax<float4>(const float4& a, const float4& b) {
  return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pmax<double2>(const double2& a, const double2& b) {
  return make_double2(fmax(a.x, b.x), fmax(a.y, b.y));
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 pload<float4>(const float* from) {
  return *reinterpret_cast<const float4*>(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 pload<double2>(const double* from) {
  return *reinterpret_cast<const double2*>(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 ploadu<float4>(const float* from) {
  return make_float4(from[0], from[1], from[2], from[3]);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 ploadu<double2>(const double* from) {
  return make_double2(from[0], from[1]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float4 ploaddup<float4>(const float*   from) {
  return make_float4(from[0], from[0], from[1], from[1]);
}
template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double2 ploaddup<double2>(const double*  from) {
  return make_double2(from[0], from[0]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<float>(float*   to, const float4& from) {
  *reinterpret_cast<float4*>(to) = from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<double>(double* to, const double2& from) {
  *reinterpret_cast<double2*>(to) = from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<float>(float*  to, const float4& from) {
  to[0] = from.x;
  to[1] = from.y;
  to[2] = from.z;
  to[3] = from.w;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const double2& from) {
  to[0] = from.x;
  to[1] = from.y;
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float4 ploadt_ro<float4, Aligned>(const float* from) {
#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
  return __ldg((const float4*)from);
#else
  return make_float4(from[0], from[1], from[2], from[3]);
#endif
}
template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double2 ploadt_ro<double2, Aligned>(const double* from) {
#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
  return __ldg((const double2*)from);
#else
  return make_double2(from[0], from[1]);
#endif
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE float4 ploadt_ro<float4, Unaligned>(const float* from) {
#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
  return make_float4(__ldg(from+0), __ldg(from+1), __ldg(from+2), __ldg(from+3));
#else
  return make_float4(from[0], from[1], from[2], from[3]);
#endif
}
template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE double2 ploadt_ro<double2, Unaligned>(const double* from) {
#if defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350
  return make_double2(__ldg(from+0), __ldg(from+1));
#else
  return make_double2(from[0], from[1]);
#endif
}

template<> EIGEN_DEVICE_FUNC inline float4 pgather<float, float4>(const float* from, Index stride) {
  return make_float4(from[0*stride], from[1*stride], from[2*stride], from[3*stride]);
}

template<> EIGEN_DEVICE_FUNC inline double2 pgather<double, double2>(const double* from, Index stride) {
  return make_double2(from[0*stride], from[1*stride]);
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<float, float4>(float* to, const float4& from, Index stride) {
  to[stride*0] = from.x;
  to[stride*1] = from.y;
  to[stride*2] = from.z;
  to[stride*3] = from.w;
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<double, double2>(double* to, const double2& from, Index stride) {
  to[stride*0] = from.x;
  to[stride*1] = from.y;
}

template<> EIGEN_DEVICE_FUNC inline float  pfirst<float4>(const float4& a) {
  return a.x;
}
template<> EIGEN_DEVICE_FUNC inline double pfirst<double2>(const double2& a) {
  return a.x;
}

template<> EIGEN_DEVICE_FUNC inline float  predux<float4>(const float4& a) {
  return a.x + a.y + a.z + a.w;
}
template<> EIGEN_DEVICE_FUNC inline double predux<double2>(const double2& a) {
  return a.x + a.y;
}

template<> EIGEN_DEVICE_FUNC inline float  predux_max<float4>(const float4& a) {
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}
template<> EIGEN_DEVICE_FUNC inline double predux_max<double2>(const double2& a) {
  return fmax(a.x, a.y);
}

template<> EIGEN_DEVICE_FUNC inline float  predux_min<float4>(const float4& a) {
  return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}
template<> EIGEN_DEVICE_FUNC inline double predux_min<double2>(const double2& a) {
  return fmin(a.x, a.y);
}

template<> EIGEN_DEVICE_FUNC inline float  predux_mul<float4>(const float4& a) {
  return a.x * a.y * a.z * a.w;
}
template<> EIGEN_DEVICE_FUNC inline double predux_mul<double2>(const double2& a) {
  return a.x * a.y;
}

template<> EIGEN_DEVICE_FUNC inline float4  pabs<float4>(const float4& a) {
  return make_float4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w));
}
template<> EIGEN_DEVICE_FUNC inline double2 pabs<double2>(const double2& a) {
  return make_double2(fabs(a.x), fabs(a.y));
}

template<> EIGEN_DEVICE_FUNC inline float4  pfloor<float4>(const float4& a) {
  return make_float4(floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w));
}
template<> EIGEN_DEVICE_FUNC inline double2 pfloor<double2>(const double2& a) {
  return make_double2(floor(a.x), floor(a.y));
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<float4,4>& kernel) {
  float tmp = kernel.packet[0].y;
  kernel.packet[0].y = kernel.packet[1].x;
  kernel.packet[1].x = tmp;

  tmp = kernel.packet[0].z;
  kernel.packet[0].z = kernel.packet[2].x;
  kernel.packet[2].x = tmp;

  tmp = kernel.packet[0].w;
  kernel.packet[0].w = kernel.packet[3].x;
  kernel.packet[3].x = tmp;

  tmp = kernel.packet[1].z;
  kernel.packet[1].z = kernel.packet[2].y;
  kernel.packet[2].y = tmp;

  tmp = kernel.packet[1].w;
  kernel.packet[1].w = kernel.packet[3].y;
  kernel.packet[3].y = tmp;

  tmp = kernel.packet[2].w;
  kernel.packet[2].w = kernel.packet[3].z;
  kernel.packet[3].z = tmp;
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<double2,2>& kernel) {
  double tmp = kernel.packet[0].y;
  kernel.packet[0].y = kernel.packet[1].x;
  kernel.packet[1].x = tmp;
}

#endif

// Packet math for Eigen::half
// Most of the following operations require arch >= 3.0
#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDACC) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
  (defined(EIGEN_HAS_HIP_FP16) && defined(EIGEN_HIPCC) && defined(EIGEN_HIP_DEVICE_COMPILE)) || \
  (defined(EIGEN_HAS_CUDA_FP16) && defined(__clang__) && defined(__CUDA__))
typedef ulonglong2 Packet4h2;
template<> struct unpacket_traits<half2> { typedef Eigen::half type; enum {size=2, alignment=Aligned16, vectorizable=true, masked_load_available=false, masked_store_available=false}; typedef half2 half; };
template<> struct is_arithmetic<half2> { enum { value = true }; };
template<> struct unpacket_traits<Packet4h2> { typedef Eigen::half type; enum {size=8, alignment=Aligned16, vectorizable=true, masked_load_available=false, masked_store_available=false}; typedef Packet4h2 half; };
template<> struct is_arithmetic<Packet4h2> { enum { value = true }; };
#if !defined(EIGEN_WIDE_FP16)
template<> struct packet_traits<Eigen::half> : default_packet_traits
{
  typedef half2 type;
  typedef half2 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=2,
    HasHalfPacket = 0,
    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasSqrt   = 1,
    HasRsqrt  = 1,
    HasExp    = 1,
    HasExpm1  = 1,
    HasLog    = 1,
    HasLog1p  = 1
  };
};

#else

template<> struct packet_traits<Eigen::half> : default_packet_traits
{
  typedef Packet4h2 type;
  typedef Packet4h2 half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=8,
    HasHalfPacket = 0,
    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasSqrt   = 1,
    HasRsqrt  = 1,
    HasExp    = 1,
    HasExpm1  = 1,
    HasLog    = 1,
    HasLog1p  = 1
  };
};

#endif


template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pset1<half2>(const Eigen::half& from) {
#if !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_HIP_DEVICE_COMPILE)
  half2 r;
  r.x = from;
  r.y = from;
  return r;
#else
  return __half2half2(from);
#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pload<half2>(const Eigen::half* from) {
  return *reinterpret_cast<const half2*>(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 ploadu<half2>(const Eigen::half* from) {
  return __halves2half2(from[0], from[1]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 ploaddup<half2>(const Eigen::half*  from) {
  return __halves2half2(from[0], from[0]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<Eigen::half>(Eigen::half* to, const half2& from) {
  *reinterpret_cast<half2*>(to) = from;
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(Eigen::half* to, const half2& from) {
#if !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_HIP_DEVICE_COMPILE)
  to[0] = from.x;
  to[1] = from.y;
#else
  to[0] = __low2half(from);
  to[1] = __high2half(from);
#endif
}

template<>
 EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half2 ploadt_ro<half2, Aligned>(const Eigen::half* from) {

#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __ldg((const half2*)from);

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 350
   return __ldg((const half2*)from);
#else
  return __halves2half2(*(from+0), *(from+1));
#endif

#endif
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE half2 ploadt_ro<half2, Unaligned>(const Eigen::half* from) {

#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __halves2half2(__ldg(from+0), __ldg(from+1));

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 350
   return __halves2half2(__ldg(from+0), __ldg(from+1));
#else
  return __halves2half2(*(from+0), *(from+1));
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pgather<Eigen::half, half2>(const Eigen::half* from, Index stride) {
  return __halves2half2(from[0*stride], from[1*stride]);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pscatter<Eigen::half, half2>(Eigen::half* to, const half2& from, Index stride) {
  to[stride*0] = __low2half(from);
  to[stride*1] = __high2half(from);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half pfirst<half2>(const half2& a) {
  return __low2half(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pabs<half2>(const half2& a) {
  half a1 = __low2half(a);
  half a2 = __high2half(a);
  half result1 = half_impl::raw_uint16_to_half(a1.x & 0x7FFF);
  half result2 = half_impl::raw_uint16_to_half(a2.x & 0x7FFF);
  return __halves2half2(result1, result2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 ptrue<half2>(const half2& a) {
  half true_half = half_impl::raw_uint16_to_half(0xffffu);
  return pset1<half2>(true_half);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pzero<half2>(const half2& a) {
  half false_half = half_impl::raw_uint16_to_half(0x0000u);
  return pset1<half2>(false_half);
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void
ptranspose(PacketBlock<half2,2>& kernel) {
  __half a1 = __low2half(kernel.packet[0]);
  __half a2 = __high2half(kernel.packet[0]);
  __half b1 = __low2half(kernel.packet[1]);
  __half b2 = __high2half(kernel.packet[1]);
  kernel.packet[0] = __halves2half2(a1, b1);
  kernel.packet[1] = __halves2half2(a2, b2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plset<half2>(const Eigen::half& a) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __halves2half2(a, __hadd(a, __float2half(1.0f)));

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  return __halves2half2(a, __hadd(a, __float2half(1.0f)));
#else
  float f = __half2float(a) + 1.0f;
  return __halves2half2(a, __float2half(f));
#endif

#endif
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pselect<half2>(const half2& mask,
                                                           const half2& a,
                                                           const half2& b) {
  half mask_low = __low2half(mask);
  half mask_high = __high2half(mask);
  half result_low = mask_low == half(0) ? __low2half(b) : __low2half(a);
  half result_high = mask_high == half(0) ? __high2half(b) : __high2half(a);
  return __halves2half2(result_low, result_high);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pcmp_eq<half2>(const half2& a,
                                                           const half2& b) {
  half true_half = half_impl::raw_uint16_to_half(0xffffu);
  half false_half = half_impl::raw_uint16_to_half(0x0000u);
  half a1 = __low2half(a);
  half a2 = __high2half(a);
  half b1 = __low2half(b);
  half b2 = __high2half(b);
  half eq1 = __half2float(a1) == __half2float(b1) ? true_half : false_half;
  half eq2 = __half2float(a2) == __half2float(b2) ? true_half : false_half;
  return __halves2half2(eq1, eq2);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pcmp_lt<half2>(const half2& a,
                                                           const half2& b) {
  half true_half = half_impl::raw_uint16_to_half(0xffffu);
  half false_half = half_impl::raw_uint16_to_half(0x0000u);
  half a1 = __low2half(a);
  half a2 = __high2half(a);
  half b1 = __low2half(b);
  half b2 = __high2half(b);
  half eq1 = __half2float(a1) < __half2float(b1) ? true_half : false_half;
  half eq2 = __half2float(a2) < __half2float(b2) ? true_half : false_half;
  return __halves2half2(eq1, eq2);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pand<half2>(const half2& a,
                                                        const half2& b) {
  half a1 = __low2half(a);
  half a2 = __high2half(a);
  half b1 = __low2half(b);
  half b2 = __high2half(b);
  half result1 = half_impl::raw_uint16_to_half(a1.x & b1.x);
  half result2 = half_impl::raw_uint16_to_half(a2.x & b2.x);
  return __halves2half2(result1, result2);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 por<half2>(const half2& a,
                                                       const half2& b) {
  half a1 = __low2half(a);
  half a2 = __high2half(a);
  half b1 = __low2half(b);
  half b2 = __high2half(b);
  half result1 = half_impl::raw_uint16_to_half(a1.x | b1.x);
  half result2 = half_impl::raw_uint16_to_half(a2.x | b2.x);
  return __halves2half2(result1, result2);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pxor<half2>(const half2& a,
                                                        const half2& b) {
  half a1 = __low2half(a);
  half a2 = __high2half(a);
  half b1 = __low2half(b);
  half b2 = __high2half(b);
  half result1 = half_impl::raw_uint16_to_half(a1.x ^ b1.x);
  half result2 = half_impl::raw_uint16_to_half(a2.x ^ b2.x);
  return __halves2half2(result1, result2);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pandnot<half2>(const half2& a,
                                                           const half2& b) {
  half a1 = __low2half(a);
  half a2 = __high2half(a);
  half b1 = __low2half(b);
  half b2 = __high2half(b);
  half result1 = half_impl::raw_uint16_to_half(a1.x & ~b1.x);
  half result2 = half_impl::raw_uint16_to_half(a2.x & ~b2.x);
  return __halves2half2(result1, result2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 padd<half2>(const half2& a, const half2& b) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __hadd2(a, b);

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  return __hadd2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 + b1;
  float r2 = a2 + b2;
  return __floats2half2_rn(r1, r2);
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 psub<half2>(const half2& a, const half2& b) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __hsub2(a, b);

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  return __hsub2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 - b1;
  float r2 = a2 - b2;
  return __floats2half2_rn(r1, r2);
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pnegate(const half2& a) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __hneg2(a);

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  return __hneg2(a);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return __floats2half2_rn(-a1, -a2);
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pconj(const half2& a) { return a; }

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmul<half2>(const half2& a, const half2& b) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __hmul2(a, b);

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  return __hmul2(a, b);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 * b1;
  float r2 = a2 * b2;
  return __floats2half2_rn(r1, r2);
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmadd<half2>(const half2& a, const half2& b, const half2& c) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

   return __hfma2(a, b, c);

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
   return __hfma2(a, b, c);
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float c1 = __low2float(c);
  float c2 = __high2float(c);
  float r1 = a1 * b1 + c1;
  float r2 = a2 * b2 + c2;
  return __floats2half2_rn(r1, r2);
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pdiv<half2>(const half2& a, const half2& b) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __h2div(a, b);

#else // EIGEN_CUDA_ARCH

  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  float r1 = a1 / b1;
  float r2 = a2 / b2;
  return __floats2half2_rn(r1, r2);

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmin<half2>(const half2& a, const half2& b) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  __half r1 = a1 < b1 ? __low2half(a) : __low2half(b);
  __half r2 = a2 < b2 ? __high2half(a) : __high2half(b);
  return __halves2half2(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pmax<half2>(const half2& a, const half2& b) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float b1 = __low2float(b);
  float b2 = __high2float(b);
  __half r1 = a1 > b1 ? __low2half(a) : __low2half(b);
  __half r2 = a2 > b2 ? __high2half(a) : __high2half(b);
  return __halves2half2(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux<half2>(const half2& a) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __hadd(__low2half(a), __high2half(a));

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  return __hadd(__low2half(a), __high2half(a));
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return Eigen::half(__float2half(a1 + a2));
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_max<half2>(const half2& a) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  __half first = __low2half(a);
  __half second = __high2half(a);
  return __hgt(first, second) ? first : second;

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  __half first = __low2half(a);
  __half second = __high2half(a);
  return __hgt(first, second) ? first : second;
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return a1 > a2 ? __low2half(a) : __high2half(a);
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_min<half2>(const half2& a) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  __half first = __low2half(a);
  __half second = __high2half(a);
  return __hlt(first, second) ? first : second;

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  __half first = __low2half(a);
  __half second = __high2half(a);
  return __hlt(first, second) ? first : second;
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return a1 < a2 ? __low2half(a) : __high2half(a);
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_mul<half2>(const half2& a) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  return __hmul(__low2half(a), __high2half(a));

#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  return __hmul(__low2half(a), __high2half(a));
#else
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  return Eigen::half(__float2half(a1 * a2));
#endif

#endif
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plog1p<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = log1pf(a1);
  float r2 = log1pf(a2);
  return __floats2half2_rn(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pexpm1<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = expm1f(a1);
  float r2 = expm1f(a2);
  return __floats2half2_rn(r1, r2);
}

#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
  defined(EIGEN_HIP_DEVICE_COMPILE)

template<>  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
half2 plog<half2>(const half2& a) {
  return h2log(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
half2 pexp<half2>(const half2& a) {
  return h2exp(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
half2 psqrt<half2>(const half2& a) {
  return h2sqrt(a);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
half2 prsqrt<half2>(const half2& a) {
  return h2rsqrt(a);
}

#else

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 plog<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = logf(a1);
  float r2 = logf(a2);
  return __floats2half2_rn(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 pexp<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = expf(a1);
  float r2 = expf(a2);
  return __floats2half2_rn(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 psqrt<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = sqrtf(a1);
  float r2 = sqrtf(a2);
  return __floats2half2_rn(r1, r2);
}

template<> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE half2 prsqrt<half2>(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = rsqrtf(a1);
  float r2 = rsqrtf(a2);
  return __floats2half2_rn(r1, r2);
}
#endif

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
pset1<Packet4h2>(const Eigen::half& from) {
  Packet4h2 r;
  half2* p_alias = reinterpret_cast<half2*>(&r);
  p_alias[0] = pset1<half2>(from);
  p_alias[1] = pset1<half2>(from);
  p_alias[2] = pset1<half2>(from);
  p_alias[3] = pset1<half2>(from);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
pload<Packet4h2>(const Eigen::half* from) {
  return *reinterpret_cast<const Packet4h2*>(from);
}
// unaligned load;
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
ploadu<Packet4h2>(const Eigen::half* from) {
  Packet4h2 r;
  half2* p_alias = reinterpret_cast<half2*>(&r);
  p_alias[0] = ploadu<half2>(from+0);
  p_alias[1] = ploadu<half2>(from+2);
  p_alias[2] = ploadu<half2>(from+4);
  p_alias[3] = ploadu<half2>(from+6);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
ploaddup<Packet4h2>(const Eigen::half* from) {
  Packet4h2 r;
  half2* p_alias = reinterpret_cast<half2*>(&r);
  p_alias[0] = ploaddup<half2>(from);
  p_alias[1] = ploaddup<half2>(from);
  p_alias[2] = ploaddup<half2>(from);
  p_alias[3] = ploaddup<half2>(from);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstore<Eigen::half>(
    Eigen::half* to, const Packet4h2& from) {
  *reinterpret_cast<Packet4h2*>(to) = from;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pstoreu<Eigen::half>(
    Eigen::half* to, const Packet4h2& from) {
#if !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_HIP_DEVICE_COMPILE)
  to[0] = from.x;
  to[1] = from.y;
#else
  const half2* from_alias = reinterpret_cast<const half2*>(&from);
  pstoreu<Eigen::half>(to,from_alias[0]);
  pstoreu<Eigen::half>(to+2,from_alias[1]);
  pstoreu<Eigen::half>(to+4,from_alias[2]);
  pstoreu<Eigen::half>(to+6,from_alias[3]);
#endif
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet4h2
ploadt_ro<Packet4h2, Aligned>(const Eigen::half* from) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  Packet4h2 r;
  r = __ldg((const Packet4h2*)from);
  return r;
#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 350
  Packet4h2 r;
  r = __ldg((const Packet4h2*)from);
  return r;
#else
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  r_alias[0] = ploadt_ro<half2,Aligned>(from+0);
  r_alias[1] = ploadt_ro<half2,Aligned>(from+2);
  r_alias[2] = ploadt_ro<half2,Aligned>(from+4);
  r_alias[3] = ploadt_ro<half2,Aligned>(from+6);
  return r;
#endif

#endif
}

template <>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet4h2
ploadt_ro<Packet4h2, Unaligned>(const Eigen::half* from) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  r_alias[0] = ploadt_ro<half2,Unaligned>(from + 0);
  r_alias[1] = ploadt_ro<half2,Unaligned>(from + 2);
  r_alias[2] = ploadt_ro<half2,Unaligned>(from + 4);
  r_alias[3] = ploadt_ro<half2,Unaligned>(from + 6);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
pgather<Eigen::half, Packet4h2>(const Eigen::half* from, Index stride) {
  Packet4h2 r;
  half2* p_alias = reinterpret_cast<half2*>(&r);
  p_alias[0] = __halves2half2(from[0 * stride], from[1 * stride]);
  p_alias[1] = __halves2half2(from[2 * stride], from[3 * stride]);
  p_alias[2] = __halves2half2(from[4 * stride], from[5 * stride]);
  p_alias[3] = __halves2half2(from[6 * stride], from[7 * stride]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void pscatter<Eigen::half, Packet4h2>(
    Eigen::half* to, const Packet4h2& from, Index stride) {
  const half2* from_alias = reinterpret_cast<const half2*>(&from);
  pscatter<Eigen::half,half2>(to,from_alias[0],stride);
  pscatter<Eigen::half,half2>(to+stride*2,from_alias[1],stride);
  pscatter<Eigen::half,half2>(to+stride*4,from_alias[2],stride);
  pscatter<Eigen::half,half2>(to+stride*6,from_alias[3],stride);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half pfirst<Packet4h2>(
    const Packet4h2& a) {
  return pfirst<half2>(*(reinterpret_cast<const half2*>(&a)));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pabs<Packet4h2>(const Packet4h2& a) {
  Packet4h2 r;
  half2* p_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  p_alias[0] = pabs<half2>(a_alias[0]);
  p_alias[1] = pabs<half2>(a_alias[1]);
  p_alias[2] = pabs<half2>(a_alias[2]);
  p_alias[3] = pabs<half2>(a_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 ptrue<Packet4h2>(const Packet4h2& a) {
  half true_half = half_impl::raw_uint16_to_half(0xffffu);
  return pset1<Packet4h2>(true_half);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pzero<Packet4h2>(const Packet4h2& a) {
  half false_half = half_impl::raw_uint16_to_half(0x0000u);
  return pset1<Packet4h2>(false_half);
}

// TODO SAMI
// EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void
// ptranspose(PacketBlock<Packet4h2,4>& kernel) {
//   __half a1 = __low2half(kernel.packet[0]);
//   __half a2 = __high2half(kernel.packet[0]);
//   __half b1 = __low2half(kernel.packet[1]);
//   __half b2 = __high2half(kernel.packet[1]);
//   kernel.packet[0] = __halves2half2(a1, b1);
//   kernel.packet[1] = __halves2half2(a2, b2);
// }

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
plset<Packet4h2>(const Eigen::half& a) {
#if defined(EIGEN_HIP_DEVICE_COMPILE)

  Packet4h2 r;
  half2* p_alias = reinterpret_cast<half2*>(&r);
  p_alias[0] = __halves2half2(a, __hadd(a, __float2half(1.0f)));
  p_alias[1] = __halves2half2(__hadd(a, __float2half(2.0f)),
                              __hadd(a, __float2half(3.0f)));
  p_alias[2] = __halves2half2(__hadd(a, __float2half(4.0f)),
                              __hadd(a, __float2half(5.0f)));
  p_alias[3] = __halves2half2(__hadd(a, __float2half(6.0f)),
                              __hadd(a, __float2half(7.0f)));
  return r;
#else  // EIGEN_CUDA_ARCH

#if EIGEN_CUDA_ARCH >= 530
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  half2 half_2pack=__halves2half2(__float2half(2.0f),__float2half(4.0f));
  half2 b=pset1<half2>(a);
  r_alias[0] = plset<half2>(__low2half(b));
  r_alias[1] = plset<half2>(__high2half(b));
  b=__hadd2(b,half_2pack);
  r_alias[2] = plset<half2>(__low2half(b));
  r_alias[3] = plset<half2>(__high2half(b));
  return r;

#else
  float f = __half2float(a);
  Packet4h2 r;
  half2* p_alias = reinterpret_cast<half2*>(&r);
  p_alias[0] = __halves2half2(a, __float2half(f + 1.0f));
  p_alias[1] = __halves2half2(__float2half(f + 2.0f), __float2half(f + 3.0f));
  p_alias[2] = __halves2half2(__float2half(f + 4.0f), __float2half(f + 5.0f));
  p_alias[3] = __halves2half2(__float2half(f + 6.0f), __float2half(f + 7.0f));
  return r;
#endif

#endif
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
pselect<Packet4h2>(const Packet4h2& mask, const Packet4h2& a, const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* mask_alias = reinterpret_cast<const half2*>(&mask);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pselect<half2>(mask_alias[0], a_alias[0], b_alias[0]);
  r_alias[1] = pselect<half2>(mask_alias[1], a_alias[1], b_alias[1]);
  r_alias[2] = pselect<half2>(mask_alias[2], a_alias[2], b_alias[2]);
  r_alias[3] = pselect<half2>(mask_alias[3], a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
pcmp_eq<Packet4h2>(const Packet4h2& a, const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pcmp_eq<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = pcmp_eq<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = pcmp_eq<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = pcmp_eq<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pand<Packet4h2>(const Packet4h2& a,
                                                            const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pand<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = pand<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = pand<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = pand<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 por<Packet4h2>(const Packet4h2& a,
                                                           const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = por<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = por<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = por<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = por<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pxor<Packet4h2>(const Packet4h2& a,
                                                            const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pxor<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = pxor<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = pxor<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = pxor<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
pandnot<Packet4h2>(const Packet4h2& a, const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pandnot<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = pandnot<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = pandnot<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = pandnot<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 padd<Packet4h2>(const Packet4h2& a,
                                                            const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = padd<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = padd<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = padd<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = padd<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 psub<Packet4h2>(const Packet4h2& a,
                                                            const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = psub<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = psub<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = psub<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = psub<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pnegate(const Packet4h2& a) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  r_alias[0] = pnegate<half2>(a_alias[0]);
  r_alias[1] = pnegate<half2>(a_alias[1]);
  r_alias[2] = pnegate<half2>(a_alias[2]);
  r_alias[3] = pnegate<half2>(a_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pconj(const Packet4h2& a) {
  return a;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pmul<Packet4h2>(const Packet4h2& a,
                                                            const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pmul<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = pmul<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = pmul<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = pmul<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pmadd<Packet4h2>(const Packet4h2& a,
                                                             const Packet4h2& b,
                                                             const Packet4h2& c) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  const half2* c_alias = reinterpret_cast<const half2*>(&c);
  r_alias[0] = pmadd<half2>(a_alias[0], b_alias[0], c_alias[0]);
  r_alias[1] = pmadd<half2>(a_alias[1], b_alias[1], c_alias[1]);
  r_alias[2] = pmadd<half2>(a_alias[2], b_alias[2], c_alias[2]);
  r_alias[3] = pmadd<half2>(a_alias[3], b_alias[3], c_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pdiv<Packet4h2>(const Packet4h2& a,
                                                            const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pdiv<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = pdiv<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = pdiv<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = pdiv<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pmin<Packet4h2>(const Packet4h2& a,
                                                            const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pmin<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = pmin<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = pmin<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = pmin<half2>(a_alias[3], b_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pmax<Packet4h2>(const Packet4h2& a,
                                                            const Packet4h2& b) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  const half2* b_alias = reinterpret_cast<const half2*>(&b);
  r_alias[0] = pmax<half2>(a_alias[0], b_alias[0]);
  r_alias[1] = pmax<half2>(a_alias[1], b_alias[1]);
  r_alias[2] = pmax<half2>(a_alias[2], b_alias[2]);
  r_alias[3] = pmax<half2>(a_alias[3], b_alias[3]);
  return r;
}
// might overflow easier
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux<Packet4h2>(
    const Packet4h2& a) {
  const half2* a_alias = reinterpret_cast<const half2*>(&a);

  return predux<half2>(a_alias[0]) + predux<half2>(a_alias[1]) +
         predux<half2>(a_alias[2]) + predux<half2>(a_alias[3]);
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_max<Packet4h2>(
    const Packet4h2& a) {
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  half2 m0 = __halves2half2(predux_max<half2>(a_alias[0]),
                          predux_max<half2>(a_alias[1]));
  half2 m1 = __halves2half2(predux_max<half2>(a_alias[2]),
                          predux_max<half2>(a_alias[3]));
  __half first=predux_max<half2>(m0);
  __half second=predux_max<half2>(m1);
#if EIGEN_CUDA_ARCH < 530
  float ffirst=__half2float(first);
  float fsecond=__half2float(second);
  return (ffirst>fsecond)?first:second;
#else
  return (__hgt(first, second) ? first : second);
#endif


}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_min<Packet4h2>(
    const Packet4h2& a) {
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  half2 m0 = __halves2half2(predux_min<half2>(a_alias[0]),
                          predux_min<half2>(a_alias[1]));
  half2 m1 = __halves2half2(predux_min<half2>(a_alias[2]),
                          predux_min<half2>(a_alias[3]));
  __half first = predux_min<half2>(m0);
  __half second = predux_min<half2>(m1);
#if EIGEN_CUDA_ARCH < 530
  float ffirst=__half2float(first);
  float fsecond=__half2float(second);
  return (ffirst<fsecond)?first:second;
#else
  return (__hlt(first, second) ? first : second);
#endif

}
// likely overflow/underflow
template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Eigen::half predux_mul<Packet4h2>(
    const Packet4h2& a) {
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  return predux_mul<half2>(pmul<half2>(pmul<half2>(a_alias[0], a_alias[1]), pmul<half2>(a_alias[2], a_alias[3])));
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
plog1p<Packet4h2>(const Packet4h2& a) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  r_alias[0] = plog1p<half2>(a_alias[0]);
  r_alias[1] = plog1p<half2>(a_alias[1]);
  r_alias[2] = plog1p<half2>(a_alias[2]);
  r_alias[3] = plog1p<half2>(a_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
pexpm1<Packet4h2>(const Packet4h2& a) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  r_alias[0] = pexpm1<half2>(a_alias[0]);
  r_alias[1] = pexpm1<half2>(a_alias[1]);
  r_alias[2] = pexpm1<half2>(a_alias[2]);
  r_alias[3] = pexpm1<half2>(a_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 plog<Packet4h2>(const Packet4h2& a) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  r_alias[0] = plog<half2>(a_alias[0]);
  r_alias[1] = plog<half2>(a_alias[1]);
  r_alias[2] = plog<half2>(a_alias[2]);
  r_alias[3] = plog<half2>(a_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 pexp<Packet4h2>(const Packet4h2& a) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  r_alias[0] = pexp<half2>(a_alias[0]);
  r_alias[1] = pexp<half2>(a_alias[1]);
  r_alias[2] = pexp<half2>(a_alias[2]);
  r_alias[3] = pexp<half2>(a_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2 psqrt<Packet4h2>(const Packet4h2& a) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  r_alias[0] = psqrt<half2>(a_alias[0]);
  r_alias[1] = psqrt<half2>(a_alias[1]);
  r_alias[2] = psqrt<half2>(a_alias[2]);
  r_alias[3] = psqrt<half2>(a_alias[3]);
  return r;
}

template <>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet4h2
prsqrt<Packet4h2>(const Packet4h2& a) {
  Packet4h2 r;
  half2* r_alias = reinterpret_cast<half2*>(&r);
  const half2* a_alias = reinterpret_cast<const half2*>(&a);
  r_alias[0] = prsqrt<half2>(a_alias[0]);
  r_alias[1] = prsqrt<half2>(a_alias[1]);
  r_alias[2] = prsqrt<half2>(a_alias[2]);
  r_alias[3] = prsqrt<half2>(a_alias[3]);
  return r;
}

#endif

} // end namespace internal

} // end namespace Eigen


#endif // EIGEN_PACKET_MATH_GPU_H
