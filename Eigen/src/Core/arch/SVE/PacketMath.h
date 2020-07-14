// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Konstantinos Margaritis <markos@freevec.org>
// Copyright (C) 2020, Arm Limited and Contributors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_SVE_H
#define EIGEN_PACKET_MATH_SVE_H

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 8
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_MADD
#endif

#ifndef EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
#define EIGEN_HAS_SINGLE_INSTRUCTION_CJMADD
#endif

#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#if EIGEN_ARCH_ARM64
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 32
#else
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 16
#endif
#endif

// SVE Sizeless types
// Depending on the EIGEN_SVE_VL, these types will have different-sized elements.
typedef svint32_t   Packeti  __attribute__((arm_sve_vector_bits(EIGEN_SVE_VL)));
typedef svuint32_t  Packetui __attribute__((arm_sve_vector_bits(EIGEN_SVE_VL)));
typedef svfloat32_t Packetf  __attribute__((arm_sve_vector_bits(EIGEN_SVE_VL)));
typedef svfloat64_t Packetd  __attribute__((arm_sve_vector_bits(EIGEN_SVE_VL)));

#define FLOAT_PACKET_SIZE EIGEN_SVE_VL/32
#define INT_PACKET_SIZE EIGEN_SVE_VL/32
#define DOUBLE_PACKET_SIZE EIGEN_SVE_VL/64


#define _EIGEN_DECLARE_CONST_Packetf(NAME,X) \
  const Packetf pf_##NAME = pset1<Packetf>(X)

#define _EIGEN_DECLARE_CONST_Packetf_FROM_INT(NAME,X) \
  const Packetf pf_##NAME = svreinterpret_f32_u32(pset1<int32_t>(X))

#define _EIGEN_DECLARE_CONST_Packeti(NAME,X) \
  const Packeti pi_##NAME = pset1<Packeti>(X)

#if EIGEN_ARCH_ARM64
  // __builtin_prefetch tends to do nothing on ARM64 compilers because the
  // prefetch instructions there are too detailed for __builtin_prefetch to map
  // meaningfully to them.
  // TODO [SVE] Confirm that this still applies to SVE (and if we should switch to PRFW)
  // In relevant functions, we have switched the prefetcher to respective SVE intrinsic
  #define EIGEN_ARM_PREFETCH(ADDR)  __asm__ __volatile__("prfm pldl1keep, [%[addr]]\n" ::[addr] "r"(ADDR) : );
#elif EIGEN_HAS_BUILTIN(__builtin_prefetch) || EIGEN_COMP_GNUC
  #define EIGEN_ARM_PREFETCH(ADDR) __builtin_prefetch(ADDR);
#elif defined __pld
  #define EIGEN_ARM_PREFETCH(ADDR) __pld(ADDR)
#elif EIGEN_ARCH_ARM32
  #define EIGEN_ARM_PREFETCH(ADDR) __asm__ __volatile__ ("pld [%[addr]]\n" :: [addr] "r" (ADDR) : );
#else
  // by default no explicit prefetching
  #define EIGEN_ARM_PREFETCH(ADDR)
#endif

template <>
struct packet_traits<float> : default_packet_traits
{
  typedef Packetf type;
  typedef Packetf half;  // Half not implemented yet
  enum
  {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = FLOAT_PACKET_SIZE,
    HasHalfPacket = 0,

    HasAdd       = 1,
    HasSub       = 1,
    HasShift     = 1,
    HasMul       = 1,
    HasNegate    = 1,
    HasAbs       = 1,
    HasArg       = 0,
    HasAbs2      = 1,
    HasMin       = 1,
    HasMax       = 1,
    HasConj      = 1,
    HasSetLinear = 0,
    HasBlend     = 0,
    HasReduxp    = 0,  // Not implemented in SVE

    HasDiv   = 1,
    HasFloor = 1,

    HasSin  = EIGEN_FAST_MATH,
    HasCos  = EIGEN_FAST_MATH,
    HasLog  = 1,
    HasExp  = 1,
    HasSqrt = 0,
    HasTanh = EIGEN_FAST_MATH,
    HasErf  = EIGEN_FAST_MATH
  };
};

template <>
struct packet_traits<int32_t> : default_packet_traits
{
  typedef Packeti type;
  typedef Packeti half; // Half not implemented yet
  enum
  {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = INT_PACKET_SIZE,
    HasHalfPacket=0,

    HasAdd       = 1,
    HasSub       = 1,
    HasShift     = 1,
    HasMul       = 1,
    HasNegate    = 1,
    HasAbs       = 1,
    HasArg       = 0,
    HasAbs2      = 1,
    HasMin       = 1,
    HasMax       = 1,
    HasConj      = 1,
    HasSetLinear = 0,
    HasBlend     = 0,
    HasReduxp    = 0  // Not implemented in SVE
  };
};

template<> struct unpacket_traits<Packetf>
{
  typedef float type;
  typedef Packetf half;  // Half not yet implemented
  typedef Packeti integer_packet;
  enum
  {
    size = FLOAT_PACKET_SIZE,
    alignment = Aligned64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template<> struct unpacket_traits<Packeti>
{
  typedef int32_t type;
  typedef Packeti half;  // Half not yet implemented
  enum
  {
    size = INT_PACKET_SIZE,
    alignment = Aligned64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template<> EIGEN_STRONG_INLINE Packetf pset1<Packetf>(const float& from) { return svdup_n_f32(from); }
template<> EIGEN_STRONG_INLINE Packeti pset1<Packeti>(const int32_t& from) { return svdup_n_s32(from); }

template<> EIGEN_STRONG_INLINE Packetf pset1frombits<Packetf>(unsigned int from)
{ return svreinterpret_f32_u32(svdup_n_u32_z(svptrue_b32(),from)); }

template<> EIGEN_STRONG_INLINE Packetf plset<Packetf>(const float& a)
{ 
  float c[FLOAT_PACKET_SIZE];
  for (int i = 0; i < FLOAT_PACKET_SIZE; i++)
    c[i] = i;
  return svadd_f32_z(svptrue_b32(), pset1<Packetf>(a), svld1_f32(svptrue_b32(), c));
}
template<> EIGEN_STRONG_INLINE Packeti plset<Packeti>(const int32_t& a)
{
  int32_t c[INT_PACKET_SIZE];
  for (int i = 0; i < INT_PACKET_SIZE; i++)
    c[i] = i;
  return svadd_s32_z(svptrue_b32(), pset1<Packeti>(a), svld1_s32(svptrue_b32(), c));
}

template<> EIGEN_STRONG_INLINE Packetf padd<Packetf>(const Packetf& a, const Packetf& b) { return svadd_f32_z(svptrue_b32(),a,b); }
template<> EIGEN_STRONG_INLINE Packeti padd<Packeti>(const Packeti& a, const Packeti& b) { return svadd_s32_z(svptrue_b32(),a,b); }

template<> EIGEN_STRONG_INLINE Packetf psub<Packetf>(const Packetf& a, const Packetf& b) { return svsub_f32_z(svptrue_b32(),a,b); }
template<> EIGEN_STRONG_INLINE Packeti psub<Packeti>(const Packeti& a, const Packeti& b) { return svsub_s32_z(svptrue_b32(),a,b); }

template<> EIGEN_STRONG_INLINE Packetf pnegate(const Packetf& a) { return svneg_f32_z(svptrue_b32(),a); }
template<> EIGEN_STRONG_INLINE Packeti pnegate(const Packeti& a) { return svneg_s32_z(svptrue_b32(),a); }

template<> EIGEN_STRONG_INLINE Packetf pconj(const Packetf& a) { return a; }
template<> EIGEN_STRONG_INLINE Packeti pconj(const Packeti& a) { return a; }

template<> EIGEN_STRONG_INLINE Packetf pmul<Packetf>(const Packetf& a, const Packetf& b) { return svmul_f32_z(svptrue_b32(),a,b); }
template<> EIGEN_STRONG_INLINE Packeti pmul<Packeti>(const Packeti& a, const Packeti& b) { return svmul_s32_z(svptrue_b32(),a,b); }

template<> EIGEN_STRONG_INLINE Packetf pdiv<Packetf>(const Packetf& a, const Packetf& b) { return svdiv_f32_z(svptrue_b32(),a,b); }
template<> EIGEN_STRONG_INLINE Packeti pdiv<Packeti>(const Packeti& a, const Packeti& b) { return svdiv_s32_z(svptrue_b32(),a,b); }

template<> EIGEN_STRONG_INLINE Packetf pmadd(const Packetf& a, const Packetf& b, const Packetf& c) { return svmad_f32_z(svptrue_b32(),c,a,b); }
template<> EIGEN_STRONG_INLINE Packeti pmadd(const Packeti& a, const Packeti& b, const Packeti& c) { return svmla_s32_z(svptrue_b32(),c,a,b); }

template<> EIGEN_STRONG_INLINE Packetf pmin<Packetf>(const Packetf& a, const Packetf& b) { return svmin_f32_z(svptrue_b32(),a,b); }
template<> EIGEN_STRONG_INLINE Packeti pmin<Packeti>(const Packeti& a, const Packeti& b) { return svmin_s32_z(svptrue_b32(),a,b); }

template<> EIGEN_STRONG_INLINE Packetf pmax<Packetf>(const Packetf& a, const Packetf& b) { return svmax_f32_z(svptrue_b32(),a,b); }
template<> EIGEN_STRONG_INLINE Packeti pmax<Packeti>(const Packeti& a, const Packeti& b) { return svmax_s32_z(svptrue_b32(),a,b); }

// Integer comparisons in SVE return svbool (predicate). Use svdup to set active lanes to 1 (0xffffffffu) and inactive lanes to 0.
template<> EIGEN_STRONG_INLINE Packetf pcmp_le<Packetf>(const Packetf& a, const Packetf& b)
{ return svreinterpret_f32_u32(svdup_n_u32_z(svcmplt_f32(svptrue_b32(),a,b),0xffffffffu)); }
template<> EIGEN_STRONG_INLINE Packeti pcmp_le<Packeti>(const Packeti& a, const Packeti& b)
{ return svdup_n_s32_z(svcmplt_s32(svptrue_b32(),a,b),0xffffffffu); }

template<> EIGEN_STRONG_INLINE Packetf pcmp_lt<Packetf>(const Packetf& a, const Packetf& b)
{ return svreinterpret_f32_u32(svdup_n_u32_z(svcmplt_f32(svptrue_b32(),a,b),0xffffffffu)); }
template<> EIGEN_STRONG_INLINE Packeti pcmp_lt<Packeti>(const Packeti& a, const Packeti& b)
{ return svdup_n_s32_z(svcmplt_s32(svptrue_b32(),a,b),0xffffffffu); }

template<> EIGEN_STRONG_INLINE Packetf pcmp_eq<Packetf>(const Packetf& a, const Packetf& b)
{ return svreinterpret_f32_u32(svdup_n_u32_z(svcmpeq_f32(svptrue_b32(),a,b),0xffffffffu)); }
template<> EIGEN_STRONG_INLINE Packeti pcmp_eq<Packeti>(const Packeti& a, const Packeti& b)
{ return svdup_n_s32_z(svcmpeq_s32(svptrue_b32(),a,b),0xffffffffu); }

// Do a predicate inverse (svnot_b_z) on the predicate resulted from the greater/equal comparison (svcmpge_f32).
// Then fill a float vector with the active elements.
template<> EIGEN_STRONG_INLINE Packetf pcmp_lt_or_nan<Packetf>(const Packetf& a, const Packetf& b)
{  return svreinterpret_f32_u32(svdup_n_u32_z(svnot_b_z(svptrue_b32(),svcmpge_f32(svptrue_b32(),a,b)),0xffffffffu)); }

template<> EIGEN_STRONG_INLINE Packetf pfloor<Packetf>(const Packetf& a)
{
  const Packetf cst_1 = pset1<Packetf>(1.0f);
  /* perform a floorf */
  Packetf tmp = svcvt_f32_s32_z(svptrue_b32(), svcvt_s32_f32_z(svptrue_b32(), a));

  /* if greater, substract 1 */
  // Integer comparisons in SVE return svbool (predicate). Use svdup to set active lanes 1 (0xffffffffu) and inactive lanes to 0.
  Packetui mask = svdup_n_u32_z(svcmpgt_f32(svptrue_b32(),tmp, a), 0xffffffffu);
  mask = svand_u32_z(svptrue_b32(), mask, svreinterpret_u32_f32(cst_1));
  return svsub_f32_z(svptrue_b32(), tmp, svreinterpret_f32_u32(mask));
}

//template<> EIGEN_STRONG_INLINE Packetf pnot<Packetf>(const Packetf& a)
//{ return svreinterpret_f32_u32(svnot_u32_z(svptrue_b32(),svreinterpret_u32_f32(a))); }
//template<> EIGEN_STRONG_INLINE Packeti pnot<Packeti>(const Packeti& a)
//{ return svnot_s32_z(svptrue_b32(),a); }

template<> EIGEN_STRONG_INLINE Packetf ptrue<Packetf>(const Packetf& /*a*/)
{ return svreinterpret_f32_u32(svdup_n_u32_z(svptrue_b32(),0xffffffffu)); }
template<> EIGEN_STRONG_INLINE Packeti ptrue<Packeti>(const Packeti& /*a*/)
{ return svdup_n_s32_z(svptrue_b32(),0xffffffffu); }

template<> EIGEN_STRONG_INLINE Packetf pzero<Packetf>(const Packetf& /*a*/)
{ return svreinterpret_f32_u32(svdup_n_u32_z(svptrue_b32(),0)); }
template<> EIGEN_STRONG_INLINE Packeti pzero<Packeti>(const Packeti& /*a*/)
{ return svdup_n_s32_z(svptrue_b32(),0); }

// Logical Operations are not supported for float, so reinterpret casts
template<> EIGEN_STRONG_INLINE Packetf pand<Packetf>(const Packetf& a, const Packetf& b)
{ return svreinterpret_f32_u32(svand_u32_z(svptrue_b32(),svreinterpret_u32_f32(a),svreinterpret_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packeti pand<Packeti>(const Packeti& a, const Packeti& b) { return svand_s32_z(svptrue_b32(),a,b); }

template<> EIGEN_STRONG_INLINE Packetf por<Packetf>(const Packetf& a, const Packetf& b)
{ return svreinterpret_f32_u32(svorr_u32_z(svptrue_b32(),svreinterpret_u32_f32(a),svreinterpret_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packeti por<Packeti>(const Packeti& a, const Packeti& b) { return svorr_s32_z(svptrue_b32(),a,b); }

template<> EIGEN_STRONG_INLINE Packetf pxor<Packetf>(const Packetf& a, const Packetf& b)
{ return svreinterpret_f32_u32(sveor_u32_z(svptrue_b32(),svreinterpret_u32_f32(a),svreinterpret_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packeti pxor<Packeti>(const Packeti& a, const Packeti& b) { return sveor_s32_z(svptrue_b32(),a,b); }

template<> EIGEN_STRONG_INLINE Packetf pandnot<Packetf>(const Packetf& a, const Packetf& b)
{ return svreinterpret_f32_u32(svbic_u32_z(svptrue_b32(),svreinterpret_u32_f32(a),svreinterpret_u32_f32(b))); }
template<> EIGEN_STRONG_INLINE Packeti pandnot<Packeti>(const Packeti& a, const Packeti& b) { return svbic_s32_z(svptrue_b32(),a,b); }

template<int N> EIGEN_STRONG_INLINE Packeti parithmetic_shift_right(Packeti a) { return svasrd_n_s32_z(svptrue_b32(), a, N); }

template<int N> EIGEN_STRONG_INLINE Packeti plogical_shift_right(Packeti a)
{ return svreinterpret_s32_u32(svlsr_u32_z(svptrue_b32(),svreinterpret_u32_s32(a),svdup_n_u32_z(svptrue_b32(),N))); }

template<int N> EIGEN_STRONG_INLINE Packeti plogical_shift_left(Packeti a) { return svlsl_s32_z(svptrue_b32(),a,svdup_n_u32_z(svptrue_b32(),N)); }

template<> EIGEN_STRONG_INLINE Packetf pload<Packetf>(const float*    from) { EIGEN_DEBUG_ALIGNED_LOAD return svld1_f32(svptrue_b32(),from); }
template<> EIGEN_STRONG_INLINE Packeti pload<Packeti>(const int32_t*  from) { EIGEN_DEBUG_ALIGNED_LOAD return svld1_s32(svptrue_b32(),from); }

template<> EIGEN_STRONG_INLINE Packetf ploadu<Packetf>(const float*   from) { EIGEN_DEBUG_UNALIGNED_LOAD return svld1_f32(svptrue_b32(),from); }
template<> EIGEN_STRONG_INLINE Packeti ploadu<Packeti>(const int32_t* from) { EIGEN_DEBUG_UNALIGNED_LOAD return svld1_s32(svptrue_b32(),from); }

template<> EIGEN_STRONG_INLINE Packetf ploaddup<Packetf>(const float* from)
{
  svuint32_t indices = svindex_u32(0, 1); // index {base=0, base+step=1, base+step*2, ...}
  indices = svzip1_u32(indices, indices); // index in the format {a0, a0, a1, a1, a2, a2, ...}
  return svld1_gather_u32index_f32(svptrue_b32(), from, indices);
}
template<> EIGEN_STRONG_INLINE Packeti ploaddup<Packeti>(const int32_t* from)
{
  svuint32_t indices = svindex_u32(0, 1); // index {base=0, base+step=1, base+step*2, ...}
  indices = svzip1_u32(indices, indices); // index in the format {a0, a0, a1, a1, a2, a2, ...}
  return svld1_gather_u32index_s32(svptrue_b32(), from, indices);
}

template<> EIGEN_STRONG_INLINE Packetf ploadquad<Packetf>(const float* from)
{
  svuint32_t indices = svindex_u32(0, 1); // index {base=0, base+step=1, base+step*2, ...}
  indices = svzip1_u32(indices, indices); // index in the format {a0, a0, a1, a1, a2, a2, ...}
  indices = svzip1_u32(indices, indices); // index in the format {a0, a0, a0, a0, a1, a1, a1, a1, ...}
  return svld1_gather_u32index_f32(svptrue_b32(), from, indices);
}
template<> EIGEN_STRONG_INLINE Packeti ploadquad<Packeti>(const int32_t* from)
{
  svuint32_t indices = svindex_u32(0, 1); // index {base=0, base+step=1, base+step*2, ...}
  indices = svzip1_u32(indices, indices); // index in the format {a0, a0, a1, a1, a2, a2, ...}
  indices = svzip1_u32(indices, indices); // index in the format {a0, a0, a0, a0, a1, a1, a1, a1, ...}
  return svld1_gather_u32index_s32(svptrue_b32(), from, indices);
}

template<> EIGEN_STRONG_INLINE void pstore<float>  (float*    to, const Packetf& from) { EIGEN_DEBUG_ALIGNED_STORE svst1_f32(svptrue_b32(),to, from); }
template<> EIGEN_STRONG_INLINE void pstore<int32_t>(int32_t*  to, const Packeti& from) { EIGEN_DEBUG_ALIGNED_STORE svst1_s32(svptrue_b32(),to, from); }

template<> EIGEN_STRONG_INLINE void pstoreu<float>  (float*   to, const Packetf& from) { EIGEN_DEBUG_UNALIGNED_STORE svst1_f32(svptrue_b32(),to, from); }
template<> EIGEN_STRONG_INLINE void pstoreu<int32_t>(int32_t* to, const Packeti& from) { EIGEN_DEBUG_UNALIGNED_STORE svst1_s32(svptrue_b32(),to, from); }

template<> EIGEN_DEVICE_FUNC inline Packetf pgather<float, Packetf>(const float* from, Index stride)
{
  // Indice format: {base=0, base+stride, base+stride*2, base+stride*3, ...}
  svint32_t indices = svindex_s32(0, stride);
  return svld1_gather_s32index_f32(svptrue_b32(), from, indices);
}
template<> EIGEN_DEVICE_FUNC inline Packeti pgather<int32_t, Packeti>(const int32_t* from, Index stride)
{
  // Indice format: {base=0, base+stride, base+stride*2, base+stride*3, ...}
  svint32_t indices = svindex_s32(0, stride);
  return svld1_gather_s32index_s32(svptrue_b32(), from, indices);
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<float, Packetf>(float* to, const Packetf& from, Index stride)
{
  // Indice format: {base=0, base+stride, base+stride*2, base+stride*3, ...}
  svint32_t indices = svindex_s32(0, stride);
  svst1_scatter_s32index_f32(svptrue_b32(), to, indices, from);
}
template<> EIGEN_DEVICE_FUNC inline void pscatter<int32_t, Packeti>(int32_t* to, const Packeti& from, Index stride)
{
  // Indice format: {base=0, base+stride, base+stride*2, base+stride*3, ...}
  svint32_t indices = svindex_s32(0, stride);
  svst1_scatter_s32index_s32(svptrue_b32(), to, indices, from);
}

template<> EIGEN_STRONG_INLINE void prefetch<int32_t>(const int32_t*  addr) { svprfw(svptrue_b32(), addr, SV_PLDL1KEEP); }

template<> EIGEN_STRONG_INLINE float   pfirst<Packetf>(const Packetf& a) { EIGEN_ALIGN64 float   x[FLOAT_PACKET_SIZE]; svst1_f32(svptrue_b32(), x, a); return x[0]; }
template<> EIGEN_STRONG_INLINE int32_t pfirst<Packeti>(const Packeti& a) { EIGEN_ALIGN64 int32_t x[INT_PACKET_SIZE]; svst1_s32(svptrue_b32(), x, a); return x[0]; }

template<> EIGEN_STRONG_INLINE Packetf preverse(const Packetf& a) { return svrev_f32(a); }
template<> EIGEN_STRONG_INLINE Packeti preverse(const Packeti& a) { return svrev_s32(a); }

template<> EIGEN_STRONG_INLINE Packetf pabs(const Packetf& a) { return svabs_f32_z(svptrue_b32(), a); }
template<> EIGEN_STRONG_INLINE Packeti pabs(const Packeti& a) { return svabs_s32_z(svptrue_b32(), a); }

template<> EIGEN_STRONG_INLINE Packetf pfrexp<Packetf>(const Packetf& a, Packetf& exponent)
{ return pfrexp_float(a,exponent); }

template<> EIGEN_STRONG_INLINE Packetf pldexp<Packetf>(const Packetf& a, const Packetf& exponent)
{ return pldexp_float(a,exponent); }

template<> EIGEN_STRONG_INLINE float predux<Packetf>(const Packetf& a)
{ return svaddv_f32(svptrue_b32(), a); }
template<> EIGEN_STRONG_INLINE int32_t predux<Packeti>(const Packeti& a)
{ return svaddv_s32(svptrue_b32(), a); }

// Other reduction functions:
// mul
// Only works for SVE Vls multiple of 128
template<> EIGEN_STRONG_INLINE float predux_mul<Packetf>(const Packetf& a)
{
  // Multiply the vector by its inverse
  svfloat32_t prod = svmul_f32_z(svptrue_b32(), a, svrev_f32(a));
  svfloat32_t half_prod;

  // Extract the high half of the vector. Depending on the VL more reductions need to be done
  if (EIGEN_SVE_VL >= 2048){
    half_prod = svtbl_f32(prod, svindex_u32(32, 1));
    prod = svmul_f32_z(svptrue_b32(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 1024){
    half_prod = svtbl_f32(prod, svindex_u32(16, 1));
    prod = svmul_f32_z(svptrue_b32(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 512){
    half_prod = svtbl_f32(prod, svindex_u32(8, 1));
    prod = svmul_f32_z(svptrue_b32(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 256){
    half_prod = svtbl_f32(prod, svindex_u32(4, 1));
    prod = svmul_f32_z(svptrue_b32(), prod, half_prod);
  }
  // Last reduction
  half_prod = svtbl_f32(prod, svindex_u32(2, 1));
  prod = svmul_f32_z(svptrue_b32(), prod, half_prod);

  // The reduction is done to the first element. Reverse and return the last element
  // (there are no SVE intrinsics to get the first active element)
  return svlastb_f32(svptrue_b32(), svrev_f32(prod));
}
// Only works for SVE Vls multiple of 128
template<> EIGEN_STRONG_INLINE int32_t predux_mul<Packeti>(const Packeti& a)
{
  // Multiply the vector by its inverse
  svint32_t prod = svmul_s32_z(svptrue_b32(), a, svrev_s32(a));
  svint32_t half_prod;

  // Extract the high half of the vector. Depending on the VL more reductions need to be done
  if (EIGEN_SVE_VL >= 2048){
    half_prod = svtbl_s32(prod, svindex_u32(32, 1));
    prod = svmul_s32_z(svptrue_b32(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 1024){
    half_prod = svtbl_s32(prod, svindex_u32(16, 1));
    prod = svmul_s32_z(svptrue_b32(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 512){
    half_prod = svtbl_s32(prod, svindex_u32(8, 1));
    prod = svmul_s32_z(svptrue_b32(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 256){
    half_prod = svtbl_s32(prod, svindex_u32(4, 1));
    prod = svmul_s32_z(svptrue_b32(), prod, half_prod);
  }
  // Last reduction
  half_prod = svtbl_s32(prod, svindex_u32(2, 1));
  prod = svmul_s32_z(svptrue_b32(), prod, half_prod);

  // The reduction is done to the first element. Reverse and return the last element
  // (there are no SVE intrinsics to get the first active element)
  return svlastb_s32(svptrue_b32(), svrev_s32(prod));
}

// min
template<> EIGEN_STRONG_INLINE float predux_min<Packetf>(const Packetf& a)
{  return svminv_f32(svptrue_b32(), a); }
template<> EIGEN_STRONG_INLINE int32_t predux_min<Packeti>(const Packeti& a)
{  return svminv_s32(svptrue_b32(), a); }

// max
template<> EIGEN_STRONG_INLINE float predux_max<Packetf>(const Packetf& a)
{ return svmaxv_f32(svptrue_b32(), a); }
template<> EIGEN_STRONG_INLINE int32_t predux_max<Packeti>(const Packeti& a)
{ return svmaxv_s32(svptrue_b32(), a); }

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packetf>& kernel)
{
  float buffer[FLOAT_PACKET_SIZE*FLOAT_PACKET_SIZE]; // PacketSize^2
  memset(buffer, 0, FLOAT_PACKET_SIZE*FLOAT_PACKET_SIZE*sizeof(float));
  int i = 0;

  Packeti stride_index = svindex_s32(0, FLOAT_PACKET_SIZE);

  for (i=0; i<FLOAT_PACKET_SIZE; i++){
    svst1_scatter_s32index_f32(svptrue_b32(), buffer+i, stride_index, kernel.packet[i]);
  }
  for (i=0; i<FLOAT_PACKET_SIZE; i++){
    kernel.packet[i] = svld1_f32(svptrue_b32(), buffer+i*FLOAT_PACKET_SIZE);
  }
}
EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packeti>& kernel)
{
  int buffer[INT_PACKET_SIZE*INT_PACKET_SIZE]; // PacketSize^2
  memset(buffer, 0, INT_PACKET_SIZE*INT_PACKET_SIZE*sizeof(int));
  int i = 0;

  Packeti stride_index = svindex_s32(0, INT_PACKET_SIZE);

  for (i=0; i<INT_PACKET_SIZE; i++){
    svst1_scatter_s32index_s32(svptrue_b32(), buffer+i, stride_index, kernel.packet[i]);
  }
  for (i=0; i<INT_PACKET_SIZE; i++){
    kernel.packet[i] = svld1_s32(svptrue_b32(), buffer+i*INT_PACKET_SIZE);
  }
}

//---------- double ----------

// Clang 3.5 in the iOS toolchain has an ICE triggered by NEON intrisics for double.
// Confirmed at least with __apple_build_version__ = 6000054.
#ifdef __apple_build_version__
// Let's hope that by the time __apple_build_version__ hits the 601* range, the bug will be fixed.
// https://gist.github.com/yamaya/2924292 suggests that the 3 first digits are only updated with
// major toolchain updates.
#define EIGEN_APPLE_DOUBLE_NEON_BUG (__apple_build_version__ < 6010000)
#else
#define EIGEN_APPLE_DOUBLE_NEON_BUG 0
#endif

#if EIGEN_ARCH_ARM64 && !EIGEN_APPLE_DOUBLE_NEON_BUG

template<> struct packet_traits<double>  : default_packet_traits
{
  typedef Packetd type;
  typedef Packetd half;  // Half package not implemented
  enum
  {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = DOUBLE_PACKET_SIZE,
    HasHalfPacket = 0,

    HasDiv  = 1,
    // FIXME check the Has*
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 0,
    HasExp  = 0,
    HasSqrt = 0,
    HasTanh = 0
  };
};

template<> struct unpacket_traits<Packetd>
{
  typedef double type;
  enum
  {
    size = DOUBLE_PACKET_SIZE,
    alignment = Aligned64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
  typedef Packetd half;
};

template<> EIGEN_STRONG_INLINE Packetd pset1<Packetd>(const double&  from) { return svdup_n_f64(from); }

template<> EIGEN_STRONG_INLINE Packetd plset<Packetd>(const double& a)
{ 
  double c[DOUBLE_PACKET_SIZE];
  for (int i = 0; i < DOUBLE_PACKET_SIZE; i++)
    c[i] = i;
  return svadd_f64_z(svptrue_b64(), pset1<Packetd>(a), svld1_f64(svptrue_b32(), c));
}

template<> EIGEN_STRONG_INLINE Packetd padd<Packetd>(const Packetd& a, const Packetd& b) { return svadd_f64_z(svptrue_b64(),a,b); }

template<> EIGEN_STRONG_INLINE Packetd psub<Packetd>(const Packetd& a, const Packetd& b) { return svsub_f64_z(svptrue_b64(),a,b); }

template<> EIGEN_STRONG_INLINE Packetd pnegate(const Packetd& a) { return svneg_f64_z(svptrue_b64(),a); }

template<> EIGEN_STRONG_INLINE Packetd pconj(const Packetd& a) { return a; }

template<> EIGEN_STRONG_INLINE Packetd pmul<Packetd>(const Packetd& a, const Packetd& b) { return svmul_f64_z(svptrue_b64(),a,b); }

template<> EIGEN_STRONG_INLINE Packetd pdiv<Packetd>(const Packetd& a, const Packetd& b) { return svdiv_f64_z(svptrue_b64(),a,b); }

template<> EIGEN_STRONG_INLINE Packetd pmadd(const Packetd& a, const Packetd& b, const Packetd& c)
{ return svmad_f64_z(svptrue_b64(),c,a,b); }

template<> EIGEN_STRONG_INLINE Packetd pmin<Packetd>(const Packetd& a, const Packetd& b) { return svmin_f64_z(svptrue_b64(),a,b); }

template<> EIGEN_STRONG_INLINE Packetd pmax<Packetd>(const Packetd& a, const Packetd& b) { return svmax_f64_z(svptrue_b64(),a,b); }

// Logical Operations are not supported for float, so we have to reinterpret casts using intrinsics
//template<> EIGEN_STRONG_INLINE Packetd pnot<Packetd>(const Packetd& a)
//{ return svreinterpret_f64_u64(svnot_u64_z(svptrue_b64(),svreinterpret_u64_f64(a))); }

template<> EIGEN_STRONG_INLINE Packetd ptrue<Packetd>(const Packetd& /*a*/)
{ return svreinterpret_f64_u64(svdup_n_u64_z(svptrue_b64(),0xffffffffffffffffu)); }

template<> EIGEN_STRONG_INLINE Packetd pzero<Packetd>(const Packetd& /*a*/)
{ return svreinterpret_f64_u64(svdup_n_u64_z(svptrue_b64(),0)); }

template<> EIGEN_STRONG_INLINE Packetd pand<Packetd>(const Packetd& a, const Packetd& b)
{ return svreinterpret_f64_u64(svand_u64_z(svptrue_b64(),svreinterpret_u64_f64(a),svreinterpret_u64_f64(b))); }

template<> EIGEN_STRONG_INLINE Packetd por<Packetd>(const Packetd& a, const Packetd& b)
{ return svreinterpret_f64_u64(svorr_u64_z(svptrue_b64(),svreinterpret_u64_f64(a),svreinterpret_u64_f64(b))); }

template<> EIGEN_STRONG_INLINE Packetd pxor<Packetd>(const Packetd& a, const Packetd& b)
{ return svreinterpret_f64_u64(sveor_u64_z(svptrue_b64(),svreinterpret_u64_f64(a),svreinterpret_u64_f64(b))); }

template<> EIGEN_STRONG_INLINE Packetd pandnot<Packetd>(const Packetd& a, const Packetd& b)
{ return svreinterpret_f64_u64(svbic_u64_z(svptrue_b64(),svreinterpret_u64_f64(a),svreinterpret_u64_f64(b))); }

// Integer comparisons in SVE return svbool (predicate). Use svdup to set active lanes 1 (0xffffffffffffffffu) and inactive lanes to 0.
template<> EIGEN_STRONG_INLINE Packetd pcmp_le(const Packetd& a, const Packetd& b)
{ return svreinterpret_f64_u64(svdup_n_u64_z(svcmplt_f64(svptrue_b64(),a,b),0xffffffffffffffffu)); }

// Integer comparisons in SVE return svbool (predicate). Use svdup to set active lanes 1 (0xffffffffffffffffu) and inactive lanes to 0.
template<> EIGEN_STRONG_INLINE Packetd pcmp_lt(const Packetd& a, const Packetd& b)
{ return svreinterpret_f64_u64(svdup_n_u64_z(svcmplt_f64(svptrue_b64(),a,b),0xffffffffffffffffu)); }

template<> EIGEN_STRONG_INLINE Packetd pcmp_eq(const Packetd& a, const Packetd& b)
{return svreinterpret_f64_u64(svdup_n_u64_z(svcmpeq_f64(svptrue_b32(),a,b),0xffffffffffffffffu)); }

template<> EIGEN_STRONG_INLINE Packetd pload<Packetd>(const double* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return svld1_f64(svptrue_b64(),from); }

template<> EIGEN_STRONG_INLINE Packetd ploadu<Packetd>(const double* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return svld1_f64(svptrue_b64(),from); }

template<> EIGEN_STRONG_INLINE Packetd ploaddup<Packetd>(const double* from)
{
  svuint64_t indices = svindex_u64(0, 1); // index {base=0, base+step=1, base+step*2, ...}
  indices = svzip1_u64(indices, indices); // index in the format {a0, a0, a1, a1, a2, a2, ...}
  return svld1_gather_u64index_f64(svptrue_b32(), from, indices);
}

template<> EIGEN_STRONG_INLINE Packetd ploadquad<Packetd>(const double* from)
{
  svuint64_t indices = svindex_u64(0, 1); // index {base=0, base+step=1, base+step*2, ...}
  indices = svzip1_u64(indices, indices); // index in the format {a0, a0, a1, a1, a2, a2, ...}
  indices = svzip1_u64(indices, indices); // index in the format {a0, a0, a0, a0, a1, a1, a1, a1, ...}
  return svld1_gather_u64index_f64(svptrue_b64(), from, indices);
}

template<> EIGEN_STRONG_INLINE void pstore<double>(double* to, const Packetd& from)
{ EIGEN_DEBUG_ALIGNED_STORE svst1_f64(svptrue_b64(),to,from); }

template<> EIGEN_STRONG_INLINE void pstoreu<double>(double* to, const Packetd& from)
{ EIGEN_DEBUG_UNALIGNED_STORE svst1_f64(svptrue_b64(),to,from); }

template<> EIGEN_DEVICE_FUNC inline Packetd pgather<double, Packetd>(const double* from, Index stride)
{
  // Indice format: {base=0, base+stride, base+stride*2, base+stride*3, ...}
  svint64_t indices = svindex_s64(0, stride);
  return svld1_gather_s64index_f64(svptrue_b64(), from, indices);
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<double, Packetd>(double* to, const Packetd& from, Index stride)
{
  // Indice format: {base=0, base+stride, base+stride*2, base+stride*3, ...}
  svint64_t indices = svindex_s64(0, stride);
  svst1_scatter_s64index_f64(svptrue_b64(), to, indices, from);
}

// Currently no SVE prefetch
// template<> EIGEN_STRONG_INLINE void prefetch<double>(const double* addr) { svprfd(svptrue_b64(), addr, SV_PLDL1KEEP); }
  // EIGEN_ARM_PREFETCH(addr);

template<> EIGEN_STRONG_INLINE double pfirst<Packetd>(const Packetd& a) { EIGEN_ALIGN64 double x[DOUBLE_PACKET_SIZE]; svst1_f64(svptrue_b64(), x, a); return x[0]; }

template<> EIGEN_STRONG_INLINE Packetd preverse(const Packetd& a)
{ return svrev_f64(a); }

template<> EIGEN_STRONG_INLINE Packetd pabs(const Packetd& a) { return svabs_f64_z(svptrue_b64(), a); }

template<> EIGEN_STRONG_INLINE double predux<Packetd>(const Packetd& a)
{ return svaddv_f64(svptrue_b64(), a); }

// Other reduction functions:
// mul
// Only works for SVE Vls multiple of 128
template<> EIGEN_STRONG_INLINE double predux_mul<Packetd>(const Packetd& a)
{
  // Multiply the vector by its inverse
  // For VL = 128, no further reduction is needed
  svfloat64_t prod = svmul_f64_z(svptrue_b64(), a, svrev_f64(a));
  svfloat64_t half_prod;

  // Extract the high half of the vector. Depending on the VL more reductions need to be done
  if (EIGEN_SVE_VL >= 2048){
    half_prod = svtbl_f64(prod, svindex_u64(16, 1));
    prod = svmul_f64_z(svptrue_b64(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 1024){
    half_prod = svtbl_f64(prod, svindex_u64(8, 1));
    prod = svmul_f64_z(svptrue_b64(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 512){
    half_prod = svtbl_f64(prod, svindex_u64(4, 1));
    prod = svmul_f64_z(svptrue_b64(), prod, half_prod);
  }
  if (EIGEN_SVE_VL >= 256){
    // Last reduction
    half_prod = svtbl_f64(prod, svindex_u64(2, 1));
    prod = svmul_f64_z(svptrue_b64(), prod, half_prod);
  }

  // The reduction is done to the first element. Reverse and return the last element
  // (there are no SVE intrinsics to get the first active element)
  return svlastb_f64(svptrue_b64(), svrev_f64(prod));
}

// min
template<> EIGEN_STRONG_INLINE double predux_min<Packetd>(const Packetd& a)
{ return svminv_f64(svptrue_b64(), a); }

// max
template<> EIGEN_STRONG_INLINE double predux_max<Packetd>(const Packetd& a)
{ return svmaxv_f64(svptrue_b64(), a); }

EIGEN_DEVICE_FUNC inline void ptranspose(PacketBlock<Packetd>& kernel)
{
  double buffer[DOUBLE_PACKET_SIZE*DOUBLE_PACKET_SIZE]; // PacketSize^2
  memset(buffer, 0, DOUBLE_PACKET_SIZE*DOUBLE_PACKET_SIZE*sizeof(double));
  int i = 0;

  svint64_t stride_index = svindex_s64(0, DOUBLE_PACKET_SIZE);

  for (i=0; i<DOUBLE_PACKET_SIZE; i++){
    svst1_scatter_s64index_f64(svptrue_b64(), buffer+i, stride_index, kernel.packet[i]);
  }
  for (i=0; i<DOUBLE_PACKET_SIZE; i++){
    kernel.packet[i] = svld1_f64(svptrue_b64(), buffer+i*DOUBLE_PACKET_SIZE);
  }
}

#endif // EIGEN_ARCH_ARM64

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKET_MATH_SVE_H
