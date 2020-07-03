// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Rasmus Munk Larsen <rmlarsen@google.com>
// Copyright (C) 2020, Arm Limited and Contributors
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_TYPE_CASTING_SVE_H
#define EIGEN_TYPE_CASTING_SVE_H

namespace Eigen {

namespace internal {

template<> struct type_casting_traits<float,numext::int32_t>
{ enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 }; };
template<> struct type_casting_traits<numext::int32_t,float>
{ enum { VectorizedCast = 1, SrcCoeffRatio = 1, TgtCoeffRatio = 1 }; };

template<> EIGEN_STRONG_INLINE Packetf pcast<Packeti,Packetf>(const Packeti& a) {
  return svcvt_f32_s32_z(svptrue_b32(), a); } // Convert integer to FP, setting inactive to zero
template<> EIGEN_STRONG_INLINE Packeti pcast<Packetf,Packeti>(const Packetf& a) { return svcvt_s32_f32_z(svptrue_b32(), a); } // Convert FP to integer, setting inactive to zero

template<> EIGEN_STRONG_INLINE Packetf preinterpret<Packetf,Packeti>(const Packeti& a)
{ return svreinterpret_f32_s32(a); }
template<> EIGEN_STRONG_INLINE Packeti preinterpret<Packeti,Packetf>(const Packetf& a)
{ return svreinterpret_s32_f32(a); }

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_TYPE_CASTING_SVE_H
