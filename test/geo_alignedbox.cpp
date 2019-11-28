// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/QR>

#include<iostream>
using namespace std;

// TODO not sure if this is actually still necessary anywhere ...
template<typename T> EIGEN_DONT_INLINE
void kill_extra_precision(T& ) {  }

// VERIFY_IS_APPROX isn't good for comparisons against zero!
#define VERIFY_ALMOST_ZERO(num) VERIFY_IS_APPROX((num) + Scalar(1), Scalar(1))


template<typename BoxType> void alignedbox(const BoxType& _box)
{
  /* this test covers the following files:
     AlignedBox.h
  */
  typedef typename BoxType::Scalar Scalar;
  typedef NumTraits<Scalar> ScalarTraits;
  typedef typename ScalarTraits::Real RealScalar;
  typedef Matrix<Scalar, BoxType::AmbientDimAtCompileTime, 1> VectorType;

  const Index dim = _box.dim();

  VectorType p0 = VectorType::Random(dim);
  VectorType p1 = VectorType::Random(dim);
  while( p1 == p0 ){
      p1 =  VectorType::Random(dim); }
  RealScalar s1 = internal::random<RealScalar>(0,1);

  BoxType b0(dim);
  BoxType b1(VectorType::Random(dim),VectorType::Random(dim));
  BoxType b2;
  
  kill_extra_precision(b1);
  kill_extra_precision(p0);
  kill_extra_precision(p1);

  b0.extend(p0);
  b0.extend(p1);
  VERIFY(b0.contains(p0*s1+(Scalar(1)-s1)*p1));
  VERIFY(b0.contains(b0.center()));
  VERIFY_IS_APPROX(b0.center(),(p0+p1)/Scalar(2));

  (b2 = b0).extend(b1);
  VERIFY(b2.contains(b0));
  VERIFY(b2.contains(b1));
  VERIFY_IS_APPROX(b2.clamp(b0), b0);

  // intersection
  BoxType box1(VectorType::Random(dim));
  box1.extend(VectorType::Random(dim));
  BoxType box2(VectorType::Random(dim));
  box2.extend(VectorType::Random(dim));

  VERIFY(box1.intersects(box2) == !box1.intersection(box2).isEmpty()); 

  // alignment -- make sure there is no memory alignment assertion
  BoxType *bp0 = new BoxType(dim);
  BoxType *bp1 = new BoxType(dim);
  bp0->extend(*bp1);
  delete bp0;
  delete bp1;

  // sampling
  for( int i=0; i<10; ++i )
  {
      VectorType r = b0.sample();
      VERIFY(b0.contains(r));
  }

}

template<typename BoxType> void alignedboxTranslatable(const BoxType& _box)
{
  typedef typename BoxType::Scalar Scalar;
  typedef Matrix<Scalar, BoxType::AmbientDimAtCompileTime, 1> VectorType;
  typedef Transform<Scalar, BoxType::AmbientDimAtCompileTime, Isometry> IsometryTransform;

  alignedbox(_box);

  const Index dim = _box.dim();

  // box((-1, -1, -1), (1, 1, 1))
  BoxType a(-VectorType::Ones(dim), VectorType::Ones(dim));

  for (Index d = 0; d < dim; ++d)
    VERIFY_IS_APPROX(a.sizes()[d], Scalar(2));

  BoxType b = a;
  VectorType translate = VectorType::Ones(dim);
  translate[0] = Scalar(2);
  b.translate(translate);
  // translate by (2, 1, 1) -> box((1, 0, 0), (3, 2, 2))

  for (Index d = 0; d < dim; ++d)
    VERIFY_IS_APPROX(b.sizes()[d], Scalar(2));

  VERIFY_IS_APPROX((b.min)()[0], Scalar(1));
  for (Index d = 1; d < dim; ++d)
    VERIFY_IS_APPROX((b.min)()[d], Scalar(0));

  VERIFY_IS_APPROX((b.max)()[0], Scalar(3));
  for (Index d = 1; d < dim; ++d)
    VERIFY_IS_APPROX((b.max)()[d], Scalar(2));

  // Test transform

  IsometryTransform tf = IsometryTransform::Identity();
  tf.translation() = -translate;

  BoxType c = b.transformed(tf);
  // translate by (-2, -1, -1) -> box((-1, -1, -1), (1, 1, 1))
  for (Index d = 0; d < dim; ++d)
  {
    VERIFY_IS_APPROX(c.sizes()[d], a.sizes()[d]);
    VERIFY_IS_APPROX((c.min)()[d], (a.min)()[d]);
    VERIFY_IS_APPROX((c.max)()[d], (a.max)()[d]);
  }

  c.transform(tf);
  // translate by (-2, -1, -1) -> box((-3, -2, -2), (-1, 0, 0))
  for (Index d = 0; d < dim; ++d)
    VERIFY_IS_APPROX(c.sizes()[d], a.sizes()[d]);

  VERIFY_IS_APPROX((c.min)()[0], Scalar(-3));
  VERIFY_IS_APPROX((c.max)()[0], Scalar(-1));
  for (Index d = 1; d < dim; ++d)
  {
    VERIFY_IS_APPROX((c.min)()[d], Scalar(-2));
    VERIFY_IS_APPROX((c.max)()[d], Scalar(0));
  }

  // test for roundoff errors
  IsometryTransform identity;
  identity.setIdentity();
  BoxType transformedC;
  transformedC.extend(c.transformed(identity));
  VERIFY(transformedC.contains(c));

  for (size_t i = 0; i < 10; ++i)
  {
    VectorType minCorner, maxCorner;
    for (Index d = 0; d < dim; ++d)
    {
      minCorner[d] = internal::random<Scalar>(-10,10);
      maxCorner[d] = minCorner[d] + internal::random<Scalar>(0, 10);
    }

    c = BoxType(minCorner, maxCorner);

    VectorType translate = VectorType::Random();
    c.translate(translate);

    for (Index d = 0; d < dim; ++d)
    {
      VERIFY_IS_APPROX((c.min)()[d], minCorner[d] + translate[d]);
      VERIFY_IS_APPROX((c.max)()[d], maxCorner[d] + translate[d]);
    }
  }
}

template<typename Scalar, typename Rotation>
Rotation rotate2D(Scalar _angle) {
  return Rotation2D<Scalar>(_angle);
}

template<typename Scalar, typename Rotation>
Rotation rotate2DIntegral(typename NumTraits<Scalar>::NonInteger _angle) {
  typedef typename NumTraits<Scalar>::NonInteger NonInteger;
  return Rotation2D<NonInteger>(_angle).toRotationMatrix().
      template cast<Scalar>();
}

template<typename Scalar, typename Rotation>
Rotation rotate3DZAxis(Scalar _angle) {
  return AngleAxis<Scalar>(_angle, Matrix<Scalar, 3, 1>(0, 0, 1));
}

template<typename Scalar, typename Rotation>
Rotation rotate3DZAxisIntegral(typename NumTraits<Scalar>::NonInteger _angle) {
  typedef typename NumTraits<Scalar>::NonInteger NonInteger;
  return AngleAxis<NonInteger>(_angle, Matrix<NonInteger, 3, 1>(0, 0, 1)).
      toRotationMatrix().template cast<Scalar>();
}

template<typename BoxType, typename Rotation> void alignedboxRotatable(
    const BoxType& _box,
    Rotation (*_rotate)(typename NumTraits<typename BoxType::Scalar>::NonInteger _angle),
    const bool allowNonIntegralRotations = true)
{
  alignedboxTranslatable(_box);

  typedef typename BoxType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::NonInteger NonInteger;
  typedef Matrix<Scalar, BoxType::AmbientDimAtCompileTime, 1> VectorType;
  typedef Transform<Scalar, BoxType::AmbientDimAtCompileTime, Isometry> IsometryTransform;

  const Index dim = _box.dim();

  // in this kind of comments the 3D case values will be illustrated
  // box((-1, -1, -1), (1, 1, 1))
  BoxType a(-VectorType::Ones(dim), VectorType::Ones(dim));

  // to allow templating this test for both 2D and 3D cases, we always set all
  // but the first coordinate to the same value; so basically 3D case works as
  // if you were looking at the scene from top

  VectorType minPoint = -2*VectorType::Ones(dim);
  minPoint[0] = -3;
  VectorType maxPoint = VectorType::Zero(dim);
  maxPoint[0] = -1;
  BoxType c(minPoint, maxPoint);
  // box((-3, -2, -2), (-1, 0, 0))

  IsometryTransform tf2 = IsometryTransform::Identity();
  // for some weird reason the following statement has to be put separate from
  // the following rotate call, otherwise precision problems arise...
  Rotation rot = _rotate(NonInteger(EIGEN_PI));
  tf2.rotate(rot);

  c.transform(tf2);
  // rotate by 180 deg around origin -> box((1, 0, -2), (3, 2, 0))

  for (Index d = 0; d < dim; ++d)
    VERIFY_IS_APPROX(c.sizes()[d], a.sizes()[d]);

  VERIFY_IS_APPROX((c.min)()[0], Scalar(1));
  VERIFY_IS_APPROX((c.max)()[0], Scalar(3));
  VERIFY_IS_APPROX((c.min)()[1], Scalar(0));
  VERIFY_IS_APPROX((c.max)()[1], Scalar(2));
  for (Index d = 2; d < dim; ++d)
  {
    VERIFY_IS_APPROX((c.min)()[2], Scalar(-2));
    VERIFY_ALMOST_ZERO((c.max)()[2]);
  }

  rot = _rotate(NonInteger(EIGEN_PI/2));
  tf2.setIdentity();
  tf2.rotate(rot);

  c.transform(tf2);
  // rotate by 90 deg around origin ->  box((-2, 1, -2), (0, 3, 0))

  for (Index d = 0; d < dim; ++d)
    VERIFY_IS_APPROX(c.sizes()[d], a.sizes()[d]);

  VERIFY_IS_APPROX((c.min)()[0], Scalar(-2));
  VERIFY_ALMOST_ZERO((c.max)()[0]);
  VERIFY_IS_APPROX((c.min)()[1], Scalar(1));
  VERIFY_IS_APPROX((c.max)()[1], Scalar(3));
  for (Index d = 2; d < dim; ++d)
  {
    VERIFY_IS_APPROX((c.min)()[2], Scalar(-2));
    VERIFY_ALMOST_ZERO((c.max)()[2]);
  }

  if (!allowNonIntegralRotations)
    return;

  VectorType cornerBL = (c.min)();
  VectorType cornerTR = (c.max)();
  VectorType cornerBR = (c.min)(); cornerBR[0] = cornerTR[0];
  VectorType cornerTL = (c.max)(); cornerTL[0] = cornerBL[0];

  NonInteger angle = NonInteger(EIGEN_PI/3);
  rot = _rotate(angle);
  tf2.setIdentity();
  tf2.rotate(rot);

  c.transform(tf2);
  // rotate by 60 deg ->  box((-3.59, -1.23, -2), (-0.86, 1.5, 0))
  // just draw the figure and these numbers will pop out

  cornerBL = tf2 * cornerBL;
  cornerBR = tf2 * cornerBR;
  cornerTL = tf2 * cornerTL;
  cornerTR = tf2 * cornerTR;
  
  VectorType minCorner;
  VectorType maxCorner;
  minCorner[0] = (min)((min)(cornerBL[0], cornerBR[0]), (min)(cornerTL[0], cornerTR[0]));
  maxCorner[0] = (max)((max)(cornerBL[0], cornerBR[0]), (max)(cornerTL[0], cornerTR[0]));
  minCorner[1] = (min)((min)(cornerBL[1], cornerBR[1]), (min)(cornerTL[1], cornerTR[1]));
  maxCorner[1] = (max)((max)(cornerBL[1], cornerBR[1]), (max)(cornerTL[1], cornerTR[1]));

  for (Index d = 2; d < dim; ++d)
    VERIFY_IS_APPROX(c.sizes()[d], a.sizes()[d]);

  VERIFY_IS_APPROX((c.min)()[0], minCorner[0]);
  VERIFY_IS_APPROX((c.min)()[0], Scalar(Scalar(-sqrt(2*2 + 3*3)) * Scalar(cos(Scalar(atan(2.0/3.0)) - angle/2))));
  VERIFY_IS_APPROX((c.max)()[0], maxCorner[0]);
  VERIFY_IS_APPROX((c.max)()[0], Scalar(-sin(angle)));
  VERIFY_IS_APPROX((c.min)()[1], minCorner[1]);
  VERIFY_IS_APPROX((c.min)()[1], Scalar(Scalar(-sqrt(1*1 + 2*2)) * Scalar(sin(Scalar(atan(2.0/1.0)) - angle/2))));
  VERIFY_IS_APPROX((c.max)()[1], maxCorner[1]);
  VERIFY_IS_APPROX((c.max)()[1], Scalar(3 * cos(angle)));
  for (Index d = 2; d < dim; ++d)
  {
    VERIFY_IS_APPROX((c.min)()[d], Scalar(-2));
    VERIFY_ALMOST_ZERO((c.max)()[d]);
  }

  // randomized test - transform the box and compare to a box made of transformed vertices
  for (size_t i = 0; i < 10; ++i)
  {
    for (Index d = 0; d < dim; ++d)
    {
      minCorner[d] = internal::random<Scalar>(-10,10);
      maxCorner[d] = minCorner[d] + internal::random<Scalar>(0, 10);
    }

    c = BoxType(minCorner, maxCorner);

    std::vector<VectorType> corners;
    const size_t numCorners = (dim == 1 ? 2 : (dim == 2 ? 4 : 8));
    for (int corner = 0; corner < static_cast<int>(numCorners); ++corner)
      corners.push_back(c.corner(static_cast<typename BoxType::CornerType>(corner)));

    angle = internal::random<NonInteger>(-EIGEN_PI, EIGEN_PI);
    rot = _rotate(angle);
    tf2.setIdentity();
    tf2.rotate(rot);

    c.transform(tf2);
    for (size_t corner = 0; corner < numCorners; ++corner)
      corners[corner] = tf2 * corners[corner];

    for (Index d = 0; d < dim; ++d)
    {
      minCorner[d] = corners[0][d];
      maxCorner[d] = corners[0][d];

      for (size_t corner = 0; corner < numCorners; ++corner)
      {
        minCorner[d] = (min)(minCorner[d], corners[corner][d]);
        maxCorner[d] = (max)(maxCorner[d], corners[corner][d]);
      }
    }

    for (Index d = 0; d < dim; ++d)
    {
      VERIFY_IS_APPROX((c.min)()[d], minCorner[d]);
      VERIFY_IS_APPROX((c.max)()[d], maxCorner[d]);
    }
  }
}

template<typename BoxType>
void alignedboxCastTests(const BoxType& _box)
{
  // casting  
  typedef typename BoxType::Scalar Scalar;
  typedef Matrix<Scalar, BoxType::AmbientDimAtCompileTime, 1> VectorType;

  const Index dim = _box.dim();

  VectorType p0 = VectorType::Random(dim);
  VectorType p1 = VectorType::Random(dim);

  BoxType b0(dim);

  b0.extend(p0);
  b0.extend(p1);

  const int Dim = BoxType::AmbientDimAtCompileTime;
  typedef typename GetDifferentType<Scalar>::type OtherScalar;
  AlignedBox<OtherScalar,Dim> hp1f = b0.template cast<OtherScalar>();
  VERIFY_IS_APPROX(hp1f.template cast<Scalar>(),b0);
  AlignedBox<Scalar,Dim> hp1d = b0.template cast<Scalar>();
  VERIFY_IS_APPROX(hp1d.template cast<Scalar>(),b0);
}


void specificTest1()
{
    Vector2f m; m << -1.0f, -2.0f;
    Vector2f M; M <<  1.0f,  5.0f;

    typedef AlignedBox2f  BoxType;
    BoxType box( m, M );

    Vector2f sides = M-m;
    VERIFY_IS_APPROX(sides, box.sizes() );
    VERIFY_IS_APPROX(sides[1], box.sizes()[1] );
    VERIFY_IS_APPROX(sides[1], box.sizes().maxCoeff() );
    VERIFY_IS_APPROX(sides[0], box.sizes().minCoeff() );

    VERIFY_IS_APPROX( 14.0f, box.volume() );
    VERIFY_IS_APPROX( 53.0f, box.diagonal().squaredNorm() );
    VERIFY_IS_APPROX( std::sqrt( 53.0f ), box.diagonal().norm() );

    VERIFY_IS_APPROX( m, box.corner( BoxType::BottomLeft ) );
    VERIFY_IS_APPROX( M, box.corner( BoxType::TopRight ) );
    Vector2f bottomRight; bottomRight << M[0], m[1];
    Vector2f topLeft; topLeft << m[0], M[1];
    VERIFY_IS_APPROX( bottomRight, box.corner( BoxType::BottomRight ) );
    VERIFY_IS_APPROX( topLeft, box.corner( BoxType::TopLeft ) );
}


void specificTest2()
{
    Vector3i m; m << -1, -2, 0;
    Vector3i M; M <<  1,  5, 3;

    typedef AlignedBox3i  BoxType;
    BoxType box( m, M );

    Vector3i sides = M-m;
    VERIFY_IS_APPROX(sides, box.sizes() );
    VERIFY_IS_APPROX(sides[1], box.sizes()[1] );
    VERIFY_IS_APPROX(sides[1], box.sizes().maxCoeff() );
    VERIFY_IS_APPROX(sides[0], box.sizes().minCoeff() );

    VERIFY_IS_APPROX( 42, box.volume() );
    VERIFY_IS_APPROX( 62, box.diagonal().squaredNorm() );

    VERIFY_IS_APPROX( m, box.corner( BoxType::BottomLeftFloor ) );
    VERIFY_IS_APPROX( M, box.corner( BoxType::TopRightCeil ) );
    Vector3i bottomRightFloor; bottomRightFloor << M[0], m[1], m[2];
    Vector3i topLeftFloor; topLeftFloor << m[0], M[1], m[2];
    VERIFY_IS_APPROX( bottomRightFloor, box.corner( BoxType::BottomRightFloor ) );
    VERIFY_IS_APPROX( topLeftFloor, box.corner( BoxType::TopLeftFloor ) );
}


void test_geo_alignedbox()
{
  for(int i = 0; i < g_repeat; i++)
  {
    CALL_SUBTEST_1( (alignedboxRotatable<AlignedBox2f, Rotation2Df>(AlignedBox2f(), &rotate2D)) );
    CALL_SUBTEST_2( alignedboxCastTests(AlignedBox2f()) );

    CALL_SUBTEST_3( (alignedboxRotatable<AlignedBox3f, AngleAxisf>(AlignedBox3f(), &rotate3DZAxis)) );
    CALL_SUBTEST_4( alignedboxCastTests(AlignedBox3f()) );

    CALL_SUBTEST_5( alignedboxTranslatable(AlignedBox4d()) );
    CALL_SUBTEST_6( alignedboxCastTests(AlignedBox4d()) );

    CALL_SUBTEST_7( alignedboxTranslatable(AlignedBox1d()) );
    CALL_SUBTEST_8( alignedboxCastTests(AlignedBox1d()) );

    CALL_SUBTEST_9( alignedboxTranslatable(AlignedBox1i()) );
    CALL_SUBTEST_10( (alignedboxRotatable<AlignedBox2i, Matrix2i>(AlignedBox2i(), &rotate2DIntegral<int, Matrix2i>, false)) );
    CALL_SUBTEST_11( (alignedboxRotatable<AlignedBox3i, Matrix3i>(AlignedBox3i(), &rotate3DZAxisIntegral<int, Matrix3i>, false)) );

    CALL_SUBTEST_14( alignedbox(AlignedBox<double,Dynamic>(4)) );
  }
  CALL_SUBTEST_12( specificTest1() );
  CALL_SUBTEST_13( specificTest2() );
}
