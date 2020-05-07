// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template<typename T, typename I_> void test_BiCGSTABL_T()
{

	BiCGSTABL<SparseMatrix<T, 0, I_>, DiagonalPreconditioner<T> > BiCGSTABL_diag;
	BiCGSTABL<SparseMatrix<T, 0, I_>, IdentityPreconditioner>     BiCGSTABL_I;
	BiCGSTABL<SparseMatrix<T, 0, I_>, IncompleteLUT<T, I_> >      BiCGSTABL_ilut;

	BiCGSTABL_diag.setTolerance(NumTraits<T>::epsilon() * 4);
	BiCGSTABL_I.setTolerance(NumTraits<T>::epsilon() * 4);
	BiCGSTABL_ilut.setTolerance(NumTraits<T>::epsilon() * 4);

	CALL_SUBTEST( check_sparse_square_solving(BiCGSTABL_diag));
	CALL_SUBTEST( check_sparse_square_solving(BiCGSTABL_I));
	CALL_SUBTEST( check_sparse_square_solving(BiCGSTABL_ilut));
}

EIGEN_DECLARE_TEST(bicgstabl)
{
	CALL_SUBTEST_1((test_BiCGSTABL_T<double, int>()) );
	CALL_SUBTEST_2((test_BiCGSTABL_T<std::complex<double>, int>()));
	CALL_SUBTEST_3((test_BiCGSTABL_T<double, long int>()));
}
