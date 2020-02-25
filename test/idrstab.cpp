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

template<typename T, typename I_> void test_idrstab_T()
{
	int L = 2;
	int S = 4;
	//for(int L=1; L<=8; L*=2)
	{
		//for(int S=1; S<=8; S*=2)
		{
			IDRStab<SparseMatrix<T, 0, I_>, DiagonalPreconditioner<T> > idrstab_colmajor_diag;
			IDRStab<SparseMatrix<T, 0, I_>, IdentityPreconditioner>     idrstab_colmajor_I;
			IDRStab<SparseMatrix<T, 0, I_>, IncompleteLUT<T, I_> >      idrstab_colmajor_ilut;

			idrstab_colmajor_diag.setL(L);
			idrstab_colmajor_diag.setS(S);
			idrstab_colmajor_I.setL(L);
			idrstab_colmajor_I.setS(S);
			idrstab_colmajor_ilut.setL(L);
			idrstab_colmajor_ilut.setS(S);

			idrstab_colmajor_diag.setTolerance(NumTraits<T>::epsilon() * 4);
			idrstab_colmajor_I.setTolerance(NumTraits<T>::epsilon() * 4);
			idrstab_colmajor_ilut.setTolerance(NumTraits<T>::epsilon() * 4);

			CALL_SUBTEST( check_sparse_square_solving(idrstab_colmajor_diag));
			CALL_SUBTEST( check_sparse_square_solving(idrstab_colmajor_I));
			CALL_SUBTEST( check_sparse_square_solving(idrstab_colmajor_ilut));
		}
	}
}

EIGEN_DECLARE_TEST(idrstab)
{
	CALL_SUBTEST_1((test_idrstab_T<double, int>()) );
	CALL_SUBTEST_2((test_idrstab_T<std::complex<double>, int>()));
	CALL_SUBTEST_3((test_idrstab_T<double, long int>()));
}
