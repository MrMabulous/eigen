/*
	Chris Schoutrop & Adithya Vijaykumar
	-working concept method on 27/02/2019
	-support for complex numbers
	-converted to Eigen instead of std::vector
	-BLAS2 operation for vector updates
	-Right preconditioning
	-fixed issue when preconditioner is "too good" and initial guess is zeros;
		precond.solve(x) would always be zero. Setting x to random initially works.
		There should be a more mathematical derivation on why and how for this though.
	-used Eigen's QR decomposition to simplify MR part
	-Added condition on L for small matrices (use L=length of x if L>length of x)
	-Added fix if r0 becomes orthogonal to rShadow
	-Added condition to stop computing if rho1 becomes nan/inf
	-Added termination condition if residual is low enough during BiCG step.
	-Removed fix if r0 becomes orthogonal to rShadow
	-Added reliable updating strategy from enhanced bicgstabl paper
	-Initial version of the convex combination from enhanced bicgstabl paper
	-Added defines to switch between original, householder QR and convex combination methods for argmin (stab-step) step.
	-Added define to switch between left and right preconditioning, note that right-preconditioning may not be compatible currently with reliable update step.


	Right-preconditioning is applied here;	Ax=b->AMinvu=b.
	instead of solving for x, we solve for MInv*u
	Then at the end x can be obtained via x=MInv*x
	See Algorithm 9.5 from Iterative methods for sparse linear systems for inspiration.

	Known issues:
	-See TODO's
	-If L is too large (~20) instabilities occur.
	-Residual gap can be significant (1e3 or more) if 1000+ iterations are needed. This is likely an algorithmic issue, not implementation issue.
		In case of IDR(s)Stab(L) this is shown in the paper of Kensuke.
		=> To fix this, consider the variant of BiCGStab(L) given in Fokkema, Diederik R. Enhanced implementation of BiCGstab (l) for solving
    	linear systems of equations. Universiteit Utrecht. Mathematisch Instituut, 1996.

	This implementation of BiCGStab(L) is based on the papers
		General algorithm:
		1. G.L.G. Sleijpen, D.R. Fokkema. (1993). BiCGstab(l) for linear equations involving unsymmetric matrices with complex spectrum. Electronic Transactions on Numerical Analysis.
		Polynomial step update:
		2. G.L.G. Sleijpen, M.B. Van Gijzen. (2010) Exploiting BiCGstab(l) strategies to induce dimension reduction SIAM Journal on Scientific Computing.
		3. Fokkema, Diederik R. Enhanced implementation of BiCGstab (l) for solving linear systems of equations. Universiteit Utrecht. Mathematisch Instituut, 1996
*/

#ifndef EIGEN_BICGSTABL_H
#define EIGEN_BICGSTABL_H

#include <algorithm>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <iostream>

#define BICGSTABL_IN_LIBRARY 1 //1 is needed to have bicgstab(L) compile with Eigen, as part of the library for the unit tests. 0 to have it as a plugin for the test suite and in Plasimo.
//TODO: Figure out a better fix.

#define BICGSTABL_REL_UPDATE 1 //0=no reliable update strategy, 1=reliable update strategy
#define BICGSTABL_ALGORITHM 0  //0: original, 1: Householder QR, 2: Convex polynomial method
#define BICGSTABL_PRECOND 1	   //0: right, 1: left. 0 may be broken, 1 is checked against Fortrain implementation of Enhanced bicgstabl
/*
        Technically it is possible to change these strategies at runtime, however this would also require the user (or some sort of automated criterion) to know why and when certain methods are best.
        Currently this is done via defines to eliminate any possible contamination between the methods.
*/
#define BICGSTABL_DEBUG_INFO 0 //Print info to console about the problem being solved.

#if BICGSTABL_DEBUG_INFO
#include <chrono>
#endif
namespace Eigen
{

	namespace internal
	{

		template <typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
		bool BiCGSTABL(const MatrixType &mat, const Rhs &rhs, Dest &x,
					   const Preconditioner &precond, Index &iters,
					   typename Dest::RealScalar &tol_error, Index L)
		{
#if BICGSTABL_DEBUG_INFO
			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			std::cout << "Matrix size: " << mat.rows() << std::endl;
#endif
			using std::abs;
			using std::sqrt;
			typedef typename Dest::RealScalar RealScalar;
			typedef typename Dest::Scalar Scalar;
			Index N = rhs.size();
			L = L < x.rows() ? L : x.rows();

			Index k = 0;

			RealScalar tol = tol_error;
			Index maxIters = iters;

			typedef Matrix<Scalar, Dynamic, 1> VectorType;
			typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrixType;

//We start with an initial guess x_0 and let us set r_0 as (residual calculated from x_0)
#if BICGSTABL_PRECOND == 0
			VectorType r0 = rhs - precond.solve(mat * x); //r_0
#elif BICGSTABL_PRECOND == 1
			VectorType r0 = precond.solve(rhs - mat * x); //r_0
#endif

			//rShadow is arbritary, but must not be orthogonal to r0.
			VectorType rShadow = r0;

			//RealScalar rShadow_sqnorm = rShadow.squaredNorm();
			//RealScalar rhs_sqnorm = rhs.squaredNorm();

			VectorType x_prime = x;
			x.setZero();
			VectorType b_prime = r0;

			//Other vectors and scalars initialization
			Scalar rho0 = 1.0;
			Scalar alpha = 0.0;
			Scalar omega = 1.0;

			//RealScalar tol2 = tol * tol * rShadow_sqnorm;
			//RealScalar eps2 = NumTraits<Scalar>::epsilon() * NumTraits<Scalar>::epsilon();

			DenseMatrixType rHat(N, L + 1);
			DenseMatrixType uHat(N, L + 1);

			rHat.col(0) = r0;
			uHat.col(0).fill(0.0);

			//Index restarts = 0;
			bool bicg_convergence = false;

			//RealScalar zeta0 = r0.norm();
			RealScalar zeta0 = rhs.norm();
			RealScalar zeta = zeta0;
			RealScalar Mx = zeta0;
			RealScalar Mr = zeta0;

			const RealScalar delta = 0.01;

			bool compute_res = false;
			bool update_app = false;

			//Bool to signal that a new rShadow was chosen and the main loop should be restarted.
			//bool reset = false;

			while (zeta > tol * zeta0 && k < maxIters)
			{
				rho0 = -omega * rho0;

				for (Index j = 0; j <= L - 1; ++j)
				{
					Scalar rho1 = rShadow.dot(rHat.col(j));

					// if (abs(rho1) < eps2 * rShadow_sqnorm)
					// {
					// 	// The new residual vector became too orthogonal to the arbitrarily chosen direction rShadow
					// 	// Let's restart with a new rShadow
					// 	std::cout << "Orthogonality reset triggered" << std::endl;

					// 	if (k != 0)
					// 	{
					// 		//Choose a new rShadow based on the current solution
					// 		r0  = precond.solve(rhs - mat * x);
					// 		rShadow = r0;
					// 		rho1 = rShadow_sqnorm = rShadow.squaredNorm();
					// 		rHat.col(0) = r0;
					// 		uHat.col(0).fill(0.0);
					// 	}
					// 	else
					// 	{
					// 		//During the first iteration there is no reason to reset, as nothing would change.
					// 		//Might as well try with a random rShadow instead, as it will be very unlikely to be orthogonal to any vector.
					// 		rShadow.setRandom();
					// 	}

					// 	if (restarts++ == 0)
					// 	{
					// 		k = 0;
					// 	}

					// 	reset = true;
					// 	break;
					// }

					if (rho1 - rho1 != 0.0 || rho0 == 0.0)
					{
						std::cout << "Internal BiCGStab(L) parameter became unfit to continue computing" << std::endl;
						tol_error = zeta / zeta0;
						return false;
					}

					Scalar beta = alpha * (rho1 / rho0);
					rho0 = rho1;

					//Update search directions
					for (Index i = 0; i <= j; ++i)
					{

						uHat.col(i) = rHat.col(i) - beta * uHat.col(i);
					}

#if BICGSTABL_PRECOND == 0
					uHat.col(j + 1) = mat * precond.solve(uHat.col(j));
#elif BICGSTABL_PRECOND == 1
					uHat.col(j + 1) = precond.solve(mat * uHat.col(j));
#endif
					alpha = rho1 / (rShadow.dot(uHat.col(j + 1)));

					//Update residuals
					for (Index i = 0; i <= j; ++i)
					{
						rHat.col(i) = rHat.col(i) - alpha * uHat.col(i + 1);
					}

#if BICGSTABL_PRECOND == 0
					rHat.col(j + 1) = mat * precond.solve(rHat.col(j));
#elif BICGSTABL_PRECOND == 1
					rHat.col(j + 1) = precond.solve(mat * rHat.col(j));
#endif

					//Complete BiCG iteration by updating x
					x = x + alpha * uHat.col(0);

					//Check for early exit
					zeta = rHat.col(0).norm();

					if (zeta < tol * zeta0)
					{
						/*
						        Convergence was achieved during BiCG step.
						        Without this check BiCGStab(L) fails for trivial matrices, such as when the preconditioner already is the inverse,
						        or the input matrix is identity.
						*/
						bicg_convergence = true;
						break;
					}
				}

#if BICGSTABL_ALGORITHM == 0

				if (bicg_convergence == false)
				{
					DenseMatrixType tau(L + 1, L + 1);
					VectorType sigma(L + 1);
					VectorType gamma(L + 1);
					VectorType gammaP(L + 1);
					VectorType gammaPP(L);
					/*
					        //TODO: Currently the first elements of tau,gamma, gammaP and gammaPP are never used
					        these do take up (a small amount of) memory. Changing indices to 0-based leads to deviations in notation with the Bicgstabl paper
					        and may be confusing.
					*/

					for (Index j = 1; j <= L; ++j)
					{
						for (Index i = 1; i <= j - 1; ++i)
						{
							tau(i, j) = (rHat.col(i).dot(rHat.col(j))) / sigma(i);
							rHat.col(j) -= tau(i, j) * rHat.col(i);
						}

						sigma(j) = rHat.col(j).squaredNorm();
						gammaP(j) = (rHat.col(j).dot(rHat.col(0))) / sigma(j);
					}

					gamma(L) = gammaP(L);
					omega = gamma(L);

					//TODO: store tau as triangular, perform backsubstitution with solve
					//instead of explicit summation.
					for (Index j = L - 1; j >= 1; --j)
					{
						Scalar sum = 0.0;

						for (Index i = j + 1; i <= L; ++i)
						{
							sum += tau(j, i) * gamma(i);
						}

						gamma(j) = gammaP(j) - sum;
					}

					//TODO: See if this can be done without nested loops and explicit sum
					for (Index j = 1; j <= L - 1; ++j)
					{
						Scalar sum = 0.0;

						for (Index i = j + 1; i <= L - 1; ++i)
						{
							sum += tau(j, i) * gamma(i + 1);
						}

						gammaPP(j) = gamma(j + 1) + sum;
					}

					x += gamma(1) * rHat.col(0);
					rHat.col(0) -= gammaP(L) * rHat.col(L);
					uHat.col(0) -= gamma(L) * uHat.col(L);

					x += rHat.block(0, 1, N, L - 1) * gammaPP.segment(1, L - 1);
					uHat.col(0) -= uHat.block(0, 1, N, L - 1) * gamma.segment(1, L - 1);
					rHat.col(0) -= rHat.block(0, 1, N, L - 1) * gammaP.segment(1, L - 1);
				}

#elif BICGSTABL_ALGORITHM == 1

				if (bicg_convergence == false) //(L == 1)
				{
					/*
						//TODO: Compare this with the Fortran code step by step in the Enhanced bicgstabL paper, before any hard conclusions are drawn about this section.
					*/
					/*
						The polynomial/minimize residual step.

						QR Householder method for argmin is more stable than (modified) Gram-Schmidt, in the sense that there is less loss of orthogonality.
						It is more accurate than solving the normal equations, since the normal equations scale with condition number squared.

						This method can also be used if L>1,  however the paper of Fokkema recommends the convex combination method to maintain convergence.
					*/
					VectorType gamma(L);
					// gamma = ((rHat.block(0, 1, N, L)).adjoint() * (rHat.block(0, 1, N,
					// 				L))).llt().solve((rHat.block(0, 1, N, L)).adjoint() * rHat.col(0));

					// DenseMatrixType Z(L, L);

					// for (Index i = 0; i < L; ++i)
					// {
					// 	for (Index j = 0; j <= i; ++j)
					// 	{
					// 		Z(i, j) = rHat.col(i + 1).dot(rHat.col(j + 1));

					// 	}

					// }

					//Z = (Z.template selfadjointView<Eigen::Lower>());

					//gamma = Z.llt().solve(Z.col(0));
					//gamma = Z.llt().solve(rHat.block(0, 1, N, L).adjoint() * rHat.col(0));
					gamma = (rHat.block(0, 1, N, L)).householderQr().solve(rHat.col(0));
					x += rHat.block(0, 0, N, L) * gamma;
					rHat.col(0) -= rHat.block(0, 1, N, L) * gamma;
					uHat.col(0) -= uHat.block(0, 1, N, L) * gamma;
					zeta = rHat.col(0).norm();
					omega = gamma(L - 1);
				}

#elif BICGSTABL_ALGORITHM == 2

				if (L != 1 && bicg_convergence == false) //else
				{
					//TODO: ADD case for L==1, refer to householder method, or analyitical bicgstab method
					/*
					        Convex combination step

					        This takes a combination of Minimum and Orthogonal residual polynomials,
						resulting in a more stable determination of BiCG coefficients [Fokkema].
					*/
					DenseMatrixType Z(L + 1, L + 1);

					//Z = rHat.adjoint() * rHat;
					for (Index i = 0; i <= L; ++i)
					{
						for (Index j = 0; j <= i; ++j)
						{
							Z(i, j) = rHat.col(i).dot(rHat.col(j));
						}
					}

					// //Z is Hermitian, therefore only one half has to be computed, the other half can be filled in, saving several DOTs.
					Z = (Z.template selfadjointView<Eigen::Lower>());
					DenseMatrixType Z0 = Z.block(1, 1, L - 1,
												 L - 1); //Copy to ensure there is no in place decomposition taking place
					//Z = (rHat.block(0, 1, N, L)).transpose() * rHat.block(0, 1, N, L);
					//Z = rHat.adjoint() * rHat;
					//TODO: strictly upper/lower doesnt work, so this is performing some unneeded work.

					//Determine the coefficients for the polynomials.
					VectorType y0(L + 1);
					VectorType yL(L + 1);
					y0(0) = -1.0;
					y0(L) = 0.0;
					yL(0) = 0.0;
					yL(L) = -1.0;

					//The block of Z is used for both y0 and yL, so the decomposition can be recycled.
					//PartialPivLU<DenseMatrixType> lu_decomposition(Z.block(1, 1, L - 1, L - 1));
					LLT<DenseMatrixType> lu_decomposition(Z0);
					y0.block(1, 0, L - 1, 1) = lu_decomposition.solve(Z.block(1, 0, L - 1, 1));
					yL.block(1, 0, L - 1, 1) = lu_decomposition.solve(Z.block(1, L, L - 1, 1));

					//y0.block(1, 0, L - 1, 1) =
					//yL.block(1, 0, L - 1, 1) = //Dit toch gwn met householder doen ipv normal eqns

					Scalar kappa0 = y0.adjoint() * (Z * y0);
					Scalar kappaL = yL.adjoint() * (Z * yL);
					kappa0 = sqrt(kappa0);
					kappaL = sqrt(kappaL);

					//Combine the MR and OR versions.
					Scalar rho = yL.adjoint() * (Z * y0);
					rho = rho / (kappa0 * kappaL);
					RealScalar abs_rho = abs(rho);
					Scalar gamma_hat = (rho / abs_rho) * (abs_rho > 0.7 ? abs_rho : 0.7);
					y0 = y0 - gamma_hat * (kappa0 / kappaL) * yL;

					//The norm of the residual can computed be more efficiently from y0 and Z than from rHat.col(0).norm().
					rho = y0.adjoint() * (Z * y0);
					zeta = sqrt(abs(rho));

					//if (bicg_convergence && zeta - zeta != 0)
					if (zeta - zeta != 0)
					{
/*
							Convergence was achieved during BiCG step, this leads to extremely small values of rHat, and Z.
							As a result the polynomial step can break down, however x is likely to still be usable.
						*/
#if BICGSTABL_DEBUG_INFO
						std::cout << "zeta breakdown" << std::endl;
						//std::cout << "rHat: \n" << rHat << std::endl;
						//std::cout << "x: \n" << x << std::endl;
						//std::cout << "uHat: \n" << uHat << std::endl;
						std::cout << "y0: \n"
								  << y0 << std::endl;
						std::cout << "yL: \n"
								  << yL << std::endl;
						std::cout << "Z: \n"
								  << Z << std::endl;
						std::cout << "zeta: \n"
								  << zeta << std::endl;
#endif
						zeta = rHat.col(0).norm();
						break;
					}
					else
					{
						//Update
						omega = y0(L);
						uHat.col(0) -= uHat.block(0, 1, N, L) * y0.block(1, 0, L, 1);
						x += rHat.block(0, 0, N, L) * y0.block(1, 0, L, 1);
						rHat.col(0) -= rHat.block(0, 1, N, L) * y0.block(1, 0, L, 1);
					}
				}

#endif

				//TODO: Duplicate update code can be removed for the L=1 and L!=1 case.
				//TODO: Use analytical expression instead of householder for L=1.
				k += L;
				//k += 1;

				/*
					Reliable update part

					The recursively computed residual can deviate from the actual residual after several iterations. However, computing the
					residual from the definition costs extra MVs and should not be done at each iteration.
					The reliable update strategy computes the true residual from the definition: r=b-A*x at strategic intervals.
					Furthermore a "group wise update" strategy is used to combine updates, which improves accuracy.
				*/

#if BICGSTABL_REL_UPDATE
				Mx = std::max(Mx, zeta); //Maximum norm of residuals since last update of x.
				Mr = std::max(Mr, zeta); //Maximum norm of residuals since last computation of the true residual.

				if (zeta < delta * zeta0 && zeta0 <= Mx)
				{
					update_app = true;
				}

				if (update_app || (zeta < delta * Mr && zeta0 <= Mr))
				{
					compute_res = true;
				}

				if (bicg_convergence)
				{
					update_app = true;
					compute_res = true;
					bicg_convergence = false;
				}

				if (compute_res)
				{
#if BICGSTABL_PRECOND == 0
					rHat.col(0) = b_prime - precond.solve(mat * x);
#elif BICGSTABL_PRECOND == 1
					//rHat.col(0) = b_prime - mat * x; //Fokkema paper pseudocode
					//Fokkema paper Fortan code L250-254
					rHat.col(0) = mat * x;
					rHat.col(0) = precond.solve(rHat.col(0));
					rHat.col(0) = b_prime - rHat.col(0);
#endif

					zeta = rHat.col(0).norm();
					Mr = zeta;

					if (update_app)
					{
						//After the group wise update, the original problem is translated to a shifted one.
						x_prime = x_prime + x;
						x.setZero();
						b_prime = rHat.col(0);
						Mx = zeta;
					}
				}

				compute_res = false;
				update_app = false;
#endif
#if BICGSTABL_DEBUG_INFO
				std::cout << "k: " << k << "res:" << zeta / zeta0 << std::endl;
#endif
			}

			//Convert internal variable to the true solution vector x
			x = x_prime + x;
#if BICGSTABL_PRECOND == 0
			x = precond.solve(x);
#endif
			tol_error = zeta / zeta0;
			//tol_error = zeta;
			//tol_error = sqrt(rHat.col(0).squaredNorm() / rhs_sqnorm);
			//tol_error = (mat * x - rhs).norm() / rhs.norm();
			iters = k;

#if BICGSTABL_DEBUG_INFO
			//Print experimental info
			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
			std::cout << "Solver time: " << time_span.count() << " seconds" << std::endl;
			std::cout << "#iterations:     " << k << std::endl;
			std::cout << "Estimated error: " << tol_error << std::endl;
			std::cout << "True error:      " << (mat * x - rhs).norm() / rhs.norm();
#endif
			return true;
		}

	} // namespace internal

	template < typename _MatrixType,
			   typename _Preconditioner = DiagonalPreconditioner<typename _MatrixType::Scalar> >
	class BiCGSTABL;

	namespace internal
	{

		template <typename _MatrixType, typename _Preconditioner>
		struct traits<Eigen::BiCGSTABL<_MatrixType,_Preconditioner> >
		{
			typedef _MatrixType MatrixType;
			typedef _Preconditioner Preconditioner;
		};

	} // namespace internal

	template <typename _MatrixType, typename _Preconditioner>
	class BiCGSTABL : public IterativeSolverBase<BiCGSTABL<_MatrixType, _Preconditioner> >
	{
		typedef IterativeSolverBase<BiCGSTABL> Base;
		using Base::m_error;
		using Base::m_info;
		using Base::m_isInitialized;
		using Base::m_iterations;
		using Base::matrix;
		Index m_L = 2;

	public:
		typedef _MatrixType MatrixType;
		typedef typename MatrixType::Scalar Scalar;
		typedef typename MatrixType::RealScalar RealScalar;
		typedef _Preconditioner Preconditioner;

	public:
		/** Default constructor. */
		BiCGSTABL() : Base() {}

		/**     Initialize the solver with matrix \a A for further \c Ax=b solving.

			        This constructor is a shortcut for the default constructor followed
			        by a call to compute().

			        \warning this class stores a reference to the matrix A as well as some
			        precomputed values that depend on it. Therefore, if \a A is changed
			        this class becomes invalid. Call compute() to update it with the new
			        matrix A, or modify a copy of A.
			*/
		template <typename MatrixDerived>
		explicit BiCGSTABL(const EigenBase<MatrixDerived> &A) : Base(A.derived()) {}

		~BiCGSTABL() {}

/** \internal */
/**     Loops over the number of columns of b and does the following:
			        1. sets the tolerence and maxIterations
			        2. Calls the function that has the core solver routine
			*/
#if BICGSTABL_IN_LIBRARY == 0
		template <typename Rhs, typename Dest>
		void _solve_with_guess_impl(const Rhs &b, Dest &x) const
		{
			_solve_vector_with_guess_impl(b, x);
		}
#endif

		template <typename Rhs, typename Dest>
		void _solve_vector_with_guess_impl(const Rhs &b, Dest &x) const
		{
			m_iterations = Base::maxIterations();
			m_error = Base::m_tolerance;

			bool ret = internal::BiCGSTABL(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error,
										   m_L);
#if BICGSTABL_DEBUG_INFO
			std::cout << "ret: " << ret << std::endl;
			std::cout << "m_error: " << m_error << std::endl;
			std::cout << "Base::m_tolerance: " << Base::m_tolerance << std::endl;
#endif

			if (ret == false)
			{
				m_info = NumericalIssue;
				x.setZero(); //x=nan does not pass Eigen's tests even if m_info=NumericalIssue :)
				m_error = ((std::isfinite)(m_error) ? m_error : 1.0);
			}
			else
			{
				m_info = (m_error <= Base::m_tolerance) ? Success
														: NoConvergence;
			}

// m_info = (!ret) ? NumericalIssue
// 	: m_error <= Base::m_tolerance ? Success
// 	: NoConvergence;
//m_info=NumericalIssue;
#if BICGSTABL_DEBUG_INFO
			std::cout << "m_error_returned: " << m_error << std::endl;
			std::cout << "m_info: " << m_info << std::endl;
#endif
			// m_info = (!ret) ? NumericalIssue
			// 	: m_error <= Base::m_tolerance ? Success
			// 	: NoConvergence;
			m_isInitialized = true;
		}

/** \internal */
/** Resizes the x vector to match the dimenstion of b and sets the elements to zero*/
#if BICGSTABL_IN_LIBRARY == 0
		using Base::_solve_impl;
		template <typename Rhs, typename Dest>
		void _solve_impl(const MatrixBase<Rhs> &b, Dest &x) const
		{

			x.resize(this->rows(), b.cols());
			x.setZero();

			_solve_with_guess_impl(b, x);
		}
#endif
		void setL(Index L)
		{
			if (L < 1)
			{
				L = 2;
			}

			m_L = L;
		}

	protected:
	};

} // namespace Eigen

#endif /* EIGEN_BICGSTABL_H */
