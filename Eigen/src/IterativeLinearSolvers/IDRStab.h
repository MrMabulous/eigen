/*
	C.E.M. Schoutrop

	The IDR(S)Stab(L) method is a combination of IDR(S) and BiCGStab(L)
	//TODO: elaborate what this improves over BiCGStab here

	Possible optimizations (PO):
	-See //PO: notes in the code

	This implementation of IDRStab is based on
		1. Aihara, K., Abe, K., & Ishiwata, E. (2014). A variant of IDRstab with reliable update strategies for solving sparse linear systems. Journal of Computational and Applied Mathematics, 259, 244-258. doi:10.1016/j.cam.2013.08.028
		2. Aihara, K., Abe, K., & Ishiwata, E. (2015). Preconditioned IDRStab Algorithms for Solving Nonsymmetric Linear Systems. International Journal of Applied Mathematics, 45(3).
		3. Saad, Y. (2003). Iterative Methods for Sparse Linear Systems: Second Edition. Philadelphia, PA: SIAM.
		4. Sonneveld, P., & Van Gijzen, M. B. (2009). IDR(s): A Family of Simple and Fast Algorithms for Solving Large Nonsymmetric Systems of Linear Equations. SIAM Journal on Scientific Computing, 31(2), 1035-1062. doi:10.1137/070685804
		5. Sonneveld, P. (2012). On the convergence behavior of IDR (s) and related methods. SIAM Journal on Scientific Computing, 34(5), A2576-A2598.

	Special acknowledgement to Mischa Senders for his work on an initial reference implementation of this algorithm in MATLAB, to Lex Kuijpers for testing the IDRStab implementation and to Adithya Vijaykumar for providing the framework for this solver.

        Right-preconditioning based on Ref. 3 is implemented here.
*/

#ifndef idrstab_h
#define idrstab_h

#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <cmath>

//TODO: Remove the debug info, and obsolete options in final version.
#define IDRSTAB_DEBUG_INFO 0 	//Print info to console about the problem being solved.
#define SAVE_FAILS 0		//Save matrices that didn't converge
#define IDRSTAB_ACCURATE true	//true: Accurate version to pass unit tests, false: time-optimized version that works for probably every reasonable case.
#define IDRSTAB_INF_NORM false	//true: Use the faster, slightly more strict and not normally used, infinity norm where possible. False: use the standard 2-norm.

#if SAVE_FAILS
#include <chrono>
#include "eigen3/unsupported/Eigen/src/SparseExtra/MarketIO.h"
#endif
#if IDRSTAB_DEBUG_INFO > 0
#include <chrono>
#endif
namespace Eigen
{

	namespace internal
	{

		template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
		bool idrstab(const MatrixType& mat, const Rhs& rhs, Dest& x,
			const Preconditioner& precond, Index& iters,
			typename Dest::RealScalar& tol_error, Index L, Index S)
		{

			/*
				Setup and type definitions.
			*/
			typedef typename Dest::Scalar Scalar;
			typedef typename Dest::RealScalar RealScalar;
			typedef Matrix<Scalar, Dynamic, 1> VectorType;
			typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrixTypeCol;
			typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> DenseMatrixTypeRow;
			#if IDRSTAB_DEBUG_INFO >0
			std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
			std::cout << "Matrix size: " << mat.rows() << std::endl;
			std::cout << mat.IsRowMajor << std::endl;
			//std::cout << "rhs\n" << rhs << std::endl;
			//std::cout << "A\n" << mat << std::endl;
			#endif

			const Index N = x.rows();

			Index k = 0; //Iteration counter
			const Index maxIters = iters;

			//PO: sqrNorm() saves 1 sqrt calculation
			#if IDRSTAB_INF_NORM
			const RealScalar rhs_norm = rhs.template lpNorm<Eigen::Infinity>();
			#else
			const RealScalar rhs_norm = rhs.norm();
			#endif
			const RealScalar tol2 = tol_error * rhs_norm;

			if (rhs_norm == 0)
			{
				/*
					If b==0, then the exact solution is x=0.
					rhs_norm is needed for other calculations anyways, this exit is a freebie.
				*/
				/*
					exit
				*/
				#if IDRSTAB_DEBUG_INFO >1
				std::cout << "rhs_norm==0 exit" << std::endl;
				#endif
				x.setZero();
				tol_error = 0.0;
				return true;
			}
			//Construct decomposition objects beforehand.
			#if IDRSTAB_ACCURATE
			ColPivHouseholderQR<DenseMatrixTypeCol> qr_solver;
			FullPivLU<DenseMatrixTypeCol> lu_solver;
			#else
			HouseholderQR<DenseMatrixTypeCol> qr_solver;
			PartialPivLU<DenseMatrixTypeCol> lu_solver;
			#endif
			/*
				Solving small S x S systems:
				From numerical experiment is follows that in some cases (when the tolerance is really too low) inverting sigma is delicate,
				since the resulting alpha will be close to zero. If this is not done accurately enough the algorithm breaks down.
				Based on the benchmark data from: https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html
				it is concluded that for the small (SxS) systems the difference between PartialPivLU and FullPivLU is negligible.
				HouseholderQR methods can be used for these systems as well, however it is about 2x as slow.

				Solving least squares problem in the polynomial step:
				ColPivHouseholderQR is sufficiently accurate, based on the benchmark data this is about 2x as fast compared to FillPivHouseholderQR.

				Resulting strategy:
				-Use the more accurate, but slower FullPivLU version for small systems
				-Use ColPivHouseholderQR for the minimal residual step.
			*/

			if (S >= N || L >= N)
			{
				/*
					The matrix is very small, or the choice of L and S is very poor
					in that case solving directly will be best.
				*/
				/*
					Exit
				*/
				#if IDRSTAB_DEBUG_INFO >1
				std::cout << "Trivial matrix exit" << std::endl;
				#endif
				lu_solver.compute(DenseMatrixTypeRow(mat));
				x = lu_solver.solve(rhs);
				#if IDRSTAB_INF_NORM
				tol_error = (rhs - mat * x).template lpNorm<Eigen::Infinity>() / rhs_norm;
				#else
				tol_error = (rhs - mat * x).norm() / rhs_norm;
				#endif
				return true;
			}

			//Define maximum sizes to prevent any reallocation later on.
			VectorType u(N * (L + 1));
			VectorType r(N * (L + 1));
			DenseMatrixTypeCol V(N * (L + 1), S);
			DenseMatrixTypeCol U(N * (L + 1), S);
			DenseMatrixTypeCol rHat(N, L + 1);

			VectorType alpha(S);
			VectorType gamma(L);
			VectorType update(N);

			/*
				Main IDRStab algorithm
			*/
			//Set up the initial residual
			r.head(N) = rhs - mat * precond.solve(x);
			#if IDRSTAB_INF_NORM
			tol_error = r.head(N).template lpNorm<Eigen::Infinity>();
			#else
			tol_error = r.head(N).norm();
			#endif

			DenseMatrixTypeRow h_FOM(S, S - 1);
			h_FOM.setZero();

			/*
				Determine an initial U matrix of size N x S
			*/
			for (Index q = 0; q < S; ++q)
			{
				//By default S=4, q!=0 case is more likely, this ordering is better for branch prediction.
				//Arnoldi-like process to generate a set of orthogonal vectors spanning {u,A*u,A*A*u,...,A^(S-1)*u}.
				//This construction can be combined with the Full Orthogonalization Method (FOM) from Ref.3 to provide a possible early exit with no additional MV.
				if (q != 0)
				{
					/*
						Original Gram-Schmidt orthogonalization strategy from Ref. 1:
					*/
					//u.head(N) -= U.topLeftCorner(N, q) * (U.topLeftCorner(N, q).adjoint() * u.head(N));

					/*
						Modified Gram-Schmidt strategy:
						Note that GS and MGS are mathematically equivalent, they are NOT numerically equivalent.

						Eventough h is a scalar, converting the dot product to Scalar is not supported:
						http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1610
					*/
					VectorType w = mat * precond.solve(u.head(N));
					for (Index i = 0; i < q; ++i)
					{
						//"Normalization factor" (v is normalized already)
						VectorType v = U.block(0, i, N, 1);

						//"How much do v and w have in common?"
						h_FOM(i, q - 1) = v.dot(w);

						//"Subtract the part they have in common"
						w = w - h_FOM(i, q - 1) * v;
					}
					u.head(N) = w;
					h_FOM(q, q - 1) = u.head(N).norm();

					if (std::abs(h_FOM(q, q - 1)) != 0.0)
					{
						/*
							This only happens if u is NOT exactly zero. In case it is exactly zero
							it would imply that that this u has no component in the direction of the current residual.

							By then setting u to zero it will not contribute any further (as it should).
							Whereas attempting to normalize results in division by zero.

							Such cases occur if:
							1. The basis of dimension <S is sufficient to exactly solve the linear system.
							I.e. the current residual is in span{r,Ar,...A^{m-1}r}, where (m-1)<=S.
							2. Two vectors vectors generated from r, Ar,... are (numerically) parallel.

							In case 1, the exact solution to the system can be obtained from the "Full Orthogonalization Method" (Algorithm 6.4 in the book of Saad), without any additional MV.

							Contrary to what one would suspect, the comparison with ==0.0 for floating-point types is intended here.
							Any arbritary non-zero u is fine to continue, however if u contains either NaN or Inf the algorithm will break down.
						*/
						u.head(N) /= h_FOM(q, q - 1);
					}

				}
				else
				{
					u.head(N) = r.head(N);
					u.head(N) /= u.head(N).norm();
				}

				U.block(0, q, N, 1) = u.head(N);
			}
			if (S > 1)
			{
				/*
					Check for early FOM exit.
				*/
				//Scalar beta = tol_error; //r.head(N).norm(): This is expected to be tol_error at this point!
				Scalar beta = r.head(N).norm(); //This must be the actual length of the vector, not the infinity norm estimate.
				VectorType e1 = VectorType::Zero(S - 1);
				e1(0) = beta;
				lu_solver.compute(h_FOM.topLeftCorner(S - 1, S - 1));
				VectorType y = lu_solver.solve(e1);
				VectorType x2 = x + U.block(0, 0, N, S - 1) * y;

				//Using proposition 6.7 in Saad, one MV can be saved to calculate the residual
				#if IDRSTAB_INF_NORM
				RealScalar FOM_residual = (h_FOM(S - 1, S - 2) * y(S - 2) * U.block(0, S - 1, N, 1)).template lpNorm<Eigen::Infinity>();
				#else
				RealScalar FOM_residual = (h_FOM(S - 1, S - 2) * y(S - 2) * U.block(0, S - 1, N, 1)).norm();
				#endif

				#if IDRSTAB_DEBUG_INFO >1
				std::cout << "h_FOM\n" << h_FOM << std::endl;
				std::cout << "h_FOM(S-1, S - 2)\n" << h_FOM(S - 1, S - 2) << std::endl;
				std::cout << "y(S-1)\n" << y(S - 2) << std::endl;
				std::cout << "y\n" << y << std::endl;
				#if IDRSTAB_INF_NORM
				std::cout << "U.block(0,S-1,N,1).template lpNorm<Eigen::Infinity>()\n" << U.block(0, S - 1, N, 1).template lpNorm<Eigen::Infinity>() << std::endl;
				RealScalar FOM_actual_res = (rhs - mat * precond.solve(x2)).template lpNorm<Eigen::Infinity>() / rhs_norm;
				#else
				std::cout << "U.block(0,S-1,N,1).norm()\n" << U.block(0, S - 1, N, 1).norm() << std::endl;
				RealScalar FOM_actual_res = (rhs - mat * precond.solve(x2)).norm() / rhs_norm;
				#endif
				std::cout << "FOM actual residual\n" << FOM_actual_res << std::endl;
				std::cout << "FOM estimated residual from Saad method\n " << FOM_residual / rhs_norm << std::endl;
				#endif
				if (FOM_residual < tol2)
				{
					/*
						Exit, the FOM algorithm was already accurate enough
					*/
					iters = k;
					x = precond.solve(x2);
					tol_error = FOM_residual / rhs_norm;
					#if IDRSTAB_DEBUG_INFO >0
					std::cout << "Speculative FOM exit" << std::endl;
					std::cout << "Estimated relative residual: " << tol_error << std::endl;
					#if IDRSTAB_INF_NORM
					std::cout << "True relative residual:      " << (mat * x - rhs).template lpNorm<Eigen::Infinity>() / rhs.template lpNorm<Eigen::Infinity>() <<
						std::endl;
					#else
					std::cout << "True relative residual:      " << (mat * x - rhs).norm() / rhs.norm() << std::endl;
					#endif

					#endif
					return true;
				}
			}

			#if IDRSTAB_DEBUG_INFO >1
			//Columns of U should be orthonormal
			std::cout << "Check orthonormality U\n" <<
				U.block(0, 0, N, S).adjoint()*U.block(0, 0, N, S) << std::endl;
			//h_FOM should not contain any NaNs
			std::cout << "h_FOM:\n" << h_FOM << std::endl;
			#endif

			/*
				Select an initial (N x S) matrix R0.
				1. Generate random R0, orthonormalize the result.
				2. This results in R0, however to save memory and compute we only need the adjoint of R0. This is given by the matrix R_T.\
				Additionally, the matrix (mat.adjoint()*R_tilde).adjoint()=R_tilde.adjoint()*mat by the anti-distributivity property of the adjoint.
				This results in AR_T, which is constant if R_T does not have to be regenerated and can be precomputed. Based on reference 4,
				this has zero probability in exact arithmetic. However in practice it does (extremely infrequently) occur,
				most notably for small matrices.
			*/
			//PO: To save on memory consumption identity can be sparse
			//PO: can this be done with colPiv/fullPiv version as well? This would save 1 construction of a HouseholderQR object

			//Original IDRStab and Kensuke choose S random vectors:
			HouseholderQR<DenseMatrixTypeCol> qr(DenseMatrixTypeCol::Random(N, S));
			DenseMatrixTypeRow R_T = (qr.householderQ() * DenseMatrixTypeCol::Identity(N,
						S)).adjoint();
			DenseMatrixTypeRow AR_T = DenseMatrixTypeRow(R_T * mat);

			#if IDRSTAB_DEBUG_INFO >1
			std::cout << "Check orthonormality R_T\n" <<
				R_T* R_T.adjoint() << std::endl;
			#endif

			//Pre-allocate sigma, this space will be recycled without additional allocations.
			DenseMatrixTypeCol sigma(S, S);

			Index rt_counter = k;				//Iteration at which R_T was reset last
			bool reset_while = false;			//Should the while loop be reset for some reason?

			#if IDRSTAB_ACCURATE
			VectorType Q(S, 1);				//Vector containing the row-scaling applied to sigma
			VectorType P(S, 1);				//Vector containing the column-scaling applied to sigma
			DenseMatrixTypeCol QAP(S, S);			//Scaled sigma
			bool repair_flag = false;
			RealScalar residual_0 = tol_error;
			//bool apply_r_exit = false;
			#endif

			while (k < maxIters)
			{

				for (Index j = 1; j <= L; ++j)
				{
					//Cache some indexing variables that occur frequently and are constant.
					const Index Nj = N * j;
					const Index Nj_plus_1 = N * (j + 1);
					const Index Nj_min_1 = N * (j - 1);

					/*
						The IDR Step
					*/
					//Construction of the sigma-matrix, and the decomposition of sigma.
					for (Index i = 0; i < S; ++i)
					{
						sigma.col(i).noalias() = AR_T * precond.solve(U.block(Nj_min_1, i, N, 1));
					}
					/*
					        Suspected is that sigma could be badly scaled, since causes alpha~=0, but the
					        update vector is not zero. To improve stability we scale with absolute row and col sums first.
						Sigma can become badly scaled (but still well-conditioned) for the ASML matrices, where sigma~=0 happens.
						A bad sigma also happens if R_T is not chosen properly, for example if R_T is zeros sigma would be zeros as well.
						The effect of this is a left-right preconditioner, instead of solving Ax=b, we solve
						Q*A*P*inv(P)*b=Q*b.
					*/
					#if IDRSTAB_ACCURATE
					Q = (sigma.cwiseAbs().rowwise().sum()).cwiseInverse();	//Calculate absolute inverse row sum
					QAP = Q.asDiagonal() * sigma;				//Scale with inverse absolute row sums
					P = (QAP.cwiseAbs().colwise().sum()).cwiseInverse();	//Calculate absolute inverse column sum
					QAP = QAP * P.asDiagonal();				//Scale with inverse absolute column sums
					lu_solver.compute(QAP);
					#else
					lu_solver.compute(sigma);
					#endif
					//Obtain the update coefficients alpha
					if (j != 1)
					{
						//alpha=inverse(sigma)*(AR_T*r_{j-2})
						#if IDRSTAB_ACCURATE
						alpha.noalias() = lu_solver.solve(Q.asDiagonal() * AR_T * precond.solve(r.segment(N * (j - 2), N)));
						#else
						alpha.noalias() = lu_solver.solve(AR_T * precond.solve(r.segment(N * (j - 2), N)));
						#endif
					}
					else
					{
						//alpha=inverse(sigma)*(R_T*r_0);
						#if IDRSTAB_ACCURATE
						alpha.noalias() = lu_solver.solve(Q.asDiagonal() * R_T * r.head(N));
						#else
						alpha.noalias() = lu_solver.solve(R_T * r.head(N));
						#endif
					}
					//Unscale the solution
					#if IDRSTAB_ACCURATE
					alpha = P.asDiagonal() * alpha;
					#endif
					//TODO: Check if another badly scaled scenario exists

					double old_res = tol_error;

					//Obtain new solution and residual from this update
					update.noalias() = U.topRows(N) * alpha;
					r.head(N) -=  mat * precond.solve(update);
					x += update;

					for (Index i = 1; i <= j - 2; ++i)
					{
						//This only affects the case L>2
						r.segment(N * i, N) -= U.block(N * (i + 1), 0, N, S) * alpha;
					}
					if (j > 1)
					{
						//r=[r;A*r_{j-2}]
						r.segment(Nj_min_1, N).noalias() = mat * precond.solve(r.segment(N * (j - 2), N));
					}
					#if IDRSTAB_DEBUG_INFO >1
					#if IDRSTAB_INF_NORM
					std::cout << "r.head(N).template lpNorm<Eigen::Infinity>()380: " << r.head(N).template lpNorm<Eigen::Infinity>() << std::endl;
					std::cout << "update.norm()" << update.template lpNorm<Eigen::Infinity>() << std::endl;
					std::cout << "update.norm()/old_res" << update.template lpNorm<Eigen::Infinity>() / old_res << std::endl;
					#else
					std::cout << "r.head(N).norm()380: " << r.head(N).norm() << std::endl;
					std::cout << "update.norm()" << update.norm() << std::endl;
					std::cout << "update.norm()/old_res" << update.norm() / old_res << std::endl;
					#endif
					std::cout << "QAP\n" << QAP << std::endl;
					std::cout << "sigma\n" << sigma << std::endl;
					std::cout << "alpha380: " << alpha << std::endl;

					//If alpha380 is the zero vector, this can be caused by R0 being no good.
					//It turns out that the update can be significant, even if alpha~=0,
					//this suggests sigma can just be really badly scaled
					//This was also found in the early version by Mischa, but thought it was resolved.
					//If R_T=identity, then sigma is all zeros.
					//A "bad" sigma can be obtained by choosing:
					//R_T = DenseMatrixTypeRow::Identity(S, N);
					//R_T = R_T+1e-12*DenseMatrixTypeRow::Random(S, N);
					#endif
					#if IDRSTAB_DEBUG_INFO >2
					VectorType x_unscaled;
					VectorType x_scaled;
					VectorType b = VectorType::Random(S, 1);
					x_unscaled = sigma.fullPivLu().solve(b);
					x_scaled = P.asDiagonal() * QAP.fullPivLu().solve(Q.asDiagonal() * b);
					std::cout << "x_unscaled\n" << x_unscaled << std::endl;
					std::cout << "x_scaled\n" << x_scaled << std::endl;
					#endif
					#if IDRSTAB_INF_NORM
					tol_error = r.head(N).template lpNorm<Eigen::Infinity>();
					#else
					tol_error = r.head(N).norm();
					#endif
					if (tol_error < tol2)
					{
						//If at this point the algorithm has converged, exit.
						reset_while = true;
						break;
					}
					#if IDRSTAB_ACCURATE
					if (repair_flag==false && tol_error > 10 * residual_0)
					{
						//Sonneveld's repair flag suggestion from [5]
						//This massively reduces problems with false residual estimates (if they'd occur)
						repair_flag = true;
						#if IDRSTAB_DEBUG_INFO > 0
						std::cout << "repair flag set true, iter: " << k << std::endl;
						#endif
					}
					if (repair_flag && 1000 * tol_error < residual_0)
					{
						r.head(N) = rhs - mat * precond.solve(x);
						repair_flag = false;
						#if IDRSTAB_DEBUG_INFO > 0
						std::cout << "repair flag reset iter: " << k << std::endl;
						#endif
					}
					#endif
					bool reset_R_T = false;
					#if IDRSTAB_INF_NORM
					if (alpha.template lpNorm<Eigen::Infinity>() * rhs_norm < S * NumTraits<Scalar>::epsilon() * old_res)
					#else
					if (alpha.norm() * rhs_norm < S * NumTraits<Scalar>::epsilon() * old_res)
					#endif
					{
						//This would indicate the update computed by alpha did nothing much to decrease the residual
						//apparantly we've also not converged either.
						//TODO: Check if there is some better criterion, the current one is a bit handwavy.
						reset_R_T = true;
					}

					if (reset_R_T)
					{
						if (k - rt_counter > 0)
						{
							/*
							        Only regenerate if it didn't already happen this iteration.
							*/
							#if IDRSTAB_DEBUG_INFO >1
							std::cout << "Generating new R_T" << std::endl;
							#endif
							//Choose new R0 and try again
							qr.compute(DenseMatrixTypeCol::Random(N, S));
							R_T = (qr.householderQ() * DenseMatrixTypeCol::Identity(N, S)).transpose(); //.adjoint() vs .transpose() makes no difference, R_T is random anyways.
							/*
							        Additionally, the matrix (mat.adjoint()*R_tilde).adjoint()=R_tilde.adjoint()*mat by the anti-distributivity property of the adjoint.
							        This results in AR_T, which can be precomputed.
							*/
							AR_T = DenseMatrixTypeRow(R_T * mat);
							j = 0;
							rt_counter = k;
							continue;
							//reset_while = true;
							//break;
						}
					}
					bool break_normalization=false;
					for (Index q = 1; q <= S; ++q)
					{
						if (q != 1)
						{
							//u=[u_1;u_2;...;u_j]
							u.head(Nj) = u.segment(N, Nj);
						}
						else
						{
							//u = r;
							u.head(Nj_plus_1) = r.topRows(Nj_plus_1);
						}
						//Obtain the update coefficients beta implicitly
						//beta=lu_sigma.solve(AR_T * u.block(Nj_min_1, 0, N, 1)
						#if IDRSTAB_ACCURATE
						u.head(Nj) -=  U.topRows(Nj) * P.asDiagonal() * lu_solver.solve(Q.asDiagonal() * AR_T * precond.solve(u.segment(Nj_min_1, N)));
						#else
						u.head(Nj) -=  U.topRows(Nj) * lu_solver.solve(AR_T * precond.solve(u.segment(Nj_min_1, N)));
						#endif
						//u=[u;Au_{j-1}]
						u.segment(Nj, N).noalias() = mat * precond.solve(u.segment(Nj_min_1, N));

						//Orthonormalize u_j to the columns of V_j(:,1:q-1)
						if (q > 1)
						{
							/*
								Original Gram-Schmidt-like procedure to make u orthogonal to the columns of V from Ref. 1.

								The vector mu from Ref. 1 is obtained implicitly:
								mu=V.block(Nj, 0, N, q - 1).adjoint() * u.block(Nj, 0, N, 1).
							*/
							#if IDRSTAB_ACCURATE==false
							u.head(Nj_plus_1) -= V.topLeftCorner(Nj_plus_1, q - 1) * (V.block(Nj, 0, N, q - 1).adjoint() * u.segment(Nj, N));
							#else
							/*
								The same, but using MGS instead of GS
							*/
							DenseMatrixTypeCol h(1, 1); //Eventhough h should be Scalar, casting from a 1x1 matrix to Scalar is not supported.
							for (Index i = 0; i <= q - 2; ++i)
							{
								//"Normalization factor"
								h = V.block(Nj, i, N, 1).adjoint() * V.block(Nj, i, N, 1);

								//"How much do u and V have in common?"
								h = V.block(Nj, i, N, 1).adjoint() * u.segment(Nj, N) / h(0, 0);

								//"Subtract the part they have in common"
								u.head(Nj_plus_1) -= h(0, 0) * V.block(0, i, Nj_plus_1, 1);
								//std::cout << "h\n" << h << std::endl;
							}
							#endif
						}
						#if IDRSTAB_ACCURATE
						//Normalize u and assign to a column of V
						Scalar normalization_constant = u.block(Nj, 0, N, 1).norm();

						if (normalization_constant != 0.0)
						{
							/*
								If u is exactly zero, this will lead to a NaN. Small, non-zero u is fine. In the case of NaN the algorithm breaks down,
								eventhough it could have continued, since u zero implies that there is no further update in a given direction.
							*/
							u.head(Nj_plus_1) /= normalization_constant;
						}
						else
						{
							// iters = k;
							// x = precond.solve(x);
							// tol_error = tol_error / rhs_norm;

							// #if IDRSTAB_DEBUG_INFO >0
							// std::cout << "normalization_constant==0 exit" << std::endl;
							// std::cout << "Estimated relative residual: " << tol_error << std::endl;
							// #if IDRSTAB_INF_NORM
							// std::cout << "True relative residual:      " << (mat * x - rhs).template lpNorm<Eigen::Infinity>() / rhs.template lpNorm<Eigen::Infinity>() <<
							// 	std::endl;
							// #else
							// std::cout << "True relative residual:      " << (mat * x - rhs).norm() / rhs.norm() << std::endl;
							// #endif
							u.head(Nj_plus_1).setZero();
							//u.head(Nj_plus_1).setRandom(); //TODO: bij gebrek aan beter >_>
							//u.head(Nj_plus_1)=u.head(Nj_plus_1)/u.head(Nj_plus_1).norm();
							#if IDRSTAB_DEBUG_INFO >0
							std::cout << "normalization_constant==0" << std::endl;
							std::cout << "tol_error/rhs_norm: " <<tol_error/rhs_norm<< std::endl;
							std::cout << "True relative residual: " << (rhs - mat * precond.solve(x)).norm() / rhs.norm() <<std::endl;
							#endif
							//if(true)
							if(tol_error < tol2 || tol_error < 1e4 * NumTraits<Scalar>::epsilon())
							//if(tol_error < tol2)
							{
								//Just quit, we've converged
								iters = k;
								x = precond.solve(x);
								tol_error = tol_error / rhs_norm;
								tol_error = 0.0;
								#if IDRSTAB_DEBUG_INFO >0
								std::cout << "normalization_constant==0 EXIT" << std::endl;
								#endif
								return true;
							}
							//#endif
							//TODO: This happens if A is singular. We may need to return the best solution and quit.
							//Currently this can happen many many times in a row (although never for the unit tests).
							//apply_r_exit = true;
							//break;
							break_normalization=true;
							break;
							//return true; //TODO
							//return false;
						}

						V.block(0, q - 1, Nj_plus_1, 1).noalias() = u.head(Nj_plus_1);
						// if(break_normalization)
						// {
						// 	break;
						// }
						#else
						//Since the segment u.head(Nj_plus_1) is not needed next q-iteration this may be combined into one (Only works for GS method, not MGS):
						V.block(0, q - 1, Nj_plus_1, 1).noalias() = u.head(Nj_plus_1) / u.segment(Nj, N).norm();
						#endif

						#if IDRSTAB_DEBUG_INFO >1
						{
							std::cout << "New u should be orthonormal to the columns of V" << std::endl;
							DenseMatrixTypeCol u_V_orthonormality = V.block(Nj, 0, N, q).adjoint() * u.block(Nj, 0, N, 1);
							std::cout << u_V_orthonormality << std::endl; //OK
							if (u_V_orthonormality(0, 0) - u_V_orthonormality(0, 0) != 0.0)
							{

								std::cout << "Internal state:" << std::endl;
								std::cout << "V:\n" << V << std::endl;
								std::cout << "U:\n" << U << std::endl;
								std::cout << "r:\n" << r << std::endl;
								std::cout << "u:\n" << u << std::endl;
								std::cout << "x:\n" << x << std::endl;
								//std::cout<<"mat:\n "<<mat<<std::endl;
								//std::cout<<"rhs:\n "<<rhs<<std::endl;
								//BUG-2-13-Nov-2019
								std::cout << "Segfault: to trigger the debugger" << std::endl;
								*(char*)0 = 0;

							}

						}
						#endif

					}

					#if IDRSTAB_DEBUG_INFO >1
					//This should be identity, since the columns of V are orthonormalized.
					std::cout << "Check if the columns of V are orthonormalized" << std::endl;
					std::cout << V.block(Nj, 0, N, S).adjoint()* V.block(Nj, 0, N, S) << std::endl;
					#endif
					if (break_normalization==false)
					{
						U = V;
					}

				}
				if (reset_while)
				{
					reset_while = false;
					#if IDRSTAB_INF_NORM
					tol_error = r.head(N).template lpNorm<Eigen::Infinity>();
					#else
					tol_error = r.head(N).norm();
					#endif
					if (tol_error < tol2)
					{
						/*
							Slightly early exit by moving the criterion before the update of U,
							after the main while loop the result of that calculation would not be needed.
						*/
						break;
					}
					continue;
				}

				//r=[r;mat*r_{L-1}]
				//Save this in rHat, the storage form for rHat is more suitable for the argmin step than the way r is stored.
				//In Eigen 3.4 this step can be compactly done via: rHat = r.reshaped(N, L + 1);
				r.segment(N * L, N).noalias() = mat * precond.solve(r.segment(N * (L - 1), N));
				for (Index i = 0; i <= L; ++i)
				{
					rHat.col(i) = r.segment(N * i, N);
				}

				/*
					The polynomial step
				*/
				#if IDRSTAB_ACCURATE
				qr_solver.compute(rHat.rightCols(L));
				gamma.noalias() = qr_solver.solve(r.head(N));
				#else
				gamma.noalias() = (rHat.rightCols(L).adjoint() * rHat.rightCols(L)).llt().solve(rHat.rightCols(L).adjoint() * r.head(N));
				#endif

				//Update solution and residual using the "minimized residual coefficients"
				update.noalias() = rHat.leftCols(L) * gamma;
				x += update;
				r.head(N) -= mat * precond.solve(update);

				//Update iteration info
				++k;
				#if IDRSTAB_INF_NORM
				tol_error = r.head(N).template lpNorm<Eigen::Infinity>();
				#else
				tol_error = r.head(N).norm();
				#endif
				#if IDRSTAB_DEBUG_INFO > 0
				// if(apply_r_exit)
				// {
				// 	//u normalization failed, cannot continue with next iteration?
				// 	iters = k;
				// 	x = precond.solve(x);
				// 	tol_error = tol_error / rhs_norm;
				// 	return true;
				// }
				if(k % 10 == 0)
				{
					std::cout << "Current residual: " << tol_error / rhs_norm << " iter: " << k << std::endl;
				}
				#endif
				if (tol_error < tol2)
				{
					//Slightly early exit by moving the criterion before the update of U,
					//after the main while loop the result of that calculation would not be needed.
					#if IDRSTAB_DEBUG_INFO >1
					std::cout << "tol_error break" << std::endl;
					#endif
					break;
				}
				#if IDRSTAB_ACCURATE
				if (repair_flag==false && tol_error > 10 * residual_0)
				{
					//Sonneveld's repair flag suggestion from [5]
					//This massively reduces problems with false residual estimates (if they'd occur)
					repair_flag = true;
					#if IDRSTAB_DEBUG_INFO > 0
					std::cout << "repair flag set true, iter: " << k << std::endl;
					#endif
				}
				if (repair_flag && 1000 * tol_error < residual_0)
				{
					r.head(N) = rhs - mat * precond.solve(x);
					repair_flag = false;
					#if IDRSTAB_DEBUG_INFO > 0
					std::cout << "repair flag reset iter: " << k << std::endl;
					#endif
				}
				#endif

				/*
					U=U0-sum(gamma_j*U_j)
					Consider the first iteration. Then U only contains U0, so at the start of the while-loop
					U should be U0. Therefore only the first N rows of U have to be updated.
				*/
				for (Index i = 1; i <= L; ++i)
				{
					U.topRows(N) -= U.block(N * i, 0, N, S) * gamma(i - 1);
				}

			}

			/*
				Exit after the while loop terminated.
			*/
			iters = k;
			x = precond.solve(x);
			tol_error = tol_error / rhs_norm;
			#if IDRSTAB_DEBUG_INFO >0
			std::cout << "Final exit" << std::endl;
			std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>
				(t2 - t1);
			std::cout << "Solver time: " << time_span.count() << " seconds" << std::endl;
			std::cout << "#iterations:     " << k << std::endl;
			std::cout << "Estimated relative residual: " << tol_error << std::endl;
			#if IDRSTAB_INF_NORM
			std::cout << "True relative residual:      " << (mat * x - rhs).template lpNorm<Eigen::Infinity>() / rhs.template lpNorm<Eigen::Infinity>() <<
				std::endl;
			#else
			std::cout << "True relative residual:      " << (mat * x - rhs).norm() / rhs.norm() <<
				std::endl;
			#endif


			#endif

			return true;
		}

	}

	template< typename _MatrixType,
		typename _Preconditioner = DiagonalPreconditioner<typename _MatrixType::Scalar> >
	class IDRStab;

	namespace internal
	{

		template< typename _MatrixType, typename _Preconditioner>
		struct traits<IDRStab<_MatrixType, _Preconditioner> >
		{
			typedef _MatrixType MatrixType;
			typedef _Preconditioner Preconditioner;
		};

	}

	template< typename _MatrixType, typename _Preconditioner>
	class IDRStab : public IterativeSolverBase<IDRStab<_MatrixType, _Preconditioner> >
	{
			typedef IterativeSolverBase<IDRStab> Base;
			using Base::matrix;
			using Base::m_error;
			using Base::m_iterations;
			using Base::m_info;
			using Base::m_isInitialized;
			Index m_L = 2;
			Index m_S = 4;
		public:
			typedef _MatrixType MatrixType;
			typedef typename MatrixType::Scalar Scalar;
			typedef typename MatrixType::RealScalar RealScalar;
			typedef _Preconditioner Preconditioner;

		public:

			/** Default constructor. */
			IDRStab() : Base() {}

			/**     Initialize the solver with matrix \a A for further \c Ax=b solving.

				This constructor is a shortcut for the default constructor followed
				by a call to compute().

				\warning this class stores a reference to the matrix A as well as some
				precomputed values that depend on it. Therefore, if \a A is changed
				this class becomes invalid. Call compute() to update it with the new
				matrix A, or modify a copy of A.
			*/
			template<typename MatrixDerived>
			explicit IDRStab(const EigenBase<MatrixDerived>& A) : Base(A.derived()) {}

			~IDRStab() {}

			/** \internal */
			/**     Loops over the number of columns of b and does the following:
				1. Sets the tolerance and maxIterations
				2. Calls the function that has the core solver routine
			*/
			// template<typename Rhs, typename Dest>
			// void _solve_with_guess_impl(const Rhs& b, Dest& x) const
			// {
			// 	_solve_vector_with_guess_impl(b, x);
			// }
			// using Base::_solve_impl;
			// template<typename Rhs, typename Dest>
			// void _solve_impl(const MatrixBase<Rhs>& b, Dest& x) const
			// {

			// 	x.resize(this->rows(), b.cols());
			// 	x.setZero();
			// 	//x.setRandom();
			// 	//TODO:This may break if b contains multiple columns, but is probably better than choosing zeros.
			// 	//Random is more reliable than zeros, one can find cases where MATLAB's bicgstabl does not converge with a zero guess either, but does with random.
			// 	//Unit tests pass more often with random compared to zero.
			// 	//x = Base::m_preconditioner.solve(b);
			// 	// for (Index i = 0; i < b.cols(); ++i)
			// 	// {
			// 	// 	x.col(i) = Base::m_preconditioner.solve(b.col(i));
			// 	// }
			// 	_solve_with_guess_impl(b, x);
			// }

			template<typename Rhs, typename Dest>
			void _solve_vector_with_guess_impl(const Rhs& b, Dest& x) const
			{
				m_iterations = Base::maxIterations();
				m_error = Base::m_tolerance;
				bool ret = internal::idrstab(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error,
						m_L, m_S);

				m_info = (!ret) ? NumericalIssue
					: m_error <= 10 * Base::m_tolerance ? Success
					: NoConvergence;

				#if IDRSTAB_DEBUG_INFO >0
				std::cout << "Matrix size: " << b.rows() << std::endl;
				std::cout << "ret: " << ret << std::endl;
				std::cout << "m_error: " << m_error << std::endl;
				std::cout << "Base::m_tolerance: " << Base::m_tolerance << std::endl;
				std::cout << "m_info: " << m_info << std::endl;
				if (m_info != Success)
				{
					if (m_error < Base::m_tolerance * 10)
					{
						std::cout << "Unsatisfactory abort" << std::endl;
					}
					else
					{
						std::cout << "Legitimate abort" << std::endl;
					}

				}
				#endif
				#if SAVE_FAILS
				#if IDRSTAB_INF_NORM
				if ((b - matrix()*x).template lpNorm<Eigen::Infinity>() / b.template lpNorm<Eigen::Infinity>() > 1e-6)
				#else
				if ((b - matrix()*x).norm() / b.norm() > 1e-6)
				#endif
				{
					#if IDRSTAB_INF_NORM
					std::cout << "True residual bad: " << (b - matrix()*x).template lpNorm<Eigen::Infinity>() / b.template lpNorm<Eigen::Infinity>() << std::endl;
					#else
					std::cout << "True residual bad: " << (b - matrix()*x).norm() / b.norm() << std::endl;
					#endif
					using namespace std::chrono;
					typedef typename Dest::Scalar Scalar;
					typedef Matrix<Scalar, Dynamic, 1> VectorType;
					int64_t timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
					Eigen::SparseMatrix<Scalar> A = matrix();
					Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int> > solver;
					// fill A and b;
					// Compute the ordering permutation vector from the structural pattern of A
					solver.analyzePattern(A);
					// Compute the numerical factorization
					solver.factorize(A);
					//Use the factors to solve the linear system
					VectorType x_LU = solver.solve(b);
					Eigen::saveMarket(A, "A_" + std::to_string(timestamp));
					Eigen::saveMarketVector(b, "b_" + std::to_string(timestamp));
					Eigen::saveMarketVector(x_LU, "x_LU_" + std::to_string(timestamp));
					Eigen::saveMarketVector(x, "x_IDRSTAB_" + std::to_string(timestamp));
					std::cout << "Segfault: to trigger the debugger" << std::endl;
					//*(char*)0 = 0;
				}
				#endif

			}

			/** \internal */
			//TODO: Should setL and setS be supported? Or should the defaults L=2,S=4 be adopted?
			void setL(Index L)
			{
				//Truncate to even number
				L = (L / 2) * 2;

				//L must be positive
				L = L < 1 ? 2 : L;

				//L may not exceed 8
				L = L > 8 ? 8 : L;

				m_L = L == 6 ? 4 : L;
			}
			void setS(Index S)
			{
				//Truncate to even number
				S = (S / 2) * 2;

				//S must be positive
				S = S < 1 ? 2 : S;

				//S may not exceed 8
				S = S > 8 ? 8 : S;

				m_S = S == 6 ? 4 : S;
			}

		protected:

	};

}

#endif /* idrstab_h */
