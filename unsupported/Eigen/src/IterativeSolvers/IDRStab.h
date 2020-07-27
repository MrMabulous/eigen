// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Chris Schoutrop <c.e.m.schoutrop@tue.nl>
// Copyright (C) 2020 Mischa Senders <m.j.senders@student.tue.nl>
// Copyright (C) 2020 Lex Kuijpers <l.kuijpers@student.tue.nl>
// Copyright (C) 2020 Jens Wehner <j.wehner@esciencecenter.nl>
// Copyright (C) 2020 Jan van Dijk <j.v.dijk@tue.nl>
// Copyright (C) 2020 Adithya Vijaykumar
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
/*

        The IDR(S)Stab(L) method is a combination of IDR(S) and BiCGStab(L)
        //TODO: elaborate what this improves over BiCGStab here

        Possible optimizations (PO):
        -See //PO: notes in the code

        This implementation of IDRStab is based on
                1. Aihara, K., Abe, K., & Ishiwata, E. (2014). A variant of IDRstab with reliable update strategies for
   solving sparse linear systems. Journal of Computational and Applied Mathematics, 259, 244-258.
   doi:10.1016/j.cam.2013.08.028
                2. Aihara, K., Abe, K., & Ishiwata, E. (2015). Preconditioned IDRStab Algorithms for Solving
   Nonsymmetric Linear Systems. International Journal of Applied Mathematics, 45(3).
                3. Saad, Y. (2003). Iterative Methods for Sparse Linear Systems: Second Edition. Philadelphia, PA: SIAM.
                4. Sonneveld, P., & Van Gijzen, M. B. (2009). IDR(s): A Family of Simple and Fast Algorithms for Solving
   Large Nonsymmetric Systems of Linear Equations. SIAM Journal on Scientific Computing, 31(2), 1035-1062.
   doi:10.1137/070685804
                5. Sonneveld, P. (2012). On the convergence behavior of IDR (s) and related methods. SIAM Journal on
   Scientific Computing, 34(5), A2576-A2598.

    Right-preconditioning based on Ref. 3 is implemented here.
*/

#ifndef EIGEN_IDRSTAB_H
#define EIGEN_IDRSTAB_H

namespace Eigen {

namespace internal {

template <typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
bool idrstab(const MatrixType &mat, const Rhs &rhs, Dest &x, const Preconditioner &precond, Index &iters,
             typename Dest::RealScalar &tol_error, Index L, Index S) {
  /*
    Setup and type definitions.
  */
  using numext::abs;
  using numext::sqrt;
  typedef typename Dest::Scalar Scalar;
  typedef typename Dest::RealScalar RealScalar;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic, ColMajor> DenseMatrixTypeCol;
  typedef Matrix<Scalar, Dynamic, Dynamic, RowMajor> DenseMatrixTypeRow;

  const Index N = x.rows();

  Index k = 0;  // Iteration counter
  const Index maxIters = iters;

  const RealScalar rhs_norm = rhs.norm();
  const RealScalar tol2 = tol_error * rhs_norm;

  if (rhs_norm == 0) {
    /*
      If b==0, then the exact solution is x=0.
      rhs_norm is needed for other calculations anyways, this exit is a freebie.
    */d
    x.setZero();
    tol_error = 0.0;
    return true;
  }
  // Construct decomposition objects beforehand.
  ColPivHouseholderQR<DenseMatrixTypeCol> qr_solver;
  FullPivLU<DenseMatrixTypeCol> lu_solver;

  if (S >= N || L >= N) {
    /*
      The matrix is very small, or the choice of L and S is very poor
      in that case solving directly will be best.
    */
    /*
      Exit
    */
    lu_solver.compute(DenseMatrixTypeRow(mat));
    x = lu_solver.solve(rhs);
    tol_error = (rhs - mat * x).norm() / rhs_norm;
    return true;
  }

  // Define maximum sizes to prevent any reallocation later on.
  VectorType u(N * (L + 1));
  VectorType r(N * (L + 1));
  DenseMatrixTypeCol V(N * (L + 1), S);

  DenseMatrixTypeCol rHat(N, L + 1);

  VectorType alpha(S);
  VectorType gamma(L);
  VectorType update(N);

  /*
    Main IDRStab algorithm
  */
  // Set up the initial residual
  r.head(N) = rhs - mat * precond.solve(x);
  tol_error = r.head(N).norm();

  DenseMatrixTypeRow h_FOM(S, S - 1);
  h_FOM.setZero();

  /*
    Determine an initial U matrix of size N x S
  */

  DenseMatrixTypeCol U(N * (L + 1), S);
  for (Index col_index = 0; col_index < S; ++col_index) {
    // Arnoldi-like process to generate a set of orthogonal vectors spanning {u,A*u,A*A*u,...,A^(S-1)*u}.
    // This construction can be combined with the Full Orthogonalization Method (FOM) from Ref.3 to provide a possible
    // early exit with no additional MV.
    if (col_index != 0) {

      /*
      Modified Gram-Schmidt strategy:
      */
      VectorType w = mat * precond.solve(u.head(N));
      for (Index i = 0; i < col_index; ++i) {
        //"Normalization factor" (v is normalized already)
        VectorType v = U.col(i).head(N);

        //"How much do v and w have in common?"
        h_FOM(i, col_index - 1) = v.dot(w);

        //"Subtract the part they have in common"
        w -= h_FOM(i, col_index - 1) * v;
      }
      u.head(N) = w;
      h_FOM(col_index, col_index - 1) = u.head(N).norm();

      if (abs(h_FOM(col_index, col_index - 1)) != 0.0) {
        /*
        This only happens if u is NOT exactly zero. In case it is exactly zero
        it would imply that that this u has no component in the direction of the current residual.

        By then setting u to zero it will not contribute any further (as it should).
        Whereas attempting to normalize results in division by zero.

        Such cases occur if:
        1. The basis of dimension <S is sufficient to exactly solve the linear system.
        I.e. the current residual is in span{r,Ar,...A^{m-1}r}, where (m-1)<=S.
        2. Two vectors vectors generated from r, Ar,... are (numerically) parallel.

        In case 1, the exact solution to the system can be obtained from the "Full Orthogonalization Method"
        (Algorithm 6.4 in the book of Saad), without any additional MV.

        Contrary to what one would suspect, the comparison with ==0.0 for floating-point types is intended here.
        Any arbritary non-zero u is fine to continue, however if u contains either NaN or Inf the algorithm will
        break down.
        */
        u.head(N) /= h_FOM(col_index, col_index - 1);
      }
    } else {
      u.head(N) = r.head(N);
      u.head(N).normalize();
    }

    U.col(col_index).head(N) = u.head(N);
  }

  if (S > 1) {
    /*
    Check for early FOM exit.
    */
    Scalar beta = r.head(N).norm();
    VectorType e1 = VectorType::Zero(S - 1);
    e1(0) = beta;
    lu_solver.compute(h_FOM.topLeftCorner(S - 1, S - 1));
    VectorType y = lu_solver.solve(e1);
    VectorType x2 = x + U.topLeftCorner(N, S - 1) * y;

    // Using proposition 6.7 in Saad, one MV can be saved to calculate the residual
    RealScalar FOM_residual = (h_FOM(S - 1, S - 2) * y(S - 2) * U.col(S - 1).head(N)).norm();

    if (FOM_residual < tol2) {
      /*
      Exit, the FOM algorithm was already accurate enough
      */
      iters = k;
      x = precond.solve(x2);
      tol_error = FOM_residual / rhs_norm;
      return true;
    }
  }

  /*
    Select an initial (N x S) matrix R0.
    1. Generate random R0, orthonormalize the result.
    2. This results in R0, however to save memory and compute we only need the adjoint of R0. This is given by the
    matrix R_T.\ Additionally, the matrix (mat.adjoint()*R_tilde).adjoint()=R_tilde.adjoint()*mat by the
    anti-distributivity property of the adjoint. This results in AR_T, which is constant if R_T does not have to be
    regenerated and can be precomputed. Based on reference 4, this has zero probability in exact arithmetic. However in
    practice it does (extremely infrequently) occur, most notably for small matrices.
  */
  // PO: To save on memory consumption identity can be sparse
  // PO: can this be done with colPiv/fullPiv version as well? This would save 1 construction of a HouseholderQR object

  // Original IDRStab and Kensuke choose S random vectors:
  HouseholderQR<DenseMatrixTypeCol> qr(DenseMatrixTypeCol::Random(N, S));
  DenseMatrixTypeRow R_T = (qr.householderQ() * DenseMatrixTypeCol::Identity(N, S)).adjoint();
  DenseMatrixTypeRow AR_T = DenseMatrixTypeRow(R_T * mat);

  // Pre-allocate sigma, this space will be recycled without additional allocations.
  DenseMatrixTypeCol sigma(S, S);

  Index rt_counter = k;      // Iteration at which R_T was reset last
  bool reset_while = false;  // Should the while loop be reset for some reason?

  VectorType Q(S, 1);            // Vector containing the row-scaling applied to sigma
  VectorType P(S, 1);            // Vector containing the column-scaling applied to sigma
  DenseMatrixTypeCol QAP(S, S);  // Scaled sigma
  bool repair_flag = false;
  RealScalar residual_0 = tol_error;
  // bool apply_r_exit = false;

  while (k < maxIters) {
    for (Index j = 1; j <= L; ++j) {
      // Cache some indexing variables that occur frequently and are constant.
      const Index Nj = N * j;
      const Index Nj_plus_1 = N * (j + 1);
      const Index Nj_min_1 = N * (j - 1);

      /*
        The IDR Step
      */
      // Construction of the sigma-matrix, and the decomposition of sigma.
      for (Index i = 0; i < S; ++i) {
        sigma.col(i).noalias() = AR_T * precond.solve(U.block(Nj_min_1, i, N, 1));
      }
      /*
        Suspected is that sigma could be badly scaled, since causes alpha~=0, but the
        update vector is not zero. To improve stability we scale with absolute row and col sums first.
        Sigma can become badly scaled (but still well-conditioned).
        A bad sigma also happens if R_T is not chosen properly, for example if R_T is zeros sigma would be zeros
        as well. The effect of this is a left-right preconditioner, instead of solving Ax=b, we solve
        Q*A*P*inv(P)*x=Q*b.
      */

      Q = (sigma.cwiseAbs().rowwise().sum()).cwiseInverse();  // Calculate absolute inverse row sum
      QAP = Q.asDiagonal() * sigma;                           // Scale with inverse absolute row sums
      P = (QAP.cwiseAbs().colwise().sum()).cwiseInverse();    // Calculate absolute inverse column sum
      QAP = QAP * P.asDiagonal();                             // Scale with inverse absolute column sums
      lu_solver.compute(QAP);
      // Obtain the update coefficients alpha
      if (j == 1) {
        // alpha=inverse(sigma)*(R_T*r_0);
        alpha.noalias() = lu_solver.solve(Q.asDiagonal() * R_T * r.head(N));
      } else {
        // alpha=inverse(sigma)*(AR_T*r_{j-2})
        alpha.noalias() = lu_solver.solve(Q.asDiagonal() * AR_T * precond.solve(r.segment(N * (j - 2), N)));
      }
      // Unscale the solution
      alpha = P.asDiagonal() * alpha;

      double old_res = tol_error;

      // Obtain new solution and residual from this update
      update.noalias() = U.topRows(N) * alpha;
      r.head(N) -= mat * precond.solve(update);
      x += update;

      for (Index i = 1; i <= j - 2; ++i) {
        // This only affects the case L>2
        r.segment(N * i, N) -= U.block(N * (i + 1), 0, N, S) * alpha;
      }
      if (j > 1) {
        // r=[r;A*r_{j-2}]
        r.segment(Nj_min_1, N).noalias() = mat * precond.solve(r.segment(N * (j - 2), N));
      }
      tol_error = r.head(N).norm();

      if (tol_error < tol2) {
        // If at this point the algorithm has converged, exit.
        reset_while = true;
        break;
      }

      if (repair_flag == false && tol_error > 10 * residual_0) {
        // Sonneveld's repair flag suggestion from [5]
        // This massively reduces problems with false residual estimates (if they'd occur)
        repair_flag = true;
      }
      if (repair_flag && 1000 * tol_error < residual_0) {
        // 1000 comes from Sonneveld's repair flag suggestion from [5]
        r.head(N) = rhs - mat * precond.solve(x);
        repair_flag = false;
      }

      bool reset_R_T = false;
      if (alpha.norm() * rhs_norm < S * NumTraits<Scalar>::epsilon() * old_res) {
        // This would indicate the update computed by alpha did nothing much to decrease the residual
        // apparantly we've also not converged either.
        // TODO: Check if there is some better criterion, the current one is a bit handwavy.
        reset_R_T = true;
      }

      if (reset_R_T) {
        if (k - rt_counter > 0) {
          /*
                  Only regenerate if it didn't already happen this iteration.
          */
          // Choose new R0 and try again
          qr.compute(DenseMatrixTypeCol::Random(N, S));
          R_T = (qr.householderQ() * DenseMatrixTypeCol::Identity(N, S))
                    .transpose();  //.adjoint() vs .transpose() makes no difference, R_T is random anyways.
          /*
            Additionally, the matrix (mat.adjoint()*R_tilde).adjoint()=R_tilde.adjoint()*mat by the
            anti-distributivity property of the adjoint. This results in AR_T, which can be precomputed.
          */
          AR_T = DenseMatrixTypeRow(R_T * mat);
          j = 0;  // WARNING reset the for loop counter
          rt_counter = k;
          continue;
        }
      }
      bool break_normalization = false;
      for (Index q = 1; q <= S; ++q) {
        if (q == 1) {
          // u = r;
          u.head(Nj_plus_1) = r.topRows(Nj_plus_1);
        } else {
          // u=[u_1;u_2;...;u_j]
          u.head(Nj) = u.segment(N, Nj);
        }
        // Obtain the update coefficients beta implicitly
        // beta=lu_sigma.solve(AR_T * u.block(Nj_min_1, 0, N, 1)

        u.head(Nj) -= U.topRows(Nj) * P.asDiagonal() *
                      lu_solver.solve(Q.asDiagonal() * AR_T * precond.solve(u.segment(Nj_min_1, N)));

        // u=[u;Au_{j-1}]
        u.segment(Nj, N).noalias() = mat * precond.solve(u.segment(Nj_min_1, N));

        // Orthonormalize u_j to the columns of V_j(:,1:q-1)
        if (q > 1) {
          /*
          Modified Gram-Schmidt-like procedure to make u orthogonal to the columns of V from Ref. 1.

          The vector mu from Ref. 1 is obtained implicitly:
          mu=V.block(Nj, 0, N, q - 1).adjoint() * u.block(Nj, 0, N, 1).
          */

          for (Index i = 0; i <= q - 2; ++i) {
            //"Normalization factor"
            Scalar h = V.col(i).segment(Nj, N).squaredNorm();

            //"How much do u and V have in common?"
            h = V.col(i).segment(Nj, N).dot(u.segment(Nj, N)) / h;

            //"Subtract the part they have in common"
            u.head(Nj_plus_1) -= h * V.block(0, i, Nj_plus_1, 1);
          }
        }
        // Normalize u and assign to a column of V
        Scalar normalization_constant = u.block(Nj, 0, N, 1).norm();

        if (normalization_constant != 0.0) {
          /*
            If u is exactly zero, this will lead to a NaN. Small, non-zero u is fine. In the case of NaN the
            algorithm breaks down, eventhough it could have continued, since u zero implies that there is no further
            update in a given direction.
          */
          u.head(Nj_plus_1) /= normalization_constant;
        } else {
          u.head(Nj_plus_1).setZero();
          if (tol_error < tol2 || tol_error < 1e4 * NumTraits<Scalar>::epsilon()) {
            // Just quit, we've converged
            iters = k;
            x = precond.solve(x);
            tol_error = (rhs - mat * x).norm() / rhs_norm;
            return true;
          }
          break_normalization = true;
          break;
        }

        V.block(0, q - 1, Nj_plus_1, 1).noalias() = u.head(Nj_plus_1);
      }

      if (break_normalization == false) {
        U = V;
      }
    }
    if (reset_while) {
      reset_while = false;
      tol_error = r.head(N).norm();
      if (tol_error < tol2) {
        /*
        Slightly early exit by moving the criterion before the update of U,
        after the main while loop the result of that calculation would not be needed.
        */
        break;
      }
      continue;
    }

    // r=[r;mat*r_{L-1}]
    // Save this in rHat, the storage form for rHat is more suitable for the argmin step than the way r is stored.
    // In Eigen 3.4 this step can be compactly done via: rHat = r.reshaped(N, L + 1);
    r.segment(N * L, N).noalias() = mat * precond.solve(r.segment(N * (L - 1), N));

    for (Index i = 0; i <= L; ++i) {
      rHat.col(i) = r.segment(N * i, N);
    }

    /*
            The polynomial step
    */
    qr_solver.compute(rHat.rightCols(L));
    gamma.noalias() = qr_solver.solve(r.head(N));

    // Update solution and residual using the "minimized residual coefficients"
    update.noalias() = rHat.leftCols(L) * gamma;
    x += update;
    r.head(N) -= mat * precond.solve(update);

    // Update iteration info
    ++k;
    tol_error = r.head(N).norm();

    if (tol_error < tol2) {
      // Slightly early exit by moving the criterion before the update of U,
      // after the main while loop the result of that calculation would not be needed.
      break;
    }

    if (repair_flag == false && tol_error > 10 * residual_0) {
      // Sonneveld's repair flag suggestion from [5]
      // This massively reduces problems with false residual estimates (if they'd occur)
      repair_flag = true;
    }
    if (repair_flag && 1000 * tol_error < residual_0) {
      r.head(N) = rhs - mat * precond.solve(x);
      repair_flag = false;
    }

    /*
    U=U0-sum(gamma_j*U_j)
    Consider the first iteration. Then U only contains U0, so at the start of the while-loop
    U should be U0. Therefore only the first N rows of U have to be updated.
    */
    for (Index i = 1; i <= L; ++i) {
      U.topRows(N) -= U.block(N * i, 0, N, S) * gamma(i - 1);
    }
  }

  /*
          Exit after the while loop terminated.
  */
  iters = k;
  x = precond.solve(x);
  tol_error = tol_error / rhs_norm;
  return true;
}

}  // namespace internal

template <typename _MatrixType, typename _Preconditioner = DiagonalPreconditioner<typename _MatrixType::Scalar> >
class IDRStab;

namespace internal {

template <typename _MatrixType, typename _Preconditioner>
struct traits<IDRStab<_MatrixType, _Preconditioner> > {
  typedef _MatrixType MatrixType;
  typedef _Preconditioner Preconditioner;
};

}  // namespace internal

template <typename _MatrixType, typename _Preconditioner>
class IDRStab : public IterativeSolverBase<IDRStab<_MatrixType, _Preconditioner> > {
  typedef IterativeSolverBase<IDRStab> Base;
  using Base::m_error;
  using Base::m_info;
  using Base::m_isInitialized;
  using Base::m_iterations;
  using Base::matrix;
  Index m_L;
  Index m_S;

 public:
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef _Preconditioner Preconditioner;

 public:
  /** Default constructor. */
  IDRStab() : Base() {
    m_L = 2;
    m_S = 4;
  }

  /**   Initialize the solver with matrix \a A for further \c Ax=b solving.

  This constructor is a shortcut for the default constructor followed
  by a call to compute().

  \warning this class stores a reference to the matrix A as well as some
  precomputed values that depend on it. Therefore, if \a A is changed
  this class becomes invalid. Call compute() to update it with the new
  matrix A, or modify a copy of A.
          */
  template <typename MatrixDerived>
  explicit IDRStab(const EigenBase<MatrixDerived> &A) : Base(A.derived()) {
    m_L = 2;
    m_S = 4;
  }

  ~IDRStab() {}

  /** \internal */

  template <typename Rhs, typename Dest>
  void _solve_vector_with_guess_impl(const Rhs &b, Dest &x) const {
    m_iterations = Base::maxIterations();
    m_error = Base::m_tolerance;
    bool ret = internal::idrstab(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error, m_L, m_S);

    m_info = (!ret) ? NumericalIssue : m_error <= 10 * Base::m_tolerance ? Success : NoConvergence;
  }

  /** \internal */
  /** Sets the parameter L, indicating the amount of minimize residual steps are used. */
  void setL(Index L) {
    if (L < 1) {
      L = 2;
    }

    m_L = L;
  }
  /** \internal */
  /** Sets the parameter S, indicating the dimension of the shadow residual space.. */
  void setS(Index S) {
    if (S > 0) {
      m_S = S;
    }
  }

 protected:
};

}  // namespace Eigen

#endif /* EIGEN_IDRSTAB_H */
