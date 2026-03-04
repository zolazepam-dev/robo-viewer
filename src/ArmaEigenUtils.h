#pragma once

#include <armadillo>
#include <Eigen/Dense>

// Hybrid Armadillo/Eigen utilities for zero-copy interop and targeted fast ops.
// Armadillo remains the primary container (arma::mat). Eigen is used via
// Eigen::Map over existing Armadillo memory for operations where benchmarks
// favor Eigen (inversion, decompositions).

namespace linalg {

// Create an Eigen view (no copy) over an Armadillo matrix.
inline auto ToEigenConst(const arma::mat& A)
{
    return Eigen::Map<const Eigen::MatrixXd>(A.memptr(), A.n_rows, A.n_cols);
}

// Mutable Eigen view (no copy) over an Armadillo matrix.
inline auto ToEigen(arma::mat& A)
{
    return Eigen::Map<Eigen::MatrixXd>(A.memptr(), A.n_rows, A.n_cols);
}

// Wrap Eigen data as an Armadillo matrix (shallow if possible).
inline arma::mat ToArma(const Eigen::MatrixXd& E)
{
    return arma::mat(E.data(), static_cast<arma::uword>(E.rows()), static_cast<arma::uword>(E.cols()), false, true);
}

// Fast dense inverse using Eigen, returning arma::mat.
inline arma::mat FastInverse(const arma::mat& A)
{
    auto eA = ToEigenConst(A);
    Eigen::MatrixXd eInv = eA.inverse();
    return ToArma(eInv);
}

// Solve Ax = b (A SPD) via Eigen LLT; falls back to armadillo solve if needed.
inline arma::mat SolveSPD(const arma::mat& A, const arma::mat& B)
{
    auto eA = ToEigenConst(A);
    auto eB = ToEigenConst(B);
    Eigen::LLT<Eigen::MatrixXd> llt(eA);
    if (llt.info() == Eigen::Success)
    {
        Eigen::MatrixXd eX = llt.solve(eB);
        return ToArma(eX);
    }
    // Fallback keeps behavior correct even if decomposition fails.
    return arma::solve(A, B);
}

// Solve general Ax = b via Eigen full pivot LU; returns arma::mat.
inline arma::mat SolveGeneral(const arma::mat& A, const arma::mat& B)
{
    auto eA = ToEigenConst(A);
    auto eB = ToEigenConst(B);
    Eigen::FullPivLU<Eigen::MatrixXd> lu(eA);
    Eigen::MatrixXd eX = lu.solve(eB);
    return ToArma(eX);
}

} // namespace linalg
