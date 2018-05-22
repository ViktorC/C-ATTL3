/*
 * EigenProxy.hpp
 *
 *  Created on: 12 Apr 2018
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_UTILS_EIGENPROXY_H_
#define CATTL3_UTILS_EIGENPROXY_H_

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cstddef>
#include <Eigen/Dense>
#include <thread>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * The namespace containing all classes and typedefs of the C-ATTL3 library.
 */
namespace cattle {

/**
 * An alias for a single row matrix of an arbitrary scalar type.
 */
template<typename Scalar>
using RowVector = Eigen::Matrix<Scalar,1,Eigen::Dynamic,Eigen::RowMajor, 1,Eigen::Dynamic>;

/**
 * An alias for a single column matrix of an arbitrary scalar type.
 */
template <typename Scalar>
using ColVector = Eigen::Matrix<Scalar,Eigen::Dynamic,1,Eigen::ColMajor,Eigen::Dynamic,1>;

/**
 * An alias for a dynamically sized matrix of an arbitrary scalar type.
 */
template<typename Scalar>
using Matrix = Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor,
		Eigen::Dynamic,Eigen::Dynamic>;

/**
 * An alias for a class that can be used to map raw pointer data to a dynamically
 * sized Matrix of an arbitrary scalar type.
 */
template<typename Scalar>
using MatrixMap = Eigen::Map<Matrix<Scalar>>;

/**
 * An alias for a tensor of arbitrary rank and scalar type with dynamic dimensionality.
 */
template<typename Scalar, std::size_t Rank>
using Tensor = Eigen::Tensor<Scalar,Rank,Eigen::ColMajor,std::size_t>;

/**
 * An for a class that can be used to map raw pointer data to a tensor of arbitrary
 * rank and scalar type with dynamic dimensionality.
 */
template<typename Scalar, std::size_t Rank>
using TensorMap = Eigen::TensorMap<Tensor<Scalar,Rank>>;

/**
 * An alias for Eigen's normal standard distribution random number generator
 * for tensors.
 */
template<typename Scalar>
using NormalRandGen = Eigen::internal::NormalRandomGenerator<Scalar>;

/**
 * An alias for permutation matrices.
 */
using PermMatrix = Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic>;

/**
 * An alias for self-adjoint eigen solvers.
 */
template<typename Scalar>
using EigenSolver = Eigen::SelfAdjointEigenSolver<Matrix<Scalar>>;

/**
 * An alias for Eigen's bi-diagonal divide and conquer singular-value decomposition.
 */
template<typename Scalar>
using SVD = Eigen::BDCSVD<Matrix<Scalar>>;

/**
 * An alias for Eigen's singular-value decomposition options.
 */
using SVDOptions = Eigen::DecompositionOptions;

/**
 * A namespace for utilities used by C-ATTL3 internally.
 */
namespace internal {

/**
 * A struct for retrieving and setting the number of threads Eigen should use
 * for matrix multiplication and other parallelized operations.
 */
struct EigenProxy {
	EigenProxy() = delete;
	/**
	 * @return The number of threads used by Eigen to accelerate operations
	 * supporting multithreading.
	 */
	inline static int num_of_eval_threads() {
		return Eigen::nbThreads();
	}
	/**
	 * @param num_of_threads The number of threads Eigen should use to accelerate
	 * operations supporting multithreading. The lower bound of the actual value
	 * applied is 1 while the upper bound is the maximum of 1 and the level of
	 * hardware concurrency detected.
	 */
	inline static void set_num_of_eval_threads(int num_of_threads) {
		int max = std::max(1, (int) std::thread::hardware_concurrency());
		Eigen::setNbThreads(std::max(1, std::min(num_of_threads, max)));
	}
};

}

}

#endif /* CATTL3_UTILS_EIGENPROXY_H_ */
