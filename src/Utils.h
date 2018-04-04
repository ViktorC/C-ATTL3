/*
 * Utils.h
 *
 *  Created on: 20.12.2017
 *      Author: Viktor Csomor
 */

#ifndef UTILS_H_
#define UTILS_H_

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "Dimensions.h"

/**
 * The namespace containing all classes and typedefs of the C-ATTL3 library.
 */
namespace cattle {

/**
 * An alias for a single row matrix of an arbitrary scalar type.
 */
template<typename Scalar>
using RowVector = Eigen::Matrix<Scalar,1,Eigen::Dynamic,Eigen::RowMajor,1,Eigen::Dynamic>;

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
 * A namespace for utilities used by C-ATTL3 internally.
 */
namespace internal {

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
 * A utility class template containing static methods and variables to help with
 * numerical issues, the manipulation of matrices and tensors, and the conversion
 * between tensors and matrices.
 */
template<typename Scalar>
class Utils {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
private:
	Utils() { };
public:
	static constexpr Scalar MIN = std::numeric_limits<Scalar>::lowest();
	static constexpr Scalar MAX = std::numeric_limits<Scalar>::max();
	static constexpr Scalar EPSILON1 = std::numeric_limits<Scalar>::epsilon();
	static constexpr Scalar EPSILON2 = 1e-8;
	static constexpr Scalar EPSILON3 = 1e-4;
	/**
	 * @return The number of threads used by Eigen to accelerate operations
	 * supporting multithreading.
	 */
	inline static int num_of_eval_threads() {
		return Eigen::nbThreads();
	}
	/**
	 * @param num_of_threads The number of threads Eigen should use to accelerate
	 * operations supporting multithreading.
	 */
	inline static void set_num_of_eval_threads(int num_of_threads) {
		Eigen::setNbThreads(num_of_threads);
	}
	/**
	 * Returns whether the two numerals are close enough to be considered equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the two numerals can be considered equal.
	 */
	inline static bool almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		Scalar diff = std::abs(n1 - n2);
		if (diff <= abs_epsilon)
			return true;
		Scalar max = std::max(std::abs(n1), std::abs(n2));
		return diff <= max * rel_epsilon;
	}
	/**
	 * Returns whether a numeral is greater than another one by a margin great enough for
	 * them not to be considered almost equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the the first numeral is sufficiently greater than the second.
	 */
	inline static bool decidedly_greater(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	}
	/**
	 * Returns whether a numeral is not lesser than another one by a margin great enough for
	 * them not to be considered almost equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the the first numeral is sufficiently great enough not to be considered
	 * decidedly smaller than the second.
	 */
	inline static bool greater_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	}
	/**
	 * Returns whether a numeral is lesser than another one by a margin great enough for
	 * them not to be considered almost equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the the first numeral is sufficiently lesser than the second.
	 */
	inline static bool decidedly_lesser(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	}
	/**
	 * Returns whether a numeral is not greater than another one by a margin great enough for
	 * them not to be considered almost equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the the first numeral is sufficiently small enough not to be considered
	 * decidedly greater than the second.
	 */
	inline static bool lesser_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	}
};

}

}

#endif /* UTILS_H_ */
