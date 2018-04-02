/*
 * Utils.h
 *
 *  Created on: 20.12.2017
 *      Author: Viktor Csomor
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
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

template<typename Derived>
using TensorExp = Eigen::TensorBase<Derived>;

/**
 * An for a class that can be used to map raw pointer data to a tensor of arbitrary
 * rank and scalar type with dynamic dimensionality.
 */
template<typename Scalar, std::size_t Rank>
using TensorMap = Eigen::TensorMap<Tensor<Scalar,Rank>>;

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
	/**
	 * It asserts that each rank of the tensor has a size greater than 0.
	 *
	 * @param tensor A constant reference to the tensor to check.
	 */
	template<std::size_t Rank>
	inline static void check_dim_validity(const Tensor<Scalar,Rank>& tensor) {
		std::array<std::size_t,Rank> dimensions = tensor.dimensions();
		for (std::size_t i = 0; i < Rank; ++i)
			assert(dimensions[i] > 0 && "illegal tensor dimension");
	}
	/**
	 * Returns a Dimensions instance representing the dimensionality of the specified tensor.
	 *
	 * @param tensor A constant reference to the tensor whose dimensionality is to be
	 * assessed.
	 * @return A Dimensions instance containing the size of the tensor along its ranks.
	 */
	template<std::size_t Rank>
	inline static Dimensions<std::size_t,Rank> get_dims(const Tensor<Scalar,Rank>& tensor) {
		static_assert(Rank > 1, "illegal tensor rank");
		return Dimensions<std::size_t,Rank>(tensor.dimensions());
	}
	/**
	 * It maps the data backing a tensor to a matrix. The product of the specified number of
	 * rows and columns should equal the product of the dimensions of the tensor along its ranks.
	 *
	 * @param tensor The tensor whose data is to be mapped to a matrix.
	 * @param rows The number of rows the returned matrix is to have.
	 * @param cols The number of columns the returned matrix is to have.
	 * @return A matrix backed by the data of the tensor.
	 */
	template<std::size_t Rank>
	inline static Matrix<Scalar> map_tensor_to_mat(Tensor<Scalar,Rank> tensor, int rows, int cols) {
		static_assert(Rank > 1, "tensor rank too low for conversion to matrix");
		assert(rows > 0 && cols > 0 && rows * cols = tensor.size());
		return MatrixMap<Scalar>(tensor.data(), rows, cols);
	}
	/**
	 * It maps the data backing a tensor to a matrix. The number of rows of the returned matrix
	 * equals the size of the tensor along its first rank and the number of columns of the matrix
	 * equals the product of the sizes of the tensor along its other ranks.
	 *
	 * @param tensor The tensor whose data is to be mapped to a matrix.
	 * @return A matrix backed by the data of the tensor.
	 */
	template<std::size_t Rank>
	inline static Matrix<Scalar> map_tensor_to_mat(Tensor<Scalar,Rank> tensor) {
		static_assert(Rank > 1, "tensor rank too low for conversion to matrix");
		int rows = tensor.dimension(0);
		return MatrixMap<Scalar>(tensor.data(), rows, tensor.size() / rows);
	}
	/**
	 * It maps the data backing a matrix to a tensor of arbitrary rank.
	 *
	 * @param mat The matrix to map to a tensor.
	 * @param dims The dimensionality of the tensor to which the matrix is to be mapped. The rank
	 * of the dimensions is one less than that of the returned tensor as the size of the tensor
	 * along its first rank takes on the number of rows the matrix has.
	 * @return A tensor backed by the data of the matrix.
	 */
	template<std::size_t Rank>
	inline static Tensor<Scalar,Rank> map_mat_to_tensor(Matrix<Scalar> mat,
			const Dimensions<std::size_t,Rank - 1>& dims) {
		static_assert(Rank > 1, "rank too low for conversion to tensor");
		assert(dims.get_volume() == mat.cols());
		Dimensions<std::size_t,Rank> promoted = dims.template promote<>();
		promoted(0) = mat.rows();
		return TensorMap<Scalar,Rank>(mat.data(), promoted);
	}
	/**
	 * It maps the data backing a matrix to a tensor of arbitrary rank with the same dimensionality
	 * along its first rank as the number of rows the matrix has and the same dimensionality along
	 * its second rank as the number of columns the matrix has.
	 *
	 * @param mat The matrix whose data is to be mapped to a tensor.
	 * @return A tensor backed by the data of the matrix.
	 */
	template<std::size_t Rank>
	inline static Tensor<Scalar,Rank> map_mat_to_tensor(Matrix<Scalar> mat) {
		static_assert(Rank > 1, "rank too low for conversion to tensor");
		Dimensions<std::size_t,Rank> dims;
		dims(0) = mat.rows();
		dims(1) = mat.cols();
		return TensorMap<Scalar,Rank>(mat.data(), dims);
	}
	/**
	 * It maps a tensor to another tensor of arbitrary rank.
	 *
	 * @param tensor The tensor whose backing data is to be mapped onto another tensor.
	 * @param dims The dimensionality of the new tensor. The volume should equal the
	 * total size of the tensor to be mapped.
	 * @return A tensor backed by the data of the original tensor.
	 */
	template<std::size_t Rank, std::size_t NewRank>
	inline static Tensor<Scalar,NewRank> map_tensor_to_tensor(Tensor<Scalar,Rank> tensor,
			const Dimensions<std::size_t,NewRank>& dims) {
		static_assert(Rank > 0 && NewRank > 0, "illegal tensor rank");
		assert(tensor.size() == dims.get_volume());
		return TensorMap<Scalar,NewRank>(tensor.data(), dims);
	}
	/**
	 * It removes the first rank of the tensor and extends the new first rank by a factor
	 * equal to the dimensionality of the removed rank.
	 *
	 * @param tensor The tensor whose first two ranks are to be joined.
	 * @return The tensor with the first two ranks joined.
	 */
	template<std::size_t Rank>
	inline static Tensor<Scalar,Rank - 1> join_first_two_ranks(Tensor<Scalar,Rank> tensor) {
		static_assert(Rank > 1, "illegal tensor rank");
		Dimensions<std::size_t,Rank> dims = get_dims<Rank>(tensor);
		int lowest_dim = dims(0);
		Dimensions<std::size_t,Rank - 1> joined_dims = dims.template demote<>();
		joined_dims(0) *= lowest_dim;
		return TensorMap<Scalar,Rank - 1>(tensor.data(), joined_dims);
	}
	/**
	 * It splits the first rank of the tensor into two ranks. The product of the two rank sizes
	 * specified must match the size of the tensor's first rank.
	 *
	 * @param tensor The tensor whose first rank is to be split.
	 * @param rank0_size The size of the new first rank.
	 * @param rank1_size The size of the new second rank.
	 * @return The tensor with the first rank split into two ranks.
	 */
	template<std::size_t Rank>
	inline static Tensor<Scalar,Rank + 1> split_first_rank(Tensor<Scalar,Rank> tensor, std::size_t rank0_size,
			std::size_t rank1_size) {
		static_assert(Rank > 0, "illegal tensor rank");
		Dimensions<std::size_t,Rank> dims = get_dims<Rank>(tensor);
		assert(dims(0) == rank0_size * rank1_size);
		Dimensions<std::size_t,Rank + 1> split_dims = dims.template promote<>();
		split_dims(0) = rank0_size;
		split_dims(1) = rank1_size;
		return TensorMap<Scalar,Rank + 1>(tensor.data(), split_dims);
	}
	/**
	 * It randomly shuffles the rows of the specified matrix.
	 *
	 * @param mat A reference to the matrix whose rows are to be shuffled.
	 */
	inline static void shuffle_mat_rows(Matrix<Scalar>& mat) {
		Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(mat.rows());
		perm.setIdentity();
		// Shuffle the indices of the identity matrix.
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		mat = perm * mat;
	}
	/**
	 * It randomly shuffles the first rank of the tensor.
	 *
	 * @param tensor A reference to the tensor whose first rank is to be shuffled.
	 */
	template<std::size_t Rank>
	inline static void shuffle_tensor_rows(Tensor<Scalar,Rank>& tensor) {
		static_assert(Rank > 1, "illegal tensor rank");
		Dimensions<std::size_t,Rank - 1> dims = get_dims<Rank>(tensor).template demote<>();
		Matrix<Scalar> mat = map_tensor_to_mat<Rank>(std::move(tensor));
		shuffle_mat_rows(mat);
		tensor = map_mat_to_tensor<Rank>(std::move(mat), dims);
	}
};

}

#endif /* UTILS_H_ */
