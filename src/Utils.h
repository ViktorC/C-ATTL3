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
#include <Dimensions.h>
#include <Eigen/Dense>
#include <limits>
#include <type_traits>
#include <utility>
#include <unsupported/Eigen/CXX11/Tensor>

namespace cattle {

template<typename Scalar>
using RowVector = Eigen::Matrix<Scalar,1,Eigen::Dynamic,Eigen::RowMajor,1,Eigen::Dynamic>;

template <typename Scalar>
using ColVector = Eigen::Matrix<Scalar,Eigen::Dynamic,1,Eigen::ColMajor,Eigen::Dynamic,1>;

template<typename Scalar>
using Matrix = Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor,Eigen::Dynamic,Eigen::Dynamic>;

template<typename Scalar>
using MatrixMap = Eigen::Map<Matrix<Scalar>>;

template<typename Scalar, size_t Rank>
using Tensor = Eigen::Tensor<Scalar,Rank,Eigen::ColMajor,int>;

template<typename Scalar, size_t Rank>
using TensorMap = Eigen::TensorMap<Tensor<Scalar,Rank>>;

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
	static const Matrix<Scalar> NULL_MATRIX;
	static const Tensor<Scalar,2> NULL_TENSOR2;
	static const Tensor<Scalar,3> NULL_TENSOR3;
	static const Tensor<Scalar,4> NULL_TENSOR4;
	static const Tensor<Scalar,5> NULL_TENSOR5;
	inline static int num_of_eigen_threads() {
		return Eigen::nbThreads();
	};
	inline static void set_num_of_eigen_threads(int num_of_threads) {
		Eigen::setNbThreads(num_of_threads);
	};
	inline static bool almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		Scalar diff = std::abs(n1 - n2);
		if (diff <= abs_epsilon)
			return true;
		Scalar max = std::max(std::abs(n1), std::abs(n2));
		return diff <= max * rel_epsilon;
	};
	inline static bool decidedly_greater(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	inline static bool greater_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	inline static bool decidedly_lesser(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	inline static bool lesser_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	template<size_t Rank>
	inline static const Tensor<Scalar,Rank>& get_null_tensor() {
		static_assert(Rank > 1 && Rank < 6, "illegal null tensor rank");
		switch (Rank) {
		case 2:
			return NULL_TENSOR2;
		case 3:
			return NULL_TENSOR3;
		case 4:
			return NULL_TENSOR4;
		case 5:
			return NULL_TENSOR5;
		}
	};
	template<size_t Rank>
	inline static void check_tensor_dims(const Tensor<Scalar,Rank>& tensor) {
		std::array<int,Rank> dimensions = tensor.dimensions();
		for (size_t i = 0; i < Rank; i++)
			assert(dimensions[i] > 0 && "illegal tensor dimension");
	};
	template<size_t Rank>
	inline static Dimensions<int,Rank> get_dims(const Tensor<Scalar,Rank>& tensor) {
		static_assert(Rank > 1, "illegal tensor rank");
		return Dimensions<int,Rank>(tensor.dimensions());
	};
	template<size_t Rank>
	inline static Matrix<Scalar> map_tensor_to_mat(Tensor<Scalar,Rank> tensor, int rows, int cols) {
		static_assert(Rank > 1, "tensor rank too low for conversion to matrix");
		assert(rows > 0 && cols > 0 && rows * cols = tensor.size());
		return MatrixMap<Scalar>(tensor.data(), rows, cols);
	};
	template<size_t Rank>
	inline static Matrix<Scalar> map_tensor_to_mat(Tensor<Scalar,Rank> tensor) {
		static_assert(Rank > 1, "tensor rank too low for conversion to matrix");
		int rows = tensor.dimension(0);
		return MatrixMap<Scalar>(tensor.data(), rows, tensor.size() / rows);
	};
	template<size_t Rank>
	inline static Tensor<Scalar,Rank> map_mat_to_tensor(Matrix<Scalar> mat, const Dimensions<int,Rank - 1>& dims) {
		static_assert(Rank > 1, "rank too low for conversion to tensor");
		assert(dims.get_volume() == mat.cols());
		Dimensions<int,Rank> promoted = dims.template promote<>();
		promoted(0) = mat.rows();
		return TensorMap<Scalar,Rank>(mat.data(), promoted);
	};
	template<size_t Rank>
	inline static Tensor<Scalar,Rank> map_mat_to_tensor(Matrix<Scalar> mat) {
		static_assert(Rank > 1, "rank too low for conversion to tensor");
		Dimensions<int,Rank> dims;
		dims(0) = mat.rows();
		dims(1) = mat.cols();
		return TensorMap<Scalar,Rank>(mat.data(), dims);
	};
	template<size_t Rank, size_t NewRank>
	inline static Tensor<Scalar,NewRank> map_tensor_to_tensor(Tensor<Scalar,Rank> tensor, const Dimensions<int,NewRank>& dims) {
		static_assert(Rank > 0 && NewRank > 0, "illegal tensor rank");
		assert(tensor.size() == dims.get_volume());
		return TensorMap<Scalar,Rank>(tensor.data(), dims);
	};
	inline static void shuffle_mat_rows(Matrix<Scalar>& mat) {
		Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(mat.rows());
		perm.setIdentity();
		// Shuffle the indices of the identity matrix.
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		mat = perm * mat;
	};
	template<size_t Rank>
	inline static void shuffle_tensor_rows(Tensor<Scalar,Rank>& tensor) {
		static_assert(Rank > 1, "illegal tensor rank");
		Dimensions<int,Rank - 1> dims = get_dims<Rank>(tensor).template demote<>();
		Matrix<Scalar> mat = map_tensor_to_mat<Rank>(std::move(tensor));
		shuffle_mat_rows(mat);
		tensor = map_mat_to_tensor<Rank>(std::move(mat), dims);
	};
};

template<typename Scalar>
const Matrix<Scalar> Utils<Scalar>::NULL_MATRIX = Matrix<Scalar>();
template<typename Scalar>
const Tensor<Scalar,2> Utils<Scalar>::NULL_TENSOR2 = Tensor<Scalar,2>();
template<typename Scalar>
const Tensor<Scalar,3> Utils<Scalar>::NULL_TENSOR3 = Tensor<Scalar,3>();
template<typename Scalar>
const Tensor<Scalar,4> Utils<Scalar>::NULL_TENSOR4 = Tensor<Scalar,4>();

}

#endif /* UTILS_H_ */
