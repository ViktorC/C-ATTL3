/*
 * Utils.h
 *
 *  Created on: 20.12.2017
 *      Author: Viktor Csomor
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <Dimensions.h>
#include <Eigen/Dense>
#include <limits>
#include <type_traits>
#include <utility>
#include <unsupported/Eigen/CXX11/Tensor>

namespace cppnn {

template<typename Scalar>
using RowVector = Eigen::Matrix<Scalar,1,Eigen::Dynamic,Eigen::RowMajor,1,Eigen::Dynamic>;

template <typename Scalar>
using ColVector = Eigen::Matrix<Scalar,Eigen::Dynamic,1,Eigen::ColMajor,Eigen::Dynamic,1>;

template<typename Scalar>
using Matrix = Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor,Eigen::Dynamic,Eigen::Dynamic>;

template<typename Scalar>
using MatrixMap = Eigen::Map<Matrix<Scalar>>;

template<typename Scalar>
using Tensor4 = Eigen::Tensor<Scalar,4,Eigen::ColMajor,int>;

template<typename Scalar>
using Tensor4Map = Eigen::TensorMap<Tensor4<Scalar>>;

template<typename Scalar>
using Array4 = Eigen::array<Scalar,4>;

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
	static const Tensor4<Scalar> NULL_TENSOR;
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
	inline static Matrix<Scalar> map_tensor4_to_mat(Tensor4<Scalar> tensor) {
		int rows = tensor.dimension(0);
		return MatrixMap<Scalar>(tensor.data(), rows, tensor.size() / rows);
	};
	inline static Tensor4<Scalar> map_mat_to_tensor4(Matrix<Scalar> mat, Dimensions<int> dims) {
		assert(dims.get_points() == mat.cols());
		return Tensor4Map<Scalar>(mat.data(), mat.rows(), dims.get_dim1(),
				dims.get_dim2(), dims.get_dim3());
	};
	inline static void shuffle_mat_rows(Matrix<Scalar>& mat) {
		Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(mat.rows());
		perm.setIdentity();
		// Shuffle the indices of the identity matrix.
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		mat = perm * mat;
	};
	inline static void shuffle_tensor_rows(Tensor4<Scalar>& tensor) {
		Dimensions<int> tensor_dims(tensor.dimension(1), tensor.dimension(2), tensor.dimension(3));
		Matrix<Scalar> mat = map_tensor4_to_mat(tensor);
		shuffle_mat_rows(mat);
		tensor = map_mat_to_tensor4(mat, tensor_dims);
	};
	inline static Dimensions<int> get_dims(const Tensor4<Scalar>& tensor) {
		return Dimensions<int>(tensor.dimension(1), tensor.dimension(2), tensor.dimension(3));
	};
};

template<typename Scalar>
const Matrix<Scalar> Utils<Scalar>::NULL_MATRIX = Matrix<Scalar>();

template<typename Scalar>
const Tensor4<Scalar> Utils<Scalar>::NULL_TENSOR = Tensor4<Scalar>();

}

#endif /* UTILS_H_ */
