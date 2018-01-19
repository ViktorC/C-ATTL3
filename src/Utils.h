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
private:
	Utils() { };
public:
	static constexpr Scalar MIN = std::numeric_limits<Scalar>::lowest();
	static constexpr Scalar MAX = std::numeric_limits<Scalar>::max();
	static constexpr Scalar EPSILON1 = std::numeric_limits<Scalar>::epsilon();
	static constexpr Scalar EPSILON2 = 1e-8;
	static constexpr Scalar EPSILON3 = 1e-4;
	static bool almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		Scalar diff = std::abs(n1 - n2);
		if (diff <= abs_epsilon)
			return true;
		Scalar max = std::max(std::abs(n1), std::abs(n2));
		return diff <= max * rel_epsilon;
	};
	static bool decidedly_greater(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	static bool greater_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	bool decidedly_lesser(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	static bool lesser_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	static Matrix<Scalar> tensor4d_to_mat(Tensor4<Scalar> tensor) {
		int rows = tensor.dimension(0);
		return MatrixMap<Scalar>(tensor.data(), rows, tensor.size() / rows);
	};
	static Tensor4<Scalar> mat_to_tensor4d(Matrix<Scalar> mat, Dimensions dims) {
		return Tensor4Map<Scalar>(mat.data(), mat.rows(), dims.get_dim1(),
				dims.get_dim2(), dims.get_dim3());
	};
};

}

#endif /* UTILS_H_ */
