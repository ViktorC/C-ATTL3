/*
 * Tensor.h
 *
 *  Created on: 13 Jan 2018
 *      Author: Viktor Csomor
 */

#ifndef TENSOR_H_
#define TENSOR_H_

#include <Dimensions.h>
#include <Eigen/Dense>
#include <Matrix.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Vector.h>

namespace cppnn {

template<typename Scalar>
using Tensor3D = Eigen::Tensor<Scalar,3,Eigen::ColMajor,int>;

template<typename Scalar>
using Tensor4D = Eigen::Tensor<Scalar,4,Eigen::ColMajor,int>;

template<typename Scalar>
static RowVector<Scalar> tensor3d_to_vec(Tensor3D<Scalar> tensor) {
	return Eigen::Map<RowVector<Scalar>>(tensor.data(), tensor.size());
};
template<typename Scalar>
static Matrix<Scalar> tensor4d_to_mat(Tensor4D<Scalar> tensor) {
	int rows = tensor.dimension(0);
	return Eigen::Map<Matrix<Scalar>>(tensor.data(), rows, tensor.size() / rows);
};
template<typename Scalar>
static Tensor3D<Scalar> vec_to_tensor3d(RowVector<Scalar> vec, Dimensions dims) {
	return Eigen::TensorMap<Tensor3D<Scalar>>(vec.data(), (int) dims.get_dim1(),
			(int) dims.get_dim2(), (int) dims.get_dim3());
};
template<typename Scalar>
static Tensor4D<Scalar> mat_to_tensor4d(Matrix<Scalar> mat, Dimensions dims) {
	return Eigen::TensorMap<Tensor4D<Scalar>>(mat.data(), mat.rows(), (int) dims.get_dim1(),
			(int) dims.get_dim2(), (int) dims.get_dim3());
};

} /* namespace cppnn */

#endif /* TENSOR_H_ */
