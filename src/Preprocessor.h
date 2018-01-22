/*
 * Preprocessor.h
 *
 *  Created on: 12.12.2017
 *      Author: Viktor Csomor
 */

#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include <string>
#include <type_traits>
#include <Utils.h>
#include <vector>

namespace cppnn {

template<typename Scalar>
class Preprocessor {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~Preprocessor() = default;
	virtual void fit(const Tensor4<Scalar>& data) = 0;
	virtual void transform(Tensor4<Scalar>& data) const = 0;
};

template<typename Scalar>
class NormalizationPreprocessor : public Preprocessor<Scalar> {
public:
	NormalizationPreprocessor(bool standardize = false,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				standardize(standardize),
				epsilon(epsilon) {
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	virtual ~NormalizationPreprocessor() = default;
	virtual void fit(const Tensor4<Scalar>& data) {
		int rows = data.dimension(0);
		assert(rows > 0);
		dims = Dimensions<int>(data.dimension(1), data.dimension(2), data.dimension(3));
		Array4<int> offsets = { 0, 0, 0, 0 };
		Array4<int> extents = { rows, dims.get_dim1(), dims.get_dim2(), 1 };
		means = Matrix<Scalar>(dims.get_dim3(), dims.get_dim1() * dims.get_dim2());
		sd = Matrix<Scalar>(means.rows(), means.cols());
		for (int i = 0; i < dims.get_dim3(); i++) {
			offsets[3] = i;
			Tensor4<Scalar> data_slice_i = data.slice(offsets, extents);
			Matrix<Scalar> data_mat = Utils<Scalar>::tensor4d_to_mat(data_slice_i);
			means.row(i) = data_mat.colwise().mean();
			if (standardize)
				sd.row(i) = (data_mat.rowwise() - means.row(i)).array().square().colwise().mean().sqrt();
		}
	};
	virtual void transform(Tensor4<Scalar>& data) const {
		int rows = data.dimension(0);
		assert(rows > 0);
		assert(dims.get_dim1() == data.dimension(1) && dims.get_dim2() == data.dimension(2) &&
				dims.get_dim3() == data.dimension(3) && "mismatched fit and transform input tensor dimensions");
		Dimensions<int> slice_dims(dims.get_dim1(), dims.get_dim2(), 1);
		Array4<int> offsets = { 0, 0, 0, 0 };
		Array4<int> extents = { rows, slice_dims.get_dim1(), slice_dims.get_dim2(), slice_dims.get_dim3() };
		for (int i = 0; i < dims.get_dim3(); i++) {
			offsets[3] = i;
			Tensor4<Scalar> data_slice_i = data.slice(offsets, extents);
			Matrix<Scalar> data_ch_i = Utils<Scalar>::tensor4d_to_mat(data_slice_i);
			data_ch_i = data_ch_i.rowwise() - means.row(i);
			if (standardize)
				data_ch_i *= (sd.row(i).array() + epsilon).inverse().matrix().asDiagonal();
			data.slice(offsets, extents) = Utils<Scalar>::mat_to_tensor4d(data_ch_i, slice_dims);
		}
	};
protected:
	bool standardize;
	Scalar epsilon;
	Dimensions<int> dims;
	Matrix<Scalar> means;
	Matrix<Scalar> sd;
};

template<typename Scalar>
class PCAPreprocessor : public NormalizationPreprocessor<Scalar> {
public:
	PCAPreprocessor(bool standardize = false, bool whiten = false, Scalar min_rel_var_to_retain = 1,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				NormalizationPreprocessor<Scalar>::NormalizationPreprocessor(standardize, epsilon),
				whiten(whiten),
				min_rel_var_to_retain(min_rel_var_to_retain) {
		assert(min_rel_var_to_retain > 0 && min_rel_var_to_retain <= 1 &&
				"the minimum relative variance to be retained must be greater "
				"then 0 and less than or equal to 1");
	};
	void fit(const Tensor4<Scalar>& data) {
		NormalizationPreprocessor<Scalar>::fit(data);
		Tensor4<Scalar> norm_data = data;
		NormalizationPreprocessor<Scalar>::transform(norm_data);
		Matrix<Scalar> norm_data_mat = Utils<Scalar>::tensor4d_to_mat(norm_data);
		// Compute the covariance matrix.
		Matrix<Scalar> cov = norm_data_mat.transpose() * norm_data_mat / norm_data_mat.rows();
		// Eigen decomposition.
		Eigen::SelfAdjointEigenSolver<Matrix<Scalar>> eigen_solver(cov);
		// Determine the number of components to retain.
		const ColVector<Scalar>& eigen_values = eigen_solver.eigenvalues();
		const Scalar min_var_to_retain = eigen_values.sum() * min_rel_var_to_retain;
		Scalar var = 0;
		int dims_to_retain = 0;
		for (; dims_to_retain < eigen_values.rows(); dims_to_retain++) {
			// The eigen values are in ascending order.
			var += eigen_values(eigen_values.rows() - (1 + dims_to_retain));
			if (Utils<Scalar>::decidedly_greater(var, min_var_to_retain))
				break;
		}
		// The eigen vectors are sorted by the magnitude of their corresponding eigen values.
		eigen_basis = eigen_solver.eigenvectors().rightCols(dims_to_retain);
		if (whiten) // The eigen values are only needed if whitening is enabled.
			this->eigen_values = eigen_values.bottomRows(dims_to_retain).transpose();
	};
	void transform(Tensor4<Scalar>& data) const {
		NormalizationPreprocessor<Scalar>::transform(data);
		Matrix<Scalar> data_mat = Utils<Scalar>::tensor4d_to_mat(data);
		data_mat *= eigen_basis;
		if (whiten)
			data_mat *= (eigen_values.array() + NormalizationPreprocessor<Scalar>::epsilon)
					.sqrt().inverse().matrix().asDiagonal();
		data = Utils<Scalar>::mat_to_tensor4d(data_mat, Dimensions<int>(data_mat.cols(), 1, 1));
	};
private:
	bool whiten;
	Scalar min_rel_var_to_retain;
	Matrix<Scalar> eigen_basis;
	RowVector<Scalar> eigen_values;
};

} /* namespace cppnn */

#endif /* PREPROCESSOR_H_ */
