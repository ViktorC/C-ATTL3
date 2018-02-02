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

namespace cattle {

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
	inline virtual void fit(const Tensor4<Scalar>& data) {
		int rows = data.dimension(0);
		assert(rows > 0);
		dims = Dimensions<int>(data.dimension(1), data.dimension(2), data.dimension(3));
		Array4<int> offsets = { 0, 0, 0, 0 };
		Array4<int> extents = { rows, dims.get_height(), dims.get_width(), 1 };
		means = Matrix<Scalar>(dims.get_depth(), dims.get_height() * dims.get_width());
		sd = Matrix<Scalar>(means.rows(), means.cols());
		for (int i = 0; i < dims.get_depth(); i++) {
			offsets[3] = i;
			Tensor4<Scalar> data_slice_i = data.slice(offsets, extents);
			Matrix<Scalar> data_mat = Utils<Scalar>::map_tensor4_to_mat(data_slice_i);
			means.row(i) = data_mat.colwise().mean();
			if (standardize)
				sd.row(i) = (data_mat.rowwise() - means.row(i)).array().square().colwise().mean().sqrt();
		}
	};
	inline virtual void transform(Tensor4<Scalar>& data) const {
		int rows = data.dimension(0);
		assert(rows > 0);
		assert(dims.get_height() == data.dimension(1) && dims.get_width() == data.dimension(2) &&
				dims.get_depth() == data.dimension(3) && "mismatched fit and transform input tensor dimensions");
		Dimensions<int> slice_dims(dims.get_height(), dims.get_width(), 1);
		Array4<int> offsets = { 0, 0, 0, 0 };
		Array4<int> extents = { rows, slice_dims.get_height(), slice_dims.get_width(), slice_dims.get_depth() };
		for (int i = 0; i < dims.get_depth(); i++) {
			offsets[3] = i;
			Tensor4<Scalar> data_slice_i = data.slice(offsets, extents);
			Matrix<Scalar> data_ch_i = Utils<Scalar>::map_tensor4_to_mat(data_slice_i);
			data_ch_i = data_ch_i.rowwise() - means.row(i);
			if (standardize)
				data_ch_i *= (sd.row(i).array() + epsilon).inverse().matrix().asDiagonal();
			data.slice(offsets, extents) = Utils<Scalar>::map_mat_to_tensor4(data_ch_i, slice_dims);
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
				min_rel_var_to_retain(min_rel_var_to_retain),
				reduce_dims(Utils<Scalar>::decidedly_lesser(min_rel_var_to_retain, (Scalar) 1)) {
		assert(min_rel_var_to_retain > 0 && min_rel_var_to_retain <= 1 &&
				"the minimum relative variance to be retained must be greater "
				"then 0 and less than or equal to 1");
	};
	inline void fit(const Tensor4<Scalar>& data) {
		assert((!reduce_dims || data.dimension(3) == 1) &&
				"cannot reduce the dimensionality of multi-channel tensors");
		NormalizationPreprocessor<Scalar>::fit(data);
		int rows = data.dimension(0);
		int channels = NormalizationPreprocessor<Scalar>::dims.get_depth();
		Array4<int> offsets({ 0, 0, 0, 0 });
		Array4<int> extents({ rows, NormalizationPreprocessor<Scalar>::dims.get_height(),
				NormalizationPreprocessor<Scalar>::dims.get_width(), 1 });
		ed_vec = std::vector<EigenDecomposition>(channels);
		for (int i = 0; i < channels; i++) {
			offsets[3] = i;
			Tensor4<Scalar> data_slice_i = data.slice(offsets, extents);
			Matrix<Scalar> normalized_data = Utils<Scalar>::map_tensor4_to_mat(data_slice_i).rowwise() -
					NormalizationPreprocessor<Scalar>::means.row(i);
			if (NormalizationPreprocessor<Scalar>::standardize)
				normalized_data *= (NormalizationPreprocessor<Scalar>::sd.row(i).array() +
						NormalizationPreprocessor<Scalar>::epsilon).inverse().matrix().asDiagonal();
			// Compute the covariance matrix.
			Matrix<Scalar> cov = normalized_data.transpose() * normalized_data / normalized_data.rows();
			// Eigen decomposition.
			Eigen::SelfAdjointEigenSolver<Matrix<Scalar>> eigen_solver(cov);
			// Determine the number of components to retain.
			const ColVector<Scalar>& eigen_values = eigen_solver.eigenvalues();
			int dims_to_retain = 0;
			if (reduce_dims) {
				const Scalar min_var_to_retain = eigen_values.sum() * min_rel_var_to_retain;
				Scalar var = 0;
				for (; dims_to_retain < eigen_values.rows(); dims_to_retain++) {
					// The eigen values are in ascending order.
					var += eigen_values(eigen_values.rows() - (1 + dims_to_retain));
					if (Utils<Scalar>::decidedly_greater(var, min_var_to_retain))
						break;
				}
			} else
				dims_to_retain = eigen_values.rows();
			// The eigen vectors are sorted by the magnitude of their corresponding eigen values.
			ed_vec[i].eigen_basis = eigen_solver.eigenvectors().rightCols(dims_to_retain);
			if (whiten) // The eigen values are only needed if whitening is enabled.
				ed_vec[i].eigen_values = eigen_values.bottomRows(dims_to_retain).transpose();
		}
	};
	inline void transform(Tensor4<Scalar>& data) const {
		NormalizationPreprocessor<Scalar>::transform(data);
		if (reduce_dims) {
			Matrix<Scalar> data_mat = Utils<Scalar>::map_tensor4_to_mat(data);
			data_mat *= ed_vec[0].eigen_basis;
			if (whiten)
				data_mat *= (ed_vec[0].eigen_values.array() + NormalizationPreprocessor<Scalar>::epsilon)
						.sqrt().inverse().matrix().asDiagonal();
			data = Utils<Scalar>::map_mat_to_tensor4(data_mat, Dimensions<int>(data_mat.cols()));
		} else {
			int rows = data.dimension(0);
			Dimensions<int> slice_dims(NormalizationPreprocessor<Scalar>::dims.get_height(),
					NormalizationPreprocessor<Scalar>::dims.get_width(), 1);
			Array4<int> offsets({ 0, 0, 0, 0 });
			Array4<int> extents({ rows, slice_dims.get_height(), slice_dims.get_width(), slice_dims.get_depth() });
			for (int i = 0; i < data.dimension(3); i++) {
				offsets[3] = i;
				Tensor4<Scalar> data_slice_i = data.slice(offsets, extents);
				Matrix<Scalar> data_ch_i = Utils<Scalar>::map_tensor4_to_mat(data_slice_i);
				data_ch_i *= ed_vec[i].eigen_basis;
				if (whiten)
					data_ch_i *= (ed_vec[i].eigen_values.array() + NormalizationPreprocessor<Scalar>::epsilon)
							.sqrt().inverse().matrix().asDiagonal();
				data.slice(offsets, extents) = Utils<Scalar>::map_mat_to_tensor4(data_ch_i, slice_dims);
			}
		}
	};
private:
	bool whiten;
	Scalar min_rel_var_to_retain;
	bool reduce_dims;
	struct EigenDecomposition {
		Matrix<Scalar> eigen_basis;
		RowVector<Scalar> eigen_values;
	};
	std::vector<EigenDecomposition> ed_vec;
};

} /* namespace cattle */

#endif /* PREPROCESSOR_H_ */
