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
#include <cstddef>
#include <Eigen/Dense>
#include <string>
#include <type_traits>
#include <Utils.h>
#include <vector>

namespace cattle {

template<typename Scalar, size_t Rank>
class Preprocessor {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal pre-processor rank");
public:
	virtual ~Preprocessor() = default;
	virtual void fit(const Tensor<Scalar,Rank + 1>& data) = 0;
	virtual void transform(Tensor<Scalar,Rank + 1>& data) const = 0;
};

template<typename Scalar, size_t Rank>
class NormalizationPreprocessor : public Preprocessor<Scalar,Rank> {
public:
	NormalizationPreprocessor(bool standardize = false,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				standardize(standardize),
				epsilon(epsilon) {
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	virtual ~NormalizationPreprocessor() = default;
	inline virtual void fit(const Tensor<Scalar,Rank + 1>& data) {
		int rows = data.dimension(0);
		assert(rows > 0);
		dims = Utils<Scalar>::get_dims(data);
		if (Rank < 3) {
			Matrix<Scalar> data_mat = Utils<Scalar>::map_tensor_to_mat(data);
			means = data_mat.colwise().mean();
			if (standardize)
				sd = (data_mat.rowwise() - means).array().square().colwise().mean().sqrt();
		} else {
			int depth = data.dimension(3);
			means = Matrix<Scalar>(depth, dims.get_volume() / depth);
			sd = Matrix<Scalar>(means.rows(), means.cols());
			Array<int,4> offsets({ 0, 0, 0, 0 });
			Array<int,4> extents({ rows, dims(0), dims(1), 1 });
			for (int i = 0; i < depth; i++) {
				offsets[3] = i;
				Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
				Matrix<Scalar> data_mat = Utils<Scalar>::map_tensor_to_mat(std::move(data_slice_i));
				means.row(i) = data_mat.colwise().mean();
				if (standardize)
					sd.row(i) = (data_mat.rowwise() - means.row(i)).array().square().colwise().mean().sqrt();
			}
		}
	};
	inline virtual void transform(Tensor<Scalar,Rank + 1>& data) const {
		int rows = data.dimension(0);
		Dimensions<int,3> demoted_dims = Utils<Scalar>::get_dims(data).demote();
		assert(demoted_dims == dims && "mismatched fit and transform input tensor dimensions");
		assert(rows > 0);
		if (Rank < 3) {
			Matrix<Scalar> data_mat = Utils<Scalar>::map_tensor_to_mat(std::move(data));
			data_mat = data_mat.rowwise() - means;
			if (standardize)
				data_mat *= (sd.array() + epsilon).inverse().matrix().asDiagonal();
			data = Utils<Scalar>::map_mat_to_tensor<Rank + 1>(data_mat, demoted_dims);
		} else {
			int depth = data.dimension(3);
			Dimensions<int,3> slice_dims(dims(0), dims(1), 1);
			Array<int,4> offsets = { 0, 0, 0, 0 };
			Array<int,4> extents = { rows, slice_dims(0), slice_dims(1), slice_dims(2) };
			for (int i = 0; i < depth; i++) {
				offsets[3] = i;
				Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
				Matrix<Scalar> data_ch_i = Utils<Scalar>::map_tensor_to_mat(std::move(data_slice_i));
				data_ch_i = data_ch_i.rowwise() - means.row(i);
				if (standardize)
					data_ch_i *= (sd.row(i).array() + epsilon).inverse().matrix().asDiagonal();
				data.slice(offsets, extents) = Utils<Scalar>::map_mat_to_tensor<4>(data_ch_i, slice_dims);
			}
		}
	};
protected:
	bool standardize;
	Scalar epsilon;
	Dimensions<int,Rank> dims;
	Matrix<Scalar> means;
	Matrix<Scalar> sd;
};

template<typename Scalar, size_t Rank>
class PCAPreprocessor : public NormalizationPreprocessor<Scalar,Rank> {
public:
	PCAPreprocessor(bool standardize = false, bool whiten = false, Scalar min_rel_var_to_retain = 1,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				NormalizationPreprocessor<Scalar,Rank>::NormalizationPreprocessor(standardize, epsilon),
				whiten(whiten),
				min_rel_var_to_retain(min_rel_var_to_retain),
				reduce_dims(Utils<Scalar>::decidedly_lesser(min_rel_var_to_retain, (Scalar) 1)) {
		assert(min_rel_var_to_retain > 0 && min_rel_var_to_retain <= 1 &&
				"the minimum relative variance to be retained must be greater "
				"then 0 and less than or equal to 1");
	};
	inline void fit(const Tensor<Scalar,Rank + 1>& data) {
		NormalizationPreprocessor<Scalar,Rank>::fit(data);
		if (Rank < 3) {
			ed_vec = std::vector<EigenDecomposition>(1);
			_fit(data, 0);
		} else {
			assert((!reduce_dims || data.dimension(3) == 1) && "cannot reduce the dimensionality of multi-channel tensors");
			int channels = NormalizationPreprocessor<Scalar,Rank>::dims(2);
			ed_vec = std::vector<EigenDecomposition>(channels);
			Array<int,4> offsets({ 0, 0, 0, 0 });
			Array<int,4> extents({ data.dimension(0), NormalizationPreprocessor<Scalar,Rank>::dims(0),
					NormalizationPreprocessor<Scalar,Rank>::dims(1), 1 });
			for (int i = 0; i < channels; i++) {
				offsets[3] = i;
				Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
				_fit(std::move(data), 0);
			}
		}
	};
	inline void transform(Tensor<Scalar,Rank + 1>& data) const {
		NormalizationPreprocessor<Scalar,Rank>::transform(data);
		if (reduce) {
			Dimensions<int,1> output_dims({ Utils<Scalar>::get_dims(data).demote().get_volume() });
			data = _transform(std::move(data), output_dims, 0);
		} else {
			if (Rank < 3)
				data = _transform(std::move(data), Utils<Scalar>::get_dims(data).demote(), 0);
			else {
				int rows = data.dimension(0);
				Dimensions<int,Rank> slice_dims(NormalizationPreprocessor<Scalar,Rank>::dims.get_height(),
						NormalizationPreprocessor<Scalar,Rank>::dims.get_width(), 1);
				Array<int,4> offsets({ 0, 0, 0, 0 });
				Array<int,4> extents({ rows, slice_dims(0), slice_dims(1), slice_dims(2) });
				for (int i = 0; i < data.dimension(3); i++) {
					offsets[3] = i;
					Tensor4<Scalar> data_slice_i = data.slice(offsets, extents);
					data.slice(offsets, extents) = _transform(std::move(data_slice_i), slice_dims, i);
				}
			}
		}
	};
private:
	inline void _fit(Tensor<Scalar,Rank + 1> data, int i) {
		Matrix<Scalar> normalized_data = Utils<Scalar>::map_tensor_to_mat(std::move(data)).rowwise() -
				NormalizationPreprocessor<Scalar,Rank>::means.row(i);
		if (NormalizationPreprocessor<Scalar,Rank>::standardize)
			normalized_data *= (NormalizationPreprocessor<Scalar,Rank>::sd.row(i).array() +
					NormalizationPreprocessor<Scalar,Rank>::epsilon).inverse().matrix().asDiagonal();
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
	};
	inline Tensor<Scalar,Rank + 1> _transform(Tensor<Scalar,Rank + 1> data, const Dimensions<int,Rank>& output_dims,
			int i) const {
		Matrix<Scalar> data_mat = Utils<Scalar>::map_tensor_to_mat(std::move(data));
		data_mat *= ed_vec[i].eigen_basis;
		if (whiten)
			data_mat *= (ed_vec[i].eigen_values.array() + NormalizationPreprocessor<Scalar,Rank>::epsilon)
					.sqrt().inverse().matrix().asDiagonal();
		return Utils<Scalar>::map_mat_to_tensor(data_mat, output_dims);
	};
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
