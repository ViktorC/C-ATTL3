/*
 * Preprocessor.h
 *
 *  Created on: 12.12.2017
 *      Author: Viktor Csomor
 */

#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <Eigen/Dense>
#include <string>
#include <type_traits>
#include <Utils.h>
#include <vector>

namespace cattle {

template<typename Scalar, size_t Rank, bool Sequential>
class Preprocessor {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal pre-processor rank");
public:
	virtual ~Preprocessor() = default;
	virtual void fit(const Tensor<Scalar,Rank + Sequential + 1>& data) = 0;
	virtual void transform(Tensor<Scalar,Rank + Sequential + 1>& data) const = 0;
};

template<typename Scalar, size_t Rank>
class NormalizationPreprocessor : public Preprocessor<Scalar,Rank,false> {
public:
	NormalizationPreprocessor(bool standardize = false, Scalar epsilon = Utils<Scalar>::EPSILON2) :
			standardize(standardize),
			epsilon(epsilon) {
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	virtual ~NormalizationPreprocessor() = default;
	inline virtual void fit(const Tensor<Scalar,Rank + 1>& data) {
		int rows = data.dimension(0);
		assert(rows > 0);
		dims = Utils<Scalar>::template get_dims<Rank + 1>(data).demote();
		Matrix<Scalar> data_mat = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(data);
		means = data_mat.colwise().mean();
		if (standardize)
			sd = (data_mat.rowwise() - means).array().square().colwise().mean().sqrt();
	};
	inline virtual void transform(Tensor<Scalar,Rank + 1>& data) const {
		int rows = data.dimension(0);
		assert(rows > 0);
		Dimensions<int,Rank> demoted_dims = Utils<Scalar>::template get_dims<Rank + 1>(data).demote();
		assert(demoted_dims == dims && "mismatched fit and transform input tensor dimensions");
		Matrix<Scalar> data_mat = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(data));
		data_mat = data_mat.rowwise() - means;
		if (standardize)
			data_mat *= (sd.array() + epsilon).inverse().matrix().asDiagonal();
		data = Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(data_mat, demoted_dims);
	};
protected:
	const bool standardize;
	const Scalar epsilon;
	Dimensions<int,Rank> dims;
	RowVector<Scalar> means;
	RowVector<Scalar> sd;
};

// Partial template specialization for batches of 3D tensors (with multiple channels).
template<typename Scalar>
class NormalizationPreprocessor<Scalar,3> {
public:
	NormalizationPreprocessor(bool standardize = false, Scalar epsilon = Utils<Scalar>::EPSILON2) :
			standardize(standardize),
			epsilon(epsilon) {
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	virtual ~NormalizationPreprocessor() = default;
	inline virtual void fit(const Tensor<Scalar,4>& data) {
		int rows = data.dimension(0);
		assert(rows > 0);
		dims = Utils<Scalar>::template get_dims<4>(data).demote();
		int depth = dims(2);
		means = Matrix<Scalar>(depth, dims.get_volume() / depth);
		sd = Matrix<Scalar>(means.rows(), means.cols());
		std::array<int,4> offsets({ 0, 0, 0, 0 });
		std::array<int,4> extents({ rows, dims(0), dims(1), 1 });
		for (int i = 0; i < depth; i++) {
			offsets[3] = i;
			Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
			Matrix<Scalar> data_mat = Utils<Scalar>::template map_tensor_to_mat<4>(std::move(data_slice_i));
			means.row(i) = data_mat.colwise().mean();
			if (standardize)
				sd.row(i) = (data_mat.rowwise() - means.row(i)).array().square().colwise().mean().sqrt();
		}
	};
	inline virtual void transform(Tensor<Scalar,4>& data) const {
		int rows = data.dimension(0);
		Dimensions<int,3> demoted_dims = Utils<Scalar>::template get_dims<4>(data).demote();
		assert(demoted_dims == dims && "mismatched fit and transform input tensor dimensions");
		assert(rows > 0);
		int depth = data.dimension(3);
		Dimensions<int,3> slice_dims({ dims(0), dims(1), 1 });
		std::array<int,4> offsets = { 0, 0, 0, 0 };
		std::array<int,4> extents = { rows, slice_dims(0), slice_dims(1), slice_dims(2) };
		for (int i = 0; i < depth; i++) {
			offsets[3] = i;
			Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
			Matrix<Scalar> data_ch_i = Utils<Scalar>::template map_tensor_to_mat<4>(std::move(data_slice_i));
			data_ch_i = data_ch_i.rowwise() - means.row(i);
			if (standardize)
				data_ch_i *= (sd.row(i).array() + epsilon).inverse().matrix().asDiagonal();
			data.slice(offsets, extents) = Utils<Scalar>::template map_mat_to_tensor<4>(data_ch_i, slice_dims);
		}
	};
protected:
	const bool standardize;
	const Scalar epsilon;
	Dimensions<int,3> dims;
	Matrix<Scalar> means;
	Matrix<Scalar> sd;
};

template<typename Scalar, size_t Rank>
class PCAPreprocessorBase : public NormalizationPreprocessor<Scalar,Rank> {
protected:
	typedef NormalizationPreprocessor<Scalar,Rank> Base;
	PCAPreprocessorBase(bool standardize, bool whiten, Scalar min_rel_var_to_retain, Scalar epsilon) :
				Base::NormalizationPreprocessor(standardize, epsilon),
				whiten(whiten),
				min_rel_var_to_retain(min_rel_var_to_retain),
				reduce_dims(Utils<Scalar>::decidedly_lesser(min_rel_var_to_retain, (Scalar) 1)) {
		assert(min_rel_var_to_retain > 0 && min_rel_var_to_retain <= 1 &&
				"the minimum relative variance to be retained must be greater "
				"then 0 and less than or equal to 1");
	};
	inline void _fit(Tensor<Scalar,Rank + 1> data, int i) {
		Matrix<Scalar> normalized_data = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(data)).rowwise() -
				Base::means.row(i);
		if (Base::standardize)
			normalized_data *= (Base::sd.row(i).array() + Base::epsilon).inverse().matrix().asDiagonal();
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
	inline Tensor<Scalar,Rank + 1> _transform(Tensor<Scalar,Rank + 1> data, int i) const {
		Dimensions<int,Rank> output_dims;
		if (!reduce_dims)
			output_dims = Utils<Scalar>::template get_dims<Rank + 1>(data).demote();
		Matrix<Scalar> data_mat = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(data));
		data_mat *= ed_vec[i].eigen_basis;
		if (whiten)
			data_mat *= (ed_vec[i].eigen_values.array() + Base::epsilon).sqrt().inverse().matrix().asDiagonal();
		if (reduce_dims)
			output_dims(0) = data_mat.cols();
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(data_mat, output_dims);
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

template<typename Scalar, size_t Rank>
class PCAPreprocessor : public PCAPreprocessorBase<Scalar,Rank> {
	typedef PCAPreprocessorBase<Scalar,Rank> Base;
	typedef typename Base::Base BaseBase;
public:
	PCAPreprocessor(bool standardize = false, bool whiten = false, Scalar min_rel_var_to_retain = 1,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				Base::PCAPreprocessorBase(standardize, whiten, min_rel_var_to_retain, epsilon) { };
	inline void fit(const Tensor<Scalar,Rank + 1>& data) {
		BaseBase::fit(data);
		Base::ed_vec = std::vector<typename Base::EigenDecomposition>(1);
		Base::_fit(data, 0);
	};
	inline void transform(Tensor<Scalar,Rank + 1>& data) const {
		BaseBase::transform(data);
		data = Base::_transform(std::move(data), 0);
	};
};

// 3D partial template specialization of the PCA preprocessor.
template<typename Scalar>
class PCAPreprocessor<Scalar,3> : public PCAPreprocessorBase<Scalar,3> {
	typedef PCAPreprocessorBase<Scalar,3> Base;
	typedef typename Base::Base BaseBase;
public:
	PCAPreprocessor(bool standardize = false, bool whiten = false, Scalar min_rel_var_to_retain = 1,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				Base::PCAPreprocessorBase(standardize, whiten, min_rel_var_to_retain, epsilon) { };
	inline void fit(const Tensor<Scalar,4>& data) {
		assert((!Base::reduce_dims || data.dimension(3) == 1) && "cannot reduce the dimensionality of multi-channel tensors");
		BaseBase::fit(data);
		int channels = BaseBase::dims(2);
		Base::ed_vec = std::vector<typename Base::EigenDecomposition>(channels);
		std::array<int,4> offsets({ 0, 0, 0, 0 });
		std::array<int,4> extents({ data.dimension(0), BaseBase::dims(0), BaseBase::dims(1), 1 });
		for (int i = 0; i < channels; i++) {
			offsets[3] = i;
			Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
			Base::_fit(std::move(data_slice_i), i);
		}
	};
	inline void transform(Tensor<Scalar,4>& data) const {
		BaseBase::transform(data);
		if (Base::reduce_dims) {
			Dimensions<int,3> output_dims({ data.dimension(1) * data.dimension(2), 1, 1 });
			data = Base::_transform(std::move(data), 0);
		} else {
			int rows = data.dimension(0);
			std::array<int,4> offsets({ 0, 0, 0, 0 });
			std::array<int,4> extents({ rows, BaseBase::dims(0), BaseBase::dims(1), 1 });
			for (int i = 0; i < BaseBase::dims(2); i++) {
				offsets[3] = i;
				Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
				data.slice(offsets, extents) = Base::_transform(std::move(data_slice_i), i);
			}
		}
	};
};

} /* namespace cattle */

#endif /* PREPROCESSOR_H_ */
