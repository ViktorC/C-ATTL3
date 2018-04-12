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
#include <string>
#include <type_traits>
#include <vector>
#include "utils/Eigen.h"
#include "utils/NumericUtils.h"

namespace cattle {

// TODO Preprocessors for sequential data.

/**
 * An abstract class template for data preprocessors.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class Preprocessor {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal pre-processor rank");
public:
	virtual ~Preprocessor() = default;
	/**
	 * It fits the preprocessor to the specified data.
	 *
	 * @param data A constant reference to a data tensor.
	 */
	virtual void fit(const Tensor<Scalar,Rank + Sequential + 1>& data) = 0;
	/**
	 * It transforms the specified tensor according to the preprocessors current state
	 * created by #fit(const Tensor<Scalar,Rank + Sequential + 1>&).
	 *
	 * @param data A non-constatn reference to a data tensor.
	 */
	virtual void transform(Tensor<Scalar,Rank + Sequential + 1>& data) const = 0;
};

/**
 * A class template for a normalization (mean-subtraction) preprocessor that optionally also
 * standardizes the data (divides it by the standard deviation).
 */
template<typename Scalar, std::size_t Rank, bool Standardize = false>
class NormalizationPreprocessor : public Preprocessor<Scalar,Rank,false> {
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline NormalizationPreprocessor(Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
			epsilon(epsilon) {
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	virtual ~NormalizationPreprocessor() = default;
	inline virtual void fit(const Tensor<Scalar,Rank + 1>& data) {
		std::size_t rows = data.dimension(0);
		assert(rows > 0);
		dims = (Dimensions<std::size_t,Rank + 1>(data.dimensions()).template demote<>());
		MatrixMap<Scalar> data_mat(data.data(), rows, data.size() / rows);
		means = data_mat.colwise().mean();
		if (Standardize)
			sd = (data_mat.rowwise() - means).array().square().colwise().mean().sqrt();
	}
	inline virtual void transform(Tensor<Scalar,Rank + 1>& data) const {
		std::size_t rows = data.dimension(0);
		assert(rows > 0);
		Dimensions<std::size_t,Rank> demoted_dims = (Dimensions<std::size_t,Rank + 1>(data.dimensions()).template demote<>());
		assert(demoted_dims == dims && "mismatched fit and transform input tensor dimensions");
		MatrixMap<Scalar> data_mat(data.data(), rows, data.size() / rows);
		data_mat = data_mat.rowwise() - means;
		if (Standardize)
			data_mat *= (sd.array() + epsilon).inverse().matrix().asDiagonal();
	}
protected:
	const Scalar epsilon;
	Dimensions<std::size_t,Rank> dims;
	RowVector<Scalar> means;
	RowVector<Scalar> sd;
};

/**
 * Partial template specialization for batches of rank-3 tensors (with multiple channels).
 */
template<typename Scalar, bool Standardize>
class NormalizationPreprocessor<Scalar,3,Standardize> {
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline NormalizationPreprocessor(Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
			epsilon(epsilon) {
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	virtual ~NormalizationPreprocessor() = default;
	inline virtual void fit(const Tensor<Scalar,4>& data) {
		std::size_t rows = data.dimension(0);
		assert(rows > 0);
		dims = (Dimensions<std::size_t,4>(data.dimensions()).template demote<>());
		std::size_t depth = dims(2);
		means = Matrix<Scalar>(depth, dims.get_volume() / depth);
		sd = Matrix<Scalar>(means.rows(), means.cols());
		std::array<std::size_t,4> offsets({ 0u, 0u, 0u, 0u });
		std::array<std::size_t,4> extents({ rows, dims(0), dims(1), 1u });
		for (std::size_t i = 0; i < depth; ++i) {
			offsets[3] = i;
			Tensor<Scalar,4> data_slice = data.slice(offsets, extents);
			MatrixMap<Scalar> data_slice_mat(data_slice.data(), rows, data_slice.size() / rows);
			means.row(i) = data_slice_mat.colwise().mean();
			if (Standardize)
				sd.row(i) = (data_slice_mat.rowwise() - means.row(i)).array().square().colwise().mean().sqrt();
		}
	}
	inline virtual void transform(Tensor<Scalar,4>& data) const {
		std::size_t rows = data.dimension(0);
		Dimensions<std::size_t,3> demoted_dims = (Dimensions<std::size_t,4>(data.dimensions()).template demote<>());
		assert(demoted_dims == dims && "mismatched fit and transform input tensor dimensions");
		assert(rows > 0);
		std::size_t depth = data.dimension(3);
		std::array<std::size_t,4> offsets({ 0u, 0u, 0u, 0u });
		std::array<std::size_t,4> extents({ rows, dims(0), dims(1), 1u });
		for (std::size_t i = 0; i < depth; ++i) {
			offsets[3] = i;
			Tensor<Scalar,4> data_slice = data.slice(offsets, extents);
			MatrixMap<Scalar> data_slice_mat(data_slice.data(), rows, data_slice.size() / rows);
			data_slice_mat = data_slice_mat.rowwise() - means.row(i);
			if (Standardize)
				data_slice_mat *= (sd.row(i).array() + epsilon).inverse().matrix().asDiagonal();
			data.slice(offsets, extents) = std::move(data_slice);
		}
	}
protected:
	const Scalar epsilon;
	Dimensions<std::size_t,3> dims;
	Matrix<Scalar> means;
	Matrix<Scalar> sd;
};

// Hide the PCA preprocessor base from other translation units.
namespace {

/**
 * An abstract base class template for a principal component analysis (PCA) preprocessor that can also
 * standardize and whiten the data.
 */
template<typename Scalar, std::size_t Rank, bool Standardize = false, bool Whiten = false>
class PCAPreprocessorBase : public NormalizationPreprocessor<Scalar,Rank,Standardize> {
protected:
	typedef NormalizationPreprocessor<Scalar,Rank,Standardize> Base;
	inline PCAPreprocessorBase(Scalar min_rel_var_to_retain, Scalar epsilon) :
				Base::NormalizationPreprocessor(epsilon),
				min_rel_var_to_retain(min_rel_var_to_retain),
				reduce_dims(internal::NumericUtils<Scalar>::decidedly_lesser(min_rel_var_to_retain, (Scalar) 1)) {
		assert(min_rel_var_to_retain > 0 && min_rel_var_to_retain <= 1 &&
				"the minimum relative variance to be retained must be greater "
				"then 0 and less than or equal to 1");
	}
	inline void _fit(Tensor<Scalar,Rank + 1> data, int i) {
		std::size_t rows = data.dimension(0);
		MatrixMap<Scalar> data_mat(data.data(), rows, data.size() / rows);
		Matrix<Scalar> normalized_data = data_mat.rowwise() - Base::means.row(i);
		if (Standardize)
			normalized_data *= (Base::sd.row(i).array() + Base::epsilon).inverse().matrix().asDiagonal();
		// Compute the covariance matrix.
		Matrix<Scalar> cov = normalized_data.transpose() * normalized_data / normalized_data.rows();
		// Eigen decomposition.
		internal::EigenSolver<Scalar> eigen_solver(cov);
		// Determine the number of components to retain.
		const ColVector<Scalar>& eigen_values = eigen_solver.eigenvalues();
		int dims_to_retain = 0;
		if (reduce_dims) {
			const Scalar min_var_to_retain = eigen_values.sum() * min_rel_var_to_retain;
			Scalar var = 0;
			for (; dims_to_retain < eigen_values.rows(); ++dims_to_retain) {
				// The eigen values are in ascending order.
				var += eigen_values(eigen_values.rows() - (1 + dims_to_retain));
				if (internal::NumericUtils<Scalar>::decidedly_greater(var, min_var_to_retain))
					break;
			}
		} else
			dims_to_retain = eigen_values.rows();
		// The eigen vectors are sorted by the magnitude of their corresponding eigen values.
		ed_vec[i].eigen_basis = eigen_solver.eigenvectors().rightCols(dims_to_retain);
		if (Whiten) // The eigen values are only needed if whitening is enabled.
			ed_vec[i].eigen_values = eigen_values.bottomRows(dims_to_retain).transpose();
	}
	inline Tensor<Scalar,Rank + 1> _transform(Tensor<Scalar,Rank + 1> data, int i) const {
		std::size_t rows = data.dimension(0);
		Dimensions<std::size_t,Rank + 1> output_dims;
		output_dims(0) = rows;
		if (!reduce_dims)
			output_dims = data.dimensions();
		MatrixMap<Scalar> data_mat(data.data(), rows, data.size() / rows);
		data_mat *= ed_vec[i].eigen_basis;
		if (Whiten)
			data_mat *= (ed_vec[i].eigen_values.array() + Base::epsilon).sqrt().inverse().matrix().asDiagonal();
		if (reduce_dims)
			output_dims(1) = data_mat.cols();
		return TensorMap<Scalar,Rank + 1>(data.data(), output_dims);
	}
	Scalar min_rel_var_to_retain;
	bool reduce_dims;
	struct EigenDecomposition {
		Matrix<Scalar> eigen_basis;
		RowVector<Scalar> eigen_values;
	};
	std::vector<EigenDecomposition> ed_vec;
};

}

/**
 * A class template for a PCA preprocessor that can also standardize and whiten.
 */
template<typename Scalar, std::size_t Rank, bool Standardize = false, bool Whiten = false>
class PCAPreprocessor : public PCAPreprocessorBase<Scalar,Rank,Standardize,Whiten> {
	typedef PCAPreprocessorBase<Scalar,Rank,Standardize,Whiten> Base;
	typedef typename Base::Base Root;
public:
	/**
	 * @param min_rel_var_to_retain The minimum relative variance in the data
	 * to retain. It is expected to be within the range (0,1]. If it is 1,
	 * the dimensionality of the preprocessed data is guaranteed not to be
	 * reduced.
	 * @param epsilon A small consant used to maintain numerical stability.
	 */
	inline PCAPreprocessor(Scalar min_rel_var_to_retain = 1, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				Base::PCAPreprocessorBase(min_rel_var_to_retain, epsilon) { }
	inline void fit(const Tensor<Scalar,Rank + 1>& data) {
		Root::fit(data);
		Base::ed_vec = std::vector<typename Base::EigenDecomposition>(1);
		Base::_fit(data, 0);
	}
	inline void transform(Tensor<Scalar,Rank + 1>& data) const {
		Root::transform(data);
		data = Base::_transform(std::move(data), 0);
	}
};

/**
 * Partial template specialization of the PCA preprocessor for rank-3 tensors.
 */
template<typename Scalar, bool Standardize, bool Whiten>
class PCAPreprocessor<Scalar,3,Standardize,Whiten> : public PCAPreprocessorBase<Scalar,3,Standardize,Whiten> {
	typedef PCAPreprocessorBase<Scalar,3,Standardize,Whiten> Base;
	typedef typename Base::Base Root;
public:
	/**
	 * @param min_rel_var_to_retain The minimum relative variance in the data
	 * to retain. It is expected to be within the range (0,1]. If it is 1,
	 * the dimensionality of the preprocessed data is guaranteed not to be
	 * reduced. If it is less than 1, the data cannot be a multi-channel
	 * tensor.
	 * @param epsilon A small consant used to maintain numerical stability.
	 */
	inline PCAPreprocessor(Scalar min_rel_var_to_retain = 1, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				Base::PCAPreprocessorBase(min_rel_var_to_retain, epsilon) { }
	inline void fit(const Tensor<Scalar,4>& data) {
		assert((!Base::reduce_dims || data.dimension(3) == 1) && "cannot reduce the dimensionality of multi-channel tensors");
		Root::fit(data);
		std::size_t channels = Root::dims(2);
		Base::ed_vec = std::vector<typename Base::EigenDecomposition>(channels);
		std::array<std::size_t,4> offsets({ 0u, 0u, 0u, 0u });
		std::array<std::size_t,4> extents({ data.dimension(0), Root::dims(0), Root::dims(1), 1u });
		for (std::size_t i = 0; i < channels; ++i) {
			offsets[3] = i;
			Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
			Base::_fit(std::move(data_slice_i), i);
		}
	}
	inline void transform(Tensor<Scalar,4>& data) const {
		Root::transform(data);
		if (Base::reduce_dims) {
			Dimensions<std::size_t,3> output_dims({ data.dimension(1) * data.dimension(2), 1u, 1u });
			data = Base::_transform(std::move(data), 0);
		} else {
			std::size_t rows = data.dimension(0);
			std::array<std::size_t,4> offsets({ 0u, 0u, 0u, 0u });
			std::array<std::size_t,4> extents({ rows, Root::dims(0), Root::dims(1), 1u });
			for (std::size_t i = 0; i < Root::dims(2); ++i) {
				offsets[3] = i;
				Tensor<Scalar,4> data_slice_i = data.slice(offsets, extents);
				data.slice(offsets, extents) = Base::_transform(std::move(data_slice_i), i);
			}
		}
	}
};

} /* namespace cattle */

#endif /* PREPROCESSOR_H_ */
