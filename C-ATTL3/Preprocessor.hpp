/*
 * Preprocessor.hpp
 *
 *  Created on: 12.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_PREPROCESSOR_H_
#define CATTL3_PREPROCESSOR_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "utils/EigenProxy.hpp"
#include "utils/NumericUtils.hpp"

namespace cattle {

// TODO Preprocessors for sequential data.
// TODO LDA.

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
	 * @param data A non-constant reference to a data tensor.
	 */
	virtual void transform(Tensor<Scalar,Rank + Sequential + 1>& data) const = 0;
};

/**
 * A class template for a normalization (mean-subtraction) preprocessor that optionally also
 * standardizes the data (divides it by the standard deviation).
 */
template<typename Scalar, std::size_t Rank, bool Standardize = false, bool PerLastRank = (Rank == 3)>
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
		auto rows = data.dimension(0);
		assert(rows > 0);
		dims = (Dimensions<std::size_t,Rank + 1>(data.dimensions()).template demote<>());
		channels = dims(Rank - 1);
		means = Matrix<Scalar>(channels, dims.get_volume() / channels);
		sd = Matrix<Scalar>(means.rows(), means.cols());
		if (channels == 1) {
			Tensor<Scalar,Rank + 1> data_copy(data);
			MatrixMap<Scalar> data_mat(data_copy.data(), rows, data.size() / rows);
			means.row(0) = data_mat.colwise().mean();
			if (Standardize)
				sd.row(0) = (data_mat.rowwise() - means.row(0)).array().square().colwise().mean().sqrt();
		} else {
			std::array<std::size_t,Rank + 1> offsets;
			offsets.fill(0);
			auto extents = data.dimensions();
			extents[Rank] = 1;
			for (std::size_t i = 0; i < channels; ++i) {
				offsets[Rank] = i;
				Tensor<Scalar,Rank + 1> data_slice = data.slice(offsets, extents);
				MatrixMap<Scalar> data_slice_mat(data_slice.data(), rows, data_slice.size() / rows);
				means.row(i) = data_slice_mat.colwise().mean();
				if (Standardize)
					sd.row(i) = (data_slice_mat.rowwise() - means.row(i)).array().square().colwise().mean().sqrt();
			}
		}
	}
	inline virtual void transform(Tensor<Scalar,Rank + 1>& data) const {
		auto rows = data.dimension(0);
		assert(rows > 0);
		assert((Dimensions<std::size_t,Rank + 1>(data.dimensions()).template demote<>()) == dims &&
				"mismatched fit and transform input tensor dimensions");
		if (channels == 1) {
			MatrixMap<Scalar> data_mat(data.data(), rows, data.size() / rows);
			data_mat = data_mat.rowwise() - means.row(0);
			if (Standardize)
				data_mat *= (sd.row(0).array() + epsilon).inverse().matrix().asDiagonal();
		} else {
			std::array<std::size_t,Rank + 1> offsets;
			offsets.fill(0);
			auto extents = data.dimensions();
			extents[Rank] = 1;
			for (std::size_t i = 0; i < channels; ++i) {
				offsets[Rank] = i;
				Tensor<Scalar,Rank + 1> data_slice = data.slice(offsets, extents);
				MatrixMap<Scalar> data_slice_mat(data_slice.data(), rows, data_slice.size() / rows);
				data_slice_mat = data_slice_mat.rowwise() - means.row(i);
				if (Standardize)
					data_slice_mat *= (sd.row(i).array() + epsilon).inverse().matrix().asDiagonal();
				data.slice(offsets, extents) = std::move(data_slice);
			}
		}
	}
protected:
	const Scalar epsilon;
	Dimensions<std::size_t,Rank> dims;
	Matrix<Scalar> means;
	Matrix<Scalar> sd;
	std::size_t channels;
};

/**
 * Partial template specialization for pre-activation normalization.
 */
template<typename Scalar, std::size_t Rank, bool Standardize>
class NormalizationPreprocessor<Scalar,Rank,Standardize,false> : public Preprocessor<Scalar,Rank,false> {
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
		Tensor<Scalar,Rank + 1> data_copy(data);
		MatrixMap<Scalar> data_mat(data_copy.data(), rows, data.size() / rows);
		means = data_mat.colwise().mean();
		if (Standardize)
			sd = (data_mat.rowwise() - means).array().square().colwise().mean().sqrt();
	}
	inline virtual void transform(Tensor<Scalar,Rank + 1>& data) const {
		std::size_t rows = data.dimension(0);
		assert(rows > 0);
		assert((Dimensions<std::size_t,Rank + 1>(data.dimensions()).template demote<>()) == dims &&
				"mismatched fit and transform input tensor dimensions");
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
 * An abstract base class template for a principal component analysis (PCA) preprocessor that can also
 * standardize and whiten the data.
 */
template<typename Scalar, std::size_t Rank, bool Standardize, bool Whiten, bool PerLastRank>
class PCAPreprocessorBase : public NormalizationPreprocessor<Scalar,Rank,Standardize,PerLastRank> {
protected:
	typedef NormalizationPreprocessor<Scalar,Rank,Standardize,PerLastRank> Base;
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
		MatrixMap<Scalar> data_mat(data.data(), rows, data.size() / rows);
		if (reduce_dims) {
			Dimensions<std::size_t,Rank + 1> output_dims;
			output_dims(0) = rows;
			Matrix<Scalar> transformed_data_mat = data_mat * ed_vec[i].eigen_basis;
			if (Whiten)
				transformed_data_mat *= (ed_vec[i].eigen_values.array() + Base::epsilon).sqrt().inverse().matrix().asDiagonal();
			output_dims(1) = transformed_data_mat.cols();
			return TensorMap<Scalar,Rank + 1>(transformed_data_mat.data(), output_dims);
		} else {
			Dimensions<std::size_t,Rank + 1> output_dims = data.dimensions();
			data_mat *= ed_vec[i].eigen_basis;
			if (Whiten)
				data_mat *= (ed_vec[i].eigen_values.array() + Base::epsilon).sqrt().inverse().matrix().asDiagonal();
			return TensorMap<Scalar,Rank + 1>(data_mat.data(), output_dims);
		}
	}
	Scalar min_rel_var_to_retain;
	bool reduce_dims;
	struct EigenDecomposition {
		Matrix<Scalar> eigen_basis;
		RowVector<Scalar> eigen_values;
	};
	std::vector<EigenDecomposition> ed_vec;
};

/**
 * A class template for a PCA preprocessor that can also be used to standardize and whiten data across
 * multiple channels.
 */
template<typename Scalar, std::size_t Rank, bool Standardize = false, bool Whiten = false,
		bool PerLastRank = (Rank == 3)>
class PCAPreprocessor : public PCAPreprocessorBase<Scalar,Rank,Standardize,Whiten,PerLastRank> {
	typedef PCAPreprocessorBase<Scalar,Rank,Standardize,Whiten,PerLastRank> Base;
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
	inline void fit(const Tensor<Scalar,Rank + 1>& data) {
		assert((!Base::reduce_dims || data.dimension(Rank) == 1) && "cannot reduce the dimensionality of multi-channel data");
		Root::fit(data);
		channels = data.dimension(Rank);
		Base::ed_vec = std::vector<typename Base::EigenDecomposition>(channels);
		if (channels == 1)
			Base::_fit(data, 0);
		else {
			std::array<std::size_t,Rank + 1> offsets;
			offsets.fill(0);
			auto extents = data.dimensions();
			extents[Rank] = 1;
			for (std::size_t i = 0; i < channels; ++i) {
				offsets[Rank] = i;
				Tensor<Scalar,Rank + 1> data_slice_i = data.slice(offsets, extents);
				Base::_fit(std::move(data_slice_i), i);
			}
		}
	}
	inline void transform(Tensor<Scalar,Rank + 1>& data) const {
		Root::transform(data);
		if (Base::reduce_dims || channels == 1)
			data = Base::_transform(std::move(data), 0);
		else {
			std::array<std::size_t,Rank + 1> offsets;
			offsets.fill(0);
			auto extents = data.dimensions();
			extents[Rank] = 1;
			for (std::size_t i = 0; i < channels; ++i) {
				offsets[Rank] = i;
				Tensor<Scalar,Rank + 1> data_slice_i = data.slice(offsets, extents);
				data.slice(offsets, extents) = Base::_transform(std::move(data_slice_i), i);
			}
		}
	}
private:
	std::size_t channels;
};

/**
 * Partial template specialization of the PCA preprocessor for single channel data.
 */
template<typename Scalar, std::size_t Rank, bool Standardize, bool Whiten>
class PCAPreprocessor<Scalar,Rank,Standardize,Whiten,false> :
		public PCAPreprocessorBase<Scalar,Rank,Standardize,Whiten,false> {
	typedef PCAPreprocessorBase<Scalar,Rank,Standardize,Whiten,false> Base;
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

} /* namespace cattle */

#endif /* CATTL3_PREPROCESSOR_H_ */
