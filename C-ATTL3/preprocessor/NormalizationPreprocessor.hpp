/*
 * NormalizationPreprocessor.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PREPROCESSOR_NORMALIZATIONPREPROCESSOR_H_
#define C_ATTL3_PREPROCESSOR_NORMALIZATIONPREPROCESSOR_H_

#include <array>
#include <cassert>

#include "core/Dimensions.hpp"
#include "core/NumericUtils.hpp"
#include "core/Preprocessor.hpp"

namespace cattle {

/**
 * A class template for a normalization (mean-subtraction) preprocessor that
 * optionally also standardizes the data (divides it by the standard deviation).
 */
template <typename Scalar, std::size_t Rank, bool Standardize = false, bool PerLastRank = (Rank == 3)>
class NormalizationPreprocessor : public Preprocessor<Scalar, Rank, false> {
 public:
  /**
   * @param epsilon A small constant used to maintain numerical stability.
   */
  inline NormalizationPreprocessor(Scalar epsilon = NumericUtils<Scalar>::EPSILON2) : epsilon(epsilon) {
    assert(epsilon > 0 && "epsilon must be greater than 0");
  }
  virtual ~NormalizationPreprocessor() = default;
  inline virtual void fit(const Tensor<Scalar, Rank + 1>& data) {
    auto rows = data.dimension(0);
    assert(rows > 0);
    dims = (Dimensions<std::size_t, Rank + 1>(data.dimensions()).template demote<>());
    channels = dims(Rank - 1);
    means = Matrix<Scalar>(channels, dims.get_volume() / channels);
    sd = Matrix<Scalar>(means.rows(), means.cols());
    if (channels == 1) {
      Tensor<Scalar, Rank + 1> data_copy(data);
      MatrixMap<Scalar> data_mat(data_copy.data(), rows, data.size() / rows);
      means.row(0) = data_mat.colwise().mean();
      if (Standardize) sd.row(0) = (data_mat.rowwise() - means.row(0)).array().square().colwise().mean().sqrt();
    } else {
      std::array<std::size_t, Rank + 1> offsets;
      offsets.fill(0);
      auto extents = data.dimensions();
      extents[Rank] = 1;
      for (std::size_t i = 0; i < channels; ++i) {
        offsets[Rank] = i;
        Tensor<Scalar, Rank + 1> data_slice = data.slice(offsets, extents);
        MatrixMap<Scalar> data_slice_mat(data_slice.data(), rows, data_slice.size() / rows);
        means.row(i) = data_slice_mat.colwise().mean();
        if (Standardize) sd.row(i) = (data_slice_mat.rowwise() - means.row(i)).array().square().colwise().mean().sqrt();
      }
    }
  }
  inline virtual void transform(Tensor<Scalar, Rank + 1>& data) const {
    auto rows = data.dimension(0);
    assert(rows > 0);
    assert((Dimensions<std::size_t, Rank + 1>(data.dimensions()).template demote<>()) == dims &&
           "mismatched fit and transform input tensor dimensions");
    if (channels == 1) {
      MatrixMap<Scalar> data_mat(data.data(), rows, data.size() / rows);
      data_mat = data_mat.rowwise() - means.row(0);
      if (Standardize) data_mat *= (sd.row(0).array() + epsilon).inverse().matrix().asDiagonal();
    } else {
      std::array<std::size_t, Rank + 1> offsets;
      offsets.fill(0);
      auto extents = data.dimensions();
      extents[Rank] = 1;
      for (std::size_t i = 0; i < channels; ++i) {
        offsets[Rank] = i;
        Tensor<Scalar, Rank + 1> data_slice = data.slice(offsets, extents);
        MatrixMap<Scalar> data_slice_mat(data_slice.data(), rows, data_slice.size() / rows);
        data_slice_mat = data_slice_mat.rowwise() - means.row(i);
        if (Standardize) data_slice_mat *= (sd.row(i).array() + epsilon).inverse().matrix().asDiagonal();
        data.slice(offsets, extents) = std::move(data_slice);
      }
    }
  }

 protected:
  const Scalar epsilon;
  Dimensions<std::size_t, Rank> dims;
  Matrix<Scalar> means;
  Matrix<Scalar> sd;
  std::size_t channels;
};

/**
 * Partial template specialization for pre-activation normalization.
 */
template <typename Scalar, std::size_t Rank, bool Standardize>
class NormalizationPreprocessor<Scalar, Rank, Standardize, false> : public Preprocessor<Scalar, Rank, false> {
 public:
  /**
   * @param epsilon A small constant used to maintain numerical stability.
   */
  inline NormalizationPreprocessor(Scalar epsilon = NumericUtils<Scalar>::EPSILON2) : epsilon(epsilon) {
    assert(epsilon > 0 && "epsilon must be greater than 0");
  }
  virtual ~NormalizationPreprocessor() = default;
  inline virtual void fit(const Tensor<Scalar, Rank + 1>& data) {
    std::size_t rows = data.dimension(0);
    assert(rows > 0);
    dims = (Dimensions<std::size_t, Rank + 1>(data.dimensions()).template demote<>());
    Tensor<Scalar, Rank + 1> data_copy(data);
    MatrixMap<Scalar> data_mat(data_copy.data(), rows, data.size() / rows);
    means = data_mat.colwise().mean();
    if (Standardize) sd = (data_mat.rowwise() - means).array().square().colwise().mean().sqrt();
  }
  inline virtual void transform(Tensor<Scalar, Rank + 1>& data) const {
    std::size_t rows = data.dimension(0);
    assert(rows > 0);
    assert((Dimensions<std::size_t, Rank + 1>(data.dimensions()).template demote<>()) == dims &&
           "mismatched fit and transform input tensor dimensions");
    MatrixMap<Scalar> data_mat(data.data(), rows, data.size() / rows);
    data_mat = data_mat.rowwise() - means;
    if (Standardize) data_mat *= (sd.array() + epsilon).inverse().matrix().asDiagonal();
  }

 protected:
  const Scalar epsilon;
  Dimensions<std::size_t, Rank> dims;
  RowVector<Scalar> means;
  RowVector<Scalar> sd;
};

}  // namespace cattle

#endif /* C_ATTL3_PREPROCESSOR_NORMALIZATIONPREPROCESSOR_H_ */
