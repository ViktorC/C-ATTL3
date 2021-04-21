/*
 * MaxPoolLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_POOL_MAXPOOLLAYER_H_
#define C_ATTL3_LAYER_POOL_MAXPOOLLAYER_H_

#include <cassert>
#include <utility>

#include "layer/PoolLayer.hpp"

namespace cattle {

/**
 * An abstract class template representing a pooling layer that reduces patches
 * of the input by taking their maxima.
 */
template <typename Scalar, std::size_t Rank>
class MaxPoolLayerBase : public PoolLayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef PoolLayer<Scalar, Rank> Base;

 public:
  inline void empty_cache() { max_inds = std::vector<std::vector<unsigned>>(0); }

 protected:
  inline MaxPoolLayerBase(const typename Root::Dims& input_dims, std::size_t receptor_height,
                          std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride)
      : Base::PoolLayer(input_dims, receptor_height, receptor_width, vertical_stride, horizontal_stride) {}
  inline void _init_cache() {
    max_inds = std::vector<std::vector<unsigned>>(Base::ext_output_dims(0) * Base::ext_output_dims(1));
  }
  inline Tensor<Scalar, 4> _reduce(const Tensor<Scalar, 4>& patch, std::size_t patch_ind) {
    std::size_t rows = patch.dimension(0);
    std::size_t depth = patch.dimension(3);
    std::vector<unsigned> inds(rows * depth);
    Tensor<Scalar, 4> reduced_patch(rows, 1u, 1u, depth);
    for (std::size_t i = 0; i < depth; ++i) {
      for (std::size_t j = 0; j < rows; ++j) {
        Scalar max = NumericUtils<Scalar>::MIN;
        unsigned max_height = 0;
        unsigned max_width = 0;
        for (std::size_t k = 0; k < Base::receptor_width; ++k) {
          for (std::size_t l = 0; l < Base::receptor_height; ++l) {
            Scalar val = patch(j, l, k, i);
            if (val > max) {
              max = val;
              max_height = l;
              max_width = k;
            }
          }
        }
        inds[i * rows + j] = max_width * Base::receptor_height + max_height;
        reduced_patch(j, 0u, 0u, i) = max;
      }
    }
    max_inds[patch_ind] = inds;
    return reduced_patch;
  }
  inline Tensor<Scalar, 4> _d_reduce(const Tensor<Scalar, 4>& grad, std::size_t patch_ind) {
    std::size_t rows = grad.dimension(0);
    std::size_t depth = grad.dimension(3);
    Tensor<Scalar, 4> patch(rows, Base::receptor_height, Base::receptor_width, depth);
    patch.setZero();
    std::vector<unsigned>& inds = max_inds[patch_ind];
    for (std::size_t i = 0; i < depth; ++i) {
      for (std::size_t j = 0; j < rows; ++j) {
        unsigned max_ind = inds[i * rows + j];
        unsigned max_height = max_ind % Base::receptor_height;
        unsigned max_width = max_ind / Base::receptor_height;
        patch(j, max_height, max_width, i) = grad(j, 0u, 0u, i);
      }
    }
    return patch;
  }

 private:
  std::vector<std::vector<unsigned>> max_inds;
};

/**
 * A class template representing a 2D max pooling layer operating on rank-3
 * data.
 *
 * \see http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf
 */
template <typename Scalar, std::size_t Rank = 3>
class MaxPoolLayer : public MaxPoolLayerBase<Scalar, Rank> {
  typedef Layer<Scalar, 3> Root;
  typedef PoolLayer<Scalar, 3> PoolBase;
  typedef MaxPoolLayerBase<Scalar, 3> MaxPoolBase;

 public:
  /**
   * @param input_dims The dimensionality of the input tensor.
   * @param receptor_height The height of the pooling receptor.
   * @param receptor_width The width of the pooling receptor.
   * @param vertical_stride The vertical stride at which the input is to be
   * pooled (i.e. the number of elements by which the receptor is to be shifted
   * along the height of the input tensor).
   * @param horizontal_stride The horizontal stride at which the input is to be
   * pooled (i.e. the number of elements by which the receptor is to be shifted
   * along the width of the input tensor).
   */
  inline MaxPoolLayer(const typename Root::Dims& input_dims, std::size_t receptor_height = 2,
                      std::size_t receptor_width = 2, std::size_t vertical_stride = 2,
                      std::size_t horizontal_stride = 2)
      : MaxPoolBase::MaxPoolLayerBase(input_dims, receptor_height, receptor_width, vertical_stride, horizontal_stride) {
  }
  inline Root* clone() const { return new MaxPoolLayer(*this); }
  inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
    assert((Dimensions<std::size_t, 4>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
    assert(in.dimension(0) > 0);
    batch_size = in.dimension(0);
    return PoolBase::_pass_forward(std::move(in), training);
  }
  inline typename Root::Data pass_back(typename Root::Data out_grad) {
    assert((Dimensions<std::size_t, 4>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
    assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
    if (PoolBase::is_input_layer()) return typename Root::Data();
    return PoolBase::_pass_back(std::move(out_grad));
  }

 private:
  std::size_t batch_size;
};

/**
 * A class template representing a 2D max pooling layer operating on rank-2
 * data.
 *
 * \see http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf
 */
template <typename Scalar>
class MaxPoolLayer<Scalar, 2> : public MaxPoolLayerBase<Scalar, 2> {
  typedef Layer<Scalar, 2> Root;
  typedef PoolLayer<Scalar, 2> PoolBase;
  typedef MaxPoolLayerBase<Scalar, 2> MaxPoolBase;

 public:
  /**
   * @param input_dims The dimensionality of the input tensor.
   * @param receptor_height The height of the pooling receptor.
   * @param receptor_width The width of the pooling receptor.
   * @param vertical_stride The vertical stride at which the input is to be
   * pooled (i.e. the number of elements by which the receptor is to be shifted
   * along the height of the input tensor).
   * @param horizontal_stride The horizontal stride at which the input is to be
   * pooled (i.e. the number of elements by which the receptor is to be shifted
   * along the width of the input tensor).
   */
  inline MaxPoolLayer(const typename Root::Dims& input_dims, std::size_t receptor_height = 2,
                      std::size_t receptor_width = 2, std::size_t vertical_stride = 2,
                      std::size_t horizontal_stride = 2)
      : MaxPoolBase::MaxPoolLayerBase(input_dims, receptor_height, receptor_width, vertical_stride, horizontal_stride) {
  }
  inline Root* clone() const { return new MaxPoolLayer(*this); }
  inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
    assert((Dimensions<std::size_t, 3>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
    assert(in.dimension(0) > 0);
    batch_size = in.dimension(0);
    return PoolBase::_pass_forward(TensorMap<Scalar, 4>(in.data(), {batch_size, in.dimension(1), in.dimension(2), 1u}),
                                   training)
        .reshape(std::array<std::size_t, 3>({batch_size, PoolBase::output_dims(0), PoolBase::output_dims(1)}));
  }
  inline typename Root::Data pass_back(typename Root::Data out_grad) {
    assert((Dimensions<std::size_t, 3>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
    assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
    if (PoolBase::is_input_layer()) return typename Root::Data();
    Tensor<Scalar, 4> prev_out_grad = PoolBase::_pass_back(TensorMap<Scalar, 4>(
        out_grad.data(),
        {batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2)}));
    return TensorMap<Scalar, 3>(prev_out_grad.data(), {batch_size, PoolBase::input_dims(0), PoolBase::input_dims(1)});
  }

 private:
  std::size_t batch_size;
};

/**
 * A class template representing a 1D max pooling layer.
 *
 * \see http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf
 */
template <typename Scalar>
class MaxPoolLayer<Scalar, 1> : public MaxPoolLayerBase<Scalar, 1> {
  typedef Layer<Scalar, 1> Root;
  typedef PoolLayer<Scalar, 1> PoolBase;
  typedef MaxPoolLayerBase<Scalar, 1> MaxPoolBase;

 public:
  /**
   * @param input_dims The dimensionality of the input tensor.
   * @param receptor_length The length of the pooling receptor.
   * @param stride The stride at which the input is to be pooled (i.e. the
   * number of elements by which the receptor is to be shifted along the length
   * of the input tensor).
   */
  inline MaxPoolLayer(const typename Root::Dims& input_dims, std::size_t receptor_length = 2, std::size_t stride = 2)
      : MaxPoolBase::MaxPoolLayerBase(input_dims, receptor_length, 1, stride, 1) {}
  inline Root* clone() const { return new MaxPoolLayer(*this); }
  inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
    assert((Dimensions<std::size_t, 2>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
    assert(in.dimension(0) > 0);
    batch_size = in.dimension(0);
    return PoolBase::_pass_forward(TensorMap<Scalar, 4>(in.data(), {batch_size, in.dimension(1), 1u, 1u}), training)
        .reshape(std::array<std::size_t, 2>({batch_size, PoolBase::output_dims(0)}));
  }
  inline typename Root::Data pass_back(typename Root::Data out_grad) {
    assert((Dimensions<std::size_t, 2>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
    assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
    if (PoolBase::is_input_layer()) return typename Root::Data();
    Tensor<Scalar, 4> prev_out_grad = PoolBase::_pass_back(TensorMap<Scalar, 4>(
        out_grad.data(),
        {batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2)}));
    return TensorMap<Scalar, 2>(prev_out_grad.data(), {batch_size, PoolBase::input_dims(0)});
  }

 private:
  std::size_t batch_size;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_POOL_MAXPOOLLAYER_H_ */
