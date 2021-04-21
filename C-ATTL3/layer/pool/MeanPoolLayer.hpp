/*
 * MeanPoolLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_POOL_MEANPOOLLAYER_H_
#define C_ATTL3_LAYER_POOL_MEANPOOLLAYER_H_

#include <cassert>
#include <utility>

#include "layer/PoolLayer.hpp"

namespace cattle {

/**
 * An abstract class template representing a pooling layer that reduces patches
 * of the input by taking their means.
 */
template <typename Scalar, std::size_t Rank>
class MeanPoolLayerBase : public PoolLayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef PoolLayer<Scalar, Rank> Base;

 public:
  inline void empty_cache() {}

 protected:
  inline MeanPoolLayerBase(const typename Root::Dims& input_dims, std::size_t receptor_height,
                           std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride)
      : Base::PoolLayer(input_dims, receptor_height, receptor_width, vertical_stride, horizontal_stride),
        receptor_area(receptor_height * receptor_width) {}
  inline void _init_cache() {}
  inline Tensor<Scalar, 4> _reduce(const Tensor<Scalar, 4>& patch, std::size_t patch_ind) {
    Tensor<Scalar, 2> reduced_patch = patch.mean(Base::reduction_ranks);
    return TensorMap<Scalar, 4>(reduced_patch.data(), Base::reduced_patch_extents);
  }
  inline Tensor<Scalar, 4> _d_reduce(const Tensor<Scalar, 4>& grad, std::size_t patch_ind) {
    return (grad / (Scalar)receptor_area).broadcast(Base::broadcast);
  }

 private:
  std::size_t receptor_area;
};

/**
 * A class template representing a 2D mean pooling layer operating on rank-3
 * data.
 */
template <typename Scalar, std::size_t Rank = 3>
class MeanPoolLayer : public MeanPoolLayerBase<Scalar, Rank> {
  typedef Layer<Scalar, 3> Root;
  typedef PoolLayer<Scalar, 3> PoolBase;
  typedef MeanPoolLayerBase<Scalar, 3> MeanPoolBase;

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
  inline MeanPoolLayer(const typename Root::Dims& input_dims, std::size_t receptor_height = 2,
                       std::size_t receptor_width = 2, std::size_t vertical_stride = 2,
                       std::size_t horizontal_stride = 2)
      : MeanPoolBase::MeanPoolLayerBase(input_dims, receptor_height, receptor_width, vertical_stride,
                                        horizontal_stride) {}
  inline Root* clone() const { return new MeanPoolLayer(*this); }
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
 * A class template representing a 2D mean pooling layer operating on rank-2
 * data.
 */
template <typename Scalar>
class MeanPoolLayer<Scalar, 2> : public MeanPoolLayerBase<Scalar, 2> {
  typedef Layer<Scalar, 2> Root;
  typedef PoolLayer<Scalar, 2> PoolBase;
  typedef MeanPoolLayerBase<Scalar, 2> MeanPoolBase;

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
  inline MeanPoolLayer(const typename Root::Dims& input_dims, std::size_t receptor_height = 2,
                       std::size_t receptor_width = 2, std::size_t vertical_stride = 2,
                       std::size_t horizontal_stride = 2)
      : MeanPoolBase::MeanPoolLayerBase(input_dims, receptor_height, receptor_width, vertical_stride,
                                        horizontal_stride) {}
  inline Root* clone() const { return new MeanPoolLayer(*this); }
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
 * A class template representing a 1D mean pooling layer.
 */
template <typename Scalar>
class MeanPoolLayer<Scalar, 1> : public MeanPoolLayerBase<Scalar, 1> {
  typedef Layer<Scalar, 1> Root;
  typedef PoolLayer<Scalar, 1> PoolBase;
  typedef MeanPoolLayerBase<Scalar, 1> MeanPoolBase;

 public:
  /**
   * @param input_dims The dimensionality of the input tensor.
   * @param receptor_length The length of the pooling receptor.
   * @param stride The stride at which the input is to be pooled (i.e. the
   * number of elements by which the receptor is to be shifted along the length
   * of the input tensor).
   */
  inline MeanPoolLayer(const typename Root::Dims& input_dims, std::size_t receptor_length = 2, std::size_t stride = 2)
      : MeanPoolBase::MeanPoolLayerBase(input_dims, receptor_length, 1, stride, 1) {}
  inline Root* clone() const { return new MeanPoolLayer(*this); }
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

#endif /* C_ATTL3_LAYER_POOL_MEANPOOLLAYER_H_ */
