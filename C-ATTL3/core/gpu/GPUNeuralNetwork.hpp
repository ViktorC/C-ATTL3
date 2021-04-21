/*
 * GPUNeuralNetwork.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_GPUNEURALNETWORK_H_
#define C_ATTL3_CORE_GPU_GPUNEURALNETWORK_H_

#include "GPULayer.hpp"
#include "core/NeuralNetwork.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar, std::size_t Rank, bool Sequential>
class GPUNeuralNetwork : public virtual NeuralNetwork<Scalar, Rank, Sequential> {
  typedef NeuralNetwork<Scalar, Rank, Sequential> Base;

 protected:
  typedef Dimensions<std::size_t, 3> GPUDims;

 public:
  virtual const GPUDims& get_gpu_input_dims() const = 0;
  virtual const GPUDims& get_gpu_output_dims() const = 0;
  virtual std::vector<const GPULayer<Scalar, Rank>*> get_gpu_layers() const = 0;
  virtual std::vector<GPULayer<Scalar, Rank>*> get_gpu_layers() = 0;
  virtual CuDNNTensor<Scalar> propagate(CuDNNTensor<Scalar> input, bool training) = 0;
  virtual CuDNNTensor<Scalar> backpropagate(CuDNNTensor<Scalar> out_grad) = 0;
  inline typename Base::Data propagate(typename Base::Data input, bool training) {
    auto rows = input.dimension(0);
    auto in_gpu_dims = get_gpu_input_dims().template extend<>();
    in_gpu_dims(0) = rows;
    Tensor<Scalar, 4> out_extended =
        propagate(CuDNNTensor<Scalar>(TensorMap<Scalar, 4>(input.data(), in_gpu_dims)), training);
    auto out_dims = Base::get_output_dims().template extend<>();
    out_dims(0) = rows;
    return TensorMap<Scalar, Base::DATA_RANK>(out_extended.data(), out_dims);
  }
  inline typename Base::Data backpropagate(typename Base::Data out_grad) {
    auto rows = out_grad.dimension(0);
    auto out_gpu_dims = get_gpu_output_dims().template extend<>();
    out_gpu_dims(0) = rows;
    Tensor<Scalar, 4> prev_out_grad_extended =
        backpropagate(CuDNNTensor<Scalar>(TensorMap<Scalar, 4>(out_grad.data(), out_gpu_dims)));
    auto in_dims = Base::get_input_dims().template extend<>();
    in_dims(0) = rows;
    return TensorMap<Scalar, Base::DATA_RANK>(prev_out_grad_extended.data(), in_dims);
  }
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPUNEURALNETWORK_H_ */
