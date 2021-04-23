/*
 * ScaledActivationGPULayer.hpp
 *
 *  Created on: 19 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_ACTIVATION_SCALEDACTIVATIONGPULAYER_H_
#define C_ATTL3_LAYER_GPU_ACTIVATION_SCALEDACTIVATIONGPULAYER_H_

#include "layer/gpu/ActivationGPULayer.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar, std::size_t Rank>
class ScaledActivationGPULayer : public ActivationGPULayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef ActivationGPULayer<Scalar, Rank> Base;

 public:
  inline ScaledActivationGPULayer(const typename Root::Dims& dims, Scalar scale) : Base(dims), scale(scale) {}
  inline GPULayer<Scalar, Rank>* gpu_clone() const { return new ScaledActivationGPULayer<Scalar, Rank>(*this); }
  inline void empty_cache() {}
  inline CuDNNTensor<Scalar> gpu_pass_forward(CuDNNTensor<Scalar> in, bool training) {
    assert(in.height() == Base::gpu_dims(0) && in.width() == Base::gpu_dims(1) && in.channels() == Base::gpu_dims(2));
    assert(in.samples() > 0);
    batch_size = in.samples();
    in *= scale;
    return in;
  }
  inline CuDNNTensor<Scalar> gpu_pass_back(CuDNNTensor<Scalar> out_grad) {
    assert(out_grad.height() == Base::gpu_dims(0) && out_grad.width() == Base::gpu_dims(1) &&
           out_grad.channels() == Base::gpu_dims(2));
    assert(out_grad.samples() == batch_size);
    if (Base::is_input_layer()) return CuDNNTensor<Scalar>();
    out_grad *= scale;
    return out_grad;
  }

 private:
  const Scalar scale;
  std::size_t batch_size;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_ACTIVATION_SCALEDACTIVATIONGPULAYER_H_ */
