/*
 * SoftmaxActivationGPULayer.hpp
 *
 *  Created on: 19 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_ACTIVATION_SOFTMAXACTIVATIONGPULAYER_H_
#define C_ATTL3_LAYER_GPU_ACTIVATION_SOFTMAXACTIVATIONGPULAYER_H_

#include <cassert>

#include "layer/gpu/ActivationGPULayer.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar, std::size_t Rank>
class SoftmaxActivationGPULayer : public ActivationGPULayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef ActivationGPULayer<Scalar, Rank> Base;

 public:
  inline SoftmaxActivationGPULayer(const typename Root::Dims& dims) : Base(dims), alpha(1), beta(0) {}
  inline GPULayer<Scalar, Rank>* gpu_clone() const { return new SoftmaxActivationGPULayer<Scalar, Rank>(*this); }
  inline void empty_cache() { out_cache = CuDNNTensor<Scalar>(); }
  inline CuDNNTensor<Scalar> gpu_pass_forward(CuDNNTensor<Scalar> in, bool training) {
    assert(in.height() == Base::gpu_dims(0) && in.width() == Base::gpu_dims(1) && in.channels() == Base::gpu_dims(2));
    assert(in.samples() > 0);
    cudnnAssert(cudnnSoftmaxForward(CuDNNHandle::get_instance(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alpha, in.desc(), in.data(), &beta, in.desc(), in.data()));
    if (training) out_cache = in;
    return in;
  }
  inline CuDNNTensor<Scalar> gpu_pass_back(CuDNNTensor<Scalar> out_grad) {
    assert(out_grad.height() == Base::gpu_dims(0) && out_grad.width() == Base::gpu_dims(1) &&
           out_grad.channels() == Base::gpu_dims(2));
    assert(out_grad.samples() == out_cache.samples());
    if (Base::is_input_layer()) return CuDNNTensor<Scalar>();
    cudnnAssert(cudnnSoftmaxBackward(CuDNNHandle::get_instance(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                     &alpha, out_cache.desc(), out_cache.data(), out_grad.desc(), out_grad.data(),
                                     &beta, out_grad.desc(), out_grad.data()));
    return out_grad;
  }

 private:
  const Scalar alpha, beta;
  CuDNNTensor<Scalar> out_cache;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_ACTIVATION_SOFTMAXACTIVATIONGPULAYER_H_ */
