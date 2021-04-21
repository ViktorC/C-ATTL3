/*
 * SimpleActivationGPULayer.hpp
 *
 *  Created on: 19 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_ACTIVATION_SIMPLEACTIVATIONGPULAYER_H_
#define C_ATTL3_LAYER_GPU_ACTIVATION_SIMPLEACTIVATIONGPULAYER_H_

#include <cassert>

#include "layer/gpu/ActivationGPULayer.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar, std::size_t Rank>
class SimpleActivationGPULayer : public ActivationGPULayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef ActivationGPULayer<Scalar, Rank> Base;

 public:
  inline SimpleActivationGPULayer(const typename Root::Dims& dims, cudnnActivationMode_t act_mode, Scalar coeff = 0)
      : Base(dims), coeff(coeff), alpha(1), beta(0), act_desc() {
    cudnnAssert(cudnnCreateActivationDescriptor(&act_desc));
    cudnnAssert(cudnnSetActivationDescriptor(act_desc, act_mode, CuDNNTensor<Scalar>::NAN_PROP, coeff));
  }
  inline virtual ~SimpleActivationGPULayer() { cudnnAssert(cudnnDestroyActivationDescriptor(act_desc)); }
  inline void empty_cache() {
    in_cache = CuDNNTensor<Scalar>();
    out_cache = CuDNNTensor<Scalar>();
  }
  inline CuDNNTensor<Scalar> pass_forward(CuDNNTensor<Scalar> in, bool training) {
    assert(in.height() == Base::gpu_dims(0) && in.width() == Base::gpu_dims(1) && in.channels() == Base::gpu_dims(2));
    assert(in.samples() > 0);
    in_cache = std::move(in);
    CuDNNTensor<Scalar> out(in_cache.samples(), in_cache.height(), in_cache.width(), in_cache.channels());
    cudnnAssert(cudnnActivationForward(CuDNNHandle::get_instance(), act_desc, &alpha, in_cache.desc(), in_cache.data(),
                                       &beta, out.desc(), out.data()));
    if (training) out_cache = out;
    return out;
  }
  inline CuDNNTensor<Scalar> pass_back(CuDNNTensor<Scalar> out_grad) {
    assert(out_grad.height() == Base::gpu_dims(0) && out_grad.width() == Base::gpu_dims(1) &&
           out_grad.channels() == Base::gpu_dims(2));
    assert(out_grad.samples() == in_cache.samples());
    if (Base::is_input_layer()) return CuDNNTensor<Scalar>();
    cudnnAssert(cudnnActivationBackward(CuDNNHandle::get_instance(), act_desc, &alpha, out_cache.desc(),
                                        out_cache.data(), out_grad.desc(), out_grad.data(), in_cache.desc(),
                                        in_cache.data(), &beta, out_grad.desc(), out_grad.data()));
    return out_grad;
  }

 private:
  const Scalar coeff, alpha, beta;
  cudnnActivationDescriptor_t act_desc;
  CuDNNTensor<Scalar> in_cache, out_cache;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_ACTIVATION_SIMPLEACTIVATIONGPULAYER_H_ */
