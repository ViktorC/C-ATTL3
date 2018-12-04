/*
 * IdentityActivationGPULayer.hpp
 *
 *  Created on: 19 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_ACTIVATION_IDENTITYACTIVATIONGPULAYER_H_
#define C_ATTL3_LAYER_GPU_ACTIVATION_IDENTITYACTIVATIONGPULAYER_H_

#include "layer/gpu/ActivationGPULayer.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar, std::size_t Rank>
class IdentityActivationGPULayer : public ActivationGPULayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationGPULayer<Scalar,Rank> Base;
public:
	inline IdentityActivationGPULayer(const typename Root::Dims& dims) :
			Base(dims) { }
	inline GPULayer<Scalar,Rank>* gpu_clone() const {
		return new IdentityActivationGPULayer<Scalar,Rank>(*this);
	}
	inline void empty_cache() { }
	inline CuDNNTensor<Scalar> pass_forward(CuDNNTensor<Scalar> in, bool training) {
		assert(in.height() == Base::gpu_dims(0) && in.width() == Base::gpu_dims(1) &&
				in.channels() == Base::gpu_dims(2));
		assert(in.samples() > 0);
		batch_size = in.samples();
		return in;
	}
	inline CuDNNTensor<Scalar> pass_back(CuDNNTensor<Scalar> out_grad) {
		assert(out_grad.height() == Base::gpu_dims(0) && out_grad.width() == Base::gpu_dims(1) &&
				out_grad.channels() == Base::gpu_dims(2));
		assert(out_grad.samples() == batch_size);
		return out_grad;
	}
private:
	std::size_t batch_size;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_ACTIVATION_IDENTITYACTIVATIONGPULAYER_H_ */
