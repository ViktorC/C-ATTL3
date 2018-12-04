/*
 * ELUActivationGPULayer.hpp
 *
 *  Created on: 19 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_ACTIVATION_ELUACTIVATIONGPULAYER_H_
#define C_ATTL3_LAYER_GPU_ACTIVATION_ELUACTIVATIONGPULAYER_H_

#include "layer/gpu/activation/SimpleActivationGPULayer.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar, std::size_t Rank>
class ELUActivationGPULayer : public SimpleActivationGPULayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef SimpleActivationGPULayer<Scalar,Rank> Base;
public:
	inline ELUActivationGPULayer(const typename Root::Dims& dims, Scalar alpha = 1e-1) :
			Base(dims, CUDNN_ACTIVATION_ELU, alpha) { }
	inline GPULayer<Scalar,Rank>* gpu_clone() const {
		return new ELUActivationGPULayer<Scalar,Rank>(*this);
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_ACTIVATION_ELUACTIVATIONGPULAYER_H_ */
