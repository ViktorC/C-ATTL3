/*
 * SigmoidActivationGPULayer.hpp
 *
 *  Created on: 19 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_ACTIVATION_SIGMOIDACTIVATIONGPULAYER_H_
#define C_ATTL3_LAYER_GPU_ACTIVATION_SIGMOIDACTIVATIONGPULAYER_H_

#include "layer/gpu/activation/SimpleActivationGPULayer.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar, std::size_t Rank>
class SigmoidActivationGPULayer : public SimpleActivationGPULayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef SimpleActivationGPULayer<Scalar,Rank> Base;
public:
	inline SigmoidActivationGPULayer(const typename Root::Dims& dims) :
			Base(dims, CUDNN_ACTIVATION_SIGMOID) { }
	inline GPULayer<Scalar,Rank>* gpu_clone() const {
		return new SigmoidActivationGPULayer<Scalar,Rank>(*this);
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_ACTIVATION_SIGMOIDACTIVATIONGPULAYER_H_ */
