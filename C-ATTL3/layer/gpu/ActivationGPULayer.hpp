/*
 * ActivationGPULayer.hpp
 *
 *  Created on: 18 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_ACTIVATIONGPULAYER_H_
#define C_ATTL3_LAYER_GPU_ACTIVATIONGPULAYER_H_

#include "core/gpu/GPULayer.hpp"
#include "layer/ActivationLayer.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar, std::size_t Rank>
class ActivationGPULayer : public GPULayer<Scalar,Rank>,
		public virtual ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef GPULayer<Scalar,Rank> GPUBase;
	typedef ActivationGPULayer<Scalar,Rank> Self;
public:
	inline virtual ~ActivationGPULayer() = default;
	inline GPUBase* gpu_clone_with_shared_params() {
		return gpu_clone();
	}
	inline const typename GPUBase::GPUDims& get_gpu_input_dims() const {
		return gpu_dims;
	}
	inline const typename GPUBase::GPUDims& get_gpu_output_dims() const {
		return gpu_dims;
	}
	inline std::vector<const GPUParameters<Scalar>*> get_gpu_params() const {
		return params ? std::vector<const GPUParameters<Scalar>*>({
				static_cast<const GPUParameters<Scalar>*>(Base::params.get()) }) :
				std::vector<const GPUParameters<Scalar>*>(0);
	}
	inline std::vector<GPUParameters<Scalar>*> get_gpu_params() {
		return params ? std::vector<GPUParameters<Scalar>*>({
				static_cast<GPUParameters<Scalar>*>(Base::params.get()) }) :
				std::vector<GPUParameters<Scalar>*>(0);
	}
protected:
	typedef std::shared_ptr<GPUParameters<Scalar>> GPUParamsSharedPtr;
	inline ActivationGPULayer(const typename Root::Dims& dims, GPUParamsSharedPtr params = nullptr) :
			Base(dims, std::static_pointer_cast<ParamsSharedPtr<Scalar>>(params)),
			gpu_dims(dims.template extend<3 - Rank>()) { }
	inline ActivationGPULayer(const Self& layer, bool share_params = false) :
			Base(layer, share_params),
			gpu_dims(layer.gpu_dims) { }
private:
	const typename GPUBase::GPUDims gpu_dims;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_ACTIVATIONGPULAYER_H_ */
