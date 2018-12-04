/*
 * KernelGPULayer.hpp
 *
 *  Created on: 18 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_KERNELGPULAYER_H_
#define C_ATTL3_LAYER_GPU_KERNELGPULAYER_H_

#include "core/gpu/GPULayer.hpp"
#include "layer/KernelLayer.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar, std::size_t Rank>
class KernelGPULayer : public GPULayer<Scalar,Rank>,
		public virtual KernelLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef KernelLayer<Scalar,Rank> Base;
	typedef GPULayer<Scalar,Rank> GPUBase;
	typedef KernelGPULayer<Scalar,Rank> Self;
public:
	inline virtual ~KernelGPULayer() = default;
	inline const typename GPUBase::GPUDims& get_gpu_input_dims() const {
		return gpu_input_dims;
	}
	inline const typename GPUBase::GPUDims& get_gpu_output_dims() const {
		return gpu_output_dims;
	}
	inline std::vector<const GPUParameters<Scalar>*> get_gpu_params() const {
		return std::vector<const GPUParameters<Scalar>*>({
				static_cast<const GPUParameters<Scalar>*>(weights.get()),
				static_cast<const GPUParameters<Scalar>*>(bias.get()) });
	}
	inline std::vector<GPUParameters<Scalar>*> get_gpu_params() {
		return std::vector<GPUParameters<Scalar>*>({
				static_cast<GPUParameters<Scalar>*>(weights.get()),
				static_cast<GPUParameters<Scalar>*>(bias.get()) });
	}
protected:
	typedef std::shared_ptr<GPUParameters<Scalar>> GPUParamsSharedPtr;
	inline KernelGPULayer(const typename Root::Dims& input_dims, const typename Root::Dims& output_dims,
			const typename GPUBase::GPUDims& gpu_input_dims, const typename GPUBase::GPUDims& gpu_output_dims,
			GPUParamsSharedPtr weights, GPUParamsSharedPtr bias) :
				Base(input_dims, output_dims, std::static_pointer_cast<ParamsSharedPtr<Scalar>>(weights),
						std::static_pointer_cast<ParamsSharedPtr<Scalar>>(bias)),
				gpu_input_dims(gpu_input_dims),
				gpu_output_dims(gpu_output_dims) { }
	inline KernelGPULayer(const Self& layer, bool share_params = false) :
			Base(layer, share_params),
			gpu_input_dims(layer.gpu_input_dims),
			gpu_output_dims(layer.gpu_output_dims) { }
	const typename GPUBase::GPUDims gpu_input_dims, gpu_output_dims;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_KERNELGPULAYER_H_ */
