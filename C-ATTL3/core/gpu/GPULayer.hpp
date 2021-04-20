/*
 * GPULayer.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_GPULAYER_H_
#define C_ATTL3_CORE_GPU_GPULAYER_H_

#include "core/Layer.hpp"
#include "cudnn/CuDNNTensor.hpp"
#include "GPUParameters.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar, std::size_t Rank>
class GPULayer : public virtual Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
protected:
	typedef Dimensions<std::size_t,3> GPUDims;
public:
	virtual ~GPULayer() = default;
	virtual GPULayer<Scalar,Rank>* gpu_clone() const = 0;
	virtual GPULayer<Scalar,Rank>* gpu_clone_with_shared_params() = 0;
	virtual const GPUDims& get_gpu_input_dims() const = 0;
	virtual const GPUDims& get_gpu_output_dims() const = 0;
	virtual std::vector<const GPUParameters<Scalar>*> get_gpu_params() const = 0;
	virtual std::vector<GPUParameters<Scalar>*> get_gpu_params() = 0;
	virtual CuDNNTensor<Scalar> pass_forward(CuDNNTensor<Scalar> in, bool training) = 0;
	virtual CuDNNTensor<Scalar> pass_back(CuDNNTensor<Scalar> out_grad) = 0;
	inline virtual Layer<Scalar,Rank>* clone() const {
		return gpu_clone();
	}
	inline virtual Layer<Scalar,Rank>* clone_with_shared_params() {
		return gpu_clone_with_shared_params();
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		auto rows = in.dimension(0);
		auto in_gpu_dims = get_gpu_input_dims().template extend<>();
		in_gpu_dims(0) = rows;
		Tensor<Scalar,4> out_extended = pass_forward(CuDNNTensor<Scalar>(TensorMap<Scalar,4>(in.data(),
				in_gpu_dims)), training);
		auto out_dims = Base::get_output_dims().template extend<>();
		out_dims(0) = rows;
		return TensorMap<Scalar,Base::DATA_RANK>(out_extended.data(), out_dims);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		auto rows = out_grad.dimension(0);
		auto out_gpu_dims = get_gpu_output_dims().template extend<>();
		out_gpu_dims(0) = rows;
		Tensor<Scalar,4> prev_out_grad_extended = pass_back(TensorMap<Scalar,4>(out_grad.data(),
				out_gpu_dims));
		auto in_dims = Base::get_input_dims().template extend<>();
		in_dims(0) = rows;
		return TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad_extended.data(), in_dims);
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPULAYER_H_ */
