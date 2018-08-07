/*
 * GPULayer.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_GPULAYER_H_
#define C_ATTL3_CORE_GPU_GPULAYER_H_

#include "core/Layer.hpp"
#include "GPUParameters.hpp"
#include "TensorConversion.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar, std::size_t Rank>
class GPULayer : public virtual Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
protected:
	typedef Dimensions<std::size_t,3> GPUDims;
public:
	virtual const GPUDims& get_gpu_input_dims() const = 0;
	virtual const GPUDims& get_gpu_output_dims() const = 0;
	virtual std::vector<const GPUParameters<Scalar>*> get_gpu_params() const = 0;
	virtual std::vector<GPUParameters<Scalar>*> get_gpu_params() = 0;
	virtual CuDNNTensor<Scalar> pass_forward(CuDNNTensor<Scalar> in, bool training) = 0;
	virtual CuDNNTensor<Scalar> pass_back(CuDNNTensor<Scalar> out_grad) = 0;
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		auto rows = in.dimension(0);
		auto in_cuda_dims = in.template extend<>();
		in_cuda_dims(0) = rows;
		auto out_extended = TensorConversion<Scalar>::convert_from_cudnn_to_eigen(
				pass_forward(TensorConversion<Scalar>::convert_from_eigen_to_cudnn(
						TensorMap<Scalar,4>(in.data(), in_cuda_dims)), training));
		auto out_dims = get_output_dims().template extend<>();
		out_dims(0) = rows;
		return TensorMap<Scalar,Base::DATA_RANK>(out_extended.data(), out_dims);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		auto rows = out_grad.dimension(0);
		auto out_cuda_dims = get_cuda_output_dims().template extend<>();
		out_cuda_dims(0) = rows;
		auto prev_out_grad_extended = TensorConversion<Scalar>::convert_from_cudnn_to_eigen(
				pass_back(TensorConversion<Scalar>::convert_from_eigen_to_cudnn(
						TensorMap<Scalar,4>(out_grad.data(), out_cuda_dims))));
		auto in_dims = get_input_dims().template extend<>();
		in_dims(0) = rows;
		return TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad_extended.data(), in_dims);
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPULAYER_H_ */
