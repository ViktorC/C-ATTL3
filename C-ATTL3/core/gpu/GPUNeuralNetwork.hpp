/*
 * GPUNeuralNetwork.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_GPUNEURALNETWORK_H_
#define C_ATTL3_CORE_GPU_GPUNEURALNETWORK_H_

#include "core/NeuralNetwork.hpp"
#include "GPULayer.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar, std::size_t Rank, bool Sequential>
class GPUNeuralNetwork : public virtual NeuralNetwork<Scalar,Rank,Sequential> {
	typedef NeuralNetwork<Scalar,Rank,Sequential> Base;
protected:
	typedef Dimensions<std::size_t,3> GPUDims;
public:
	virtual const GPUDims& get_gpu_input_dims() const = 0;
	virtual const GPUDims& get_gpu_output_dims() const = 0;
	virtual std::vector<const GPULayer<Scalar,Rank>*> get_gpu_layers() const = 0;
	virtual std::vector<GPULayer<Scalar,Rank>*> get_gpu_layers() = 0;
	virtual CuDNNTensor<Scalar> propagate(CuDNNTensor<Scalar> input, bool training) = 0;
	virtual CuDNNTensor<Scalar> backpropagate(CuDNNTensor<Scalar> out_grad) = 0;
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		auto rows = input.dimension(0);
		auto in_cuda_dims = input.template extend<>();
		in_cuda_dims(0) = rows;
		auto out_extended = TensorConversion<Scalar>::convert_from_cudnn_to_eigen(
				propagate(TensorConversion<Scalar>::convert_from_eigen_to_cudnn(
						TensorMap<Scalar,4>(input.data(), in_cuda_dims)), training));
		auto out_dims = get_output_dims().template extend<>();
		out_dims(0) = rows;
		return TensorMap<Scalar,Base::DATA_RANK>(out_extended.data(), out_dims);
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grad) {
		auto rows = out_grad.dimension(0);
		auto out_cuda_dims = get_cuda_output_dims().template extend<>();
		out_cuda_dims(0) = rows;
		auto prev_out_grad_extended = TensorConversion<Scalar>::convert_from_cudnn_to_eigen(
				backpropagate(TensorConversion<Scalar>::convert_from_eigen_to_cudnn(
						TensorMap<Scalar,4>(out_grad.data(), out_cuda_dims))));
		auto in_dims = get_input_dims().template extend<>();
		in_dims(0) = rows;
		return TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad_extended.data(), in_dims);
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPUNEURALNETWORK_H_ */
