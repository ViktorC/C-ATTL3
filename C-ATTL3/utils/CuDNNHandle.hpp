/*
 * CuDNNHandle.hpp
 *
 *  Created on: 28 May 2018
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_UTILS_CUDNNHANDLE_H_
#define CATTL3_UTILS_CUDNNHANDLE_H_

#include <cassert>
#include <cstddef>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <exception>
#include <string>
#include <type_traits>

#include "EigenProxy.hpp"

// TODO Convolution
// TODO Pooling
// TODO Batch norm
// TODO Dropout
// TODO RNN, LSTM

namespace cattle {
namespace internal {

/**
 * A singleton utility class providing methods for GPU accelerated deep neural network
 * operations.
 */
template<typename Scalar, std::size_t Rank>
class CuDNNHandle {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 1 && Rank < 5, "illegal rank value");
public:
	CuDNNHandle(const CuDNNHandle&) = delete;
	~CuDNNHandle() {
		// Destroy the cuDNN handle.
		cudnnStatus_t cudnn_stat = cudnnDestroy(handle);
		assert(cudnn_stat == CUDNN_STATUS_SUCCESS);
	}
	CuDNNHandle& operator=(const CuDNNHandle&) = delete;
	/**
	 * @return A reference to the only instance of the class.
	 */
	inline static CuDNNHandle& get_instance() {
		static CuDNNHandle instance;
		return instance;
	}
	inline Tensor<Scalar,Rank> activation_fwd(Tensor<Scalar,Rank>& input, cudnnActivationMode_t act_mode,
			Scalar coeff) {
		cudnnActivationDescriptor_t act_desc;
		cudnnStatus_t cudnn_stat = cudnnCreateActivationDescriptor(&act_desc);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS)
			throw std::runtime_error("cudnn activation descriptor creation failure: " + std::to_string(cudnn_stat));
		cudnn_stat = cudnnSetActivationDescriptor(act_desc, act_mode, CUDNN_PROPAGATE_NAN, (double) coeff);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyActivationDescriptor(act_desc);
			throw std::runtime_error("cudnn activation descriptor setting failure: " + std::to_string(cudnn_stat));
		}
		cudnnTensorDescriptor_t tens_desc;
		cudnn_stat = cudnnCreateTensorDescriptor(&tens_desc);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyActivationDescriptor(act_desc);
			throw std::runtime_error("cudnn tensor descriptor creation failure: " + std::to_string(cudnn_stat));
		}
		Dimensions<std::size_t,4> dims_4d = Dimensions<std::size_t,Rank>(input.dimensions()).template extend<4 - Rank>();
		cudnn_stat = cudnnSetTensor4dDescriptor(tens_desc, CUDNN_TENSOR_NHWC, resolve_cudnn_data_type(),
				dims_4d(0), dims_4d(3), dims_4d(1), dims_4d(2));
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			throw std::runtime_error("cudnn tensor descriptor setting failure: " + std::to_string(cudnn_stat));
		}
		Scalar* dev_array;
		const std::size_t size = input.size() * sizeof(Scalar);
		cudaError_t cuda_stat = cudaMalloc(&dev_array, size);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			throw std::runtime_error("cuda malloc failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMemcpy(dev_array, input.data(), size, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_array);
			throw std::runtime_error("cuda copy from host to device failure: " + std::to_string(cuda_stat));
		}
		const Scalar alpha = 1;
		const Scalar beta = 0;
		cudnn_stat = cudnnActivationForward(handle, act_desc, &alpha, tens_desc, dev_array, &beta, tens_desc, dev_array);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_array);
			throw std::runtime_error("cudnn activation failure: " + std::to_string(cudnn_stat));
		}
		Tensor<Scalar,Rank> out(input.dimensions());
		cuda_stat = cudaMemcpy(out.data(), dev_array, size, cudaMemcpyDeviceToHost);
		cudnnDestroyActivationDescriptor(act_desc);
		cudnnDestroyTensorDescriptor(tens_desc);
		cudaFree(dev_array);
		if (cuda_stat != cudaSuccess)
			throw std::runtime_error("cuda copy from device to host failure: " + std::to_string(cuda_stat));
		return out;
	}
	inline Tensor<Scalar,Rank> activation_bwd(Tensor<Scalar,Rank>& input, Tensor<Scalar,Rank>& output,
			Tensor<Scalar,Rank>& out_grads, cudnnActivationMode_t act_mode, Scalar coeff) {
		cudnnActivationDescriptor_t act_desc;
		cudnnStatus_t cudnn_stat = cudnnCreateActivationDescriptor(&act_desc);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS)
			throw std::runtime_error("cudnn activation descriptor creation failure: " + std::to_string(cudnn_stat));
		cudnn_stat = cudnnSetActivationDescriptor(act_desc, act_mode, CUDNN_PROPAGATE_NAN, (double) coeff);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyActivationDescriptor(act_desc);
			throw std::runtime_error("cudnn activation descriptor setting failure: " + std::to_string(cudnn_stat));
		}
		cudnnTensorDescriptor_t tens_desc;
		cudnn_stat = cudnnCreateTensorDescriptor(&tens_desc);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyActivationDescriptor(act_desc);
			throw std::runtime_error("cudnn tensor descriptor creation failure: " + std::to_string(cudnn_stat));
		}
		Dimensions<std::size_t,4> dims_4d = Dimensions<std::size_t,Rank>(input.dimensions()).template extend<4 - Rank>();
		cudnn_stat = cudnnSetTensor4dDescriptor(tens_desc, CUDNN_TENSOR_NHWC, resolve_cudnn_data_type(),
				dims_4d(0), dims_4d(3), dims_4d(1), dims_4d(2));
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			throw std::runtime_error("cudnn tensor descriptor setting failure: " + std::to_string(cudnn_stat));
		}
		Scalar* dev_in_array;
		Scalar* dev_out_array;
		Scalar* dev_out_grads_array;
		const std::size_t size = input.size() * sizeof(Scalar);
		cudaError_t cuda_stat = cudaMalloc(&dev_in_array, size);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			throw std::runtime_error("cuda malloc failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMalloc(&dev_out_array, size);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_in_array);
			throw std::runtime_error("cuda malloc failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMalloc(&dev_out_grads_array, size);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_in_array);
			cudaFree(dev_out_array);
			throw std::runtime_error("cuda malloc failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMemcpy(dev_in_array, input.data(), size, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_in_array);
			cudaFree(dev_out_array);
			cudaFree(dev_out_grads_array);
			throw std::runtime_error("cuda copy from host to device failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMemcpy(dev_out_array, output.data(), size, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_in_array);
			cudaFree(dev_out_array);
			cudaFree(dev_out_grads_array);
			throw std::runtime_error("cuda copy from host to device failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMemcpy(dev_out_grads_array, out_grads.data(), size, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_in_array);
			cudaFree(dev_out_array);
			cudaFree(dev_out_grads_array);
			throw std::runtime_error("cuda copy from host to device failure: " + std::to_string(cuda_stat));
		}
		const Scalar alpha = 1;
		const Scalar beta = 0;
		cudnn_stat = cudnnActivationBackward(handle, act_desc, &alpha, tens_desc, dev_out_array, tens_desc, dev_out_grads_array,
				tens_desc, dev_in_array, &beta, tens_desc, dev_out_grads_array);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyActivationDescriptor(act_desc);
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_in_array);
			cudaFree(dev_out_array);
			cudaFree(dev_out_grads_array);
			throw std::runtime_error("cudnn activation failure: " + std::to_string(cudnn_stat));
		}
		Tensor<Scalar,Rank> prev_out_grads(input.dimensions());
		cuda_stat = cudaMemcpy(prev_out_grads.data(), dev_out_grads_array, size, cudaMemcpyDeviceToHost);
		cudnnDestroyActivationDescriptor(act_desc);
		cudnnDestroyTensorDescriptor(tens_desc);
		cudaFree(dev_in_array);
		cudaFree(dev_out_array);
		cudaFree(dev_out_grads_array);
		if (cuda_stat != cudaSuccess)
			throw std::runtime_error("cudnn copy from device to host failure: " + std::to_string(cuda_stat));
		return prev_out_grads;
	}
	inline Tensor<Scalar,Rank> sigmoid_activation_fwd(Tensor<Scalar,Rank>& input) {
		return activation_fwd(input, CUDNN_ACTIVATION_SIGMOID, 0);
	}
	inline Tensor<Scalar,Rank> sigmoid_activation_bwd(Tensor<Scalar,Rank>& input, Tensor<Scalar,Rank>& output,
			Tensor<Scalar,Rank>& out_grads) {
		return activation_bwd(input, output, out_grads, CUDNN_ACTIVATION_SIGMOID, 0);
	}
	inline Tensor<Scalar,Rank> tanh_activation_fwd(Tensor<Scalar,Rank>& input) {
		return activation_fwd(input, CUDNN_ACTIVATION_TANH, 0);
	}
	inline Tensor<Scalar,Rank> tanh_activation_bwd(Tensor<Scalar,Rank>& input, Tensor<Scalar,Rank>& output,
			Tensor<Scalar,Rank>& out_grads) {
		return activation_bwd(input, output, out_grads, CUDNN_ACTIVATION_TANH, 0);
	}
	inline Tensor<Scalar,Rank> relu_activation_fwd(Tensor<Scalar,Rank>& input) {
		return activation_fwd(input, CUDNN_ACTIVATION_RELU, 0);
	}
	inline Tensor<Scalar,Rank> relu_activation_bwd(Tensor<Scalar,Rank>& input, Tensor<Scalar,Rank>& output,
			Tensor<Scalar,Rank>& out_grads) {
		return activation_bwd(input, output, out_grads, CUDNN_ACTIVATION_RELU, 0);
	}
	inline Tensor<Scalar,Rank> elu_activation_fwd(Tensor<Scalar,Rank>& input, Scalar alpha) {
		return activation_fwd(input, CUDNN_ACTIVATION_ELU, alpha);
	}
	inline Tensor<Scalar,Rank> elu_activation_bwd(Tensor<Scalar,Rank>& input, Tensor<Scalar,Rank>& output,
			Tensor<Scalar,Rank>& out_grads, Scalar alpha) {
		return activation_bwd(input, output, out_grads, CUDNN_ACTIVATION_ELU, alpha);
	}
	inline Tensor<Scalar,Rank> softmax_fwd(Tensor<Scalar,Rank>& input) {
		cudnnTensorDescriptor_t tens_desc;
		cudnnStatus_t cudnn_stat = cudnnCreateTensorDescriptor(&tens_desc);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS)
			throw std::runtime_error("cudnn tensor descriptor creation failure: " + std::to_string(cudnn_stat));
		Dimensions<std::size_t,4> dims_4d = Dimensions<std::size_t,Rank>(input.dimensions()).template extend<4 - Rank>();
		cudnn_stat = cudnnSetTensor4dDescriptor(tens_desc, CUDNN_TENSOR_NHWC, resolve_cudnn_data_type(),
				dims_4d(0), dims_4d(3), dims_4d(1), dims_4d(2));
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyTensorDescriptor(tens_desc);
			throw std::runtime_error("cudnn tensor descriptor setting failure: " + std::to_string(cudnn_stat));
		}
		Scalar* dev_array;
		const std::size_t size = input.size() * sizeof(Scalar);
		cudaError_t cuda_stat = cudaMalloc(&dev_array, size);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyTensorDescriptor(tens_desc);
			throw std::runtime_error("cuda malloc failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMemcpy(dev_array, input.data(), size, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_array);
			throw std::runtime_error("cuda copy from host to device failure: " + std::to_string(cuda_stat));
		}
		const Scalar alpha = 1;
		const Scalar beta = 0;
		cudnn_stat = cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
				tens_desc, dev_array, &beta, tens_desc, dev_array);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_array);
			throw std::runtime_error("cudnn softmax failure: " + std::to_string(cudnn_stat));
		}
		Tensor<Scalar,Rank> out(input.dimensions());
		cuda_stat = cudaMemcpy(out.data(), dev_array, size, cudaMemcpyDeviceToHost);
		cudnnDestroyTensorDescriptor(tens_desc);
		cudaFree(dev_array);
		if (cuda_stat != cudaSuccess)
			throw std::runtime_error("cuda copy from device to host failure: " + std::to_string(cuda_stat));
		return out;
	}
	inline Tensor<Scalar,Rank> softmax_bwd(Tensor<Scalar,Rank>& output, Tensor<Scalar,Rank>& out_grads) {
		cudnnTensorDescriptor_t tens_desc;
		cudnnStatus_t cudnn_stat = cudnnCreateTensorDescriptor(&tens_desc);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS)
			throw std::runtime_error("cudnn tensor descriptor creation failure: " + std::to_string(cudnn_stat));
		Dimensions<std::size_t,4> dims_4d = Dimensions<std::size_t,Rank>(output.dimensions()).template extend<4 - Rank>();
		cudnn_stat = cudnnSetTensor4dDescriptor(tens_desc, CUDNN_TENSOR_NHWC, resolve_cudnn_data_type(),
				dims_4d(0), dims_4d(3), dims_4d(1), dims_4d(2));
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyTensorDescriptor(tens_desc);
			throw std::runtime_error("cudnn tensor descriptor setting failure: " + std::to_string(cudnn_stat));
		}
		Scalar* dev_out_array;
		Scalar* dev_out_grads_array;
		const std::size_t size = output.size() * sizeof(Scalar);
		cudaError_t cuda_stat = cudaMalloc(&dev_out_array, size);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyTensorDescriptor(tens_desc);
			throw std::runtime_error("cuda malloc failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMalloc(&dev_out_grads_array, size);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_out_array);
			throw std::runtime_error("cuda malloc failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMemcpy(dev_out_array, output.data(), size, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_out_array);
			cudaFree(dev_out_grads_array);
			throw std::runtime_error("cuda copy from host to device failure: " + std::to_string(cuda_stat));
		}
		cuda_stat = cudaMemcpy(dev_out_grads_array, out_grads.data(), size, cudaMemcpyHostToDevice);
		if (cuda_stat != cudaSuccess) {
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_out_array);
			cudaFree(dev_out_grads_array);
			throw std::runtime_error("cuda copy from host to device failure: " + std::to_string(cuda_stat));
		}
		const Scalar alpha = 1;
		const Scalar beta = 0;
		cudnn_stat = cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, tens_desc,
				dev_out_array, tens_desc, dev_out_grads_array, &beta, tens_desc, dev_out_grads_array);
		if (cudnn_stat != CUDNN_STATUS_SUCCESS) {
			cudnnDestroyTensorDescriptor(tens_desc);
			cudaFree(dev_out_array);
			cudaFree(dev_out_grads_array);
			throw std::runtime_error("cudnn softmax failure: " + std::to_string(cudnn_stat));
		}
		Tensor<Scalar,Rank> prev_out_grads(output.dimensions());
		cuda_stat = cudaMemcpy(prev_out_grads.data(), dev_out_grads_array, size, cudaMemcpyDeviceToHost);
		cudnnDestroyTensorDescriptor(tens_desc);
		cudaFree(dev_out_array);
		cudaFree(dev_out_grads_array);
		if (cuda_stat != cudaSuccess)
			throw std::runtime_error("cudnn copy from device to host failure: " + std::to_string(cuda_stat));
		return prev_out_grads;
	}
private:
	cudnnHandle_t handle;
	CuDNNHandle() :
			handle() {
		// Create the cuBLAS handle.
		cudnnStatus_t cudnn_stat = cudnnCreate(&handle);
		assert(cudnn_stat == CUDNN_STATUS_SUCCESS);
	}
	__inline__ static cudnnDataType_t resolve_cudnn_data_type() {
		if (std::is_same<Scalar,float>::value)
			return CUDNN_DATA_FLOAT;
		else
			return CUDNN_DATA_DOUBLE;
	}
};

}
} /* namespace cattle */

#endif /* CATTL3_UTILS_CUDNNHANDLE_H_ */
