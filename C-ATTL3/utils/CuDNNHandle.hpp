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
// TODO Sigmoid, tanh, ReLU, ELU, softmax
// TODO Batch norm
// TODO Dropout
// TODO RNN, LSTM

namespace cattle {
namespace internal {

/**
 * A singleton utility class providing methods for GPU accelerated deep neural network
 * operations.
 */
template<typename Scalar>
class CuDNNHandle {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	CuDNNHandle(const CuDNNHandle&) = delete;
	~CuDNNHandle() {
		// Destroy the cuBLAS handle.
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
	inline Tensor<Scalar,4> fwd_convolution(Tensor<Scalar,4>& input, Matrix<Scalar>& filter) {

	}
private:
	cudnnHandle_t handle;
	CuDNNHandle() :
			handle() {
		// Create the cuBLAS handle.
		cudnnStatus_t cudnn_stat = cudnnCreate(&handle);
		assert(cudnn_stat == CUDNN_STATUS_SUCCESS);
	}
};

}
} /* namespace cattle */

#endif /* CATTL3_UTILS_CUDNNHANDLE_H_ */
