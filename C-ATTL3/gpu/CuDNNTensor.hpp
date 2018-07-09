/*
 * CuDNNTensor.h
 *
 *  Created on: 8 Jul 2018
 *      Author: Viktor Csomor
 */

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <type_traits>
#include <utility>

#include "CUDAError.hpp"
#include "CuDNNError.hpp"

#ifndef C_ATTL3_GPU_CUDNNTENSOR_H_
#define C_ATTL3_GPU_CUDNNTENSOR_H_

namespace cattle {
namespace internal {

template<bool Filter = false>
struct CuDNNTensorDescriptorManager {
	typedef cudnnTensorDescriptor_t Type;
	__inline__ static Type create_descriptor(cudnnDataType_t data_type, cudnnTensorFormat_t format,
			std::size_t n, std::size_t h, std::size_t w, std::size_t c) {
		Type desc;
		cudnnAssert(cudnnCreateTensorDescriptor(&desc));
		cudnnAssert(cudnnSetTensor4dDescriptor(desc, format, data_type, n, c, h, w));
	}
	__inline__ static void destroy_descriptor(Type& desc) {
		cudnnAssert(cudnnDestroyTensorDescriptor(desc));
	}
};

template<>
struct CuDNNTensorDescriptorManager<true> {
	typedef cudnnFilterDescriptor_t Type;
	__inline__ static Type create_descriptor(cudnnDataType_t data_type, cudnnTensorFormat_t format,
			std::size_t n, std::size_t h, std::size_t w, std::size_t c) {
		Type desc;
		cudnnAssert(cudnnCreateFilterDescriptor(&desc));
		cudnnAssert(cudnnSetFilter4dDescriptor(desc, data_type, format, n, c, h, w));
	}
	__inline__ static void destroy_descriptor(Type& desc) {
		cudnnAssert(cudnnDestroyFilterDescriptor(desc));
	}
};

/**
 * A template class for representing cuDNN device tensors of different data types.
 */
template<typename Scalar, bool Filter = false>
class CuDNNTensor {
	static constexpr cudnnDataType_t DATA_TYPE = std::is_same<Scalar,float>::value ?
			CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	typedef CuDNNTensor<Scalar> Self;
	typedef CuDNNTensorDescriptorManager<Filter> DescriptorManager;
public:
	typedef typename DescriptorManager::Type DescriptorType;
	inline CuDNNTensor() { }
	/**
	 * @param format The tensor format to use.
	 * @param n The batch size.
	 * @param h The height.
	 * @param w The width.
	 * @param c The number of channels.
	 */
	inline CuDNNTensor(cudnnTensorFormat_t format, std::size_t n, std::size_t h, std::size_t w,
			std::size_t c) :
				format(format),
				n(n),
				h(h),
				w(w),
				c(c),
				size(n * h * w * c),
				desc(DescriptorManager::create_descriptor(DATA_TYPE, format, n, h, w, c)) {
		assert(size > 0);
		cudaAssert(cudaMalloc(&data, size));
	}
	inline CuDNNTensor(const Self& tensor) :
			CuDNNTensor(tensor.format, tensor.n, tensor.h, tensor.w, tensor.c) {
		cudaAssert(cudaMemcpy(data, tensor.data, size, cudaMemcpyDeviceToDevice));
	}
	inline CuDNNTensor(Self&& tensor) {
		swap(*this, tensor);
	}
	inline ~CuDNNTensor() {
		cudaAssert(cudaFree(data));
		DescriptorManager::destroy_descriptor(desc);
	}
	inline Self& operator=(Self tensor) {
		swap(*this, tensor);
		return *this;
	}
	/**
	 * @return A device memory address pointer pointing to the (constant) first element of
	 * the tensor.
	 */
	inline const Scalar* get_data() const {
		return data;
	}
	/**
	 * @return A device memory address pointer pointing to the first element of the tensor.
	 */
	inline Scalar* get_data() {
		return data;
	}
	/**
	 * It populates the entire device tensor with data from the host memory.
	 *
	 * @param host_data A pointer to the first element of the host tensor.
	 */
	inline void copy_from_host(const Scalar* host_data) {
		cudaAssert(cudaMemcpy(data, host_data, size, cudaMemcpyHostToDevice));
	}
	/**
	 * It copies the entire device tensor to the host memory.
	 *
	 * @param host_data A pointer pointing to the beginning of a contiguous host memory
	 * block to which the device tensor is to be copied.
	 */
	inline void copy_to_host(Scalar* host_data) const {
		cudaAssert(cudaMemcpy(host_data, data, size, cudaMemcpyDeviceToHost));
	}
	inline friend void swap(Self& tensor1, Self& tensor2) {
		using std::swap;
		swap(tensor1.format, tensor2.format);
		swap(tensor1.dims, tensor2.dims);
		swap(tensor1.size, tensor2.size);
		swap(tensor1.desc, tensor2.desc);
		swap(tensor1.data, tensor2.data);
	}
	const cudnnTensorFormat_t format;
	const std::size_t n, h, w, c;
	const std::size_t size;
	const DescriptorType desc;
private:
	Scalar* data;
};

} /* namespace internal */
} /* namespace cattle */

#endif /* C_ATTL3_GPU_CUDNNTENSOR_H_ */
