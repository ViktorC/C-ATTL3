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
#include <ostream>
#include <sstream>
#include <string>
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
	__inline__ static void create_descriptor(Type& desc, cudnnDataType_t data_type, cudnnTensorFormat_t format,
			std::size_t n, std::size_t h, std::size_t w, std::size_t c) {
		cudnnAssert(cudnnCreateTensorDescriptor(&desc));
		cudnnAssert(cudnnSetTensor4dDescriptor(desc, format, data_type, n, c, h, w));
	}
	__inline__ static void destroy_descriptor(const Type& desc) {
		cudnnAssert(cudnnDestroyTensorDescriptor(desc));
	}
};

template<>
struct CuDNNTensorDescriptorManager<true> {
	typedef cudnnFilterDescriptor_t Type;
	__inline__ static void create_descriptor(Type& desc, cudnnDataType_t data_type, cudnnTensorFormat_t format,
			std::size_t n, std::size_t h, std::size_t w, std::size_t c) {
		cudnnAssert(cudnnCreateFilterDescriptor(&desc));
		cudnnAssert(cudnnSetFilter4dDescriptor(desc, data_type, format, n, c, h, w));
	}
	__inline__ static void destroy_descriptor(const Type& desc) {
		cudnnAssert(cudnnDestroyFilterDescriptor(desc));
	}
};

/**
 * A template class for representing cuDNN device tensors of different data types.
 */
template<typename Scalar, bool Filter = false>
class CuDNNTensor {
	typedef CuDNNTensor<Scalar,Filter> Self;
	typedef CuDNNTensorDescriptorManager<Filter> DescriptorManager;
public:
	typedef typename DescriptorManager::Type DescriptorType;
	static constexpr cudnnDataType_t DATA_TYPE = std::is_same<Scalar,float>::value ?
			CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	/**
	 * @param format The tensor format to use.
	 * @param n The batch size.
	 * @param h The height.
	 * @param w The width.
	 * @param c The number of channels.
	 */
	inline CuDNNTensor(std::size_t n, std::size_t h, std::size_t w,
			std::size_t c, cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC) :
				n(n),
				h(h),
				w(w),
				c(c),
				size(n * h * w * c),
				format(format),
				desc(),
				data(nullptr) {
		if (size > 0) {
			DescriptorManager::create_descriptor(desc, DATA_TYPE, format, n, h, w, c);
			cudaAssert(cudaMalloc(&data, size * sizeof(Scalar)));
		}
	}
	inline CuDNNTensor() :
			CuDNNTensor(0u, 0u, 0u, 0u) { }
	inline CuDNNTensor(const Self& tensor) :
			CuDNNTensor(tensor.n, tensor.h, tensor.w, tensor.c, tensor.format) {
		if (size > 0)
			cudaAssert(cudaMemcpy(data, tensor.data, size * sizeof(Scalar), cudaMemcpyDeviceToDevice));
	}
	inline CuDNNTensor(Self&& tensor) :
			CuDNNTensor() {
		swap(*this, tensor);
	}
	inline ~CuDNNTensor() {
		if (size > 0) {
			cudaAssert(cudaFree(data));
			DescriptorManager::destroy_descriptor(desc);
		}
	}
	inline Self& operator=(Self tensor) {
		swap(*this, tensor);
		return *this;
	}
	/**
	 * @return The batch size of the tensor.
	 */
	inline std::size_t get_n() const {
		return n;
	}
	/**
	 * @return The height of the tensor.
	 */
	inline std::size_t get_h() const {
		return h;
	}
	/**
	 * @return The width of the tensor.
	 */
	inline std::size_t get_w() const {
		return w;
	}
	/**
	 * @return The depth/number of channels of the tensor.
	 */
	inline std::size_t get_c() const {
		return c;
	}
	/**
	 * @return The total size of the tensor.
	 */
	inline std::size_t get_size() const {
		return size;
	}
	/**
	 * @return The format of the tensor.
	 */
	inline cudnnTensorFormat_t get_format() const {
		return format;
	}
	/**
	 * @return A constant reference to the tensor descriptor.
	 */
	inline const DescriptorType& get_desc() const {
		return desc;
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
		if (size > 0)
			cudaAssert(cudaMemcpy(data, host_data, size * sizeof(Scalar), cudaMemcpyHostToDevice));
	}
	/**
	 * It copies the entire device tensor to the host memory.
	 *
	 * @param host_data A pointer pointing to the beginning of a contiguous host memory
	 * block to which the device tensor is to be copied.
	 */
	inline void copy_to_host(Scalar* host_data) const {
		if (size > 0)
			cudaAssert(cudaMemcpy(host_data, data, size * sizeof(Scalar), cudaMemcpyDeviceToHost));
	}
	/**
	 * @return A string representation of the tensor.
	 */
	std::string to_string() const {
		std::stringstream strm;
		strm << "data type: " << DATA_TYPE << "; format: " << format << "; " <<
				"[N:" << n << ", H:" << h << ", W:" << w << ", C:" << c << "]";
		return strm.str();
	}
	inline friend std::ostream& operator<<(std::ostream& os, const Self& tensor) {
		return os << tensor.to_string() << std::endl;
	}
	inline friend void swap(Self& tensor1, Self& tensor2) {
		using std::swap;
		swap(tensor1.n, tensor2.n);
		swap(tensor1.h, tensor2.h);
		swap(tensor1.w, tensor2.w);
		swap(tensor1.c, tensor2.c);
		swap(tensor1.size, tensor2.size);
		swap(tensor1.format, tensor2.format);
		swap(tensor1.desc, tensor2.desc);
		swap(tensor1.data, tensor2.data);
	}
private:
	std::size_t n, h, w, c;
	std::size_t size;
	cudnnTensorFormat_t format;
	DescriptorType desc;
	Scalar* data;
};

} /* namespace internal */
} /* namespace cattle */

#endif /* C_ATTL3_GPU_CUDNNTENSOR_H_ */
