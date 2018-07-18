/*
 * CuDNNTensor.h
 *
 *  Created on: 8 Jul 2018
 *      Author: Viktor Csomor
 */

#include <cstddef>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "CUDAArray.hpp"
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
class CuDNNTensor : public CUDAArray<Scalar> {
	typedef CUDAArray<Scalar> Base;
	typedef CuDNNTensor<Scalar,Filter> Self;
	typedef CuDNNTensorDescriptorManager<Filter> DescriptorManager;
public:
	typedef typename DescriptorManager::Type DescriptorType;
	static constexpr cudnnDataType_t DATA_TYPE = std::is_same<Scalar,float>::value ?
			CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	static constexpr cudnnTensorFormat_t TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
	/**
	 * @param format The tensor format to use.
	 * @param samples The batch size.
	 * @param height The height.
	 * @param width The width.
	 * @param channels The number of channels.
	 */
	inline CuDNNTensor(std::size_t samples, std::size_t height, std::size_t width, std::size_t channels) :
				Base(samples * height * width * channels),
				_samples(samples),
				_height(height),
				_width(width),
				_channels(channels) {
		if (Base::size() > 0)
			DescriptorManager::create_descriptor(_desc, DATA_TYPE, TENSOR_FORMAT, samples, height, width, channels);
	}
	inline CuDNNTensor() :
			CuDNNTensor(0u, 0u, 0u, 0u) { }
	inline CuDNNTensor(const Self& tensor) :
			Base(tensor),
			_samples(tensor._samples),
			_height(tensor._height),
			_width(tensor._width),
			_channels(tensor._channels),
			_desc(tensor._desc) { }
	inline CuDNNTensor(Self&& tensor) :
			CuDNNTensor() {
		swap(*this, tensor);
	}
	inline ~CuDNNTensor() {
		if (Base::size() > 0)
			DescriptorManager::destroy_descriptor(_desc);
	}
	inline Self& operator=(Self tensor) {
		swap(*this, tensor);
		return *this;
	}
	/**
	 * @return A constant reference to the tensor descriptor.
	 */
	inline const DescriptorType& desc() const {
		return _desc;
	}
	/**
	 * @return A string representation of the tensor.
	 */
	std::string to_string() const {
		std::stringstream strm;
		strm << "data type: " << DATA_TYPE << "; format: " << TENSOR_FORMAT << "; " <<
				"[N:" << _samples << ", C:" << _channels << ", H:" << _height << ", W:" << _width << "]";
		return strm.str();
	}
	inline friend std::ostream& operator<<(std::ostream& os, const Self& tensor) {
		return os << tensor.to_string() << std::endl;
	}
	inline friend void swap(Self& tensor1, Self& tensor2) {
		using std::swap;
        swap(static_cast<Base&>(tensor1), static_cast<Base&>(tensor2));
		swap(tensor1._samples, tensor2._samples);
		swap(tensor1._height, tensor2._height);
		swap(tensor1._width, tensor2._width);
		swap(tensor1._channels, tensor2._channels);
		swap(tensor1._desc, tensor2._desc);
	}
private:
	std::size_t _samples, _height, _width, _channels;
	DescriptorType _desc;
};

} /* namespace internal */
} /* namespace cattle */

#endif /* C_ATTL3_GPU_CUDNNTENSOR_H_ */
