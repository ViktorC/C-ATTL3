/*
 * CuDNNTensor.hpp
 *
 *  Created on: 8 Jul 2018
 *      Author: Viktor Csomor
 */

#include <cstddef>
#include <cudnn.h>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "core/gpu/cuda/CUDAArray.hpp"
#include "core/gpu/cuda/CUDAError.hpp"
#include "CuDNNError.hpp"

#ifndef C_ATTL3_CORE_GPU_CUDNN_CUDNNTENSOR_H_
#define C_ATTL3_CORE_GPU_CUDNN_CUDNNTENSOR_H_

namespace cattle {
namespace gpu {

/**
 * A template class for representing cuDNN device tensors of different data types.
 */
template<typename Scalar>
class CuDNNTensor : public CUDAArray<Scalar> {
	typedef CUDAArray<Scalar> Base;
	typedef CuDNNTensor<Scalar> Self;
public:
	static constexpr cudnnDataType_t DATA_TYPE = std::is_same<Scalar,float>::value ?
			CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	static constexpr cudnnTensorFormat_t TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
	/**
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
				_channels(channels),
				_desc() {
		if (Base::size() > 0)
			create_tensor_descriptor(_desc, samples, height, width, channels);
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
			destroy_tensor_descriptor(_desc);
	}
	inline Self& operator=(Self tensor) {
		swap(*this, tensor);
		return *this;
	}
	/**
	 * @return The batch size of the tensor.
	 */
	inline std::size_t samples() const {
		return _samples;
	}
	/**
	 * @return The height of the tensor.
	 */
	inline std::size_t height() const {
		return _height;
	}
	/**
	 * @return The width of the tensor.
	 */
	inline std::size_t width() const {
		return _width;
	}
	/**
	 * @return The number of channels of the tensor.
	 */
	inline std::size_t channels() const {
		return _channels;
	}
	/**
	 * @return A constant reference to the tensor descriptor.
	 */
	inline const cudnnTensorDescriptor_t& desc() const {
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
	/**
	 * @param desc A reference to the tensor descri ptor object.
	 * @param samples The batch size.
	 * @param height The height.
	 * @param width The width.
	 * @param channels The number of channels.
	 */
	inline static void create_tensor_descriptor(cudnnTensorDescriptor_t& desc, std::size_t samples,
			std::size_t height, std::size_t width, std::size_t channels) {
		cudnnAssert(cudnnCreateTensorDescriptor(&desc));
		cudnnAssert(cudnnSetTensor4dDescriptor(desc, TENSOR_FORMAT, DATA_TYPE, samples, channels,
				height, width));
	}
	/**
	 * @param desc A constant reference to the tensor descriptor object.
	 */
	inline static void destroy_tensor_descriptor(const cudnnTensorDescriptor_t& desc) {
		cudnnAssert(cudnnDestroyTensorDescriptor(desc));
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
	cudnnTensorDescriptor_t _desc;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CUDNN_CUDNNTENSOR_H_ */
