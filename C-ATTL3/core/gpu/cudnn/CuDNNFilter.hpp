/*
 * CuDNNFilter.hpp
 *
 *  Created on: 5 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_CUDNN_CUDNNFILTER_H_
#define C_ATTL3_CORE_GPU_CUDNN_CUDNNFILTER_H_

#include "CuDNNTensor.hpp"

namespace cattle {
namespace gpu {

/**
 * A template class representing cuDNN filter tensors for convolution.
 */
template<typename Scalar>
class CuDNNFilter : public CuDNNTensor<Scalar> {
	typedef CUDAArray<Scalar> Root;
	typedef CuDNNTensor<Scalar> Base;
	typedef CuDNNFilter<Scalar> Self;
public:
	/**
	 * @param samples The batch size.
	 * @param height The height.
	 * @param width The width.
	 * @param channels The number of channels.
	 */
	inline CuDNNFilter(std::size_t samples, std::size_t height, std::size_t width, std::size_t channels) :
			Base(samples, height, width, channels),
			_filter_desc() {
		if (Root::size() > 0)
			create_filter_descriptor(_filter_desc, samples, height, width, channels);
	}
	inline CuDNNFilter() :
			CuDNNFilter(0u, 0u, 0u, 0u) { }
	inline CuDNNFilter(const Self& filter) :
			Base(filter),
			_filter_desc(filter._filter_desc) { }
	inline CuDNNFilter(Self&& filter) :
			CuDNNFilter() {
		swap(*this, filter);
	}
	inline ~CuDNNFilter() {
		if (Root::size() > 0)
			destroy_filter_descriptor(_filter_desc);
	}
	inline Self& operator=(Self filter) {
		swap(*this, filter);
		return *this;
	}
	/**
	 * @return A constant reference to the filter descriptor.
	 */
	inline const cudnnFilterDescriptor_t& filter_desc() const {
		return _filter_desc;
	}
	/**
	 * @param filter_desc A reference to the filter descriptor object.
	 * @param samples The batch size.
	 * @param height The height.
	 * @param width The width.
	 * @param channels The number of channels.
	 */
	inline static void create_filter_descriptor(cudnnFilterDescriptor_t& filter_desc, std::size_t samples,
			std::size_t height, std::size_t width, std::size_t channels) {
		cudnnAssert(cudnnCreateFilterDescriptor(&filter_desc));
		cudnnAssert(cudnnSetFilter4dDescriptor(filter_desc, Base::DATA_TYPE, Base::TENSOR_FORMAT, samples,
				channels, height, width));
	}
	/**
	 * @param filter_desc A constant reference to the filter descriptor object.
	 */
	inline static void destroy_filter_descriptor(const cudnnFilterDescriptor_t& filter_desc) {
		cudnnAssert(cudnnDestroyFilterDescriptor(filter_desc));
	}
	inline friend void swap(Self& filter1, Self& filter2) {
		using std::swap;
        swap(static_cast<Base&>(filter1), static_cast<Base&>(filter2));
		swap(filter1._filter_desc, filter2._filter_desc);
	}
private:
	cudnnFilterDescriptor_t _filter_desc;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CUDNN_CUDNNFILTER_H_ */
