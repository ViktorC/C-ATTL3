/*
 * CUDAArray.hpp
 *
 *  Created on: 18.07.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_CUDA_CUDAARRAY_H_
#define C_ATTL3_CORE_GPU_CUDA_CUDAARRAY_H_

#include <cstddef>
#include <cuda_runtime.h>
#include <utility>

#include "CUDAError.hpp"

namespace cattle {
namespace gpu {

/**
 * A template class for CUDA device arrays of different data types.
 */
template<typename Scalar>
class CUDAArray {
	typedef CUDAArray<Scalar> Self;
public:
	/**
	 * @param data The device array to wrap in a CUDAArray. The ownership of the
	 * pointer is not transfered to the created CUDAArray instance.
	 * @param size The number of elements the array is to have.
	 */
	inline CUDAArray(Scalar* data, std::size_t size) :
			_size(size),
			_data(data),
			_wrapper(true) {
		assert(data || size == 0);
	}
	/**
	 * @param size The number of elements the array is to have.
	 */
	inline CUDAArray(std::size_t size) :
			_size(size),
			_data(nullptr),
			_wrapper(false) {
		if (size > 0)
			cudaAssert(cudaMalloc(&_data, size * sizeof(Scalar)));
	}
	inline CUDAArray() :
			CUDAArray(0u) { }
	inline CUDAArray(const Self& array) :
			CUDAArray(array._size) {
		if (_size > 0)
			cudaAssert(cudaMemcpy(_data, array._data, _size * sizeof(Scalar), cudaMemcpyDeviceToDevice));
	}
	inline CUDAArray(Self&& array) :
			CUDAArray() {
		swap(*this, array);
	}
	inline virtual ~CUDAArray() {
		if (_size > 0 && !_wrapper)
			cudaAssert(cudaFree(_data));
	}
	inline Self& operator=(Self array) {
		swap(*this, array);
		return *this;
	}
	/**
	 * @return The size of the array.
	 */
	inline std::size_t size() const {
		return _size;
	}
	/**
	 * @return A device memory address pointer pointing to the (constant) first element of
	 * the array.
	 */
	inline const Scalar* data() const {
		return _data;
	}
	/**
	 * @return A device memory address pointer pointing to the first element of the array.
	 */
	inline Scalar* data() {
		return _data;
	}
	/**
	 * @return Whether the instance is just a wrapper over a device array that it does not
	 * own.
	 */
	inline bool wrapper() const {
		return _wrapper;
	}
	/**
	 * @param value The integer value to which the values of the array are to be set.
	 */
	inline void set_values(int value) {
		if (_size > 0)
			cudaAssert(cudaMemset(_data, value, _size * sizeof(Scalar)));
	}
	/**
	 * It populates the entire device array with data from the host memory.
	 *
	 * @param host_array A pointer to the first element of the host array.
	 */
	inline void copy_from_host(const Scalar* host_array) {
		if (_size > 0)
			cudaAssert(cudaMemcpy(_data, host_array, _size * sizeof(Scalar), cudaMemcpyHostToDevice));
	}
	/**
	 * It copies the entire device array to the host memory.
	 *
	 * @param host_array A pointer pointing to the beginning of a contiguous host memory
	 * block to which the device tensor is to be copied.
	 */
	inline void copy_to_host(Scalar* host_array) const {
		if (_size > 0)
			cudaAssert(cudaMemcpy(host_array, _data, _size * sizeof(Scalar), cudaMemcpyDeviceToHost));
	}
	inline friend void swap(Self& array1, Self& array2) {
		using std::swap;
		swap(array1._size, array2._size);
		swap(array1._data, array2._data);
		swap(array1._wrapper, array2._wrapper);
	}
private:
	std::size_t _size;
	Scalar* _data;
	bool _wrapper;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CUDA_CUDAARRAY_H_ */
