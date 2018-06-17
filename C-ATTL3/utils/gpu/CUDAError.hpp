/*
 * CUDAError.hpp
 *
 *  Created on: 14.06.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_UTILS_GPU_CUDAERROR_HPP_
#define C_ATTL3_UTILS_GPU_CUDAERROR_HPP_

#include <cuda_runtime.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#define cudaErrorCheck(code) { _cuda_error_check(__FILE__, __LINE__, code); }
#define cudaAssert(code) { _cuda_assert(__FILE__, __LINE__, code); }

namespace cattle {
namespace internal {

/**
 * A class representing a CUDA runtime error.
 */
class CUDAError : public std::runtime_error {
public:
	/**
	 * @param code The CUDA error code.
	 * @param file The name of the file in which the error occurred.
	 * @param line The number of the line at which the error occurred.
	 */
	CUDAError(cudaError_t code, const char* file, int line) :
		std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(code)) + "; File: " +
				std::string(file) + "; Line: " + std::to_string(line)) { }
};

namespace {

__inline__ void _cuda_error_check(const char* file, int line, cudaError_t code = cudaGetLastError()) {
	if (code != cudaSuccess)
		throw CUDAError(code, file, line);
}

__inline__ void _cuda_assert(const char* file, int line, cudaError_t code = cudaGetLastError()) {
	try {
		_cuda_error_check(file, line, code);
	} catch (const CUDAError& e) {
		std::cout << e.what() << std::endl;
		exit(-1);
	}
}

}

} /* namespace internal */
} /* namespace cattle */

#endif /* C_ATTL3_UTILS_GPU_CUDAERROR_HPP_ */
