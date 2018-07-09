/*
 * CuDNNError.hpp
 *
 *  Created on: 14.06.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_GPU_CUDNNERROR_H_
#define C_ATTL3_GPU_CUDNNERROR_H_

#include <cudnn.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#define cudnnErrorCheck(status) { _cudnn_error_check(status, __FILE__, __LINE__); }
#define cudnnAssert(status) { _cudnn_assert(status, __FILE__, __LINE__); }

namespace cattle {
namespace internal {

/**
 * A class representing a cuDNN runtime error.
 */
class CuDNNError : public std::runtime_error {
public:
	/**
	 * @param status The cuDNN status code.
	 * @param file The name of the file in which the error occurred.
	 * @param line The number of the line at which the error occurred.
	 */
	CuDNNError(cudnnStatus_t status, const char* file, int line) :
		std::runtime_error("cuDNN Error: " + std::string(cudnnGetErrorString(status)) + "; File: " +
				std::string(file) + "; Line: " + std::to_string(line)) { }
};

namespace {

__inline__ void _cudnn_error_check(cudnnStatus_t status, const char* file, int line) {
	if (status != CUDNN_STATUS_SUCCESS)
		throw CuDNNError(status, file, line);
}

__inline__ void _cudnn_assert(cudnnStatus_t status, const char* file, int line) {
	try {
		_cudnn_error_check(status, file, line);
	} catch (const CuDNNError& e) {
		std::cout << e.what() << std::endl;
		exit(-1);
	}
}

}

} /* namespace internal */
} /* namespace cattle */

#endif /* C_ATTL3_GPU_CUDNNERROR_H_ */
