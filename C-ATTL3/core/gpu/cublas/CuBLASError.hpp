/*
 * CuBLASError.hpp
 *
 *  Created on: 14.06.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_CUBLAS_CUBLASERROR_H_
#define C_ATTL3_CORE_GPU_CUBLAS_CUBLASERROR_H_

#include <cublas_v2.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#define cublasErrorCheck(status) { _cublas_error_check(status, __FILE__, __LINE__); }
#define cublasAssert(status) { _cublas_assert(status, __FILE__, __LINE__); }

namespace cattle {
namespace gpu {

/**
 * A class representing a cuBLAS runtime error.
 */
class CuBLASError : public std::runtime_error {
public:
	/**
	 * @param status The cuBLAS status code.
	 * @param file The name of the file in which the error occurred.
	 * @param line The number of the line at which the error occurred.
	 */
	CuBLASError(cublasStatus_t status, const char* file, int line) :
		std::runtime_error("cuBLAS Error: " + cublas_status_to_string(status) + "; File: " +
				std::string(file) + "; Line: " + std::to_string(line)) { }
private:
	/**
	 * It returns a string representation of the provided cuBLAS status code.
	 *
	 * @param status The cuBLAS status code.
	 * @return A string describing the status code.
	 */
	inline static std::string cublas_status_to_string(cublasStatus_t status) {
		switch (status) {
			case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
			case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
			case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
			case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
			case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
			case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
			case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
			case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
			case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
			case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
			default: return "<unknown>";
	    }
	}
};

namespace {

__inline__ void _cublas_error_check(cublasStatus_t status, const char* file, int line) {
	if (status != CUBLAS_STATUS_SUCCESS)
		throw CuBLASError(status, file, line);
}

__inline__ void _cublas_assert(cublasStatus_t status, const char* file, int line) {
	try {
		_cublas_error_check(status, file, line);
	} catch (const CuBLASError& e) {
		std::cout << e.what() << std::endl;
		exit(-1);
	}
}

}

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CUBLAS_CUBLASERROR_H_ */
