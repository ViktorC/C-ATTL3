/*
 * CuRANDError.hpp
 *
 *  Created on: 05.08.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_CURAND_CURANDERROR_H_
#define C_ATTL3_CORE_GPU_CURAND_CURANDERROR_H_

#include <curand.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#define curandErrorCheck(status) \
  { _curand_error_check(status, __FILE__, __LINE__); }
#define curandAssert(status) \
  { _curand_assert(status, __FILE__, __LINE__); }

namespace cattle {
namespace gpu {

/**
 * A class representing a cuRAND runtime error.
 */
class CuRANDError : public std::runtime_error {
 public:
  /**
   * @param status The cuRAND status code.
   * @param file The name of the file in which the error occurred.
   * @param line The number of the line at which the error occurred.
   */
  CuRANDError(curandStatus_t status, const char* file, int line)
      : std::runtime_error("cuRAND Error: " + curand_status_to_string(status) + "; File: " + std::string(file) +
                           "; Line: " + std::to_string(line)) {}

 private:
  /**
   * It returns a string representation of the provided cuRAND status code.
   *
   * @param status The cuRAND status code.
   * @return A string describing the status code.
   */
  inline static std::string curand_status_to_string(curandStatus_t status) {
    switch (status) {
      case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";
      case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
      case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
      case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
      case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";
      case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
      case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
      case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
      case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
      case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
      case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
      case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
      case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
      default:
        return "<unknown>";
    }
  }
};

namespace {

__inline__ void _curand_error_check(curandStatus_t status, const char* file, int line) {
  if (status != CURAND_STATUS_SUCCESS) throw CuRANDError(status, file, line);
}

__inline__ void _curand_assert(curandStatus_t status, const char* file, int line) {
  try {
    _curand_error_check(status, file, line);
  } catch (const CuRANDError& e) {
    std::cout << e.what() << std::endl;
    exit(-1);
  }
}

}  // namespace

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CURAND_CURANDERROR_H_ */
