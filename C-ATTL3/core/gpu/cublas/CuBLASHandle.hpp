/*
 * CuBLASHandle.hpp
 *
 *  Created on: 12 Apr 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_CUBLAS_CUBLASHANDLE_H_
#define C_ATTL3_CORE_GPU_CUBLAS_CUBLASHANDLE_H_

#include <cublas_v2.h>

#include "CuBLASError.hpp"

namespace cattle {
namespace gpu {

/**
 * A singleton cuBLAS handle class.
 */
class CuBLASHandle {
 public:
  inline CuBLASHandle(const CuBLASHandle&) = delete;
  inline ~CuBLASHandle() { cublasAssert(cublasDestroy(handle)); }
  inline CuBLASHandle& operator=(const CuBLASHandle&) = delete;
  inline operator const cublasHandle_t&() const { return handle; }
  /**
   * @return A reference to the only instance of the class.
   */
  inline static const CuBLASHandle& get_instance() {
    static CuBLASHandle instance;
    return instance;
  }

 private:
  cublasHandle_t handle;
  inline CuBLASHandle() : handle() { cublasAssert(cublasCreate(&handle)); }
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CUBLAS_CUBLASHANDLE_H_ */
