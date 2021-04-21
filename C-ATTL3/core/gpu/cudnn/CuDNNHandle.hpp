/*
 * CuDNNHandle.hpp
 *
 *  Created on: 28 May 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_CUDNN_CUDNNHANDLE_H_
#define C_ATTL3_CORE_GPU_CUDNN_CUDNNHANDLE_H_

#include <cudnn.h>

#include "CuDNNError.hpp"

namespace cattle {
namespace gpu {

/**
 * A singleton utility class representing a handle to the cuDNN library.
 */
class CuDNNHandle {
 public:
  CuDNNHandle(const CuDNNHandle&) = delete;
  ~CuDNNHandle() { cudnnAssert(cudnnDestroy(handle)); }
  CuDNNHandle& operator=(const CuDNNHandle&) = delete;
  inline operator const cudnnHandle_t&() const { return handle; }
  /**
   * @return A reference to the only instance of the class.
   */
  inline static const CuDNNHandle& get_instance() {
    static CuDNNHandle instance;
    return instance;
  }

 private:
  inline CuDNNHandle() : handle() { cudnnAssert(cudnnCreate(&handle)); }
  cudnnHandle_t handle;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CUDNN_CUDNNHANDLE_H_ */
