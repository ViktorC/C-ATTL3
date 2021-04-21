/*
 * ConstantGPUParameterInitialization.hpp
 *
 *  Created on: 12 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_GPU_CONSTANTGPUPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_GPU_CONSTANTGPUPARAMETERINITIALIZATION_H_

#include "core/gpu/GPUParameterInitialization.hpp"
#include "core/gpu/cudnn/CuDNNTensor.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar>
class ConstantGPUParameterInitialization : public GPUParameterInitialization<Scalar> {
 public:
  /**
   * @param constant The value to which all elements of the parameter matrix are
   * to be initialized.
   */
  ConstantGPUParameterInitialization(Scalar constant) : constant(constant) {}
  inline void apply(CuBLASMatrix<Scalar>& params) const {
    CuDNNTensor<Scalar> cudnn_params(params.data(), params.cols(), params.rows(), 1u, 1u);
    cudnn_params.set_values(constant);
  }

 private:
  Scalar constant;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_GPU_CONSTANTGPUPARAMETERINITIALIZATION_H_ \
        */
