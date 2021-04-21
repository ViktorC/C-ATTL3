/*
 * L2GPUParameterRegularization.hpp
 *
 *  Created on: 15 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_REGULARIZATION_GPU_L2GPUPARAMETERREGULARIZATION_H_
#define C_ATTL3_PARAMETER_REGULARIZATION_GPU_L2GPUPARAMETERREGULARIZATION_H_

#include <cmath>

#include "core/gpu/GPUParameterRegularization.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar>
class L2GPUParameterRegularization : public GPUParameterRegularization<Scalar> {
 public:
  /**
   * @param lambda The constant by which the penalty is to be scaled.
   */
  inline L2GPUParameterRegularization(Scalar lambda = 1e-2) : lambda(lambda) {}
  inline Scalar function(const CuBLASMatrix<Scalar>& params) const {
    return pow(params.l2_norm(), (Scalar)2) * (Scalar).5 * lambda;
  }
  inline CuBLASMatrix<Scalar> d_function(const CuBLASMatrix<Scalar>& params) const { return params * lambda; }

 private:
  const Scalar lambda;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_PARAMETER_REGULARIZATION_GPU_L2GPUPARAMETERREGULARIZATION_H_ \
        */
