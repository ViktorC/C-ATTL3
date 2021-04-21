/*
 * GlorotGPUParameterInitialization.hpp
 *
 *  Created on: 12 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_GPU_GLOROTGPUPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_GPU_GLOROTGPUPARAMETERINITIALIZATION_H_

#include <cmath>

#include "parameter_initialization/gpu/GaussianGPUParameterInitialization.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar>
class GlorotGPUParameterInitialization : public GaussianGPUParameterInitialization<Scalar> {
 public:
  /**
   * @param sd_scaling_factor The standard deviation scaling factor.
   */
  GlorotGPUParameterInitialization(Scalar sd_scaling_factor = 1)
      : GaussianGPUParameterInitialization<Scalar>(0, sd_scaling_factor) {}

 protected:
  inline Scalar _sd(unsigned fan_ins, unsigned fan_outs) const { return sqrt(2 / (Scalar)(fan_ins + fan_outs)); }
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_GPU_GLOROTGPUPARAMETERINITIALIZATION_H_ \
        */
