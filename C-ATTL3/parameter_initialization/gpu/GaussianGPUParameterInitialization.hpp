/*
 * GaussianGPUParameterInitialization.hpp
 *
 *  Created on: 12 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_GPU_GAUSSIANGPUPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_GPU_GAUSSIANGPUPARAMETERINITIALIZATION_H_

#include <cassert>

#include "core/gpu/GPUParameterInitialization.hpp"
#include "core/gpu/curand/CuRANDGenerator.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar>
class GaussianGPUParameterInitialization : public GPUParameterInitialization<Scalar> {
 public:
  /**
   * @param mean The mean of the distribution.
   * @param sd_scaling_factor The standard deviation scaling factor.
   */
  GaussianGPUParameterInitialization(Scalar mean = 0, Scalar sd_scaling_factor = 1)
      : mean(mean), sd_scaling_factor(sd_scaling_factor) {
    assert(sd_scaling_factor > 0);
  }
  inline void apply(CuBLASMatrix<Scalar>& params) const {
    CuRANDGenerator<Scalar> gen;
    gen.generate_normal(0, sd_scaling_factor * _sd(params.rows(), params.cols()), params);
  }

 protected:
  /**
   * It computes the standard deviation of the distribution to sample from.
   *
   * @param fan_ins The input size of the kernel (the number of rows in the
   * weight matrix excluding the bias row).
   * @param fan_outs The output size of the kernel (the number of columns in the
   * weight matrix).
   * @return The standard deviation of the normal distribution from which the
   * values of the initialized weight matrix are to be sampled.
   */
  inline virtual Scalar _sd(unsigned fan_ins, unsigned fan_outs) const { return 1; }

 private:
  const Scalar mean, sd_scaling_factor;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_GPU_GAUSSIANGPUPARAMETERINITIALIZATION_H_ \
        */
