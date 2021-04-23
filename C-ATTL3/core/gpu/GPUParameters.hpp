/*
 * GPUParameters.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor Csommor
 */

#ifndef C_ATTL3_CORE_GPU_GPUPARAMETERS_H_
#define C_ATTL3_CORE_GPU_GPUPARAMETERS_H_

#include "core/Parameters.hpp"
#include "cublas/CuBLASMatrix.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar>
class GPUParameters : public virtual Parameters<Scalar> {
 public:
  virtual ~GPUParameters() = default;
  virtual GPUParameters<Scalar>* gpu_clone() const = 0;
  virtual const CuBLASMatrix<Scalar>& get_gpu_values() const = 0;
  virtual void set_gpu_values(CuBLASMatrix<Scalar> values) = 0;
  virtual const CuBLASMatrix<Scalar>& get_gpu_grad() const = 0;
  virtual void accumulate_gpu_grad(const CuBLASMatrix<Scalar>& grad) = 0;
  inline Parameters<Scalar>* clone() const { return gpu_clone(); }
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPUPARAMETERS_H_ */
