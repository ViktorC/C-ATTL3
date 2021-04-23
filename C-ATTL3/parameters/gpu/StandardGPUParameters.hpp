/*
 * StandardGPUParameters.hpp
 *
 *  Created on: 16 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETERS_GPU_STANDARDGPUPARAMETERS_H_
#define C_ATTL3_PARAMETERS_GPU_STANDARDGPUPARAMETERS_H_

#include <cassert>
#include <memory>

#include "core/gpu/GPUParameterInitialization.hpp"
#include "core/gpu/GPUParameterRegularization.hpp"
#include "core/gpu/GPUParameters.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar>
using GPUParamInitSharedPtr = std::shared_ptr<GPUParameterInitialization<Scalar>>;

template <typename Scalar>
using GPUParamRegSharedPtr = std::shared_ptr<GPUParameterRegularization<Scalar>>;

template <typename Scalar>
class StandardGPUParameters : public GPUParameters<Scalar> {
 public:
  inline StandardGPUParameters(std::size_t rows, std::size_t cols, bool optimizable = true,
                               GPUParamInitSharedPtr<Scalar> init = nullptr, GPUParamRegSharedPtr<Scalar> reg = nullptr)
      : rows(rows), cols(cols), optimizable(optimizable), param_init(init), param_reg(reg), frozen(false) {}
  inline GPUParameters<Scalar>* gpu_clone() const { return new StandardGPUParameters<Scalar>(*this); }
  inline const CuBLASMatrix<Scalar>& get_gpu_values() const { return values; }
  inline void set_gpu_values(CuBLASMatrix<Scalar> values) {
    assert(values.rows() == rows && values.cols() == cols);
    this->values = std::move(values);
  }
  inline const CuBLASMatrix<Scalar>& get_gpu_grad() const { return grad; }
  inline void accumulate_gpu_grad(const CuBLASMatrix<Scalar>& grad) {
    if (!optimizable) return;
    assert(grad.rows() == rows && grad.cols() == cols);
    this->grad += grad;
  }
  inline bool are_optimizable() const { return optimizable; }
  inline std::size_t get_rows() const { return rows; }
  inline std::size_t get_cols() const { return cols; }
  inline void init_values() {
    values = CuBLASMatrix<Scalar>(rows, cols);
    if (param_init) param_init->apply(values);
    cpu_values = Matrix<Scalar>(rows, cols);
    sync_values_to_host();
  }
  inline void init_grad() {
    if (optimizable) {
      grad = CuBLASMatrix<Scalar>(rows, cols);
      cpu_grad = Matrix<Scalar>(rows, cols);
      reset_grad();
    }
  }
  inline const Matrix<Scalar>& get_values() const { return cpu_values; }
  inline void set_values(Matrix<Scalar> values) {
    assert(values.rows() == rows && values.cols() == cols);
    cpu_values = std::move(values);
    this->values = cpu_values;
  }
  inline const Matrix<Scalar>& get_grad() const { return cpu_grad; }
  inline void accumulate_grad(const Matrix<Scalar>& grad) {
    if (!optimizable) return;
    assert(grad.rows() == rows && grad.cols() == cols);
    cpu_grad += grad;
    this->grad = cpu_grad;
  }
  inline void reset_grad() {
    grad.set_values(0);
    cpu_grad.setZero();
  }
  inline Scalar get_regularization_penalty() const {
    if (optimizable && param_reg) return param_reg->function(values);
    return 0;
  }
  inline void regularize() {
    if (optimizable && param_reg) {
      CuBLASMatrix<Scalar> reg_grad = param_reg->d_function(values);
      accumulate_gpu_grad(reg_grad);
    }
  }
  inline bool are_frozen() const { return frozen; }
  inline void set_frozen(bool frozen) { this->frozen = frozen; }
  inline void sync_values_to_host() { values.copy_to_host(cpu_values); }

 protected:
  const std::size_t rows, cols;
  const bool optimizable;
  const GPUParamInitSharedPtr<Scalar> param_init;
  const GPUParamRegSharedPtr<Scalar> param_reg;
  CuBLASMatrix<Scalar> values, grad;
  Matrix<Scalar> cpu_values, cpu_grad;
  bool frozen;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_PARAMETERS_GPU_STANDARDGPUPARAMETERS_H_ */
