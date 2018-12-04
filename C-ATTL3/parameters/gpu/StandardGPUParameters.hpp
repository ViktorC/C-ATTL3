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

template<typename Scalar>
using GPUParamInitSharedPtr = std::shared_ptr<GPUParameterInitialization<Scalar>>;

template<typename Scalar>
using GPUParamRegSharedPtr = std::shared_ptr<GPUParameterRegularization<Scalar>>;

template<typename Scalar>
class StandardGPUParameters : GPUParameters<Scalar> {
public:
	inline StandardGPUParameters(std::size_t samples, std::size_t height, std::size_t width, std::size_t channels,
			bool optimizable = true, GPUParamInitSharedPtr<Scalar> init = nullptr,
			GPUParamRegSharedPtr<Scalar> reg = nullptr) :
				samples(samples),
				height(height),
				width(width),
				channels(channels),
				rows(height * width * channels),
				cols(samples),
				optimizable(optimizable),
				param_init(init),
				param_reg(reg),
				frozen(false),
				cpu_values_up_to_date(false),
				cpu_grad_up_to_date(false) {
		assert(samples > 0 && height > 0 && width > 0 && channels > 0);
	}
	inline GPUParameters<Scalar>* gpu_clone() const {
		return new StandardGPUParameters<Scalar>(*this);
	}
	inline const CuDNNTensor<Scalar>& get_gpu_values() const {
		return tensor_values;
	}
	inline void set_gpu_tensorvalues(CuDNNTensor<Scalar> values) {
		assert(values.samples() == samples && values.height() == height &&
				values.width() == width && values.channels() == channels);
		this->values = std::move(values);
		cpu_values_up_to_date = false;
	}
	inline const CuDNNTensor<Scalar>& get_gpu_grad() const {
		return grad;
	}
	inline void accumulate_gpu_grad(const CuDNNTensor<Scalar>& grad) {
		if (!optimizable)
			return;
		assert(grad.samples() == samples && grad.height() == height &&
				grad.width() == width && grad.channels() == channels);
		this->grad += grad;
		cpu_grad_up_to_date = false;
	}
	inline bool are_optimizable() const {
		return optimizable;
	}
	inline std::size_t get_rows() const {
		return rows;
	}
	inline std::size_t get_cols() const {
		return cols;
	}
	inline void init_values() {
		values = CuDNNTensor<Scalar>(samples, height, width, channels);
		if (param_init)
			param_init->apply(CuBLASMatrix<Scalar>(values.data(), rows, cols));
		cpu_values_up_to_date = false;
	}
	inline void init_grad() {
		if (optimizable) {
			grad = CuDNNTensor<Scalar>(samples, height, width, channels);
			reset_grad();
		}
	}
	inline const Matrix<Scalar>& get_values() const {
		if (!cpu_values_up_to_date) {
			Tensor<Scalar,4> cpu_tensor_values = values;
			cpu_values = MatrixMap<Scalar>(cpu_tensor_values.data(), rows, cols);
		}
		return cpu_values;
	}
	inline void set_values(Matrix<Scalar> values) {
		assert(values.rows() == rows && values.cols() == cols);
		cpu_values = std::move(values);
		this->values = TensorMap<Scalar,4>(cpu_values.data(), samples, height, width, channels);
	}
	inline const Matrix<Scalar>& get_grad() const {
		if (!cpu_grad_up_to_date) {
			Tensor<Scalar,4> cpu_tensor_grad = grad;
			cpu_grad = MatrixMap<Scalar>(cpu_tensor_grad.data(), rows, cols);
		}
		return cpu_grad;
	}
	inline void accumulate_grad(const Matrix<Scalar>& grad) {
		if (!optimizable)
			return;
		assert(grad.rows() == rows && grad.cols() == cols);
		cpu_grad += grad;
		this->grad = TensorMap<Scalar,4>(cpu_grad.data(), samples, height, width, channels);
	}
	inline void reset_grad() {
		grad.set_values(0);
		cpu_grad_up_to_date = false;
	}
	inline Scalar get_regularization_penalty() const {
		if (optimizable && param_reg)
			return param_reg->function(CuBLASMatrix<Scalar>(values.data(), rows, cols));
		return 0;
	}
	inline void regularize() {
		if (optimizable && param_reg)
			accumulate_grad(param_reg->d_function(CuBLASMatrix<Scalar>(values.data(), rows, cols)));
	}
	inline bool are_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
protected:
	const std::size_t samples, height, width, channels;
	const std::size_t rows, cols;
	const bool optimizable;
	const GPUParamInitSharedPtr<Scalar> param_init;
	const GPUParamRegSharedPtr<Scalar> param_reg;
	CuDNNTensor<Scalar> values, grad;
	Matrix<Scalar> cpu_values, cpu_grad;
	bool frozen;
	bool cpu_values_up_to_date, cpu_grad_up_to_date;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_PARAMETERS_GPU_STANDARDGPUPARAMETERS_H_ */
