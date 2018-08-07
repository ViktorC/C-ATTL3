/*
 * GPUParameterRegularization.hpp
 *
 *  Created on: 6 Aug 2018
 *      Author: Viktor Csommor
 */

#ifndef C_ATTL3_CORE_GPU_GPUPARAMETERREGULARIZATION_H_
#define C_ATTL3_CORE_GPU_GPUPARAMETERREGULARIZATION_H_

#include "core/ParameterRegularization.hpp"
#include "TensorConversion.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar>
class GPUParameterRegularization : public virtual ParameterRegularization<Scalar> {
public:
	virtual Scalar function(const CuDNNTensor<Scalar>& params) const;
	virtual CuDNNTensor<Scalar> d_function(const CuDNNTensor<Scalar>& params) const;
	inline Scalar function(const Matrix<Scalar>& params) const {
		TensorMap<Scalar,4> params_tensor(params.data(), params.rows(), params.cols(), 1, 1);
		return function(TensorConversion<Scalar>::convert_from_eigen_to_cudnn(params_tensor));
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		TensorMap<Scalar,4> params_tensor(params.data(), params.rows(), params.cols(), 1, 1);
		auto params_gpu_tensor = TensorConversion<Scalar>::convert_from_eigen_to_cudnn(params_tensor);
		auto params_gpu_grad_tensor = TensorConversion<Scalar>::convert_from_cudnn_to_eigen(
				d_function(params_gpu_tensor));
		return MatrixMap<Scalar>(params_gpu_grad_tensor.data(), params.rows(), params.cols());
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPUPARAMETERREGULARIZATION_H_ */
