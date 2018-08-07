/*
 * GPUParameterInitialization.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor Csommor
 */

#ifndef C_ATTL3_CORE_GPU_GPUPARAMETERINITIALIZATION_H_
#define C_ATTL3_CORE_GPU_GPUPARAMETERINITIALIZATION_H_

#include "core/ParameterInitialization.hpp"
#include "TensorConversion.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar>
class GPUParameterInitialization : public virtual ParameterInitialization<Scalar> {
public:
	virtual void apply(CuDNNTensor<Scalar>& params) const = 0;
	inline void apply(Matrix<Scalar>& params) const {
		TensorMap<Scalar,4> params_tensor(params.data(), params.rows(), params.cols(), 1, 1);
		auto params_gpu_tensor = TensorConversion<Scalar>::convert_from_eigen_to_cudnn(params_tensor);
		apply(params_gpu_tensor);
		params_tensor = TensorConversion<Scalar>::convert_from_cudnn_to_eigen(params_gpu_tensor);
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPUPARAMETERINITIALIZATION_H_ */
