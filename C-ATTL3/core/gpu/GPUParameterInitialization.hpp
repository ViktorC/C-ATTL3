/*
 * GPUParameterInitialization.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor Csommor
 */

#ifndef C_ATTL3_CORE_GPU_GPUPARAMETERINITIALIZATION_H_
#define C_ATTL3_CORE_GPU_GPUPARAMETERINITIALIZATION_H_

#include "core/ParameterInitialization.hpp"
#include "cublas/CuBLASMatrix.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar>
class GPUParameterInitialization : public virtual ParameterInitialization<Scalar> {
public:
	virtual void apply(CuBLASMatrix<Scalar>& params) const = 0;
	inline void apply(Matrix<Scalar>& params) const {
		CuBLASMatrix<Scalar> gpu_params = params;
		apply(gpu_params);
		gpu_params.copy_to_host(params.data());
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPUPARAMETERINITIALIZATION_H_ */
