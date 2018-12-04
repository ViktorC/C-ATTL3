/*
 * ZeroGPUParameterInitialization.hpp
 *
 *  Created on: 12 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_GPU_ZEROGPUPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_GPU_ZEROGPUPARAMETERINITIALIZATION_H_

#include "core/gpu/GPUParameterInitialization.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar>
class ZeroGPUParameterInitialization : public GPUParameterInitialization<Scalar> {
public:
	inline void apply(CuBLASMatrix<Scalar>& params) const {
		params.set_values(0);
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_GPU_ZEROGPUPARAMETERINITIALIZATION_H_ */
