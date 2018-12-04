/*
 * OneGPUParameterInitialization.hpp
 *
 *  Created on: 12 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_GPU_ONEGPUPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_GPU_ONEGPUPARAMETERINITIALIZATION_H_

#include "core/gpu/GPUParameterInitialization.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar>
class OneGPUParameterInitialization : public GPUParameterInitialization<Scalar> {
public:
	inline void apply(CuBLASMatrix<Scalar>& params) const {
		params.set_values(1);
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_GPU_ONEGPUPARAMETERINITIALIZATION_H_ */
