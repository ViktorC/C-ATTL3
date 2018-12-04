/*
 * GPUParameterRegularization.hpp
 *
 *  Created on: 6 Aug 2018
 *      Author: Viktor Csommor
 */

#ifndef C_ATTL3_CORE_GPU_GPUPARAMETERREGULARIZATION_H_
#define C_ATTL3_CORE_GPU_GPUPARAMETERREGULARIZATION_H_

#include "core/ParameterRegularization.hpp"
#include "cublas/CuBLASMatrix.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar>
class GPUParameterRegularization : public virtual ParameterRegularization<Scalar> {
public:
	virtual Scalar function(const CuBLASMatrix<Scalar>& params) const;
	virtual CuBLASMatrix<Scalar> d_function(const CuBLASMatrix<Scalar>& params) const;
	inline Scalar function(const Matrix<Scalar>& params) const {
		return function(CuBLASMatrix<Scalar>(params));
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return d_function(CuBLASMatrix<Scalar>(params));
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPUPARAMETERREGULARIZATION_H_ */
