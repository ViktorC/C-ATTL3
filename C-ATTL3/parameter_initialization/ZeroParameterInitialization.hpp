/*
 * ZeroParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_ZEROPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_ZEROPARAMETERINITIALIZATION_H_

#include "core/ParameterInitialization.hpp"

namespace cattle {

/**
 * A class template for a weight initialization that sets all values to 0.
 */
template<typename Scalar>
class ZeroParameterInitialization : public ParameterInitialization<Scalar> {
public:
	inline void apply(Matrix<Scalar>& weights) const {
		weights.setZero(weights.rows(), weights.cols());
	}
};

}

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_ZEROPARAMETERINITIALIZATION_H_ */
