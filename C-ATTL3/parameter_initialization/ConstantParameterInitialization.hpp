/*
 * ConstantParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_CONSTANTPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_CONSTANTPARAMETERINITIALIZATION_H_

#include "core/ParameterInitialization.hpp"

namespace cattle {

/**
 * A class template for a parameter initialization that sets all values to a constant.
 */
template<typename Scalar>
class ConstantParameterInitialization : public ParameterInitialization<Scalar> {
public:
	/**
	 * @param constant The value to which all elements of the parameter matrix are to be
	 * initialized.
	 */
	ConstantParameterInitialization(Scalar constant) :
			constant(constant) { }
	inline void apply(Matrix<Scalar>& params) const {
		params.setConstant(params.rows(), params.cols(), constant);
	}
private:
	Scalar constant;
};

}

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_CONSTANTPARAMETERINITIALIZATION_H_ */
