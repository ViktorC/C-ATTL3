/*
 * GlorotParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_GLOROTPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_GLOROTPARAMETERINITIALIZATION_H_

#include <cmath>

#include "GaussianParameterInitialization.hpp"

namespace cattle {

/**
 * An abstract class representing the Xavier/Glorot weight initialization method.
 *
 * \f$\sigma = c \sqrt{\frac{2}{fan_{in} + fan_{out}}}\f$
 *
 * \see http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
 */
template<typename Scalar>
class GlorotParameterInitialization : public GaussianParameterInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 */
	inline GlorotParameterInitialization(Scalar sd_scaling_factor = 1) :
		GaussianParameterInitialization<Scalar>::GaussianParameterInitialization(sd_scaling_factor) { }
protected:
	inline Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(2.0 / (Scalar) (fan_ins + fan_outs));
	}
};

}

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_GLOROTPARAMETERINITIALIZATION_H_ */
