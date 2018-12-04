/*
 * LeCunParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_LECUNPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_LECUNPARAMETERINITIALIZATION_H_

#include <cmath>

#include "GaussianParameterInitialization.hpp"

namespace cattle {

/**
 * An abstract class representing the LeCun weight initialization method.
 *
 * \f$\sigma = c \sqrt{\frac{1}{fan_{in}}}\f$
 *
 * \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
template<typename Scalar>
class LeCunParameterInitialization : public GaussianParameterInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 */
	inline LeCunParameterInitialization(Scalar sd_scaling_factor = 1) :
			GaussianParameterInitialization<Scalar>(0, sd_scaling_factor) { }
protected:
	inline Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(1 / (Scalar) fan_ins);
	}
};

}

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_LECUNPARAMETERINITIALIZATION_H_ */
