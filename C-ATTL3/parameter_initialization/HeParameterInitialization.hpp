/*
 * HeParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_HEPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_HEPARAMETERINITIALIZATION_H_

#include <cmath>

#include "GaussianParameterInitialization.hpp"

namespace cattle {

/**
 * An abstract class representing the He weight initialization method.
 *
 * \f$\sigma = c \sqrt{\frac{2}{fan_{in}}}\f$
 *
 * \see https://arxiv.org/abs/1502.01852
 */
template<typename Scalar>
class HeParameterInitialization : public GaussianParameterInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 */
	inline HeParameterInitialization(Scalar sd_scaling_factor = 1) :
			GaussianParameterInitialization<Scalar>(0, sd_scaling_factor) { }
protected:
	inline Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(2 / (Scalar) fan_ins);
	}
};

}

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_HEPARAMETERINITIALIZATION_H_ */
