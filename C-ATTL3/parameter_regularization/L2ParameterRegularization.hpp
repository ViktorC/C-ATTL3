/*
 * L2ParameterRegularization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_REGULARIZATION_L2PARAMETERREGULARIZATION_H_
#define C_ATTL3_PARAMETER_REGULARIZATION_L2PARAMETERREGULARIZATION_H_

#include "core/ParameterRegularization.hpp"

namespace cattle {

/**
 * A class template for an L2 (second-norm) regularization penalty.
 *
 * \f$P = \frac{\lambda_2}{2} \sum\limits_{i = 1}^n w_i^2\f$
 */
template<typename Scalar>
class L2ParameterRegularization : public ParameterRegularization<Scalar> {
public:
	/**
	 * @param lambda The constant by which the penalty is to be scaled.
	 */
	inline L2ParameterRegularization(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return params.squaredNorm() * ((Scalar) .5 * lambda);
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return params * lambda;
	}
private:
	const Scalar lambda;
};

}

#endif /* C_ATTL3_PARAMETER_REGULARIZATION_L2PARAMETERREGULARIZATION_H_ */
