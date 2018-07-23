/*
 * SquaredParameterRegularization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_REGULARIZATION_SQUAREDPARAMETERREGULARIZATION_H_
#define C_ATTL3_PARAMETER_REGULARIZATION_SQUAREDPARAMETERREGULARIZATION_H_

#include "core/ParameterRegularization.hpp"

namespace cattle {

/**
 * A class template for an L2 (second-norm) regularization penalty.
 *
 * \f$P = \sum\limits_{i = 1}^n w_i^2\f$
 */
template<typename Scalar>
class SquaredParameterRegularization : public ParameterRegularization<Scalar> {
public:
	/**
	 * @param lambda The constant by which the penalty is to be scaled.
	 */
	inline SquaredParameterRegularization(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return (Scalar) .5 * lambda * params.array().square().sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return lambda * params;
	}
private:
	const Scalar lambda;
};

}

#endif /* C_ATTL3_PARAMETER_REGULARIZATION_SQUAREDPARAMETERREGULARIZATION_H_ */
