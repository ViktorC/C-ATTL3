/*
 * L1ParameterRegularization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_REGULARIZATION_L1PARAMETERREGULARIZATION_H_
#define C_ATTL3_PARAMETER_REGULARIZATION_L1PARAMETERREGULARIZATION_H_

#include "core/ParameterRegularization.hpp"

namespace cattle {

/**
 * A class template for an L1 (first-norm) regularization penalty.
 *
 * \f$P = \lambda \sum\limits_{i = 1}^n \left|w_i\right|\f$
 */
template<typename Scalar>
class L1ParameterRegularization : public ParameterRegularization<Scalar> {
public:
	/**
	 * @param lambda The constant by which the penalty is to be scaled.
	 */
	inline L1ParameterRegularization(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return params.cwiseAbs().sum() * lambda;
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return params.unaryExpr([this](Scalar e) { return e >= 0 ? lambda : -lambda; });
	}
private:
	const Scalar lambda;
};

}

#endif /* C_ATTL3_PARAMETER_REGULARIZATION_L1PARAMETERREGULARIZATION_H_ */
