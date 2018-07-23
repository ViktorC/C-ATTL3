/*
 * AbsoluteParameterRegularization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_REGULARIZATION_ABSOLUTEPARAMETERREGULARIZATION_H_
#define C_ATTL3_PARAMETER_REGULARIZATION_ABSOLUTEPARAMETERREGULARIZATION_H_

#include "core/ParameterRegularization.hpp"

namespace cattle {

/**
 * A class template for an L1 (first-norm) regularization penalty.
 *
 * \f$P = \sum\limits_{i = 1}^n \left|w_i\right|\f$
 */
template<typename Scalar>
class AbsoluteParameterRegularization : public ParameterRegularization<Scalar> {
public:
	/**
	 * @param lambda The constant by which the penalty is to be scaled.
	 */
	inline AbsoluteParameterRegularization(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return lambda * params.cwiseAbs().sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return params.unaryExpr([lambda](Scalar m) { return m >= 0 ? lambda : -lambda; });
	}
private:
	const Scalar lambda;
};

}

#endif /* C_ATTL3_PARAMETER_REGULARIZATION_ABSOLUTEPARAMETERREGULARIZATION_H_ */
