/*
 * ElasticNetParameterRegularization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_REGULARIZATION_ELASTICNETPARAMETERREGULARIZATION_H_
#define C_ATTL3_PARAMETER_REGULARIZATION_ELASTICNETPARAMETERREGULARIZATION_H_

#include "core/ParameterRegularization.hpp"

namespace cattle {

/**
 * A class template for the elastic net regularization penalty which is a combination of
 * the L1 and L2 regularization penalties.
 *
 * \f$P = \sum\limits_{i = 1}^n \left|w_i\right| + w_i^2\f$
 *
 * \see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696
 */
template<typename Scalar>
class ElasticNetParameterRegularization : public ParameterRegularization<Scalar> {
public:
	/**
	 * @param abs_lambda The constant by which the L1 penalty is to be scaled.
	 * @param sqrd_lambda The constant by which the L2 penalty is to be scaled.
	 */
	inline ElasticNetParameterRegularization(Scalar abs_lambda = 1e-2, Scalar sqrd_lambda = 1e-2) :
			abs_lambda(abs_lambda),
			sqrd_lambda(sqrd_lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return abs_lambda * params.array().abs().sum() +
				(Scalar) .5 * sqrd_lambda * params.array().square().sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return params.unaryExpr([this](Scalar e) { return e >= 0 ? abs_lambda : -abs_lambda; }) +
				sqrd_lambda * params;
	}
private:
	const Scalar abs_lambda;
	const Scalar sqrd_lambda;
};

}

#endif /* C_ATTL3_PARAMETER_REGULARIZATION_ELASTICNETPARAMETERREGULARIZATION_H_ */
