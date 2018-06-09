/*
 * ParameterRegularization.hpp
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_PARAMETERREGULARIZATION_H_
#define CATTL3_PARAMETERREGULARIZATION_H_

#include <memory>
#include <type_traits>

#include "utils/EigenProxy.hpp"

namespace cattle {

/**
 * An abstract template class for different regularization penalties for neural network
 * layer parameters. Implementations of this class should be stateless.
 */
template<typename Scalar>
class ParamaterRegularization {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~ParamaterRegularization() = default;
	/**
	 * It computes the regularization penalty for the given parameter values.
	 *
	 * @param params A constant reference to the parameter matrix.
	 * @return The regularization penalty as a single scalar.
	 */
	virtual Scalar function(const Matrix<Scalar>& params) const;
	/**
	 * It differentiates the regularization function and returns its derivative
	 * w.r.t. the parameters.
	 *
	 * @param params A constant reference to the parameter matrix.
	 * @return The gradient matrix.
	 */
	virtual Matrix<Scalar> d_function(const Matrix<Scalar>& params) const;
};

/**
 * A class template for a no-operation regularization penalty.
 */
template<typename Scalar>
class NoParameterRegularization : public ParamaterRegularization<Scalar> {
	typedef NoParameterRegularization<Scalar> Self;
public:
	inline Scalar function(const Matrix<Scalar>& params) const {
		return 0;
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return Matrix<Scalar>::Zero(params.rows(), params.cols());
	}
};

/**
 * A class template for an L1 (first-norm) regularization penalty.
 *
 * \f$P = \sum\limits_{i = 1}^n \left|w_i\right|\f$
 */
template<typename Scalar>
class AbsoluteParameterRegularization : public ParamaterRegularization<Scalar> {
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
		return params.unaryExpr([this](Scalar i) { return i >= 0 ? lambda : -lambda; });
	}
private:
	const Scalar lambda;
};

/**
 * A class template for an L2 (second-norm) regularization penalty.
 *
 * \f$P = \sum\limits_{i = 1}^n w_i^2\f$
 */
template<typename Scalar>
class SquaredParameterRegularization : public ParamaterRegularization<Scalar> {
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

/**
 * A class template for the elastic net regularization penalty which is a combination of
 * the L1 and L2 regularization penalties.
 *
 * \f$P = \sum\limits_{i = 1}^n \left|w_i\right| + w_i^2\f$
 *
 * \see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696
 */
template<typename Scalar>
class ElasticNetParameterRegularization : public ParamaterRegularization<Scalar> {
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
		return params.unaryExpr([this](Scalar i) { return i >= 0 ? abs_lambda : -abs_lambda; }) +
				sqrd_lambda * params;
	}
private:
	const Scalar abs_lambda;
	const Scalar sqrd_lambda;
};

} /* namespace cattle */

#endif /* CATTL3_PARAMETERREGULARIZATION_H_ */
