/*
 * Regularization.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef PARAMETERREGULARIZATION_H_
#define PARAMETERREGULARIZATION_H_

#include <memory>
#include <type_traits>
#include "Eigen.h"

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
 */
template<typename Scalar>
class L1ParameterRegularization : public ParamaterRegularization<Scalar> {
public:
	/**
	 * @param lambda The constant by which the penalty is to be scaled.
	 */
	inline L1ParameterRegularization(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return lambda * params.cwiseAbs().sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return lambda * params.array() / params.array().abs();
		return params.unaryExpr([this](Scalar i) { return i >= 0 ? lambda : -lambda; });
	}
private:
	const Scalar lambda;
};

/**
 * A class template for an L2 (second-norm) regularization penalty.
 */
template<typename Scalar>
class L2ParameterRegularization : public ParamaterRegularization<Scalar> {
public:
	/**
	 * @param lambda The constant by which the penalty is to be scaled.
	 */
	inline L2ParameterRegularization(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return (Scalar) .5 * lambda * params.array().square().mean();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return (lambda / params.size()) * params;
	}
private:
	const Scalar lambda;
};

/**
 * A class template for the elastic net regularization penalty which is a combination of
 * the L1 and L2 regularization penalties.
 */
template<typename Scalar>
class ElasticNetParameterRegularization : public ParamaterRegularization<Scalar> {
public:
	/**
	 * @param l1_lambda The constant by which the L1 penalty is to be scaled.
	 * @param l2_lambda The constant by which the L2 penalty is to be scaled.
	 */
	inline ElasticNetParameterRegularization(Scalar l1_lambda = 1e-2, Scalar l2_lambda = 1e-2) :
			l1_lambda(l1_lambda),
			l2_lambda(l2_lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return l1_lambda * params.array().abs().sum() +
				(Scalar) .5 * l2_lambda * params.array().square().sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return params.unaryExpr([this](Scalar i) { return i >= 0 ? l1_lambda : -l1_lambda; }) +
				l2_lambda * params;
	}
private:
	const Scalar l1_lambda;
	const Scalar l2_lambda;
};

} /* namespace cattle */

#endif /* PARAMETERREGULARIZATION_H_ */
