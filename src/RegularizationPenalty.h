/*
 * Regularization.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef REGULARIZATIONPENALTY_H_
#define REGULARIZATIONPENALTY_H_

#include <type_traits>
#include "Utils.h"

namespace cattle {

/**
 * An abstract template class for different regularization penalties for neural network
 * layer parameters. Implementations of this class should be stateless.
 */
template<typename Scalar>
class RegularizationPenalty {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~RegularizationPenalty() = default;
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
class NoRegularizationPenalty : public RegularizationPenalty<Scalar> {
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
class L1RegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	/**
	 * @param lambda The constant by which the penalty is to be scaled.
	 */
	inline L1RegularizationPenalty(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return lambda * params.cwiseAbs().sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return params.unaryExpr([this](Scalar i) { return (Scalar) (i >= .0 ? lambda : -lambda); });
	}
private:
	const Scalar lambda;
};

/**
 * A class template for an L2 (second-norm) regularization penalty.
 */
template<typename Scalar>
class L2RegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	/**
	 * @param lambda The constant by which the penalty is to be scaled.
	 */
	inline L2RegularizationPenalty(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return .5 * lambda * params.array().square().sum();
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
 */
template<typename Scalar>
class ElasticNetRegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	/**
	 * @param l1_lambda The constant by which the L1 penalty is to be scaled.
	 * @param l2_lambda The constant by which the L2 penalty is to be scaled.
	 */
	inline ElasticNetRegularizationPenalty(Scalar l1_lambda = 1e-2, Scalar l2_lambda = 1e-2) :
			l1_lambda(l1_lambda),
			l2_lambda(l2_lambda) { }
	inline Scalar function(const Matrix<Scalar>& params) const {
		return l1_lambda * params.array().abs().sum() + .5 * l2_lambda * params.array().square().sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
		return params.unaryExpr([this](Scalar i) { return (Scalar) (i >= .0 ? l1_lambda :
				-l1_lambda); }) + l2_lambda * params;
	}
private:
	const Scalar l1_lambda;
	const Scalar l2_lambda;
};

} /* namespace cattle */

#endif /* REGULARIZATIONPENALTY_H_ */
