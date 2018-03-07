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

template<typename Scalar>
class RegularizationPenalty {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~RegularizationPenalty() = default;
	virtual Scalar function(const Matrix<Scalar>& weights) const;
	virtual Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const;
};

template<typename Scalar>
class NoRegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	inline Scalar function(const Matrix<Scalar>& weights) const {
		return 0;
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return Matrix<Scalar>::Zero(weights.rows(), weights.cols());
	}
};

template<typename Scalar>
class L1RegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	inline L1RegularizationPenalty(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& weights) const {
		return (lambda * weights.cwiseAbs()).sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return weights.unaryExpr([this](Scalar i) { return (Scalar) (i >= .0 ? lambda : -lambda); });
	}
private:
	const Scalar lambda;
};

template<typename Scalar>
class L2RegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	inline L2RegularizationPenalty(Scalar lambda = 1e-2) :
			lambda(lambda) { }
	inline Scalar function(const Matrix<Scalar>& weights) const {
		return (.5 * lambda * weights.array().square()).sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return lambda * weights;
	}
private:
	const Scalar lambda;
};

template<typename Scalar>
class ElasticNetRegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	inline ElasticNetRegularizationPenalty(Scalar l1_lambda = 1e-2, Scalar l2_lambda = 1e-2) :
			l1_lambda(l1_lambda),
			l2_lambda(l2_lambda) { }
	inline Scalar function(const Matrix<Scalar>& weights) const {
		return (l1_lambda * weights.array().abs() + .5 * l2_lambda * weights.array().square()).sum();
	}
	inline Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return weights.unaryExpr([this](Scalar i) { return (Scalar) (i >= .0 ? l1_lambda :
				-l1_lambda); }) + l2_lambda * weights;
	}
private:
	const Scalar l1_lambda;
	const Scalar l2_lambda;
};

} /* namespace cattle */

#endif /* REGULARIZATIONPENALTY_H_ */
