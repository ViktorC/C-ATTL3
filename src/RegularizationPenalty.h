/*
 * Regularization.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef REGULARIZATIONPENALTY_H_
#define REGULARIZATIONPENALTY_H_

#include <type_traits>
#include <Utils.h>

namespace cppnn {

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
	Scalar function(const Matrix<Scalar>& weights) const {
		return 0;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return Matrix<Scalar>::Zero(weights.rows(), weights.cols());
	};
};

template<typename Scalar>
class L1RegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	L1RegularizationPenalty(Scalar lambda = 1e-2) :
			lambda(lambda) { };
	Scalar function(const Matrix<Scalar>& weights) const {
		return (lambda * weights.cwiseAbs()).sum();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return weights.unaryExpr([this](Scalar i) { return (Scalar) (i >= .0 ? lambda : -lambda); });
	};
private:
	Scalar lambda;
};

template<typename Scalar>
class L2RegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	L2RegularizationPenalty(Scalar lambda = 1e-2) :
			lambda(lambda) { };
	Scalar function(const Matrix<Scalar>& weights) const {
		return (.5 * lambda * weights.array().square()).sum();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return lambda * weights;
	};
private:
	Scalar lambda;
};

template<typename Scalar>
class ElasticNetRegularizationPenalty : public RegularizationPenalty<Scalar> {
public:
	ElasticNetRegularizationPenalty(Scalar l1_lambda = 1e-2, Scalar l2_lambda = 1e-2) :
			l1_lambda(l1_lambda),
			l2_lambda(l2_lambda) { };
	Scalar function(const Matrix<Scalar>& weights) const {
		return (l1_lambda * weights.array().abs() + .5 * l2_lambda * weights.array().square()).sum();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return weights.unaryExpr([this](Scalar i) { return (Scalar) (i >= .0 ? l1_lambda :
				-l1_lambda); }) + l2_lambda * weights;
	};
private:
	Scalar l1_lambda;
	Scalar l2_lambda;
};

} /* namespace cppnn */

#endif /* REGULARIZATIONPENALTY_H_ */
