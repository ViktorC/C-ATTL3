/*
 * Regularization.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef REGULARIZATION_H_
#define REGULARIZATION_H_

#include <Matrix.h>
#include <Vector.h>

namespace cppnn {

template<typename Scalar>
class Regularization {
public:
	virtual ~Regularization() = default;
	virtual ColVector<Scalar> function(const Matrix<Scalar>& weights) const;
	virtual ColVector<Scalar> d_function(const Matrix<Scalar>& weights) const;
};

template<typename Scalar>
class L1Regularization : public Regularization<Scalar> {
public:
	L1Regularization(Scalar lambda = 1e-2) :
		lambda(lambda) { };
	ColVector<Scalar> function(const Matrix<Scalar>& weights) const {
		return (lambda * weights.cwiseAbs()).rowwise().sum();
	};
	ColVector<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return x.unaryExpr([this](Scalar i) { return greater_or_equal(i, .0) ? lambda : -lambda; });
	};
private:
	Scalar lambda;
};

template<typename Scalar>
class L2Regularization : public Regularization<Scalar> {
public:
	L2Regularization(Scalar lambda = 1e-2) :
		lambda(lambda) { };
	ColVector<Scalar> function(const Matrix<Scalar>& weights) const {
		return (.5 * lambda * weights.array().square()).matrix().rowwise().sum();
	};
	ColVector<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return lambda * weights;
	};
private:
	Scalar lambda;
};

template<typename Scalar>
class ElasticNetRegularization : public Regularization<Scalar> {
public:
	ElasticNetRegularization(Scalar l1_lambda = 1e-2, Scalar l2_lambda = 1e-2) :
		l1_lambda(l1_lambda),
		l2_lambda(l2_lambda) { };
	ColVector<Scalar> function(const Matrix<Scalar>& weights) const {
		return (l1_lambda * weights.array().abs() + .5 * l2_lambda * weights.array().square())
				.matrix().rowwise().sum();
	};
	ColVector<Scalar> d_function(const Matrix<Scalar>& weights) const {
		return x.unaryExpr([this](Scalar i) { return greater_or_equal(i, .0) ? l1_lambda : -l1_lambda; }) +
				l2_lambda * weights;
	};
private:
	Scalar l1_lambda;
	Scalar l2_lambda;
};

} /* namespace cppnn */

#endif /* REGULARIZATION_H_ */
