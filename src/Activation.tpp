/*
 * Activation.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include <Activation.h>
#include <Eigen/Dense>
#include <cmath>
#include <Matrix.h>
#include <stdexcept>
#include <Vector.h>
#include <iostream>

namespace cppnn {

// TODO Address numeric stability issues.

template<typename Scalar>
Activation<Scalar>::~Activation() { };

template<typename Scalar>
IdentityActivation<Scalar>::~IdentityActivation() { };
template<typename Scalar>
Vector<Scalar> IdentityActivation<Scalar>::function(Vector<Scalar> x) {
	return x;
};
template<typename Scalar>
Vector<Scalar> IdentityActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) {
	return Vector<Scalar>::Ones(x.cols());
};

template<typename Scalar>
BinaryStepActivation<Scalar>::~BinaryStepActivation() { };
template<typename Scalar>
Vector<Scalar> BinaryStepActivation<Scalar>::function(Vector<Scalar> x) {
	for (unsigned i = 0; i < x.cols(); i++) {
		x(i) = x(i) > .0;
	}
	return x;
};
template<typename Scalar>
Vector<Scalar> BinaryStepActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) {
	return Vector<Scalar>::Zero(x.cols());
};

template<typename Scalar>
SigmoidActivation<Scalar>::~SigmoidActivation() { };
template<typename Scalar>
Vector<Scalar> SigmoidActivation<Scalar>::function(Vector<Scalar> x) {
	Vector<Scalar> exp = (-x).array().exp();
	return (Vector<Scalar>::Ones(x.cols()) + exp).cwiseInverse();
};
template<typename Scalar>
Vector<Scalar> SigmoidActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) {
	return y.cwiseProduct(Vector<Scalar>::Ones(y.cols()) - y);
};

template<typename Scalar>
SoftmaxActivation<Scalar>::~SoftmaxActivation() { };
template<typename Scalar>
Vector<Scalar> SoftmaxActivation<Scalar>::function(Vector<Scalar> x) {
	Vector<Scalar> out = x.array().exp();
	return out / out.sum();
};
template<typename Scalar>
Vector<Scalar> SoftmaxActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) {
	Matrix<Scalar> jacobian(y.cols(), y.cols());
	for (unsigned i = 0; i < jacobian.rows(); i++) {
		for (unsigned j = 0; j < jacobian.cols(); j++) {
			jacobian(i,j) = y(j) * ((i == j) - y(i));
		}
	}
	Vector<Scalar> out(y.cols());
	for (unsigned i = 0; i < out.cols(); i++) {
		out(i) = jacobian.row(i).sum();
	}
	return out;
};

template<typename Scalar>
TanhActivation<Scalar>::~TanhActivation() { };
template<typename Scalar>
Vector<Scalar> TanhActivation<Scalar>::function(Vector<Scalar> x) {
	return x.array().tanh();
};
template<typename Scalar>
Vector<Scalar> TanhActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) {
	return Vector<Scalar>::Ones(y.cols()) - y.cwiseProduct(y);
};

template<typename Scalar>
ReLUActivation<Scalar>::~ReLUActivation() { };
template<typename Scalar>
Vector<Scalar> ReLUActivation<Scalar>::function(Vector<Scalar> x) {
	return x.cwiseMax(.0);
};
template<typename Scalar>
Vector<Scalar> ReLUActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) {
	Vector<Scalar> out(x);
	for (unsigned i = 0; i < out.cols(); i++) {
		out(0,i) = out(0,i) > .0;
	}
	return out;
};

template<typename Scalar>
LeakyReLUActivation<Scalar>::LeakyReLUActivation(Scalar alpha) :
		alpha(alpha) {
	if (alpha >= 1)
		throw std::invalid_argument("alpha must be less than 1.");
};
template<typename Scalar>
LeakyReLUActivation<Scalar>::~LeakyReLUActivation() { };
template<typename Scalar>
Vector<Scalar> LeakyReLUActivation<Scalar>::function(Vector<Scalar> x) {
	return x.cwiseMax(x * alpha);
};

}
