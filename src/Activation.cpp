/*
 * Activation.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include <Activation.h>
#include <Eigen/Dense>
#include <Matrix.h>
#include <stdexcept>
#include <Vector.h>

namespace cppnn {

// TODO Address numeric stability issues.

//template<typename Scalar>
//Activation<Scalar>::~Activation() { };

template<typename Scalar>
IdentityActivation<Scalar>::~IdentityActivation() { };
template<typename Scalar>
Vector<Scalar> IdentityActivation<Scalar>::function(Vector<Scalar> x) const {
	return x;
};
template<typename Scalar>
Vector<Scalar> IdentityActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) const {
	return Vector<Scalar>::Ones(x.cols());
};

template<typename Scalar>
BinaryStepActivation<Scalar>::~BinaryStepActivation() { };
template<typename Scalar>
Vector<Scalar> BinaryStepActivation<Scalar>::function(Vector<Scalar> x) const {
	for (unsigned i = 0; i < x.cols(); i++) {
		x(i) = x(i) > .0;
	}
	return x;
};
template<typename Scalar>
Vector<Scalar> BinaryStepActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) const {
	return Vector<Scalar>::Zero(x.cols());
};

template<typename Scalar>
SigmoidActivation<Scalar>::~SigmoidActivation() { };
template<typename Scalar>
Vector<Scalar> SigmoidActivation<Scalar>::function(Vector<Scalar> x) const {
	return 1/(1 + (x * -1).exp());
};
template<typename Scalar>
Vector<Scalar> SigmoidActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) const {
	Vector<Scalar> out(y);
	return out * (1 - out);
};

template<typename Scalar>
Vector<Scalar> SoftmaxActivation<Scalar>::function(Vector<Scalar> x) const {
	Vector<Scalar> out = x.exp();
	return out / out.sum();
	return x;
};
template<typename Scalar>
Vector<Scalar> SoftmaxActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) const {
	Matrix<Scalar> jacobian;
	jacobian = y.asDiagonal() - y.transpose().dot(y);
	Vector<Scalar> out(x.cols());
	for (unsigned i = 0; i < out.cols(); i++) {
		out(i) = jacobian.row(i).sum();
	}
	return out;
};

template<typename Scalar>
TanhActivation<Scalar>::~TanhActivation() { };
template<typename Scalar>
Vector<Scalar> TanhActivation<Scalar>::function(Vector<Scalar> x) const {
	return x.tanh();
};
template<typename Scalar>
Vector<Scalar> TanhActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) const {
	Vector<Scalar> out(y);
	return 1 - out * out;
};

template<typename Scalar>
ReLUActivation<Scalar>::~ReLUActivation() { };
template<typename Scalar>
Vector<Scalar> ReLUActivation<Scalar>::function(Vector<Scalar> x) const {
	for (unsigned i = 0; i < x.cols(); i++) {
		x(i) = std::max(.0, x(i));
	}
	return x;
};
template<typename Scalar>
Vector<Scalar> ReLUActivation<Scalar>::d_function(Vector<Scalar>& x,
		Vector<Scalar>& y) const {
	Vector<Scalar> out(x);
	for (unsigned i = 0; i < out.size2(); i++) {
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
Vector<Scalar> LeakyReLUActivation<Scalar>::function(Vector<Scalar> x) const {
	for (unsigned i = 0; i < x.cols(); i++) {
		Scalar val = x(i);
		x(i) = std::max(val * alpha, val);
	}
	return x;
};

}
