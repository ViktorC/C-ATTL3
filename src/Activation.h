/*
 * Activation.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include <cassert>
#include <Eigen/Dense>
#include <Matrix.h>
#include <string>
#include <Vector.h>

// TODO Address numeric stability issues.
namespace cppnn {

static const std::string* VEC_SIZE_ERR_MSG_PTR =
		new std::string("mismatched x and y vector lengths for derivation");

template<typename Scalar>
class Activation {
public:
	virtual ~Activation() = default;
	virtual Activation<Scalar>* clone() = 0;
	virtual Vector<Scalar> function(Vector<Scalar> x) const = 0;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const = 0;
};

template<typename Scalar>
class IdentityActivation : public Activation<Scalar> {
public:
	virtual ~IdentityActivation() = default;
	virtual Activation<Scalar>* clone() {
		return new IdentityActivation();
	};
	virtual Vector<Scalar> function(Vector<Scalar> x) const {
		return x;
	};
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		return Vector<Scalar>::Ones(x.cols());
	};
};

template<typename Scalar>
class BinaryStepActivation : public Activation<Scalar> {
public:
	virtual ~BinaryStepActivation() = default;
	virtual Activation<Scalar>* clone() {
		return new BinaryStepActivation();
	};
	virtual Vector<Scalar> function(Vector<Scalar> x) const {
		for (int i = 0; i < x.cols(); i++) {
			x(i) = x(i) > .0;
		}
		return x;
	};
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		return Vector<Scalar>::Zero(x.cols());
	};
};

template<typename Scalar>
class SigmoidActivation : public Activation<Scalar> {
public:
	virtual ~SigmoidActivation() = default;
	virtual Activation<Scalar>* clone() {
		return new SigmoidActivation();
	};
	virtual Vector<Scalar> function(Vector<Scalar> x) const {
		Vector<Scalar> exp = (-x).array().exp();
		return (Vector<Scalar>::Ones(x.cols()) + exp).cwiseInverse();
	};
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		return y.cwiseProduct(Vector<Scalar>::Ones(y.cols()) - y);
	};
};

template<typename Scalar>
class SoftmaxActivation : public Activation<Scalar> {
public:
	virtual ~SoftmaxActivation() = default;
	virtual Activation<Scalar>* clone() {
		return new SoftmaxActivation();
	};
	virtual Vector<Scalar> function(Vector<Scalar> x) const {
		Vector<Scalar> out = x.array().exp();
		return out / out.sum();
	};
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		Matrix<Scalar> jacobian(y.cols(), y.cols());
		for (int i = 0; i < jacobian.rows(); i++) {
			for (int j = 0; j < jacobian.cols(); j++) {
				jacobian(i,j) = y(j) * ((i == j) - y(i));
			}
		}
		Vector<Scalar> out(y.cols());
		for (int i = 0; i < out.cols(); i++) {
			out(i) = jacobian.row(i).sum();
		}
		return out;
	};
};

template<typename Scalar>
class TanhActivation : public Activation<Scalar> {
public:
	virtual ~TanhActivation() = default;
	virtual Activation<Scalar>* clone() {
		return new TanhActivation();
	};
	virtual Vector<Scalar> function(Vector<Scalar> x) const {
		return x.array().tanh();
	};
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		return Vector<Scalar>::Ones(y.cols()) - y.cwiseProduct(y);
	};
};

template<typename Scalar>
class ReLUActivation : public Activation<Scalar> {
public:
	virtual ~ReLUActivation() = default;
	virtual Activation<Scalar>* clone() {
		return new ReLUActivation();
	};
	virtual Vector<Scalar> function(Vector<Scalar> x) const {
		return x.cwiseMax(.0);
	};
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		Vector<Scalar> out(x);
		for (int i = 0; i < out.cols(); i++) {
			out(0,i) = out(0,i) > .0;
		}
		return out;
	};
};

template<typename Scalar>
class LeakyReLUActivation : public ReLUActivation<Scalar> {
	static constexpr Scalar DEF_LRELU_ALPHA = 1e-1;
protected:
	Scalar alpha;
public:
	LeakyReLUActivation(Scalar alpha = DEF_LRELU_ALPHA) :
			alpha(alpha) {
		assert(alpha < 1 && "alpha must be less than 1");
	};
	virtual ~LeakyReLUActivation() = default;
	virtual Activation<Scalar>* clone() {
		return new LeakyReLUActivation(alpha);
	};
	virtual Vector<Scalar> function(Vector<Scalar> x) const {
		return x.cwiseMax(x * alpha);
	};
};

}

#endif /* ACTIVATION_H_ */
