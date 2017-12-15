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

// TODO Address numerical stability issues.
namespace cppnn {

static const std::string* VEC_SIZE_ERR_MSG_PTR =
		new std::string("mismatched x and y vector lengths for derivation");

template<typename Scalar>
class Activation {
public:
	virtual ~Activation() = default;
	virtual RowVector<Scalar> function(const RowVector<Scalar>& x) const = 0;
	virtual RowVector<Scalar> d_function(const RowVector<Scalar>& x,
			const RowVector<Scalar>& y) const = 0;
};

template<typename Scalar>
class IdentityActivation : public Activation<Scalar> {
public:
	RowVector<Scalar> function(const RowVector<Scalar>& x) const {
		return x;
	};
	RowVector<Scalar> d_function(const RowVector<Scalar>& x,
			const RowVector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		return RowVector<Scalar>::Ones(x.cols());
	};
};

template<typename Scalar>
class BinaryStepActivation : public Activation<Scalar> {
public:
	RowVector<Scalar> function(const RowVector<Scalar>& x) const {
		RowVector<Scalar> out = x;
		for (int i = 0; i < out.cols(); i++) {
			out(i) = out(i) > .0;
		}
		return out;
	};
	RowVector<Scalar> d_function(const RowVector<Scalar>& x,
			const RowVector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		return RowVector<Scalar>::Zero(x.cols());
	};
};

template<typename Scalar>
class SigmoidActivation : public Activation<Scalar> {
public:
	RowVector<Scalar> function(const RowVector<Scalar>& x) const {
		RowVector<Scalar> exp = (-x).array().exp();
		return (RowVector<Scalar>::Ones(x.cols()) + exp).cwiseInverse();
	};
	RowVector<Scalar> d_function(const RowVector<Scalar>& x,
			const RowVector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		return y.cwiseProduct(RowVector<Scalar>::Ones(y.cols()) - y);
	};
};

template<typename Scalar>
class SoftmaxActivation : public Activation<Scalar> {
public:
	RowVector<Scalar> function(const RowVector<Scalar>& x) const {
		RowVector<Scalar> out = x.array().exp();
		return out / out.sum();
	};
	RowVector<Scalar> d_function(const RowVector<Scalar>& x,
			const RowVector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		// TODO Vectorize the computation of the Jacobian.
		Matrix<Scalar> jacobian(y.cols(), y.cols());
		for (int i = 0; i < jacobian.rows(); i++) {
			for (int j = 0; j < jacobian.cols(); j++) {
				jacobian(i,j) = y(j) * ((i == j) - y(i));
			}
		}
		RowVector<Scalar> out(y.cols());
		for (int i = 0; i < out.cols(); i++) {
			out(i) = jacobian.row(i).sum();
		}
		return out;
	};
};

template<typename Scalar>
class TanhActivation : public Activation<Scalar> {
public:
	RowVector<Scalar> function(const RowVector<Scalar>& x) const {
		return x.array().tanh();
	};
	RowVector<Scalar> d_function(const RowVector<Scalar>& x,
			const RowVector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		return RowVector<Scalar>::Ones(y.cols()) - y.cwiseProduct(y);
	};
};

template<typename Scalar>
class ReLUActivation : public Activation<Scalar> {
public:
	virtual ~ReLUActivation() = default;
	virtual RowVector<Scalar> function(const RowVector<Scalar>& x) const {
		return x.cwiseMax(.0);
	};
	virtual RowVector<Scalar> d_function(const RowVector<Scalar>& x,
			const RowVector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);
		RowVector<Scalar> out(x);
		for (int i = 0; i < out.cols(); i++) {
			out(0,i) = out(0,i) > .0;
		}
		return out;
	};
};

template<typename Scalar>
class LeakyReLUActivation : public ReLUActivation<Scalar> {
public:
	LeakyReLUActivation(float alpha = DEF_LRELU_ALPHA) :
			alpha(alpha) {
		assert(alpha < 1 && "alpha must be less than 1");
	};
	RowVector<Scalar> function(const RowVector<Scalar>& x) const {
		return x.cwiseMax(x * alpha);
	};
private:
	static constexpr float DEF_LRELU_ALPHA = 1e-1;
	Scalar alpha;
};

template<typename Scalar>
class BatchNormalizedActivation : public Activation<Scalar> {
public:
	BatchNormalizedActivation(const Activation<Scalar>& act) :
		act(act) { };
	RowVector<Scalar> function(const RowVector<Scalar>& x) const {

	};
	RowVector<Scalar> d_function(const RowVector<Scalar>& x,
			const RowVector<Scalar>& y) const {
		assert(x.cols() == y.cols() && VEC_SIZE_ERR_MSG_PTR);

	};
private:
	const Activation<Scalar>& act;
};

} /* namespace cppnn */

#endif /* ACTIVATION_H_ */
