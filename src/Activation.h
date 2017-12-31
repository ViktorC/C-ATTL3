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
#include <Utils.h>
#include <string>
#include <Vector.h>

namespace cppnn {

template<typename Scalar>
class Activation {
public:
	virtual ~Activation() = default;
	virtual Matrix<Scalar> function(const Matrix<Scalar>& x) const = 0;
	virtual Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			const Matrix<Scalar>& out_grads) const = 0;
};

template<typename Scalar>
class IdentityActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			const Matrix<Scalar>& out_grads) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				x.rows() == out_grads.rows() && x.cols() == out_grads.cols());
		return out_grads;
	};
};

template<typename Scalar>
class BinaryStepActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x.unaryExpr([](Scalar i) { return i >= .0 ? 1.0 : .0; });
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			const Matrix<Scalar>& out_grads) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				x.rows() == out_grads.rows() && x.cols() == out_grads.cols());
		return Matrix<Scalar>::Zero(x.rows(), x.cols());
	};
};

template<typename Scalar>
class SigmoidActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return ((-x).array().exp() + 1).inverse();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			const Matrix<Scalar>& out_grads) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				x.rows() == out_grads.rows() && x.cols() == out_grads.cols());
		return (y.array() *  (-y.array() + 1)) * out_grads.array();
	};
};

template<typename Scalar>
class SoftmaxActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		/* First subtract the value of the greatest coefficient from each element row-wise
		 * to avoid an overflow due to raising e to great powers. */
		Matrix<Scalar> out = (x.array().colwise() - x.array().rowwise().maxCoeff()).exp();
		return out.array().colwise() / out.array().rowwise().sum();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			const Matrix<Scalar>& out_grads) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				x.rows() == out_grads.rows() && x.cols() == out_grads.cols());
		Matrix<Scalar> out(y.rows(), y.cols());
		for (int i = 0; i < out.rows(); i++) {
			Matrix<Scalar> jacobian = y.row(i).asDiagonal();
			jacobian -= y.row(i).transpose() * y.row(i);
			out.row(i) = out_grads.row(i) * jacobian;
		}
		return out;
	};
};

template<typename Scalar>
class TanhActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x.array().tanh();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			const Matrix<Scalar>& out_grads) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				x.rows() == out_grads.rows() && x.cols() == out_grads.cols());
		return (-y.array() * y.array() + 1) * out_grads.array();
	};
};

template<typename Scalar>
class ReLUActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x.cwiseMax(.0);
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			const Matrix<Scalar>& out_grads) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				x.rows() == out_grads.rows() && x.cols() == out_grads.cols());
		return x.unaryExpr([](Scalar i) { return i >= .0 ? 1.0 : .0; })
				.cwiseProduct(out_grads);
	};
};

template<typename Scalar>
class LeakyReLUActivation : public Activation<Scalar> {
public:
	LeakyReLUActivation(Scalar alpha = 1e-1) :
			alpha(alpha) {
		assert(alpha < 1 && alpha > 0);
	};
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x.cwiseMax(x * alpha);
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			const Matrix<Scalar>& out_grads) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				x.rows() == out_grads.rows() && x.cols() == out_grads.cols());
		return x.unaryExpr([this](Scalar i) { return i >= .0 ? 1.0 : alpha; })
				.cwiseProduct(out_grads);
	};
private:
	Scalar alpha;
};

} /* namespace cppnn */

#endif /* ACTIVATION_H_ */
