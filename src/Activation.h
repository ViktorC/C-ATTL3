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
#include <NumericUtils.h>
#include <string>
#include <Vector.h>

namespace cppnn {

template<typename Scalar>
class Activation {
public:
	virtual ~Activation() = default;
	virtual Matrix<Scalar> function(const Matrix<Scalar>& x) const = 0;
	virtual Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y) const = 0;
	virtual std::string to_string() const = 0;
};

template<typename Scalar>
class IdentityActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols());
		return Matrix<Scalar>::Ones(x.rows(), x.cols());
	};
	std::string to_string() const {
		return "identity";
	};
};

template<typename Scalar>
class BinaryStepActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x.unaryExpr([](Scalar i) { return decidedly_greater(i, .0) ? 1.0 : .0; });
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols());
		return Matrix<Scalar>::Zero(x.rows(), x.cols());
	};
	std::string to_string() const {
		return "binary step";
	};
};

template<typename Scalar>
class SigmoidActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return (Matrix<Scalar>::Ones(x.rows(), x.cols()).array() + (-x).array().exp()).cwiseInverse();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols());
		return y.cwiseProduct(Matrix<Scalar>::Ones(y.rows(), y.cols()) - y);
	};
	std::string to_string() const {
		return "sigmoid";
	};
};

template<typename Scalar>
class SoftmaxActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		Matrix<Scalar> out = (x.colwise() - x.rowwise().maxCoeff()).array().exp();
		return out.array().colwise() / out.array().rowwise().sum();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols());
		// TODO Vectorize the computation of the Jacobians.
		Matrix<Scalar> out(y.rows(), y.cols());
		for (int i = 0; i < y.rows(); i++) {
			Matrix<Scalar> jacobian(y.cols(), y.cols());
			for (int j = 0; j < jacobian.rows(); j++) {
				for (int k = 0; k < jacobian.cols(); k++) {
					jacobian(j,k) = y(i,k) * ((j == k) - y(i,j));
				}
			}
			for (int j = 0; j < out.cols(); j++) {
				out(i,j) = jacobian.row(j).sum();
			}
		}
		return out;
	};
	std::string to_string() const {
		return "softmax";
	};
};

template<typename Scalar>
class TanhActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x.array().tanh();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols());
		return Matrix<Scalar>::Ones(y.rows(), y.cols()) - y.cwiseProduct(y);
	};
	std::string to_string() const {
		return "tanh";
	};
};

template<typename Scalar>
class ReLUActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(const Matrix<Scalar>& x) const {
		return x.cwiseMax(.0);
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols());
		return x.unaryExpr([](Scalar i) { return decidedly_greater(i, .0) ? 1.0 : .0; });
	};
	std::string to_string() const {
		return "relu";
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
	Matrix<Scalar> d_function(const Matrix<Scalar>& x, const Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols());
		return x.unaryExpr([this](Scalar i) { return (Scalar) (decidedly_greater(i, .0) ? 1.0 :
				almost_equal(i, .0) ? .0 : alpha); });
	};
	std::string to_string() const {
		return "leaky relu; alpha: " + std::to_string(alpha);
	};
private:
	Scalar alpha;
};

} /* namespace cppnn */

#endif /* ACTIVATION_H_ */
