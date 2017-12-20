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
#include <NumericalUtils.h>
#include <string>
#include <Vector.h>

// TODO Address numerical stability issues.
namespace cppnn {

static std::string VEC_SIZE_ERR_MSG_PTR = "mismatched x and y vector lengths for derivation";

template<typename Scalar>
class Activation {
public:
	virtual ~Activation() = default;
	virtual Matrix<Scalar> function(Matrix<Scalar>& x) const = 0;
	virtual Matrix<Scalar> d_function(Matrix<Scalar>& x,
			Matrix<Scalar>& y) const = 0;
	virtual std::string to_string() const = 0;
};

template<typename Scalar>
class IdentityActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(Matrix<Scalar>& x) const {
		return x;
	};
	Matrix<Scalar> d_function(Matrix<Scalar>& x,
			Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				&VEC_SIZE_ERR_MSG_PTR);
		return Matrix<Scalar>::Ones(x.rows(), x.cols());
	};
	std::string to_string() const {
		return "identity";
	};
};

template<typename Scalar>
class BinaryStepActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(Matrix<Scalar>& x) const {
		return x.unaryExpr([](Scalar i) { return (Scalar) greater(i, 0); });
	};
	Matrix<Scalar> d_function(Matrix<Scalar>& x,
			Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				&VEC_SIZE_ERR_MSG_PTR);
		return Matrix<Scalar>::Zero(x.rows(), x.cols());
	};
	std::string to_string() const {
		return "binary step";
	};
};

template<typename Scalar>
class SigmoidActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(Matrix<Scalar>& x) const {
		return (Matrix<Scalar>::Ones(x.rows(), x.cols()) +
				(-x).array().exp()).cwiseInverse();
	};
	Matrix<Scalar> d_function(Matrix<Scalar>& x,
			Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				&VEC_SIZE_ERR_MSG_PTR);
		return y.cwiseProduct(Matrix<Scalar>::Ones(y.rows(),
				y.cols()) - y);
	};
	std::string to_string() const {
		return "sigmoid";
	};
};

template<typename Scalar>
class SoftmaxActivation : public Activation<Scalar> {
public:
	SoftmaxActivation(Scalar epsilon = EPSILON) :
		epsilon(epsilon) { };
	Matrix<Scalar> function(Matrix<Scalar>& x) const {
		Matrix<Scalar> out = x.array().exp();
		for (int i = 0; i < x.rows(); i++) {
			out.row(i) /= (out.row(i).sum() + epsilon);
		}
		return out;
	};
	Matrix<Scalar> d_function(Matrix<Scalar>& x,
			Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				&VEC_SIZE_ERR_MSG_PTR);
		// TODO Vectorize the computation of the Jacobian.
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
		return "softmax; epsilon: " + std::to_string(epsilon);
	};
private:
	static constexpr Scalar EPSILON = 1e-5;
	Scalar epsilon;
};

template<typename Scalar>
class TanhActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(Matrix<Scalar>& x) const {
		return x.array().tanh();
	};
	Matrix<Scalar> d_function(Matrix<Scalar>& x,
			Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				&VEC_SIZE_ERR_MSG_PTR);
		return Matrix<Scalar>::Ones(y.rows(), y.cols()) - y.cwiseProduct(y);
	};
	std::string to_string() const {
		return "tanh";
	};
};

template<typename Scalar>
class ReLUActivation : public Activation<Scalar> {
public:
	Matrix<Scalar> function(Matrix<Scalar>& x) const {
		return x.cwiseMax(.0);
	};
	Matrix<Scalar> d_function(Matrix<Scalar>& x,
			Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				&VEC_SIZE_ERR_MSG_PTR);
		return x.unaryExpr([](Scalar i) { return (Scalar) greater(i, 0); });
	};
	std::string to_string() const {
		return "relu";
	};
};

template<typename Scalar>
class LeakyReLUActivation : public Activation<Scalar> {
public:
	LeakyReLUActivation(Scalar alpha = DEF_LRELU_ALPHA) :
			alpha(alpha) {
		assert(alpha < 1 && alpha > 0 && "alpha must be less than "
				"1 and greater than or equal to 0");
	};
	Matrix<Scalar> function(Matrix<Scalar>& x) const {
		return x.cwiseMax(x * alpha);
	};
	Matrix<Scalar> d_function(Matrix<Scalar>& x,
			Matrix<Scalar>& y) const {
		assert(x.rows() == y.rows() && x.cols() == y.cols() &&
				&VEC_SIZE_ERR_MSG_PTR);
		return x.unaryExpr([this](Scalar i) { return (Scalar) (greater(i, 0) ? 1 :
				almost_equal(i, 0) ? 0 : alpha); });
	};
	std::string to_string() const {
		return "leaky relu; alpha: " + std::to_string(alpha);
	};
private:
	static constexpr Scalar DEF_LRELU_ALPHA = 1e-1;
	Scalar alpha;
};

} /* namespace cppnn */

#endif /* ACTIVATION_H_ */
