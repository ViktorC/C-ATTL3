/*
 * Cost.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <Matrix.h>
#include <Utils.h>
#include <string>
#include <Vector.h>

namespace cppnn {

template<typename Scalar>
class Loss {
public:
	virtual ~Loss() = default;
	virtual ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const = 0;
	virtual Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const = 0;
};

template<typename Scalar>
class QuadraticLoss : public Loss<Scalar> {
public:
	ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		return (out - obj).array().square().rowwise().sum();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		return 2 * (out - obj) / out.rows();
	}
};

/**
 * One-hot objective.
 */
template<typename Scalar>
class HingeLoss : public Loss<Scalar> {
public:
	HingeLoss(bool squared = false) :
		squared(squared) { };
	ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		ColVector<Scalar> loss(out.rows());
		for (int i = 0; i < obj.rows(); i++) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < obj.cols(); j++) {
				Scalar obj_ij = obj(i,j);
				assert((almost_equal(obj_ij, .0) || almost_equal(obj_ij, 1.0)));
				if (almost_equal(obj_ij, 1.0)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar loss_i = 0;
			Scalar correct_class_score = out(i,correct_class_ind);
			for (int j = 0; j < obj.cols(); j++) {
				if (j == correct_class_ind) {
					continue;
				}
				Scalar loss_ij = std::max(.0, out(i,j) - correct_class_score + 1);
				loss_i += squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		Matrix<Scalar> out_grads(out.rows(), out.cols());
		for (int i = 0; i < out.rows(); i++) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < out.cols(); j++) {
				Scalar obj_ij = obj(i,j);
				assert((almost_equal(obj_ij, .0) || almost_equal(obj_ij, 1.0)));
				if (almost_equal(obj_ij, 1.0)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar total_out_grad = 0;
			Scalar correct_class_score = out(i,correct_class_ind);
			for (int j = 0; j < out.cols(); j++) {
				if (j == correct_class_ind) {
					continue;
				}
				Scalar out_ij = out(i,j);
				Scalar margin = out_ij - correct_class_score + 1;
				if (decidedly_greater(margin, .0)) {
					Scalar out_grad = squared ? 2 * (out_ij - correct_class_score) : 1;
					total_out_grad += out_grad;
					out_grads(i,j) = out_grad;
				} else {
					out_grads(i,j) = 0;
				}
			}
			out_grads(i,correct_class_ind) = -total_out_grad;
		}
		return out_grads / out.rows();
	};
private:
	bool squared;
};

/**
 * Objective values between 0 and 1 (inclusive). Use with softmax activation.
 */
template<typename Scalar>
class CrossEntropyLoss : public Loss<Scalar> {
public:
	CrossEntropyLoss(Scalar epsilon = 1e-8) :
		epsilon(epsilon) { };
	ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		return (out.array().log() * obj.array()).matrix().rowwise().sum() * -1;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		return (-obj.array() / (out.array() + epsilon)) / out.rows();
	}
private:
	Scalar epsilon;
};

/**
 * True label: 1; false label: -1. Use with tanh activation.
 */
template<typename Scalar>
class MultiLabelHingeLoss : public Loss<Scalar> {
public:
	MultiLabelHingeLoss(bool squared = false) :
		squared(squared) { };
	ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		ColVector<Scalar> loss(out.rows());
		for (int i = 0; i < obj.rows(); i++) {
			Scalar loss_i = 0;
			for (int j = 0; j < obj.cols(); j++) {
				Scalar obj_ij = obj(i,j);
				assert((almost_equal(obj_ij, -1.0) || almost_equal(obj_ij, 1.0)));
				Scalar loss_ij = std::max(.0, 1 - obj_ij * out(i,j));
				loss_i += squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		Matrix<Scalar> out_grads(out.rows(), out.cols());
		for (int i = 0; i < out.rows(); i++) {
			for (int j = 0; j < out.cols(); j++) {
				Scalar obj_ij = obj(i,j);
				assert((almost_equal(obj_ij, -1.0) || almost_equal(obj_ij, 1.0)));
				Scalar out_ij = out(i,j);
				Scalar margin = 1 - obj_ij * out_ij;
				if (decidedly_greater(margin, .0)) {
					out_grads(i,j) = squared ? 2 * out_ij - 2 * obj_ij : -obj_ij;
				} else {
					out_grads(i,j) = 0;
				}
			}
		}
		return out_grads / out.rows();
	};
private:
	bool squared;
};

/**
 * True label: 1; false label: 0. Use with sigmoid activation.
 */
template<typename Scalar>
class MultiLabelLogLoss : public Loss<Scalar> {
public:
	MultiLabelLogLoss(Scalar epsilon = 1e-8) :
		epsilon(epsilon) { };
	ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		ColVector<Scalar> loss(out.rows());
		for (int i = 0; i < out.rows(); i++) {
			Scalar loss_i = 0;
			for (int j = 0; j < out.cols(); j++) {
				Scalar obj_ij = obj(i,j);
				assert(almost_equal(obj_ij, .0) || almost_equal(obj_ij, 1.0));
				Scalar out_ij = out(i,j);
				loss_i += (obj_ij * log(out_ij) + (1 - obj_ij) * log(1 - out_ij));
			}
			loss(i) = loss_i;
		}
		return loss;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols());
		int rows = out.rows();
		Matrix<Scalar> out_grads(rows, out.cols());
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < out_grads.cols(); j++) {
				Scalar obj_ij = obj(i,j);
				assert(almost_equal(obj_ij, .0) || almost_equal(obj_ij, 1.0));
				Scalar denominator = out(i,j) - (Scalar) (almost_equal(obj_ij, .0));
				if (denominator == 0) {
					denominator += (rand() % 2 == 0 ? epsilon : -epsilon);
				}
				out_grads(i,j) = 1 / (denominator * rows);
			}
		}
		return out_grads;
	}
private:
	Scalar epsilon;
};

} /* namespace cppnn */

#endif /* LOSS_H_ */
