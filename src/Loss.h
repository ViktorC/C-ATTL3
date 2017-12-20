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
#include <Matrix.h>
#include <NumericalUtils.h>
#include <string>
#include <Vector.h>

namespace cppnn {

static const std::string INCOMPATIBLE_DIM_ERR_MSG = "incompatible out and object matrix dimensions";

template<typename Scalar>
class Loss {
public:
	virtual ~Loss() = default;
	virtual ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const = 0;
	virtual ColVector<Scalar> d_function(const Matrix<Scalar>& out) const = 0;
};

template<typename Scalar>
class QuadraticLoss : public Loss<Scalar> {
public:
	ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols() &&
				&INCOMPATIBLE_DIM_ERR_MSG);
		return (out - obj).array().square().rowwise().sum();
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		return 2 * (out - obj);
	}
};

template<typename Scalar>
class SVMLoss : public Loss<Scalar> {
public:
	SVMLoss(bool l2) :
		l2(l2) { };
	ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols() &&
				&INCOMPATIBLE_DIM_ERR_MSG);
		ColVector<Scalar> loss(out.rows());
		for (int i = 0; i < obj.rows(); i++) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < obj.cols(); j++) {
				Scalar obj_ij = obj(i,j);
				assert((almost_equal(obj_ij, 0) || almost_equal(obj_ij, 1)) &&
						"illegal svm loss object values");
				if (almost_equal(obj_ij, 1)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1 && "invalid number of correct classes for svm loss");
			Scalar loss_i = 0;
			Scalar correct_class_score = out(i,correct_class_ind);
			for (int j = 0; j < obj.cols(); j++) {
				if (j == correct_class_ind) {
					continue;
				}
				Scalar loss_ij = std::max(0, out(i,j) - correct_class_score + 1);
				loss_i += l2 ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		Matrix<Scalar> loss_grads(out.rows(), out.cols());
		for (int i = 0; i < obj.rows(); i++) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < obj.cols(); j++) {
				if (almost_equal(obj(i,j), 1)) {
					correct_class_ind = j;
					break;
				}
			}
			int contributing_elements = 0;
			Scalar correct_class_score = out(i,correct_class_ind);
			for (int j = 0; j < obj.cols(); j++) {
				if (j == correct_class_ind) {
					continue;
				}
				Scalar out_ij = out(i,j);
				Scalar margin = out_ij - correct_class_score + 1;
				if (greater(margin, 0)) {
					contributing_elements++;
					loss_grads(i,j) = l2 ? 2 * (out_ij - correct_class_score) : 1;
				} else {
					loss_grads(i,j) = 0;
				}
			}
			loss_grads(i,correct_class_ind) = -contributing_elements;
		}
		return loss_grads;
	};
private:
	bool l2;
};

template<typename Scalar>
class CrossEntropyLoss : public Loss<Scalar> {
public:
	ColVector<Scalar> function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols() &&
				&INCOMPATIBLE_DIM_ERR_MSG);
		return 0;
	};
	Matrix<Scalar> d_function(const Matrix<Scalar>& out, const Matrix<Scalar>& obj) const;
};

} /* namespace cppnn */

#endif /* LOSS_H_ */
