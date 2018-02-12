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
#include <cstddef>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <Utils.h>

namespace cattle {

template<typename Scalar, size_t Rank>
class Loss {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal loss rank");
public:
	virtual ~Loss() = default;
	virtual ColVector<Scalar> function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const = 0;
	virtual Tensor<Scalar,Rank + 1> d_function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const = 0;
};

template<typename Scalar, size_t Rank>
class QuadraticLoss : public Loss<Scalar,Rank> {
public:
	inline ColVector<Scalar> function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		return (Utils<Scalar>::map_tensor_to_mat(out) - Utils<Scalar>::map_tensor_to_mat(obj))
				.array().square().rowwise().sum();
	};
	inline Tensor<Scalar,Rank + 1> d_function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		return Utils<Scalar>::map_mat_to_tensor((2 * (Utils<Scalar>::map_tensor_to_mat(out) -
				Utils<Scalar>::map_tensor_to_mat(obj))).eval(), Utils<Scalar>::get_dims(out));
	}
};

/**
 * One-hot objective.
 */
template<typename Scalar, size_t Rank>
class HingeLoss : public Loss<Scalar,Rank> {
public:
	HingeLoss(bool squared = false) :
		squared(squared) { };
	inline ColVector<Scalar> function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		Matrix<Scalar> out_mat = Utils<Scalar>::map_tensor_to_mat(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::map_tensor_to_mat(obj);
		ColVector<Scalar> loss(out_mat.rows());
		for (int i = 0; i < obj_mat.rows(); i++) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < obj_mat.cols(); j++) {
				Scalar obj_ij = obj_mat(i,j);
				assert((Utils<Scalar>::almost_equal(obj_ij, .0) || Utils<Scalar>::almost_equal(obj_ij, 1.0)));
				if (Utils<Scalar>::almost_equal(obj_ij, 1.0)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar loss_i = 0;
			Scalar correct_class_score = out_mat(i,correct_class_ind);
			for (int j = 0; j < obj_mat.cols(); j++) {
				if (j == correct_class_ind)
					continue;
				Scalar loss_ij = std::max(.0, out_mat(i,j) - correct_class_score + 1);
				loss_i += squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	};
	inline Tensor<Scalar,Rank + 1> d_function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		Matrix<Scalar> out_mat = Utils<Scalar>::map_tensor_to_mat(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::map_tensor_to_mat(obj);
		Matrix<Scalar> out_grads(out_mat.rows(), out_mat.cols());
		for (int i = 0; i < out_mat.rows(); i++) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < out_mat.cols(); j++) {
				Scalar obj_ij = obj_mat(i,j);
				assert((Utils<Scalar>::almost_equal(obj_ij, .0) || Utils<Scalar>::almost_equal(obj_ij, 1.0)));
				if (Utils<Scalar>::almost_equal(obj_ij, 1.0)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar total_out_grad = 0;
			Scalar correct_class_score = out_mat(i,correct_class_ind);
			for (int j = 0; j < out_mat.cols(); j++) {
				if (j == correct_class_ind)
					continue;
				Scalar out_ij = out_mat(i,j);
				Scalar margin = out_ij - correct_class_score + 1;
				if (Utils<Scalar>::decidedly_greater(margin, .0)) {
					Scalar out_grad = squared ? 2 * (out_ij - correct_class_score) : 1;
					total_out_grad += out_grad;
					out_grads(i,j) = out_grad;
				} else
					out_grads(i,j) = 0;
			}
			out_grads(i,correct_class_ind) = -total_out_grad;
		}
		return Utils<Scalar>::map_mat_to_tensor(out_grads, Utils<Scalar>::get_dims(out));
	};
private:
	bool squared;
};

/**
 * Objective values between 0 and 1 (inclusive). Use with softmax activation.
 */
template<typename Scalar, size_t Rank>
class CrossEntropyLoss : public Loss<Scalar,Rank> {
public:
	CrossEntropyLoss(Scalar epsilon = Utils<Scalar>::EPSILON2) :
		epsilon(epsilon) { };
	inline ColVector<Scalar> function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		return (Utils<Scalar>::map_tensor_to_mat(out).array().log() * Utils<Scalar>::map_tensor_to_mat(obj).array())
				.matrix().rowwise().sum() * -1;
	};
	inline Tensor<Scalar,Rank + 1> d_function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		return Utils<Scalar>::map_mat_to_tensor((-Utils<Scalar>::map_tensor_to_mat(obj).array() /
				(Utils<Scalar>::map_tensor_to_mat(out).array() + epsilon)).eval(), Utils<Scalar>::get_dims(out));
	}
private:
	Scalar epsilon;
};

/**
 * True label: 1; false label: -1.
 */
template<typename Scalar, size_t Rank>
class MultiLabelHingeLoss : public Loss<Scalar,Rank> {
public:
	MultiLabelHingeLoss(bool squared = false) :
		squared(squared) { };
	inline ColVector<Scalar> function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		Matrix<Scalar> out_mat = Utils<Scalar>::map_tensor_to_mat(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::map_tensor_to_mat(obj);
		ColVector<Scalar> loss(out_mat.rows());
		for (int i = 0; i < obj_mat.rows(); i++) {
			Scalar loss_i = 0;
			for (int j = 0; j < obj_mat.cols(); j++) {
				Scalar obj_ij = obj_mat(i,j);
				assert((Utils<Scalar>::almost_equal(obj_ij, -1.0) || Utils<Scalar>::almost_equal(obj_ij, 1.0)));
				Scalar loss_ij = std::max(.0, 1 - obj_ij * out_mat(i,j));
				loss_i += squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	};
	inline Tensor<Scalar,Rank + 1> d_function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		Matrix<Scalar> out_mat = Utils<Scalar>::map_tensor_to_mat(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::map_tensor_to_mat(obj);
		Matrix<Scalar> out_grads(out_mat.rows(), out_mat.cols());
		for (int i = 0; i < out_mat.rows(); i++) {
			for (int j = 0; j < out_mat.cols(); j++) {
				Scalar obj_ij = obj_mat(i,j);
				assert((Utils<Scalar>::almost_equal(obj_ij, -1.0) || Utils<Scalar>::almost_equal(obj_ij, 1.0)));
				Scalar out_ij = out_mat(i,j);
				Scalar margin = 1 - obj_ij * out_ij;
				if (Utils<Scalar>::decidedly_greater(margin, .0))
					out_grads(i,j) = squared ? 2 * out_ij - 2 * obj_ij : -obj_ij;
				else
					out_grads(i,j) = 0;
			}
		}
		return Utils<Scalar>::map_mat_to_tensor(out_grads, Utils<Scalar>::get_dims(out));
	};
private:
	bool squared;
};

/**
 * True label: 1; false label: 0. Use with sigmoid activation.
 */
template<typename Scalar, size_t Rank>
class MultiLabelLogLoss : public Loss<Scalar,Rank> {
public:
	MultiLabelLogLoss(Scalar epsilon = Utils<Scalar>::EPSILON2) :
		epsilon(epsilon) { };
	inline ColVector<Scalar> function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		Matrix<Scalar> out_mat = Utils<Scalar>::map_tensor_to_mat(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::map_tensor_to_mat(obj);
		ColVector<Scalar> loss(out_mat.rows());
		for (int i = 0; i < out_mat.rows(); i++) {
			Scalar loss_i = 0;
			for (int j = 0; j < out_mat.cols(); j++) {
				Scalar obj_ij = obj_mat(i,j);
				assert(Utils<Scalar>::almost_equal(obj_ij, .0) || Utils<Scalar>::almost_equal(obj_ij, 1.0));
				Scalar out_ij = out_mat(i,j);
				loss_i += (obj_ij * log(out_ij) + (1 - obj_ij) * log(1 - out_ij));
			}
			loss(i) = loss_i;
		}
		return loss;
	};
	inline Tensor<Scalar,Rank + 1> d_function(const Tensor<Scalar,Rank + 1>& out, const Tensor<Scalar,Rank + 1>& obj) const {
		assert(Utils<Scalar>::get_dims(out) == Utils<Scalar>::get_dims(obj));
		assert((Rank < 2 || out.dimension(2) == 1) && (Rank < 3 || out.dimension(3) == 1));
		Matrix<Scalar> out_mat = Utils<Scalar>::map_tensor_to_mat(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::map_tensor_to_mat(obj);
		int rows = out_mat.rows();
		Matrix<Scalar> out_grads(rows, out_mat.cols());
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < out_grads.cols(); j++) {
				Scalar obj_ij = obj_mat(i,j);
				assert(Utils<Scalar>::almost_equal(obj_ij, .0) || Utils<Scalar>::almost_equal(obj_ij, 1.0));
				Scalar denominator = out_mat(i,j) - (Scalar) (Utils<Scalar>::almost_equal(obj_ij, .0));
				if (denominator == 0)
					denominator += (rand() % 2 == 0 ? epsilon : -epsilon);
				out_grads(i,j) = 1 / (denominator * rows);
			}
		}
		return Utils<Scalar>::map_mat_to_tensor(out_grads, Utils<Scalar>::get_dims(out));
	}
private:
	Scalar epsilon;
};

} /* namespace cattle */

#endif /* LOSS_H_ */
