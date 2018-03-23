/*
 * Cost.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <type_traits>
#include "Utils.h"

namespace cattle {

// TODO CTC loss.

/**
 * An abstract class template for loss functions. Implementations of this class should be stateless.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class Loss {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal loss rank");
protected:
	static constexpr std::size_t DATA_RANKS = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANKS> Data;
public:
	virtual ~Loss() = default;
	/**
	 * It calculates the error on each sample given the output and the objective tensors.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor (of the same dimensionality as the output).
	 * @return A column vector containing the loss for each sample.
	 */
	virtual ColVector<Scalar> function(const Data& out, const Data& obj) const = 0;
	/**
	 * It calculates the derivative of the loss function w.r.t. the output.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor (of the same dimensionality as the output).
	 * @return The derivative of the loss function w.r.t. the output.
	 */
	virtual Data d_function(const Data& out, const Data& obj) const = 0;
};

/**
 * An abstract class template for loss functions for both sequential and non-sequential data.
 * Implementations of this class should be stateless.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class UniversalLoss : public Loss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Base;
protected:
	/**
	 * It computes the loss of a batch of non-sequential data.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor.
	 * @return A column vector representing the losses of the samples in the batch.
	 */
	virtual ColVector<Scalar> _function(const typename Base::Data& out, const typename Base::Data& obj) const = 0;
	/**
	 * It computes the gradient of the output batch.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor.
	 * @param grad_dims The dimensions of the gradient tensor.
	 * @return The gradient tensor of the output batch.
	 */
	virtual typename Base::Data _d_function(const typename Base::Data& out, const typename Base::Data& obj,
			const Dimensions<int,Base::DATA_RANKS - 1>& grad_dims) const = 0;
public:
	virtual ~UniversalLoss() = default;
	inline ColVector<Scalar> function(const typename Base::Data& out, const typename Base::Data& obj) const {
		assert(Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out) ==
				Utils<Scalar>::template get_dims<Base::DATA_RANKS>(obj));
		return _function(out, obj);
	}
	inline typename Base::Data d_function(const typename Base::Data& out, const typename Base::Data& obj) const {
		Dimensions<int,Base::DATA_RANKS> dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out);
		assert(dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(obj));
		return _d_function(out, obj, dims.template demote<>());
	}
};

/**
 * Partial template specialization for sequential data. Implementations
 * of this class should be stateless.
 */
template<typename Scalar, std::size_t Rank>
class UniversalLoss<Scalar,Rank,true> : public Loss<Scalar,Rank,true> {
	typedef Loss<Scalar,Rank,true> Base;
protected:
	/**
	 * It computes the loss of a single time step in a batch. The total loss of the batch is the sum of the losses
	 * of all its time steps.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor.
	 * @return A column vector representing the losses of the samples in the batch for the given time step.
	 */
	virtual ColVector<Scalar> _function(const typename Base::Data& out, const typename Base::Data& obj) const = 0;
	/**
	 * It computes the gradient of a single time step of the output sequence batch.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor.
	 * @param grad_dims The dimensions of the gradient tensor.
	 * @return The gradient tensor of the provided time step of the output batch.
	 */
	virtual typename Base::Data _d_function(const typename Base::Data& out, const typename Base::Data& obj,
			const Dimensions<int,Base::DATA_RANKS - 1>& grad_dims) const = 0;
public:
	virtual ~UniversalLoss() = default;
	inline ColVector<Scalar> function(const typename Base::Data& out, const typename Base::Data& obj) const {
		Dimensions<int,Base::DATA_RANKS> dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out);
		assert(dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(obj));
		int time_steps = dims(1);
		if (time_steps == 1)
			return _function(out, obj);
		std::array<int,Base::DATA_RANKS> offsets;
		std::array<int,Base::DATA_RANKS> extents = dims;
		offsets.fill(0);
		extents[1] = 1;
		ColVector<Scalar> loss = ColVector<Scalar>::Zero(dims(0), 1);
		for (int i = 0; i < time_steps; ++i) {
			offsets[1] = i;
			typename Base::Data out_i = out.slice(offsets, extents);
			typename Base::Data obj_i = obj.slice(offsets, extents);
			loss += _function(out_i, obj_i);
		}
		return loss;
	}
	inline typename Base::Data d_function(const typename Base::Data& out, const typename Base::Data& obj) const {
		Dimensions<int,Base::DATA_RANKS> dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out);
		assert(dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(obj));
		int time_steps = dims(1);
		if (time_steps == 1)
			return _d_function(out, obj, dims.template demote<>());
		typename Base::Data grads(dims);
		grads.setZero();
		dims(1) = 1;
		std::array<int,Base::DATA_RANKS> offsets;
		std::array<int,Base::DATA_RANKS> extents = dims;
		offsets.fill(0);
		Dimensions<int,Base::DATA_RANKS - 1> grad_dims = dims.template demote<>();
		for (int i = 0; i < time_steps; ++i) {
			offsets[1] = i;
			typename Base::Data out_i = out.slice(offsets, extents);
			typename Base::Data obj_i = obj.slice(offsets, extents);
			grads.slice(offsets, extents) += _d_function(out_i, obj_i, grad_dims);
		}
		return grads;
	}
};

/**
 * A template class representing the squared error loss function.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class QuadraticLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
protected:
	inline ColVector<Scalar> _function(const typename Root::Data& out,
			const typename Root::Data& obj) const {
		return (Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out) -
				Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj))
				.array().square().rowwise().sum();
	}
	inline typename Root::Data _d_function(const typename Root::Data& out,
			const typename Root::Data& obj, const Dimensions<int,Root::DATA_RANKS - 1>& grad_dims) const {
		return Utils<Scalar>::template map_mat_to_tensor<Root::DATA_RANKS>((2 *
				(Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out) -
				Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj))).eval(), grad_dims);
	}
};

/**
 * A template class representing the hinge loss function. This class assumes the objectives for
 * each sample (and time step) to be a one-hot vector (tensor rank).
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool Squared = false>
class HingeLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
protected:
	inline ColVector<Scalar> _function(const typename Root::Data& out, const typename Root::Data& obj) const {
		Matrix<Scalar> out_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj);
		ColVector<Scalar> loss(out_mat.rows());
		for (int i = 0; i < obj_mat.rows(); ++i) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < obj_mat.cols(); ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((Utils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						Utils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				if (Utils<Scalar>::almost_equal(obj_ij, (Scalar) 1)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar loss_i = 0;
			Scalar correct_class_score = out_mat(i,correct_class_ind);
			for (int j = 0; j < obj_mat.cols(); ++j) {
				if (j == correct_class_ind)
					continue;
				Scalar loss_ij = std::max(.0, out_mat(i,j) - correct_class_score + 1);
				loss_i += Squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	}
	inline typename Root::Data _d_function(const typename Root::Data& out,
			const typename Root::Data& obj, const Dimensions<int,Root::DATA_RANKS - 1>& grad_dims) const {
		Matrix<Scalar> out_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj);
		Matrix<Scalar> out_grads(out_mat.rows(), out_mat.cols());
		for (int i = 0; i < out_mat.rows(); ++i) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < out_mat.cols(); ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((Utils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						Utils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				if (Utils<Scalar>::almost_equal(obj_ij, (Scalar) 1)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar total_out_grad = 0;
			Scalar correct_class_score = out_mat(i,correct_class_ind);
			for (int j = 0; j < out_mat.cols(); ++j) {
				if (j == correct_class_ind)
					continue;
				Scalar out_ij = out_mat(i,j);
				Scalar margin = out_ij - correct_class_score + 1;
				if (Utils<Scalar>::decidedly_greater(margin, (Scalar) 0)) {
					Scalar out_grad = Squared ? 2 * (out_ij - correct_class_score) : 1;
					total_out_grad += out_grad;
					out_grads(i,j) = out_grad;
				} else
					out_grads(i,j) = 0;
			}
			out_grads(i,correct_class_ind) = -total_out_grad;
		}
		return Utils<Scalar>::template map_mat_to_tensor<Root::DATA_RANKS>(out_grads, grad_dims);
	}
};

/**
 * A template class representing the cross entropy loss function. This class assumes the objective
 * values for each sample (and time step) to be in the range [0, 1].
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class CrossEntropyLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	CrossEntropyLoss(Scalar epsilon = Utils<Scalar>::EPSILON2) : epsilon(epsilon) { };
protected:
	inline ColVector<Scalar> _function(const typename Root::Data& out,
			const typename Root::Data& obj) const {
		return (Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out).array().log() *
				Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj).array())
				.matrix().rowwise().sum() * -1;
	}
	inline typename Root::Data _d_function(const typename Root::Data& out,
			const typename Root::Data& obj, const Dimensions<int,Root::DATA_RANKS - 1>& grad_dims) const {
		return Utils<Scalar>::template map_mat_to_tensor<Root::DATA_RANKS>(
				(-Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj).array() /
				(Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out).array() + epsilon)).eval(),
				grad_dims);
	}
private:
	Scalar epsilon;
};

/**
 * A class template representing the hinge loss function for multi-label objectives. True labels
 * are expected to have the value 1, while false labels are expected to correspond to the value -1.
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool Squared = false>
class MultiLabelHingeLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
protected:
	inline ColVector<Scalar> _function(const typename Root::Data& out,
			const typename Root::Data& obj) const {
		Matrix<Scalar> out_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj);
		ColVector<Scalar> loss(out_mat.rows());
		for (int i = 0; i < obj_mat.rows(); ++i) {
			Scalar loss_i = 0;
			for (int j = 0; j < obj_mat.cols(); ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((Utils<Scalar>::almost_equal(obj_ij, (Scalar) -1) ||
						Utils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				Scalar loss_ij = std::max((Scalar) 0, (Scalar) 1 - obj_ij * out_mat(i,j));
				loss_i += Squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	}
	inline typename Root::Data _d_function(const typename Root::Data& out,
			const typename Root::Data& obj, const Dimensions<int,Root::DATA_RANKS - 1>& grad_dims) const {
		Matrix<Scalar> out_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj);
		Matrix<Scalar> out_grads(out_mat.rows(), out_mat.cols());
		for (int i = 0; i < out_mat.rows(); ++i) {
			for (int j = 0; j < out_mat.cols(); ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((Utils<Scalar>::almost_equal(obj_ij, (Scalar) -1) ||
						Utils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				Scalar out_ij = out_mat(i,j);
				Scalar margin = 1 - obj_ij * out_ij;
				if (Utils<Scalar>::decidedly_greater(margin, (Scalar) 0))
					out_grads(i,j) = Squared ? 2 * out_ij - 2 * obj_ij : -obj_ij;
				else
					out_grads(i,j) = 0;
			}
		}
		return Utils<Scalar>::template map_mat_to_tensor<Root::DATA_RANKS>(out_grads, grad_dims);
	}
};

/**
 * A class template representing the logarithmic loss function for multi-label objectives. True
 * labels are expected to have the value 1, while false labels are expected to correspond to the
 * value 0.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class MultiLabelLogLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	MultiLabelLogLoss(Scalar epsilon = Utils<Scalar>::EPSILON2) :
		epsilon(epsilon) { };
protected:
	inline ColVector<Scalar> _function(const typename Root::Data& out,
			const typename Root::Data& obj) const {
		Matrix<Scalar> out_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj);
		ColVector<Scalar> loss(out_mat.rows());
		for (int i = 0; i < out_mat.rows(); ++i) {
			Scalar loss_i = 0;
			for (int j = 0; j < out_mat.cols(); ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert(Utils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						Utils<Scalar>::almost_equal(obj_ij, (Scalar) 1));
				Scalar out_ij = out_mat(i,j);
				loss_i += (obj_ij * log(out_ij) + (1 - obj_ij) * log(1 - out_ij));
			}
			loss(i) = loss_i;
		}
		return loss;
	}
	inline typename Root::Data _d_function(const typename Root::Data& out,
			const typename Root::Data& obj, const Dimensions<int,Root::DATA_RANKS - 1>& grad_dims) const {
		Matrix<Scalar> out_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(out);
		Matrix<Scalar> obj_mat = Utils<Scalar>::template map_tensor_to_mat<Root::DATA_RANKS>(obj);
		int rows = out_mat.rows();
		Matrix<Scalar> out_grads(rows, out_mat.cols());
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < out_grads.cols(); ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert(Utils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						Utils<Scalar>::almost_equal(obj_ij, (Scalar) 1));
				Scalar denominator = out_mat(i,j) - (Scalar) (Utils<Scalar>::almost_equal(obj_ij, (Scalar) 0));
				if (denominator == 0)
					denominator += (rand() % 2 == 0 ? epsilon : -epsilon);
				out_grads(i,j) = 1 / (denominator * rows);
			}
		}
		return Utils<Scalar>::template map_mat_to_tensor<Root::DATA_RANKS>(out_grads, grad_dims);
	}
private:
	Scalar epsilon;
};

//template<typename Scalar, std::size_t Rank>
//class CTCLoss : public Loss<Scalar,Rank,true> {
//	typedef Loss<Scalar,Rank,true> Base;
//public:
//	inline ColVector<Scalar> function(const typename Base::Data& out, const typename Base::Data& obj) const {
//		assert(Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out) ==
//				Utils<Scalar>::template get_dims<Base::DATA_RANKS>(obj));
//
//	}
//	inline typename Base::Data d_function(const typename Base::Data& out, const typename Base::Data& obj) const {
//		Dimensions<int,Base::DATA_RANKS> dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out);
//		assert(dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(obj));
//
//	}
//};

} /* namespace cattle */

#endif /* LOSS_H_ */
