/*
 * Loss.hpp
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_LOSS_H_
#define CATTL3_LOSS_H_

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "utils/EigenProxy.hpp"
#include "utils/NumericUtils.hpp"

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
	static constexpr std::size_t DATA_RANK = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
public:
	virtual ~Loss() = default;
	/**
	 * It calculates the error on each sample given the output and the objective tensors.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor (of the same dimensionality as the output).
	 * @return A column vector containing the loss for each sample.
	 */
	virtual ColVector<Scalar> function(Data out, Data obj) const = 0;
	/**
	 * It calculates the derivative of the loss function w.r.t. the output.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor (of the same dimensionality as the output).
	 * @return The derivative of the loss function w.r.t. the output.
	 */
	virtual Data d_function(Data out, Data obj) const = 0;
};


/**
 * An alias for a unique pointer to a loss function of arbitrary rank, scalar type and
 * sequentiality.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using LossSharedPtr = std::shared_ptr<Loss<Scalar,Rank,Sequential>>;

/**
 * A wrapper class template for negating losses and thus allowing for their maximization
 * via the standard optimization methods.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NegatedLoss : public Loss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param loss A shared pointer to the loss instance to negate.
	 */
	NegatedLoss(LossSharedPtr<Scalar,Rank,Sequential> loss) :
			loss(loss) {
		assert(loss);
	}
	inline ColVector<Scalar> function(typename Base::Data out, typename Base::Data obj) const {
		return -(loss->function(std::move(out), std::move(obj)));
	}
	inline typename Base::Data d_function(typename Base::Data out, typename Base::Data obj) const {
		return -(loss->d_function(std::move(out), std::move(obj)));
	}
private:
	LossSharedPtr<Scalar,Rank,Sequential> loss;
};

/**
 * An abstract class template for loss functions for both sequential and non-sequential data.
 * Implementations of this class should be stateless.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class UniversalLoss : public Loss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Base;
protected:
	typedef std::array<std::size_t,Base::DATA_RANK> RankwiseArray;
	/**
	 * It computes the loss of a batch of non-sequential data.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor.
	 * @return A column vector representing the losses of the samples in the batch.
	 */
	virtual ColVector<Scalar> _function(typename Base::Data out, typename Base::Data obj) const = 0;
	/**
	 * It computes the gradient of the output batch.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor.
	 * @param grad_dims The dimensions of the gradient tensor.
	 * @return The gradient tensor of the output batch.
	 */
	virtual typename Base::Data _d_function(typename Base::Data out, typename Base::Data obj,
			const RankwiseArray& grad_dims) const = 0;
public:
	virtual ~UniversalLoss() = default;
	inline ColVector<Scalar> function(typename Base::Data out, typename Base::Data obj) const {
		assert(out.dimensions() == obj.dimensions());
		return _function(std::move(out), std::move(obj));
	}
	inline typename Base::Data d_function(typename Base::Data out, typename Base::Data obj) const {
		assert(out.dimensions() == obj.dimensions());
		RankwiseArray dims = out.dimensions();
		return _d_function(std::move(out), std::move(obj), dims);
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
	typedef std::array<std::size_t,Base::DATA_RANK> RankwiseArray;
	/**
	 * It computes the loss of a single time step in a batch. The total loss of the batch is the sum of the losses
	 * of all its time steps.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor.
	 * @return A column vector representing the losses of the samples in the batch for the given time step.
	 */
	virtual ColVector<Scalar> _function(typename Base::Data out, typename Base::Data obj) const = 0;
	/**
	 * It computes the gradient of a single time step of the output sequence batch.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor.
	 * @param grad_dims The dimensions of the gradient tensor.
	 * @return The gradient tensor of the provided time step of the output batch.
	 */
	virtual typename Base::Data _d_function(typename Base::Data out, typename Base::Data obj,
			const RankwiseArray& grad_dims) const = 0;
public:
	virtual ~UniversalLoss() = default;
	inline ColVector<Scalar> function(typename Base::Data out, typename Base::Data obj) const {
		assert(out.dimensions() == obj.dimensions());
		int time_steps = out.dimension(1);
		if (time_steps == 1)
			return _function(std::move(out), std::move(obj));
		RankwiseArray offsets;
		RankwiseArray extents = out.dimensions();
		offsets.fill(0);
		extents[1] = 1;
		ColVector<Scalar> loss = ColVector<Scalar>::Zero(out.dimension(0), 1);
		for (int i = 0; i < time_steps; ++i) {
			offsets[1] = i;
			typename Base::Data out_i = out.slice(offsets, extents);
			typename Base::Data obj_i = obj.slice(offsets, extents);
			loss += _function(std::move(out_i), std::move(obj_i));
		}
		return loss;
	}
	inline typename Base::Data d_function(const typename Base::Data out, const typename Base::Data obj) const {
		assert(out.dimensions() == obj.dimensions());
		int time_steps = out.dimension(1);
		if (time_steps == 1)
			return _d_function(std::move(out), std::move(obj), out.dimensions());
		RankwiseArray offsets;
		RankwiseArray extents = out.dimensions();
		offsets.fill(0);
		typename Base::Data grads(extents);
		extents[1] = 1;
		grads.setZero();
		for (int i = 0; i < time_steps; ++i) {
			offsets[1] = i;
			typename Base::Data out_i = out.slice(offsets, extents);
			typename Base::Data obj_i = obj.slice(offsets, extents);
			grads.slice(offsets, extents) += _d_function(std::move(out_i), std::move(obj_i), extents);
		}
		return grads;
	}
};

/**
 * A template class representing the absolute error (L1) loss function.
 *
 * \f$L_i = \left|\hat{y_i} - y_i\right|\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AbsoluteLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		return (MatrixMap<Scalar>(out.data(), rows, cols) - MatrixMap<Scalar>(obj.data(), rows, cols))
				.array().abs().rowwise().sum();
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		typename Root::Data diff = out - obj;
		return diff.unaryExpr([this](Scalar i) { return (Scalar) (i >= 0 ? 1 : -1); });
	}
};

/**
 * A template class representing the squared error (L2) loss function.
 *
 * \f$L_i = (\hat{y_i} - y_i)^2\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class SquaredLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		return (MatrixMap<Scalar>(out.data(), rows, cols) - MatrixMap<Scalar>(obj.data(), rows, cols))
				.array().square().rowwise().sum();
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		return 2 * (out - obj);
	}
};

/**
 * A template class representing the hinge loss function. This class assumes the objectives for
 * each sample (and time step) to be a one-hot vector (tensor rank).
 *
 * \f$L_i = \sum_{j \neq y_i} \max(0, \hat{y_i}_j - \hat{y_i}_{y_i} + 1)\f$ or
 * \f$L_i = \sum_{j \neq y_i} \max(0, \hat{y_i}_j - \hat{y_i}_{y_i} + 1)^2\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool Squared = false>
class HingeLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		ColVector<Scalar> loss(rows);
		for (int i = 0; i < rows; ++i) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < cols; ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				if (internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar loss_i = 0;
			Scalar correct_class_score = out_mat(i,correct_class_ind);
			for (int j = 0; j < cols; ++j) {
				if (j == correct_class_ind)
					continue;
				Scalar loss_ij = std::max((Scalar) 0, (Scalar) (out_mat(i,j) - correct_class_score + 1));
				loss_i += Squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		Matrix<Scalar> out_grad(rows, cols);
		for (int i = 0; i < rows; ++i) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < cols; ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				if (internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar total_out_grad = 0;
			Scalar correct_class_score = out_mat(i,correct_class_ind);
			for (int j = 0; j < cols; ++j) {
				if (j == correct_class_ind)
					continue;
				Scalar out_ij = out_mat(i,j);
				Scalar margin = out_ij - correct_class_score + 1;
				if (internal::NumericUtils<Scalar>::decidedly_greater(margin, (Scalar) 0)) {
					Scalar out_grad_ij = Squared ? 2 * margin : 1;
					total_out_grad += out_grad_ij;
					out_grad(i,j) = out_grad_ij;
				} else
					out_grad(i,j) = 0;
			}
			out_grad(i,correct_class_ind) = -total_out_grad;
		}
		return TensorMap<Scalar,Root::DATA_RANK>(out_grad.data(), grad_dims);
	}
};

/**
 * A template class representing the binary cross entropy loss function. The objective
 * is expected to be a size-1 tensor with values in the range [0, 1].
 *
 * \f$L_i = -(y_i \ln(\hat{y_i} + \epsilon) + (1 - y_i) \ln(1 + \epsilon - \hat{y_i}))\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class BinaryCrossEntropyLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	BinaryCrossEntropyLoss(Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
			epsilon(epsilon) { };
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		assert(out.size() == out.dimension(0));
		typename Root::Data loss = -(obj * (out + out.constant(epsilon)).log() +
				(obj.constant(1) - obj) * (out.constant(1 + epsilon) - out).log());
		return MatrixMap<Scalar>(loss.data(), out.dimension(0), 1);
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		assert(out.size() == out.dimension(0));
		return -(obj / (out + out.constant(epsilon)) -
				(obj.constant(1) - obj) / (out.constant(1 + epsilon) - out));
	}
private:
	Scalar epsilon;
};

/**
 * A template class representing the cross entropy loss function. This class assumes the objective
 * values for each sample (and time step) to be in the range [0, 1].
 *
 * \f$L_i = -\ln(\hat{y_i} + \epsilon) y_i\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class CrossEntropyLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	CrossEntropyLoss(Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
			epsilon(epsilon) { };
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		return -((MatrixMap<Scalar>(out.data(), rows, cols).array() + epsilon).log() *
				MatrixMap<Scalar>(obj.data(), rows, cols).array()).matrix().rowwise().sum();
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		return -obj / (out + epsilon);
	}
private:
	Scalar epsilon;
};

/**
 * A loss function template that applies the softmax function to its input before calculating the cross
 * entropy loss. This allows for increased numerical stability and faster computation.
 *
 * \f$L_i = -\ln(\text{softmax}(\hat{y_i}) + \epsilon) y_i\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class SoftmaxCrossEntropyLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	SoftmaxCrossEntropyLoss(Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
			epsilon(epsilon) { };
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		Matrix<Scalar> out_exp = (out_mat.array().colwise() - out_mat.array().rowwise().maxCoeff()).exp();
		return -((out_exp.array().colwise() / (out_exp.array().rowwise().sum() + epsilon)).log() *
				MatrixMap<Scalar>(obj.data(), rows, cols).array()).matrix().rowwise().sum();
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		Matrix<Scalar> out_exp = (out_mat.array().colwise() - out_mat.array().rowwise().maxCoeff()).exp();
		Matrix<Scalar> grads = (out_exp.array().colwise() / (out_exp.array().rowwise().sum() + epsilon)) -
				MatrixMap<Scalar>(obj.data(), rows, cols).array();
		return TensorMap<Scalar,Root::DATA_RANK>(grads.data(), grad_dims);
	}
private:
	Scalar epsilon;
};

/**
 * A template class representing the cross Kullback-Leibler divergence loss function.
 *
 * \f$L_i = -\ln(\frac{-\hat{y_i}}{y_i + \epsilon} + \epsilon) y_i\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class KullbackLeiblerLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	KullbackLeiblerLoss(Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
			epsilon(epsilon) { };
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		return -((MatrixMap<Scalar>(out.data(), rows, cols).array() /
				(obj_mat.array() + epsilon) + epsilon).log() *
				obj_mat.array()).matrix().rowwise().sum();
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		return -obj / (out + epsilon);
	}
private:
	Scalar epsilon;
};

/**
 * A class template representing the hinge loss function for multi-label objectives. True labels
 * are expected to have the value 1, while false labels are expected to correspond to the value -1.
 *
 * \f$L_i = \sum_j \max(0, 1 - {y_i}_j \hat{y_i}_j)\f$ or
 * \f$L_i = \sum_j \max(0, 1 - {y_i}_j \hat{y_i}_j)^2\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool Squared = false>
class MultiLabelHingeLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		ColVector<Scalar> loss(rows);
		for (int i = 0; i < rows; ++i) {
			Scalar loss_i = 0;
			for (int j = 0; j < cols; ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) -1) ||
						internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				Scalar loss_ij = std::max((Scalar) 0, (Scalar) 1 - obj_ij * out_mat(i,j));
				loss_i += Squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		Matrix<Scalar> out_grad(rows, cols);
		for (int i = 0; i < cols; ++i) {
			for (int j = 0; j < rows; ++j) {
				Scalar obj_ji = obj_mat(j,i);
				assert((internal::NumericUtils<Scalar>::almost_equal(obj_ji, (Scalar) -1) ||
						internal::NumericUtils<Scalar>::almost_equal(obj_ji, (Scalar) 1)));
				Scalar out_ji = out_mat(j,i);
				Scalar margin = 1 - obj_ji * out_ji;
				if (internal::NumericUtils<Scalar>::decidedly_greater(margin, (Scalar) 0))
					out_grad(j,i) = Squared ? 2 * out_ji - 2 * obj_ji : -obj_ji;
				else
					out_grad(j,i) = 0;
			}
		}
		return TensorMap<Scalar,Root::DATA_RANK>(out_grad.data(), grad_dims);
	}
};

/**
 * A class template representing the logarithmic loss function for multi-label objectives. True
 * labels are expected to have the value 1, while false labels are expected to correspond to the
 * value 0.
 *
 * \f$L_i = \sum_j {y_i}_j \ln(\hat{y_i}_j + \epsilon) + (1 - {y_i}_j) \ln(1 + \epsilon - \hat{y_i}_j)\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class MultiLabelLogLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	MultiLabelLogLoss(Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
		epsilon(epsilon) { };
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		ColVector<Scalar> loss(rows);
		for (int i = 0; i < rows; ++i) {
			Scalar loss_i = 0;
			for (int j = 0; j < cols; ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert(internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						internal::NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1));
				Scalar out_ij = out_mat(i,j);
				loss_i += obj_ij * log(out_ij + epsilon) + (1 - obj_ij) * log(1 + epsilon - out_ij);
			}
			loss(i) = loss_i;
		}
		return loss;
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		Matrix<Scalar> out_grad(rows, cols);
		for (int i = 0; i < cols; ++i) {
			for (int j = 0; j < rows; ++j) {
				Scalar obj_ji = obj_mat(j,i);
				assert(internal::NumericUtils<Scalar>::almost_equal(obj_ji, (Scalar) 0) ||
						internal::NumericUtils<Scalar>::almost_equal(obj_ji, (Scalar) 1));
				Scalar out_ji = out_mat(j,i);
				Scalar denominator = out_ji * (1 - out_ji);
				if (out_ji == 0)
					out_ji += epsilon;
				out_grad(j,i) = (obj_ji - out_ji) / denominator;
			}
		}
		return TensorMap<Scalar,Root::DATA_RANK>(out_grad.data(), grad_dims);
	}
private:
	Scalar epsilon;
};

} /* namespace cattle */

#endif /* CATTL3_LOSS_H_ */
