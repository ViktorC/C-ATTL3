/*
 * UniversalLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_UNIVERSALLOSS_H_
#define C_ATTL3_LOSS_UNIVERSALLOSS_H_

#include <array>
#include <cassert>
#include <utility>

#include "core/Loss.hpp"

namespace cattle {

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

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_UNIVERSALLOSS_H_ */
