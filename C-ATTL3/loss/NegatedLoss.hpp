/*
 * NegatedLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_NEGATEDLOSS_H_
#define C_ATTL3_LOSS_NEGATEDLOSS_H_

#include <cassert>
#include <memory>
#include <utility>

#include "core/Loss.hpp"

namespace cattle {

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

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_NEGATEDLOSS_H_ */
