/*
 * AbsoluteLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_ABSOLUTELOSS_H_
#define C_ATTL3_LOSS_ABSOLUTELOSS_H_

#include "UniversalLoss.hpp"

namespace cattle {

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

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_ABSOLUTELOSS_H_ */
