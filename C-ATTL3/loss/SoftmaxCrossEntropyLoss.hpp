/*
 * SoftmaxCrossEntropyLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_SOFTMAXCROSSENTROPYLOSS_H_
#define C_ATTL3_LOSS_SOFTMAXCROSSENTROPYLOSS_H_

#include "core/NumericUtils.hpp"
#include "UniversalLoss.hpp"

namespace cattle {

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
	SoftmaxCrossEntropyLoss(Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
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

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_SOFTMAXCROSSENTROPYLOSS_H_ */
