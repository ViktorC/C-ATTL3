/*
 * AdaDeltaOptimizer.hpp
 *
 *  Created on: 27 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_ADADELTAOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_ADADELTAOPTIMIZER_H_

#include "optimizer/SGDOptimizer.hpp"

namespace cattle {

/**
 * A class template for the ADADELTA optimization algorithm.
 *
 * \see https://arxiv.org/abs/1212.5701
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdaDeltaOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param decay The decay rate of the accelerated accumulated parameter gradients.
	 * It is expected to be in the range [0,1]. The greater it is, the faster the accumulated
	 * gradients decay.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AdaDeltaOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar decay = 5e-2, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				decay(decay),
				epsilon(epsilon) {
		assert(decay >= 0 && decay <= 1);
		assert(epsilon > 0);
	}
protected:
	inline void _fit(const std::vector<Parameters<Scalar>*>& params_vec) {
		pgus_vec = std::vector<ParamsGradAndUpdateSqrs>();
		for (auto params_ptr : params_vec) {
			if (!params_ptr->are_optimizable() || params_ptr->are_frozen())
				continue;
			ParamsGradAndUpdateSqrs pgus;
			pgus.params_grad = Matrix<Scalar>::Zero(params_ptr->get_rows(), params_ptr->get_cols());
			pgus.params_update = Matrix<Scalar>::Zero(params_ptr->get_rows(), params_ptr->get_cols());
			pgus_vec.push_back(std::move(pgus));
		}
	}
	inline void _update_params(const std::vector<Parameters<Scalar>*>& params_vec, std::size_t epoch, std::size_t timestep) {
		std::size_t i = 0;
		for (auto params_ptr : params_vec) {
			ParamsGradAndUpdateSqrs& pgus = pgus_vec[i++];
			const Matrix<Scalar>& params_grad = params_ptr->get_grad();
			pgus.params_grad = pgus.params_grad * (1 - decay) + params_grad.cwiseProduct(params_grad) * decay;
			Matrix<Scalar> weight_updates = -params_grad.array() * (pgus.params_update.array() + epsilon).sqrt() /
					(pgus.params_grad.array() + epsilon).sqrt();
			params_ptr->set_values(params_ptr->get_values() + weight_updates);
			pgus.params_update = pgus.params_update * (1 - decay) +
					weight_updates.cwiseProduct(weight_updates) * decay;
		}
	}
	const Scalar decay, epsilon;
	/**
	 * A struct containing the moving averages of the squared gradients and squared updates of a layer.
	 */
	struct ParamsGradAndUpdateSqrs {
		Matrix<Scalar> params_grad, params_update;
	};
	std::vector<ParamsGradAndUpdateSqrs> pgus_vec;
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_ADADELTAOPTIMIZER_H_ */
