/*
 * NesterovMomentumSGDOptimizer.hpp
 *
 *  Created on: 26 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_NESTEROVMOMENTUMSGDOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_NESTEROVMOMENTUMSGDOPTIMIZER_H_

#include "optimizer/MomentumSGDOptimizer.hpp"

namespace cattle {

/**
 * A class template for Nesterov momentum accelerated SGD optimizers.
 *
 * \see https://arxiv.org/abs/1212.0901
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NesterovMomentumSGDOptimizer : public MomentumSGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
	typedef MomentumSGDOptimizer<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param init_learning_rate The initial learning rate (a.k.a. step size) to use. It is
	 * expected to be greater than 0.
	 * @param annealing_rate The rate at which the learning rate is to be annealed. It is
	 * expected to be greater than or equal to 0. The greater it is, the faster the learning
	 * rate decreases.
	 * @param momentum The momentum rate to use. The greater the momentum, the lesser the
	 * effect of newer gradients. It is expected to be greater than 0 and less than 1.
	 */
	inline NesterovMomentumSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss,
			std::size_t batch_size = 1, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3,
			Scalar momentum = .9) :
				Base::MomentumSGDOptimizer(loss, batch_size, init_learning_rate, annealing_rate,
						momentum) { };
protected:
	inline void _update_params(std::vector<Parameters<Scalar>*>& params_vec, std::size_t epoch) {
		Scalar learning_rate = Base::calculate_learning_rate(epoch);
		std::size_t i = 0;
		for (auto params_ptr : params_vec) {
			if (params_ptr->are_frozen())
				continue;
			Matrix<Scalar> old_acc_params_grad = Base::params_grad_vec[i];
			Base::params_grad_vec[i] = old_acc_params_grad * Base::momentum -
					params_ptr->get_grad() * learning_rate;
			params_ptr->set_values(params_ptr->get_values() + old_acc_params_grad * -Base::momentum +
					Base::params_grad_vec[i++] * (1 + Base::momentum));
		}
	}
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_NESTEROVMOMENTUMSGDOPTIMIZER_H_ */
