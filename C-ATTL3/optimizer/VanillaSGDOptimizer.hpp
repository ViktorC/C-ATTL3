/*
 * VanillaSGDOptimizer.hpp
 *
 *  Created on: 26 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_VANILLASGDOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_VANILLASGDOPTIMIZER_H_

#include "optimizer/SGDOptimizer.hpp"

namespace cattle {

/**
 * A class template for a vanilla SGD optimizer.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class VanillaSGDOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
	typedef SGDOptimizer<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 */
	inline VanillaSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar learning_rate = 1e-3) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				learning_rate(learning_rate) {
		assert(learning_rate > 0);
	}
protected:
	inline void _fit(std::vector<Parameters<Scalar>*>& params_vec) { }
	inline void _update_params(std::vector<Parameters<Scalar>*>& params_vec, std::size_t epoch) {
		for (auto params_ptr : params_vec) {
			if (!params_ptr->are_frozen())
				params_ptr->set_values(params_ptr->get_values() - params_ptr->get_grad() * learning_rate);
		}
	}
	const Scalar learning_rate;
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_VANILLASGDOPTIMIZER_H_ */
