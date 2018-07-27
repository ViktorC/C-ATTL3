/*
 * AdamOptimizer.hpp
 *
 *  Created on: 27 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_ADAMOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_ADAMOPTIMIZER_H_

#include <cmath>

#include "optimizer/SGDOptimizer.hpp"

namespace cattle {

/**
 * A class template for the Adam optimization algorithm.
 *
 * \see https://arxiv.org/abs/1412.6980
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdamOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param l1_decay The decay rate of the accumulated parameter gradients. It is expected
	 * to be in the range [0,1]. The greater it is, the faster the accumulated gradients
	 * decay.
	 * @param l2_decay The decay rate of the accumulated squared parameter gradients. It is
	 * expected to be in the range [0,1]. The greater it is, the faster the accumulated
	 * squared gradients decay.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AdamOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				learning_rate(learning_rate),
				l1_decay(l1_decay),
				l2_decay(l2_decay),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(l1_decay >= 0 && l1_decay <= 1);
		assert(l2_decay >= 0 && l2_decay <= 1);
		assert(epsilon > 0);
	}
	virtual ~AdamOptimizer() = default;
protected:
	inline void _fit(std::vector<Parameters<Scalar>*>& params_vec) {
		pgn_vec = std::vector<ParamsGradNorms>();
		for (auto params_ptr : params_vec) {
			if (params_ptr->are_frozen())
				continue;
			ParamsGradNorms pgn;
			pgn.params_grad_l1 = Matrix<Scalar>::Zero(params_ptr->rows(), params_ptr->cols());
			pgn.params_grad_l2 = Matrix<Scalar>::Zero(params_ptr->rows(), params_ptr->cols());
			pgn_vec.push_back(std::move(pgn));
		}
	}
	inline void _update_params(std::vector<Parameters<Scalar>*>& params_vec, std::size_t epoch) {
		Scalar l1_corr = (Scalar) 1 / (1 - pow(1 - l1_decay, epoch + 1) + epsilon);
		Scalar l2_corr = (Scalar) 1 / (1 - pow(1 - l2_decay, epoch + 1) + epsilon);
		std::size_t i = 0;
		for (auto params_ptr : params_vec) {
			if (params_ptr->are_frozen())
				continue;
			ParamsGradNorms& grad_norms = pgn_vec[i];
			const Matrix<Scalar>& params_grad = params_ptr->get_grad();
			grad_norms.params_grad_l1 = grad_norms.params_grad_l1 * (1 - l1_decay) + params_grad * l1_decay;
			grad_norms.params_grad_l2 = grad_norms.params_grad_l2 * (1 - l2_decay) +
					params_grad.cwiseProduct(params_grad) * l2_decay;
			params_ptr->set_values(params_ptr->get_values() -
					((grad_norms.params_grad_l1 * (learning_rate * l1_corr)).array() /
					((grad_norms.params_grad_l2 * l2_corr).array() + epsilon).sqrt()).matrix());
		}
	}
	const Scalar learning_rate, l1_decay, l2_decay, epsilon;
	/**
	 * A struct containing the moving averages of the first and second norms of the parameter gradients
	 * of a layer.
	 */
	struct ParamsGradNorms {
		Matrix<Scalar> params_grad_l1, params_grad_l2;
	};
	std::vector<ParamsGradNorms> pgn_vec;
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_ADAMOPTIMIZER_H_ */
