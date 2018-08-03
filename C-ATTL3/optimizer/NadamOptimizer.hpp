/*
 * NadamOptimizer.hpp
 *
 *  Created on: 27 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_NADAMOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_NADAMOPTIMIZER_H_

#include "optimizer/AdamOptimizer.hpp"

namespace cattle {

/**
 * A class template for the Nesterov accelerated Adam (Nadam) optimization algorithm.
 *
 * \see http://cs229.stanford.edu/proj2015/054_report.pdf
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NadamOptimizer : public AdamOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
	typedef AdamOptimizer<Scalar,Rank,Sequential> Base;
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
	inline NadamOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				Base::AdamOptimizer(loss, batch_size, learning_rate, l1_decay, l2_decay, epsilon) { }
protected:
	inline void _update_params(const std::vector<Parameters<Scalar>*>& params_vec, std::size_t epoch) {
		Scalar l1_corr = (Scalar) 1 / (1 - pow(1 - Base::l1_decay, epoch + 1) + Base::epsilon);
		Scalar l1_next_corr = (Scalar) 1 / (1 - pow(1 - Base::l1_decay, epoch + 2) + Base::epsilon);
		Scalar l2_corr = (Scalar) 1 / (1 - pow(1 - Base::l2_decay, epoch + 1) + Base::epsilon);
		std::size_t i = 0;
		for (auto params_ptr : params_vec) {
			if (!params_ptr->are_optimizable() || params_ptr->are_frozen())
				continue;
			typename Base::ParamsGradNorms& grad_norms = Base::pgn_vec[i++];
			const Matrix<Scalar>& params_grad = params_ptr->get_grad();
			grad_norms.params_grad_l1 = grad_norms.params_grad_l1 * (1 - Base::l1_decay) +
					params_grad * Base::l1_decay;
			grad_norms.params_grad_l2 = grad_norms.params_grad_l2 * (1 - Base::l2_decay) +
					params_grad.cwiseProduct(params_grad) * Base::l2_decay;
			params_ptr->set_values(params_ptr->get_values() -
					((params_grad * (Base::l1_decay * l1_corr) + grad_norms.params_grad_l1 *
					((1 - Base::l1_decay) * l1_next_corr)).array() * Base::learning_rate /
					((grad_norms.params_grad_l2 * l2_corr).array() + Base::epsilon).sqrt()).matrix());
		}
	}
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_NADAMOPTIMIZER_H_ */
