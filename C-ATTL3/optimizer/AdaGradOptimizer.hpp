/*
 * AdaGradOptimizer.hpp
 *
 *  Created on: 27 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_ADAGRADOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_ADAGRADOPTIMIZER_H_

#include "optimizer/SGDOptimizer.hpp"

namespace cattle {

/**
 * A class template for the AdaGrad optimization algorithm.
 *
 * \see http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdaGradOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AdaGradOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar learning_rate = 1e-2, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				learning_rate(learning_rate),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(epsilon > 0);
	}
	virtual ~AdaGradOptimizer() = default;
protected:
	inline void _fit(const std::vector<Parameters<Scalar>*>& params_vec) {
		params_grad_sqrs_vec = std::vector<Matrix<Scalar>>();
		for (auto params_ptr : params_vec) {
			if (params_ptr->are_optimizable() && !params_ptr->are_frozen())
				params_grad_sqrs_vec.push_back(Matrix<Scalar>::Zero(params_ptr->get_rows(), params_ptr->get_cols()));
		}
	}
	inline void _update_params(const std::vector<Parameters<Scalar>*>& params_vec, std::size_t epoch) {
		std::size_t i = 0;
		for (auto params_ptr : params_vec) {
			if (!params_ptr->are_optimizable() || params_ptr->are_frozen())
				continue;
			const Matrix<Scalar>& params_grad = params_ptr->get_grad();
			Matrix<Scalar>& params_grad_sqrs = params_grad_sqrs_vec[i++];
			_update_acc_params_grad_sqrs(params_grad_sqrs, params_grad);
			params_ptr->set_values(params_ptr->get_values() - (params_grad.array() * learning_rate /
					(params_grad_sqrs.array().sqrt() + epsilon)).matrix());
		}
	}
	/**
	 * It updates the accumulated squared parameter gradients.
	 *
	 * @param acc_params_grad_sqrs The accumulated squared parameter gradients.
	 * @param params_grad The new parameter gradients.
	 */
	inline virtual void _update_acc_params_grad_sqrs(Matrix<Scalar>& acc_params_grad_sqrs,
			const Matrix<Scalar>& params_grad) {
		// Accumulate the squares of the gradients.
		acc_params_grad_sqrs += params_grad.cwiseProduct(params_grad);
	}
	const Scalar learning_rate, epsilon;
	std::vector<Matrix<Scalar>> params_grad_sqrs_vec;
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_ADAGRADOPTIMIZER_H_ */
