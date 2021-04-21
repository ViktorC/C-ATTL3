/*
 * MomentumSGDOptimizer.hpp
 *
 *  Created on: 26 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_MOMENTUMSGDOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_MOMENTUMSGDOPTIMIZER_H_

#include "optimizer/SGDOptimizer.hpp"

namespace cattle {

/**
 * A class template for a momentum accelerated SGD optimizer.
 */
template <typename Scalar, std::size_t Rank, bool Sequential>
class MomentumSGDOptimizer : public SGDOptimizer<Scalar, Rank, Sequential> {
  typedef Optimizer<Scalar, Rank, Sequential> Root;
  typedef SGDOptimizer<Scalar, Rank, Sequential> Base;

 public:
  /**
   * @param loss A shared pointer to the loss function to use.
   * @param batch_size The batch size to use for training and testing. It is
   * expected to be greater than 0.
   * @param init_learning_rate The initial learning rate (a.k.a. step size) to
   * use. It is expected to be greater than 0.
   * @param annealing_rate The rate at which the learning rate is to be
   * annealed. It is expected to be greater than or equal to 0. The greater it
   * is, the faster the learning rate decreases.
   * @param momentum The momentum rate to use. The greater the momentum, the
   * lesser the effect of newer gradients. It is expected to be greater than 0
   * and less than 1.
   */
  inline MomentumSGDOptimizer(LossSharedPtr<Scalar, Rank, Sequential> loss, std::size_t batch_size = 1,
                              Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3, Scalar momentum = .9)
      : SGDOptimizer<Scalar, Rank, Sequential>::SGDOptimizer(loss, batch_size),
        init_learning_rate(init_learning_rate),
        annealing_rate(annealing_rate),
        momentum(momentum) {
    assert(init_learning_rate > 0);
    assert(annealing_rate >= 0);
    assert(momentum > 0 && momentum < 1);
  }
  virtual ~MomentumSGDOptimizer() = default;

 protected:
  inline void _fit(const std::vector<Parameters<Scalar>*>& params_vec) {
    params_grad_vec = std::vector<Matrix<Scalar>>();
    for (auto params_ptr : params_vec) {
      if (params_ptr->are_optimizable() && !params_ptr->are_frozen())
        params_grad_vec.push_back(Matrix<Scalar>::Zero(params_ptr->get_rows(), params_ptr->get_cols()));
    }
  }
  inline void _update_params(const std::vector<Parameters<Scalar>*>& params_vec, std::size_t epoch,
                             std::size_t timestep) {
    Scalar learning_rate = calculate_learning_rate(epoch);
    std::size_t i = 0;
    for (auto params_ptr : params_vec) {
      params_grad_vec[i] = params_grad_vec[i] * momentum + params_ptr->get_grad() * learning_rate;
      params_ptr->set_values(params_ptr->get_values() - params_grad_vec[i++]);
    }
  }
  /**
   * It calculates the annealed learning rate as a function of the epoch index.
   *
   * @param epoch The epoch index.
   * @return The learning rate to use.
   */
  Scalar calculate_learning_rate(std::size_t epoch) { return init_learning_rate / (1 + annealing_rate * epoch); }
  const Scalar init_learning_rate, annealing_rate, momentum;
  std::vector<Matrix<Scalar>> params_grad_vec;
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_MOMENTUMSGDOPTIMIZER_H_ */
