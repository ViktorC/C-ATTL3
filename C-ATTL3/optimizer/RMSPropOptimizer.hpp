/*
 * RMSPropOptimizer.hpp
 *
 *  Created on: 27 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_RMSPROPOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_RMSPROPOPTIMIZER_H_

#include "optimizer/AdaGradOptimizer.hpp"

namespace cattle {

/**
 * A class template for the RMSProp optimizer.
 *
 * \see https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 */
template <typename Scalar, std::size_t Rank, bool Sequential>
class RMSPropOptimizer : public AdaGradOptimizer<Scalar, Rank, Sequential> {
 public:
  /**
   * @param loss A shared pointer to the loss function to use.
   * @param batch_size The batch size to use for training and testing. It is
   * expected to be greater than 0.
   * @param learning_rate The learning rate (a.k.a. step size) to use. It is
   * expected to be greater than 0.
   * @param l2_decay The decay rate of the accumulated squared parameter
   * gradients. It is expected to be in the range [0,1]. The greater it is, the
   * faster the accumulated gradients decay.
   * @param epsilon A small constant used to maintain numerical stability.
   */
  inline RMSPropOptimizer(LossSharedPtr<Scalar, Rank, Sequential> loss, std::size_t batch_size = 1,
                          Scalar learning_rate = 1e-3, Scalar l2_decay = 1e-1,
                          Scalar epsilon = NumericUtils<Scalar>::EPSILON2)
      : AdaGradOptimizer<Scalar, Rank, Sequential>::AdaGradOptimizer(loss, batch_size, learning_rate, epsilon),
        l2_decay(l2_decay) {
    assert(l2_decay >= 0 && l2_decay <= 1);
  }

 protected:
  inline void _update_acc_params_grad_sqrs(Matrix<Scalar>& acc_params_grad_sqrs, const Matrix<Scalar>& params_grad) {
    acc_params_grad_sqrs = acc_params_grad_sqrs * (1 - l2_decay) + params_grad.cwiseProduct(params_grad) * l2_decay;
  }
  const Scalar l2_decay;
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_RMSPROPOPTIMIZER_H_ */
