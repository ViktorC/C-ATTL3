/*
 * KullbackLeiblerLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_KULLBACKLEIBLERLOSS_H_
#define C_ATTL3_LOSS_KULLBACKLEIBLERLOSS_H_

#include "UniversalLoss.hpp"
#include "core/NumericUtils.hpp"

namespace cattle {

/**
 * A template class representing the cross Kullback-Leibler divergence loss
 * function.
 *
 * \f$L_i = -\ln(\frac{-\hat{y_i}}{y_i + \epsilon} + \epsilon) y_i\f$
 */
template <typename Scalar, std::size_t Rank, bool Sequential>
class KullbackLeiblerLoss : public UniversalLoss<Scalar, Rank, Sequential> {
  typedef Loss<Scalar, Rank, Sequential> Root;
  typedef UniversalLoss<Scalar, Rank, Sequential> Base;

 public:
  /**
   * @param epsilon A small constant used to maintain numerical stability.
   */
  KullbackLeiblerLoss(Scalar epsilon = NumericUtils<Scalar>::EPSILON2) : epsilon(epsilon){};

 protected:
  inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
    std::size_t rows = out.dimension(0);
    std::size_t cols = out.size() / rows;
    MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
    return -((MatrixMap<Scalar>(out.data(), rows, cols).array() / (obj_mat.array() + epsilon) + epsilon).log() *
             obj_mat.array())
                .matrix()
                .rowwise()
                .sum();
  }
  inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
                                         const typename Base::RankwiseArray& grad_dims) const {
    return -obj / (out + epsilon);
  }

 private:
  Scalar epsilon;
};

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_KULLBACKLEIBLERLOSS_H_ */
