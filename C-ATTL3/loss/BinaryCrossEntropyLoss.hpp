/*
 * BinaryCrossEntropyLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_BINARYCROSSENTROPYLOSS_H_
#define C_ATTL3_LOSS_BINARYCROSSENTROPYLOSS_H_

#include <cassert>

#include "UniversalLoss.hpp"
#include "core/NumericUtils.hpp"

namespace cattle {

/**
 * A template class representing the binary cross entropy loss function. The
 * objective is expected to be a size-1 tensor with values in the range [0, 1].
 *
 * \f$L_i = -(y_i \ln(\hat{y_i} + \epsilon) + (1 - y_i) \ln(1 + \epsilon -
 * \hat{y_i}))\f$
 */
template <typename Scalar, std::size_t Rank, bool Sequential>
class BinaryCrossEntropyLoss : public UniversalLoss<Scalar, Rank, Sequential> {
  typedef Loss<Scalar, Rank, Sequential> Root;
  typedef UniversalLoss<Scalar, Rank, Sequential> Base;

 public:
  /**
   * @param epsilon A small constant used to maintain numerical stability.
   */
  BinaryCrossEntropyLoss(Scalar epsilon = NumericUtils<Scalar>::EPSILON2) : epsilon(epsilon){};

 protected:
  inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
    assert(out.size() == out.dimension(0));
    typename Root::Data loss = -(obj * (out + out.constant(epsilon)).log() +
                                 (obj.constant(1) - obj) * (out.constant(1 + epsilon) - out).log());
    return MatrixMap<Scalar>(loss.data(), out.dimension(0), 1);
  }
  inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
                                         const typename Base::RankwiseArray& grad_dims) const {
    assert(out.size() == out.dimension(0));
    return -(obj / (out + out.constant(epsilon)) - (obj.constant(1) - obj) / (out.constant(1 + epsilon) - out));
  }

 private:
  Scalar epsilon;
};

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_BINARYCROSSENTROPYLOSS_H_ */
