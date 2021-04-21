/*
 * MultiLabelLogLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_MULTILABELLOGLOSS_H_
#define C_ATTL3_LOSS_MULTILABELLOGLOSS_H_

#include <cassert>
#include <cmath>

#include "UniversalLoss.hpp"
#include "core/NumericUtils.hpp"

namespace cattle {

/**
 * A class template representing the logarithmic loss function for multi-label
 * objectives. True labels are expected to have the value 1, while false labels
 * are expected to correspond to the value 0.
 *
 * \f$L_i = \sum_j {y_i}_j \ln(\hat{y_i}_j + \epsilon) + (1 - {y_i}_j) \ln(1 +
 * \epsilon - \hat{y_i}_j)\f$
 */
template <typename Scalar, std::size_t Rank, bool Sequential>
class MultiLabelLogLoss : public UniversalLoss<Scalar, Rank, Sequential> {
  typedef Loss<Scalar, Rank, Sequential> Root;
  typedef UniversalLoss<Scalar, Rank, Sequential> Base;

 public:
  /**
   * @param epsilon A small constant used to maintain numerical stability.
   */
  MultiLabelLogLoss(Scalar epsilon = NumericUtils<Scalar>::EPSILON2) : epsilon(epsilon){};

 protected:
  inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
    std::size_t rows = out.dimension(0);
    std::size_t cols = out.size() / rows;
    MatrixMap<Scalar> out_mat(out.data(), rows, cols);
    MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
    ColVector<Scalar> loss(rows);
    for (int i = 0; i < rows; ++i) {
      Scalar loss_i = 0;
      for (int j = 0; j < cols; ++j) {
        Scalar obj_ij = obj_mat(i, j);
        assert(NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar)0) ||
               NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar)1));
        Scalar out_ij = out_mat(i, j);
        loss_i += obj_ij * log(out_ij + epsilon) + (1 - obj_ij) * log(1 + epsilon - out_ij);
      }
      loss(i) = loss_i;
    }
    return loss;
  }
  inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
                                         const typename Base::RankwiseArray& grad_dims) const {
    std::size_t rows = out.dimension(0);
    std::size_t cols = out.size() / rows;
    MatrixMap<Scalar> out_mat(out.data(), rows, cols);
    MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
    Matrix<Scalar> out_grad(rows, cols);
    for (int i = 0; i < cols; ++i) {
      for (int j = 0; j < rows; ++j) {
        Scalar obj_ji = obj_mat(j, i);
        assert(NumericUtils<Scalar>::almost_equal(obj_ji, (Scalar)0) ||
               NumericUtils<Scalar>::almost_equal(obj_ji, (Scalar)1));
        Scalar out_ji = out_mat(j, i);
        Scalar denominator = out_ji * (1 - out_ji);
        if (out_ji == 0) out_ji += epsilon;
        out_grad(j, i) = (obj_ji - out_ji) / denominator;
      }
    }
    return TensorMap<Scalar, Root::DATA_RANK>(out_grad.data(), grad_dims);
  }

 private:
  Scalar epsilon;
};

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_MULTILABELLOGLOSS_H_ */
