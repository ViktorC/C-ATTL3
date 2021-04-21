/*
 * MultiLabelHingeLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_MULTILABELHINGELOSS_H_
#define C_ATTL3_LOSS_MULTILABELHINGELOSS_H_

#include <algorithm>
#include <cassert>

#include "UniversalLoss.hpp"
#include "core/NumericUtils.hpp"

namespace cattle {

/**
 * A class template representing the hinge loss function for multi-label
 * objectives. True labels are expected to have the value 1, while false labels
 * are expected to correspond to the value -1.
 *
 * \f$L_i = \sum_j \max(0, 1 - {y_i}_j \hat{y_i}_j)\f$ or
 * \f$L_i = \sum_j \max(0, 1 - {y_i}_j \hat{y_i}_j)^2\f$
 */
template <typename Scalar, std::size_t Rank, bool Sequential, bool Squared = false>
class MultiLabelHingeLoss : public UniversalLoss<Scalar, Rank, Sequential> {
  typedef Loss<Scalar, Rank, Sequential> Root;
  typedef UniversalLoss<Scalar, Rank, Sequential> Base;

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
        assert((NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar)-1) ||
                NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar)1)));
        Scalar loss_ij = std::max((Scalar)0, (Scalar)1 - obj_ij * out_mat(i, j));
        loss_i += Squared ? loss_ij * loss_ij : loss_ij;
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
        assert((NumericUtils<Scalar>::almost_equal(obj_ji, (Scalar)-1) ||
                NumericUtils<Scalar>::almost_equal(obj_ji, (Scalar)1)));
        Scalar out_ji = out_mat(j, i);
        Scalar margin = 1 - obj_ji * out_ji;
        if (NumericUtils<Scalar>::decidedly_greater(margin, (Scalar)0))
          out_grad(j, i) = Squared ? 2 * out_ji - 2 * obj_ji : -obj_ji;
        else
          out_grad(j, i) = 0;
      }
    }
    return TensorMap<Scalar, Root::DATA_RANK>(out_grad.data(), grad_dims);
  }
};

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_MULTILABELHINGELOSS_H_ */
