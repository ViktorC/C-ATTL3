/*
 * ElasticNetParameterRegularization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_REGULARIZATION_ELASTICNETPARAMETERREGULARIZATION_H_
#define C_ATTL3_PARAMETER_REGULARIZATION_ELASTICNETPARAMETERREGULARIZATION_H_

#include "core/ParameterRegularization.hpp"

namespace cattle {

/**
 * A class template for the elastic net regularization penalty which is a
 * combination of the L1 and L2 regularization penalties.
 *
 * \f$P = \lambda_1 \sum\limits_{i = 1}^n \left|w_i\right| + \frac{\lambda_2}{2}
 * \sum\limits_{i = 1}^n w_i^2\f$
 *
 * \see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696
 */
template <typename Scalar>
class ElasticNetParameterRegularization : public ParameterRegularization<Scalar> {
 public:
  /**
   * @param l1_lambda The constant by which the L1 penalty is to be scaled.
   * @param l2_lambda The constant by which the L2 penalty is to be scaled.
   */
  inline ElasticNetParameterRegularization(Scalar l1_lambda = 1e-2, Scalar l2_lambda = 1e-2)
      : l1_lambda(l1_lambda), l2_lambda(l2_lambda) {}
  inline Scalar function(const Matrix<Scalar>& params) const {
    return params.array().abs().sum() * l1_lambda + params.squaredNorm() * ((Scalar).5 * l2_lambda);
  }
  inline Matrix<Scalar> d_function(const Matrix<Scalar>& params) const {
    return params.unaryExpr([this](Scalar e) { return e >= 0 ? l1_lambda : -l1_lambda; }) + params * l2_lambda;
  }

 private:
  const Scalar l1_lambda;
  const Scalar l2_lambda;
};

}  // namespace cattle

#endif /* C_ATTL3_PARAMETER_REGULARIZATION_ELASTICNETPARAMETERREGULARIZATION_H_ \
        */
