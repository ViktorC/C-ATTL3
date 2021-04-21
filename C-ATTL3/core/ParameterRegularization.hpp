/*
 * ParameterRegularization.hpp
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_PARAMETERREGULARIZATION_H_
#define C_ATTL3_CORE_PARAMETERREGULARIZATION_H_

#include <type_traits>

#include "EigenProxy.hpp"

namespace cattle {

/**
 * An abstract template class for different regularization penalties for neural
 * network layer parameters. Implementations of this class should be stateless.
 */
template <typename Scalar>
class ParameterRegularization {
  static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");

 public:
  virtual ~ParameterRegularization() = default;
  /**
   * It computes the regularization penalty for the given parameter values.
   *
   * @param params A constant reference to the parameter matrix.
   * @return The regularization penalty as a single scalar.
   */
  virtual Scalar function(const Matrix<Scalar>& params) const;
  /**
   * It differentiates the regularization function and returns its derivative
   * w.r.t. the parameters.
   *
   * @param params A constant reference to the parameter matrix.
   * @return The gradient matrix.
   */
  virtual Matrix<Scalar> d_function(const Matrix<Scalar>& params) const;
};

} /* namespace cattle */

#endif /* C_ATTL3_CORE_PARAMETERREGULARIZATION_H_ */
