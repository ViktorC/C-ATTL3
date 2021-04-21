/*
 * IncrementalParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_INCREMENTALPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_INCREMENTALPARAMETERINITIALIZATION_H_

#include "core/ParameterInitialization.hpp"

namespace cattle {

/**
 * A weight initializer that assigns linearly increasing values to the elements
 * of the parameter matrix. It is meant to be used for testing.
 */
template <typename Scalar>
class IncrementalParameterInitialization : public ParameterInitialization<Scalar> {
 public:
  /**
   * @param start The starting value.
   * @param inc The value by which the parameter value is to be incremented.
   */
  inline IncrementalParameterInitialization(Scalar start, Scalar inc) : start(start), inc(inc) {}
  inline void apply(Matrix<Scalar>& params) const {
    Scalar val = start;
    for (std::size_t i = 0; i < params.cols(); ++i) {
      for (std::size_t j = 0; j < params.rows(); ++j) {
        params(j, i) = val;
        val += inc;
      }
    }
  }

 private:
  const Scalar start;
  const Scalar inc;
};

}  // namespace cattle

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_INCREMENTALPARAMETERINITIALIZATION_H_ \
        */
