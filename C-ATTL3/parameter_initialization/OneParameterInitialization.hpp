/*
 * OneParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_ONEPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_ONEPARAMETERINITIALIZATION_H_

#include "core/ParameterInitialization.hpp"

namespace cattle {

/**
 * A class template for a parameter initialization that sets all values to 1.
 */
template <typename Scalar>
class OneParameterInitialization : public ParameterInitialization<Scalar> {
 public:
  inline void apply(Matrix<Scalar>& params) const { params.setOnes(params.rows(), params.cols()); }
};

}  // namespace cattle

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_ONEPARAMETERINITIALIZATION_H_ */
