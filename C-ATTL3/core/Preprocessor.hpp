/*
 * Preprocessor.hpp
 *
 *  Created on: 12.12.2017
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_PREPROCESSOR_H_
#define C_ATTL3_CORE_PREPROCESSOR_H_

#include <cstddef>
#include <type_traits>

#include "EigenProxy.hpp"

namespace cattle {

/**
 * An abstract class template for data preprocessors.
 */
template <typename Scalar, std::size_t Rank, bool Sequential>
class Preprocessor {
  static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
  static_assert(Rank > 0 && Rank < 4, "illegal pre-processor rank");

 public:
  virtual ~Preprocessor() = default;
  /**
   * It fits the preprocessor to the specified data.
   *
   * @param data A constant reference to a data tensor.
   */
  virtual void fit(const Tensor<Scalar, Rank + Sequential + 1>& data) = 0;
  /**
   * It transforms the specified tensor according to the preprocessors current
   * state created by #fit(const Tensor<Scalar,Rank + Sequential + 1>&).
   *
   * @param data A non-constant reference to a data tensor.
   */
  virtual void transform(Tensor<Scalar, Rank + Sequential + 1>& data) const = 0;
};

} /* namespace cattle */

#endif /* C_ATTL3_CORE_PREPROCESSOR_H_ */
