/*
 * Loss.hpp
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_LOSS_H_
#define C_ATTL3_CORE_LOSS_H_

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "EigenProxy.hpp"
#include "NumericUtils.hpp"

namespace cattle {

/**
 * An abstract class template for loss functions. Implementations of this class should be stateless.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class Loss {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal loss rank");
protected:
	static constexpr std::size_t DATA_RANK = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
public:
	virtual ~Loss() = default;
	/**
	 * It calculates the error on each sample given the output and the objective tensors.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor (of the same dimensionality as the output).
	 * @return A column vector containing the loss for each sample.
	 */
	virtual ColVector<Scalar> function(Data out, Data obj) const = 0;
	/**
	 * It calculates the derivative of the loss function w.r.t. the output.
	 *
	 * @param out The output tensor.
	 * @param obj The objective tensor (of the same dimensionality as the output).
	 * @return The derivative of the loss function w.r.t. the output.
	 */
	virtual Data d_function(Data out, Data obj) const = 0;
};

} /* namespace cattle */

#endif /* C_ATTL3_CORE_LOSS_H_ */
