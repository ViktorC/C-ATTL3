/*
 * ParameterInitialization.hpp
 *
 *  Created on: 13.12.2017
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_PARAMETERINITIALIZATION_H_
#define C_ATTL3_CORE_PARAMETERINITIALIZATION_H_

#include <type_traits>

#include "EigenProxy.hpp"

namespace cattle {

/**
 * An abstract class template for different weight initialization methods for kernel layers.
 */
template<typename Scalar>
class ParameterInitialization {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~ParameterInitialization() = default;
	/**
	 * It initializes the values of the kernel. It is assumed that the last row of the kernel
	 * is the bias row and is thus treated differently.
	 *
	 * @param weights A reference to the weight matrix.
	 */
	virtual void apply(Matrix<Scalar>& weights) const = 0;
};

} /* namespace cattle */

#endif /* C_ATTL3_CORE_PARAMETERINITIALIZATION_H_ */
