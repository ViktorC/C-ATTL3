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
 * of the weight matrix. It is meant to be used for testing.
 */
template<typename Scalar>
class IncrementalParameterInitialization : public ParameterInitialization<Scalar> {
public:
	/**
	 * @param start The starting value.
	 * @param inc The value by which the parameter value is to be incremented.
	 */
	inline IncrementalParameterInitialization(Scalar start, Scalar inc) :
			start(start),
			inc(inc) { }
	inline void apply(Matrix<Scalar>& weights) const {
		Scalar val = start;
		for (std::size_t i = 0; i < weights.cols(); ++i) {
			for (std::size_t j = 0; j < weights.rows(); ++j) {
				weights(j,i) = val;
				val += inc;
			}
		}
	}
private:
	const Scalar start;
	const Scalar inc;
};

}

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_INCREMENTALPARAMETERINITIALIZATION_H_ */
