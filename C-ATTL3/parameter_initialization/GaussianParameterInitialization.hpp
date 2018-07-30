/*
 * GaussianParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_GAUSSIANPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_GAUSSIANPARAMETERINITIALIZATION_H_

#include <random>

#include "core/ParameterInitialization.hpp"

namespace cattle {

/**
 * An abstract class template representing a random weight initialization method
 * that samples from a Gaussian distribution.
 */
template<typename Scalar>
class GaussianParameterInitialization : public ParameterInitialization<Scalar> {
public:
	inline GaussianParameterInitialization(Scalar sd_scaling_factor = 1) :
		sd_scaling_factor(sd_scaling_factor) { }
	virtual ~GaussianParameterInitialization() = default;
	inline virtual void apply(Matrix<Scalar>& weights) const {
		int rows = weights.rows();
		int cols = weights.cols();
		std::default_random_engine gen;
		std::normal_distribution<Scalar> dist(0, sd_scaling_factor * _sd(rows, cols));
		for (int i = 0; i < cols; ++i) {
			for (int j = 0; j < rows; ++j)
				weights(j,i) = dist(gen);
		}
	}
protected:
	/**
	 * It computes the standard deviation of the distribution to sample from.
	 *
	 * @param fan_ins The input size of the kernel (the number of rows in the weight matrix
	 * excluding the bias row).
	 * @param fan_outs The output size of the kernel (the number of columns in the weight
	 * matrix).
	 * @return The standard deviation of the normal distribution from which the values
	 * of the initialized weight matrix are to be sampled.
	 */
	inline virtual Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return 1;
	}
private:
	const Scalar sd_scaling_factor;
};

}

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_GAUSSIANPARAMETERINITIALIZATION_H_ */
