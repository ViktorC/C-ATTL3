/*
 * ParameterInitialization.hpp
 *
 *  Created on: 13.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_PARAMETERINITIALIZATION_H_
#define CATTL3_PARAMETERINITIALIZATION_H_

#include <cmath>
#include <random>
#include <string>
#include <type_traits>

#include "utils/EigenProxy.hpp"

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

/**
 * A class template for a weight initialization that sets all values to 0.
 */
template<typename Scalar>
class ZeroParameterInitialization : public ParameterInitialization<Scalar> {
public:
	inline void apply(Matrix<Scalar>& weights) const {
		weights.setZero(weights.rows(), weights.cols());
	}
};

/**
 * A class template for a weight initialization that sets all values to 1.
 */
template<typename Scalar>
class OneParameterInitialization : public ParameterInitialization<Scalar> {
public:
	inline void apply(Matrix<Scalar>& weights) const {
		weights.setOnes(weights.rows(), weights.cols());
	}
};

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
	const Scalar max;
};

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
		std::normal_distribution<Scalar> dist(0, sd_scaling_factor * _sd(rows -  1, cols));
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j)
				weights(i,j) = dist(gen);
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
	virtual Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return 1;
	}
private:
	const Scalar sd_scaling_factor;
};

/**
 * An abstract class representing the LeCun weight initialization method.
 *
 * \f$\sigma = c \sqrt{\frac{1}{fan_{in}}}\f$
 *
 * \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
template<typename Scalar>
class LeCunParameterInitialization : public GaussianParameterInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 */
	inline LeCunParameterInitialization(Scalar sd_scaling_factor = 1) :
		GaussianParameterInitialization<Scalar>::GaussianParameterInitialization(sd_scaling_factor) { }
protected:
	inline Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(1.0 / (Scalar) fan_ins);
	}
};

/**
 * An abstract class representing the Xavier/Glorot weight initialization method.
 *
 * \f$\sigma = c \sqrt{\frac{2}{fan_{in} + fan_{out}}}\f$
 *
 * \see http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
 */
template<typename Scalar>
class GlorotParameterInitialization : public GaussianParameterInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 */
	inline GlorotParameterInitialization(Scalar sd_scaling_factor = 1) :
		GaussianParameterInitialization<Scalar>::GaussianParameterInitialization(sd_scaling_factor) { }
protected:
	inline Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(2.0 / (Scalar) (fan_ins + fan_outs));
	}
};

/**
 * An abstract class representing the He weight initialization method.
 *
 * \f$\sigma = c \sqrt{\frac{2}{fan_{in}}}\f$
 *
 * \see https://arxiv.org/abs/1502.01852
 */
template<typename Scalar>
class HeParameterInitialization : public GaussianParameterInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 */
	inline HeParameterInitialization(Scalar sd_scaling_factor = 1) :
		GaussianParameterInitialization<Scalar>::GaussianParameterInitialization(sd_scaling_factor) { }
protected:
	inline Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(2.0 / (Scalar) fan_ins);
	}
};

/**
 * A class template representing the orthogonal weight initialization method.
 *
 * \see https://arxiv.org/abs/1312.6120
 */
template<typename Scalar>
class OrthogonalParameterInitialization : public GaussianParameterInitialization<Scalar> {
public:
	/**
	 * @param sd The standard deviation of the normal distribution to sample from.
	 */
	inline OrthogonalParameterInitialization(Scalar sd = 1) :
			GaussianParameterInitialization<Scalar>::GaussianParameterInitialization(1),
			sd(sd) { }
	inline void apply(Matrix<Scalar>& weights) const {
		GaussianParameterInitialization<Scalar>::apply(weights);
		int rows = weights.rows();
		int cols = weights.cols();
		bool more_rows = rows > cols;
		SVD<Scalar> svd;
		weights.block(0, 0, rows, cols) = more_rows ?
				svd.compute(weights, SVDOptions::ComputeFullU).matrixU().block(0, 0, rows, cols) :
				svd.compute(weights, SVDOptions::ComputeFullV).matrixV().block(0, 0, rows, cols);
	}
protected:
	Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return sd;
	}
private:
	const Scalar sd;
};

} /* namespace cattle */

#endif /* CATTL3_PARAMETERINITIALIZATION_H_ */
