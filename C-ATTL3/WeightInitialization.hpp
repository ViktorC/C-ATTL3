/*
 * WeightInitialization.hpp
 *
 *  Created on: 13.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_WEIGHTINITIALIZATION_H_
#define CATTL3_WEIGHTINITIALIZATION_H_

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
class WeightInitialization {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~WeightInitialization() = default;
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
class ZeroWeightInitialization : public WeightInitialization<Scalar> {
public:
	/**
	 * @param bias The value to which the elements of the bias row are to be
	 * set.
	 */
	inline ZeroWeightInitialization(Scalar bias = 0) :
			bias(bias) { }
	inline void apply(Matrix<Scalar>& weights) const {
		weights.setZero(weights.rows(), weights.cols());
		weights.row(weights.rows() - 1).setConstant(bias);
	}
private:
	const Scalar bias;
};

/**
 * A class template for a weight initialization that sets all values to 1.
 */
template<typename Scalar>
class OneWeightInitialization : public WeightInitialization<Scalar> {
public:
	/**
	 * @param bias The value to which the elements of the bias row are to be
	 * set.
	 */
	inline OneWeightInitialization(Scalar bias = 0) :
			bias(bias) { }
	inline void apply(Matrix<Scalar>& weights) const {
		weights.setOnes(weights.rows(), weights.cols());
		weights.row(weights.rows() - 1).setConstant(bias);
	}
private:
	const Scalar bias;
};

/**
 * A weight initializer that assigns linearly increasing values to the elements
 * of the weight matrix. It is meant to be used for testing.
 */
template<typename Scalar>
class IncrementalWeightInitialization : public WeightInitialization<Scalar> {
public:
	/**
	 * @param min The starting value.
	 * @param max The value to assign to the last element.
	 * @param bias The value to which the elements of the bias row are to be
	 * set.
	 */
	inline IncrementalWeightInitialization(Scalar min = 0, Scalar max = .5, Scalar bias = 0) :
			bias(bias),
			min(min),
			max(max) {
		assert(max > min);
	}
	inline void apply(Matrix<Scalar>& weights) const {
		std::size_t rows = weights.rows() - 1;
		std::size_t cols = weights.cols();
		std::size_t elements = (weights.rows() - 1) * weights.cols();
		Scalar step_size = (max - min) / (rows * cols);
		Scalar val = min;
		for (std::size_t i = 0; i < cols; ++i) {
			for (std::size_t j = 0; j < rows; ++j) {
				weights(j,i) = val;
				val += step_size;
			}
		}
		weights.row(rows).setConstant(bias);
	}
private:
	const Scalar min;
	const Scalar max;
	const Scalar bias;
};

/**
 * An abstract class template representing a random weight initialization method
 * that samples from a Gaussian distribution.
 */
template<typename Scalar>
class GaussianWeightInitialization : public WeightInitialization<Scalar> {
public:
	inline GaussianWeightInitialization(Scalar sd_scaling_factor = 1, Scalar bias = 0) :
		bias(bias),
		sd_scaling_factor(sd_scaling_factor) { }
	virtual ~GaussianWeightInitialization() = default;
	inline virtual void apply(Matrix<Scalar>& weights) const {
		int rows = weights.rows();
		int cols = weights.cols();
		std::default_random_engine gen;
		std::normal_distribution<Scalar> dist(0, sd_scaling_factor * _sd(rows -  1, cols));
		for (int i = 0; i < rows; ++i) {
			if (i == rows - 1) // Bias row.
				weights.row(i).setConstant(bias);
			else {
				for (int j = 0; j < cols; ++j)
					weights(i,j) = dist(gen);
			}
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
	const Scalar bias;
};

/**
 * An abstract class representing the LeCun weight initialization method.
 *
 * \f$\sigma = c \sqrt{\frac{1}{fan_{in}}}\f$
 *
 * \see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
 */
template<typename Scalar>
class LeCunWeightInitialization : public GaussianWeightInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 * @param bias The value to which the elements of the bias row are to be set.
	 */
	inline LeCunWeightInitialization(Scalar sd_scaling_factor = 1, Scalar bias = 0) :
		GaussianWeightInitialization<Scalar>::GaussianWeightInitialization(sd_scaling_factor, bias) { }
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
class GlorotWeightInitialization : public GaussianWeightInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 * @param bias The value to which the elements of the bias row are to be set.
	 */
	inline GlorotWeightInitialization(Scalar sd_scaling_factor = 1, Scalar bias = 0) :
		GaussianWeightInitialization<Scalar>::GaussianWeightInitialization(sd_scaling_factor, bias) { }
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
class HeWeightInitialization : public GaussianWeightInitialization<Scalar> {
public:
	/**
	 * @param sd_scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 * @param bias The value to which the elements of the bias row are to be set.
	 */
	inline HeWeightInitialization(Scalar sd_scaling_factor = 1, Scalar bias = 0) :
		GaussianWeightInitialization<Scalar>::GaussianWeightInitialization(sd_scaling_factor, bias) { }
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
class OrthogonalWeightInitialization : public GaussianWeightInitialization<Scalar> {
public:
	/**
	 * @param sd The standard deviation of the normal distribution to sample from.
	 * @param bias The value to which the elements of the bias row are to be
	 * set.
	 */
	inline OrthogonalWeightInitialization(Scalar sd = 1, Scalar bias = 0) :
			GaussianWeightInitialization<Scalar>::GaussianWeightInitialization(1, bias),
			sd(sd) { }
	inline void apply(Matrix<Scalar>& weights) const {
		GaussianWeightInitialization<Scalar>::apply(weights);
		int rows = weights.rows() - 1;
		int cols = weights.cols();
		bool more_rows = rows > cols;
		internal::SVD<Scalar> svd;
		weights.block(0, 0, rows, cols) = more_rows ?
				svd.compute(weights, internal::SVDOptions::ComputeFullU).matrixU().block(0, 0, rows, cols) :
				svd.compute(weights, internal::SVDOptions::ComputeFullV).matrixV().block(0, 0, rows, cols);
	}
protected:
	Scalar _sd(unsigned fan_ins, unsigned fan_outs) const {
		return sd;
	}
private:
	const Scalar sd;
};

} /* namespace cattle */

#endif /* CATTL3_WEIGHTINITIALIZATION_H_ */
