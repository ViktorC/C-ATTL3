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
 * An abstract class template representing a random weight initialization method
 * that samples from a Gaussian distribution.
 */
template<typename Scalar>
class GaussianWeightInitialization : public WeightInitialization<Scalar> {
public:
	inline GaussianWeightInitialization(Scalar scaling_factor, Scalar bias) :
		bias(bias),
		scaling_factor(scaling_factor) { }
	virtual ~GaussianWeightInitialization() = default;
	inline virtual void apply(Matrix<Scalar>& weights) const {
		int rows = weights.rows();
		int cols = weights.cols();
		std::default_random_engine gen;
		std::normal_distribution<Scalar> dist(0, scaling_factor * sd(rows -  1, cols));
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
	virtual Scalar sd(unsigned fan_ins, unsigned fan_outs) const = 0;
private:
	const Scalar scaling_factor;
	const Scalar bias;
};

/**
 * An abstract class representing the LeCun weight initialization method.
 */
template<typename Scalar>
class LeCunWeightInitialization : public GaussianWeightInitialization<Scalar> {
public:
	/**
	 * @param scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 * @param bias The value to which the elements of the bias row are to be set.
	 */
	inline LeCunWeightInitialization(Scalar scaling_factor = 1, Scalar bias = 0) :
		GaussianWeightInitialization<Scalar>::GaussianWeightInitialization(scaling_factor, bias) { }
protected:
	inline Scalar sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(1.0 / (Scalar) fan_ins);
	}
};

/**
 * An abstract class representing the Xavier/Glorot weight initialization method.
 */
template<typename Scalar>
class GlorotWeightInitialization : public GaussianWeightInitialization<Scalar> {
public:
	/**
	 * @param scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 * @param bias The value to which the elements of the bias row are to be set.
	 */
	inline GlorotWeightInitialization(Scalar scaling_factor = 1, Scalar bias = 0) :
		GaussianWeightInitialization<Scalar>::GaussianWeightInitialization(scaling_factor, bias) { }
protected:
	inline Scalar sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(2.0 / (Scalar) (fan_ins + fan_outs));
	}
};

/**
 * An abstract class representing the He weight initialization method.
 */
template<typename Scalar>
class HeWeightInitialization : public GaussianWeightInitialization<Scalar> {
public:
	/**
	 * @param scaling_factor The value by which the randomly initialized weights
	 * are to be scaled.
	 * @param bias The value to which the elements of the bias row are to be set.
	 */
	inline HeWeightInitialization(Scalar scaling_factor = 1, Scalar bias = 0) :
		GaussianWeightInitialization<Scalar>::GaussianWeightInitialization(scaling_factor, bias) { }
protected:
	inline Scalar sd(unsigned fan_ins, unsigned fan_outs) const {
		return sqrt(2.0 / (Scalar) fan_ins);
	}
};

/**
 * A class template representing the orthogonal weight initialization method.
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
			_sd(sd) { }
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
	Scalar sd(unsigned fan_ins, unsigned fan_outs) const {
		return _sd;
	}
private:
	const Scalar _sd;
};

} /* namespace cattle */

#endif /* CATTL3_WEIGHTINITIALIZATION_H_ */
