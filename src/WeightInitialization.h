/*
 * Initialization.h
 *
 *  Created on: 13.12.2017
 *      Author: A6714
 */

#ifndef WEIGHTINITIALIZATION_H_
#define WEIGHTINITIALIZATION_H_

#include <cmath>
#include <Matrix.h>
#include <random>
#include <string>

namespace cppnn {

template<typename Scalar>
class WeightInitialization {
public:
	virtual ~WeightInitialization() = default;
	virtual void apply(Matrix<Scalar>& weights) const = 0;
};

template<typename Scalar>
class ZeroWeightInitialization : public WeightInitialization<Scalar> {
public:
	void apply(Matrix<Scalar>& weights) const {
		weights.setZero(weights.rows(), weights.cols());
	};
};

template<typename Scalar>
class NDRandWeightInitialization : public WeightInitialization<Scalar> {
public:
	NDRandWeightInitialization(Scalar bias_value) :
			bias_value(bias_value) { };
	virtual ~NDRandWeightInitialization() = default;
	virtual void apply(Matrix<Scalar>& weights) const {
		int rows = weights.rows();
		std::default_random_engine gen;
		std::normal_distribution<Scalar> dist(0, sd(rows -  1));
		for (int i = 0; i < rows; i++) {
			if (i == rows - 1) { // Bias row.
				weights.row(i).setConstant(bias_value);
			} else {
				for (int j = 0; j < weights.cols(); j++)
					weights(i,j) = dist(gen);
			}
		}
	};
protected:
	Scalar bias_value;
	virtual Scalar sd(int inputs) const = 0;
};

template<typename Scalar>
class XavierWeightInitialization : public NDRandWeightInitialization<Scalar> {
public:
	XavierWeightInitialization(Scalar bias_value = 0) :
		NDRandWeightInitialization<Scalar>::NDRandWeightInitialization(bias_value) { };
protected:
	Scalar sd(int inputs) const {
		return 1.0 / sqrt((Scalar) inputs);
	};
};

template<typename Scalar>
class ReLUWeightInitialization : public NDRandWeightInitialization<Scalar> {
public:
	ReLUWeightInitialization(Scalar bias_value = 0) :
		NDRandWeightInitialization<Scalar>::NDRandWeightInitialization(bias_value) { };
protected:
	Scalar sd(int inputs) const {
		return sqrt(2.0 / (Scalar) inputs);
	};
};

} /* namespace cppnn */

#endif /* WEIGHTINITIALIZATION_H_ */
