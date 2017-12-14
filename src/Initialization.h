/*
 * Initialization.h
 *
 *  Created on: 13.12.2017
 *      Author: A6714
 */

#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include <cmath>
#include <Matrix.h>
#include <random>

namespace cppnn {

template<typename Scalar>
class Initialization {
public:
	virtual ~Initialization() = default;
	virtual void init(Matrix<Scalar>& weights) const = 0;
};

template<typename Scalar>
class NDRandInitialization : public Initialization<Scalar> {
public:
	NDRandInitialization(Scalar center, Scalar sd, Scalar bias_value) :
			gen(),
			normal_distribution(center, sd),
			bias_value(bias_value) { };
	virtual ~NDRandInitialization() = default;
	virtual void init(Matrix<Scalar>& weights) const {
		int rows = weights.rows();
		Scalar r_factor = range_factor(rows -  1);
		for (int j = 0; j < rows; j++) {
			if (j == rows - 1) { // Bias row.
				weights.row(i).setConstant(bias_value);
			} else {
				for (int k = 0; k < weights.cols(); k++) {
					weights(j,k) = normal_distribution(gen) * r_factor;
				}
			}
		}
	};
protected:
	virtual Scalar range_factor(int inputs) const = 0;
	std::default_random_engine gen;
	std::normal_distribution<Scalar> normal_distribution;
	Scalar bias_value;
};

template<typename Scalar>
class StandardInitialization : public NDRandInitialization<Scalar> {
public:
	StandardInitialization() :
		NDRandInitialization(0, 0.33, 0) { };
	virtual ~StandardInitialization() = default;
protected:
	virtual Scalar range_factor(int inputs) const {
		return 1 / sqrt(inputs);
	};
};

template<typename Scalar>
class ReLUInitialization : public NDRandInitialization<Scalar> {
public:
	ReLUInitialization() :
		NDRandInitialization(0, 0.33, 0) { };
protected:
	Scalar range_factor(int inputs) const {
		return sqrt(2 / inputs);
	};
};

} /* namespace cppnn */

#endif /* INITIALIZATION_H_ */
