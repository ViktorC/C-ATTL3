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
#include <string>

namespace cppnn {

template<typename Scalar>
class Initialization {
public:
	virtual ~Initialization() = default;
	virtual void init(Matrix<Scalar>& weights) const = 0;
	virtual std::string to_string() const = 0;
};

template<typename Scalar>
class NDRandInitialization : public Initialization<Scalar> {
public:
	NDRandInitialization(Scalar bias_value) :
			bias_value(bias_value) { };
	virtual ~NDRandInitialization() = default;
	virtual void init(Matrix<Scalar>& weights) const {
		int rows = weights.rows();
		std::default_random_engine gen;
		std::normal_distribution<Scalar> dist(0, sd(rows -  1));
		for (int i = 0; i < rows; i++) {
			if (i == rows - 1) { // Bias row.
				weights.row(i).setConstant(bias_value);
			} else {
				for (int j = 0; j < weights.cols(); j++) {
					weights(i,j) = dist(gen);
				}
			}
		}
	};
protected:
	Scalar bias_value;
	virtual Scalar sd(int inputs) const = 0;
};

template<typename Scalar>
class XavierInitialization : public NDRandInitialization<Scalar> {
public:
	XavierInitialization(Scalar bias_value = 0) :
		NDRandInitialization<Scalar>::NDRandInitialization(bias_value) { };
	std::string to_string() const {
		return "xavier ND; bias: " + std::to_string(
				NDRandInitialization<Scalar>::bias_value);
	};
protected:
	Scalar sd(int inputs) const {
		return 1 / sqrt(inputs);
	};
};

template<typename Scalar>
class ReLUInitialization : public NDRandInitialization<Scalar> {
public:
	ReLUInitialization(Scalar bias_value = 0) :
		NDRandInitialization<Scalar>::NDRandInitialization(bias_value) { };
	std::string to_string() const {
		return "relu ND; bias: " + std::to_string(
				NDRandInitialization<Scalar>::bias_value);
	};
protected:
	Scalar sd(int inputs) const {
		return sqrt(2 / inputs);
	};
};

} /* namespace cppnn */

#endif /* INITIALIZATION_H_ */
