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
	NDRandInitialization(Scalar center, Scalar sd, Scalar bias_value) :
			center(center),
			sd(sd),
			bias_value(bias_value) { };
	virtual ~NDRandInitialization() = default;
	virtual void init(Matrix<Scalar>& weights) const {
		std::default_random_engine gen;
		std::normal_distribution<Scalar> dist(center, sd);
		int rows = weights.rows();
		Scalar r_factor = range_factor(rows -  1);
		for (int i = 0; i < rows; i++) {
			if (i == rows - 1) { // Bias row.
				weights.row(i).setConstant(bias_value);
			} else {
				for (int j = 0; j < weights.cols(); j++) {
					weights(i,j) = dist(gen) * r_factor;
				}
			}
		}
	};
protected:
	Scalar center;
	Scalar sd;
	Scalar bias_value;
	virtual Scalar range_factor(int inputs) const = 0;
};

template<typename Scalar>
class StandardInitialization : public NDRandInitialization<Scalar> {
public:
	StandardInitialization() :
		NDRandInitialization<Scalar>::NDRandInitialization(0, .33, 0) { };
	std::string to_string() const {
		return "standard ND; center: .0; sd: 0.33; bias: .0";
	};
protected:
	Scalar range_factor(int inputs) const {
		return 1 / sqrt(inputs);
	};
};

template<typename Scalar>
class ReLUInitialization : public NDRandInitialization<Scalar> {
public:
	ReLUInitialization() :
		NDRandInitialization<Scalar>::NDRandInitialization(0, .33, 0) { };
	std::string to_string() const {
		return "relu ND; center: .0; sd: 0.33; bias: .0";
	};
protected:
	Scalar range_factor(int inputs) const {
		return sqrt(2 / inputs);
	};
};

} /* namespace cppnn */

#endif /* INITIALIZATION_H_ */
