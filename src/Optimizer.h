/*
 * Optimizer.h
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <algorithm>
#include <cmath>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <random>
#include <Regularization.h>

namespace cppnn {

template<typename Scalar>
class Optimizer {
public:
	Optimizer(Loss<Scalar>& loss, Regularization<Scalar>& reg) :
		loss(loss),
		reg(reg) { };
	virtual ~Optimizer();
	virtual void init_weights(NeuralNetwork<Scalar>& net) {
		std::default_random_engine gen;
		for (unsigned i = 0; i < net.layers.size(); i++) {
			Matrix<double>& weights = net.layers[i].weights;
			unsigned rows = weights.rows();
			unsigned cols = weights.cols();
			double const abs_dist_Range = INIT_WEIGHT_ABS_MAX / sqrt(2 / rows);
			double const sd = abs_dist_Range * .34;
			std::normal_distribution<> normal_distribution(0, sd);
			for (unsigned j = 0; j < rows; j++) {
				for (unsigned k = 0; k < cols; k++) {
					if (j == rows - 1) {
						// Set initial bias value to 0.
						weights(j,k) = 0;
					} else {
						/* Initialize weights using normal distribution centered
						 * around 0 with a small SD. */
						double const rand_weight = normal_distribution(gen);
						double init_weight = rand_weight >= .0 ?
								std::max(INIT_WEIGHT_ABS_MIN, rand_weight) :
								std::min(-INIT_WEIGHT_ABS_MIN, rand_weight);
						weights(j,k) = init_weight;
					}
				}
			}
		}
	}
	virtual void update_weights(NeuralNetwork<Scalar>& net, Scalar error) = 0;
	virtual void train(NeuralNetwork<Scalar>& net, Matrix<Scalar>& x,
			Matrix<Scalar>& y, unsigned epochs) = 0;
protected:
	static constexpr Scalar INIT_WEIGHT_ABS_MIN = 1e-6;
	static constexpr Scalar INIT_WEIGHT_ABS_MAX = 1e-1;
	Loss<Scalar>& loss;
	Regularization<Scalar>& reg;
};

template<typename Scalar>
class SGDOptimizer : public Optimizer<Scalar> {
public:
	SGDOptimizer(Loss<Scalar>& loss, Regularization<Scalar>& reg);
	virtual ~SGDOptimizer();
	virtual void update_weights(NeuralNetwork<Scalar>& net, Scalar error);
	virtual void train(NeuralNetwork<Scalar>& net, Matrix<Scalar>& x,
				Matrix<Scalar>& y, unsigned epochs);
};

template<typename Scalar>
class NadamOptimizer : public Optimizer<Scalar> {
public:
	NadamOptimizer(Loss<Scalar>& loss, Regularization<Scalar>& reg);
	virtual ~NadamOptimizer();
	virtual void update_weights(NeuralNetwork<Scalar>& net, Scalar error);
	virtual void train(NeuralNetwork<Scalar>& net, Matrix<Scalar>& x,
					Matrix<Scalar>& y, unsigned epochs);
};

} /* namespace cppnn */

#endif /* OPTIMIZER_H_ */
