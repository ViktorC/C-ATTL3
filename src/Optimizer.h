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
#include <Initialization.h>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <random>
#include <Regularization.h>

namespace cppnn {

template<typename Scalar>
class Optimizer {
public:
	Optimizer(const Initialization<Scalar>& init, const Regularization<Scalar>& reg,
			const Loss<Scalar>& loss) :
				init(init),
				loss(loss),
				reg(reg) { };
	virtual ~Optimizer() = default;
	virtual void init_weights(NeuralNetwork<Scalar>& net) {
		std::default_random_engine gen;
		for (unsigned i = 0; i < net.layers.size(); i++) {
			Matrix<Scalar>& weights = net.layers[i].weights;
			init.init(weights);
		}
	};
	virtual void update_weights(NeuralNetwork<Scalar>& net, Scalar error) = 0;
	virtual void train(NeuralNetwork<Scalar>& net, Matrix<Scalar>& x,
			Matrix<Scalar>& y, unsigned epochs) = 0;
protected:
	const Initialization<Scalar>& init;
	const Regularization<Scalar>& reg;
	const Loss<Scalar>& loss;
};

template<typename Scalar>
class SGDOptimizer : public Optimizer<Scalar> {
public:
	SGDOptimizer(const Initialization<Scalar>& init, const Regularization<Scalar>& reg,
			const Loss<Scalar>& loss) :
				Optimizer(init, reg, loss) {

	};
	virtual ~SGDOptimizer();
	virtual void update_weights(NeuralNetwork<Scalar>& net, Scalar error);
	virtual void train(NeuralNetwork<Scalar>& net, Matrix<Scalar>& x,
				Matrix<Scalar>& y, unsigned epochs);
};

template<typename Scalar>
class NadamOptimizer : public Optimizer<Scalar> {
public:
	NadamOptimizer(const Initialization<Scalar>& init, const Regularization<Scalar>& reg,
			const Loss<Scalar>& loss) :
				Optimizer(init, reg, loss) {

	};
	virtual ~NadamOptimizer();
	virtual void update_weights(NeuralNetwork<Scalar>& net, Scalar error);
	virtual void train(NeuralNetwork<Scalar>& net, Matrix<Scalar>& x,
					Matrix<Scalar>& y, unsigned epochs);
};

} /* namespace cppnn */

#endif /* OPTIMIZER_H_ */
