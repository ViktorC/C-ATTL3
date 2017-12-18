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
	Optimizer(const Regularization<Scalar>& reg, const Loss<Scalar>& loss) :
			reg(reg),
			loss(loss) { };
	virtual ~Optimizer() = default;
	virtual void train(NeuralNetwork<Scalar>& net, Matrix<Scalar>& x,
			Matrix<Scalar>& y, unsigned epochs, unsigned batch_size) = 0;
	virtual bool check_gradient(NeuralNetwork<Scalar>& net, Matrix<Scalar>& x,
			Matrix<Scalar>& y, unsigned max_params_to_check, Scalar step_size,
			Scalar max_rel_diff) const = 0;
protected:
	const Regularization<Scalar>& reg;
	const Loss<Scalar>& loss;
};

template<typename Scalar>
class SGDOptimizer : public Optimizer<Scalar> {
public:
	SGDOptimizer(const Regularization<Scalar>& reg, const Loss<Scalar>& loss) :
				Optimizer(reg, loss) {

	};
};

template<typename Scalar>
class NadamOptimizer : public Optimizer<Scalar> {
public:
	NadamOptimizer(const Regularization<Scalar>& reg, const Loss<Scalar>& loss) :
				Optimizer(reg, loss) {

	};
};

} /* namespace cppnn */

#endif /* OPTIMIZER_H_ */
