/*
 * Optimizer.h
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Loss.h>
#include <NeuralNetwork.h>
#include <Regularization.h>
#include <vector>

namespace cppnn {

class Optimizer {
protected:
	Loss* cost;
	Regularization* reg;
public:
	Optimizer(Loss* cost, Regularization* reg);
	virtual ~Optimizer();
	virtual void init_weights(NeuralNetwork& net);
	virtual double calculate_error(NeuralNetwork& net, std::vector<double>& input,
			std::vector<double>& obj);
	virtual void update_weights(NeuralNetwork& net, double error) = 0;
	virtual void train(NeuralNetwork& net,
			std::vector<std::pair<std::vector<double>,std::vector<double>>>& training_data,
			int epochs) = 0;
};

class SGDOptimizer : public Optimizer {
public:
	SGDOptimizer(Loss* cost, Regularization* reg);
	virtual ~SGDOptimizer();
	virtual void update_weights(NeuralNetwork& net, double error);
	virtual void train(NeuralNetwork& net,
			std::vector<std::pair<std::vector<double>,std::vector<double>>>& training_data,
			int epochs);
};

class NadamOptimizer : public Optimizer {
public:
	NadamOptimizer(Loss* cost, Regularization* reg);
	virtual ~NadamOptimizer();
	virtual void update_weights(NeuralNetwork& net, double error);
	virtual void train(NeuralNetwork& net,
			std::vector<std::pair<std::vector<double>,std::vector<double>>>& training_data,
			int epochs);
};

}

#endif /* OPTIMIZER_H_ */
