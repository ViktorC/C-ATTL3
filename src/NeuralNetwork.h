/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <Layer.h>
#include <string>
#include <vector>

namespace cppnn {

static const double INIT_WEIGHT_ABS_MIN = 1e-6;
static const double INIT_WEIGHT_ABS_MAX = 1e-1;

class NeuralNetwork {
protected:
	int input_size;
	std::vector<Layer*>* layers;
public:
	NeuralNetwork(int input_size, std::vector<Layer*>* layers);
	virtual ~NeuralNetwork();
	int get_input_size();
	std::vector<Layer*>& get_layers();
	virtual std::vector<double>* feed_forward(std::vector<double>& input);
	virtual void feed_back(std::vector<double>& out_grads);
	virtual std::string to_string();
};

}

#endif /* NEURALNETWORK_H_ */
