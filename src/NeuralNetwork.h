/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <Layer.h>
#include <matrix.hpp>
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
	virtual void initialize_weights();
	virtual viennacl::matrix<double>* feed_forward(viennacl::matrix<double>* input);
	virtual void feed_back(viennacl::matrix<double>* out_grads);
	virtual std::string to_string();
};

}

#endif /* NEURALNETWORK_H_ */
