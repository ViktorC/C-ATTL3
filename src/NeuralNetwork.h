/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Layer.h"
#include "matrix.hpp"

namespace cppnn {

static const double INIT_WEIGHT_ABS_MIN = 1e-5;
static const double INIT_WEIGHT_ABS_MAX = 2e-1;

class NeuralNetwork {
protected:
	int input_size;
	std::vector<Layer>* layers;
public:
	NeuralNetwork(int input_size, std::vector<Layer>* layers);
	virtual ~NeuralNetwork();
	virtual void initialize_weights();
	viennacl::matrix<double>* feed_forward(viennacl::matrix<double>* input);
	void feed_back(viennacl::matrix<double>* out_grads);
};

}

#endif /* NEURALNETWORK_H_ */
