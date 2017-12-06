/*
 * NeuralNetwork.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include "NeuralNetwork.h"
#include "Layer.h"
#include "matrix.hpp"
#include <stdexcept>

namespace cppnn {

NeuralNetwork::NeuralNetwork(int input_size, std::vector<Layer>* layers) :
		input_size(input_size),
		layers(layers) {
	if (input_size <= 0)
		throw std::invalid_argument("input size must be greater than 0.");
	if (layers == NULL || layers->size() < 0)
		throw std::invalid_argument("layers cannot be null and must contain at least 1 element.");
};
NeuralNetwork::~NeuralNetwork() {
	delete layers;
};
void NeuralNetwork::initialize_weights() {
	std::default_random_engine gen;
	double const abs_dist_Range = INIT_WEIGHT_ABS_MAX / input_size;
	double const sd = abs_dist_Range * .34;
	std::normal_distribution<> normal_distribution(0, sd);
	for (unsigned i = 0; i < layers->size(); i++) {
		viennacl::matrix<double>* weights = (*layers)[i].get_weights();
		for (unsigned j = 0; j < weights->size1(); j++) {
			for (unsigned k = 0; k < weights->size2(); k++) {
				if (j == 0) {
					// Set initial bias value to 0.
					(*weights)(j,k) = 0;
				} else {
					// Initialize weights using normal distribution centered around 0 with a small SD.
					double const rand_weight = normal_distribution(gen);
					double init_weight = rand_weight >= .0 ? std::max(INIT_WEIGHT_ABS_MIN, rand_weight) :
							std::min(-INIT_WEIGHT_ABS_MIN, rand_weight);
					(*weights)(j,k) = init_weight;
				}
			}
		}
	}
};
viennacl::matrix<double>* NeuralNetwork::feed_forward(viennacl::matrix<double>* input) {
	for (unsigned i = 0; i < layers->size(); i++) {
		input = (*layers)[i].feed_forward(input);
	}
	return input;
};
void NeuralNetwork::feed_back(viennacl::matrix<double>* out_grads) {
	for (unsigned i = 0; i < layers->size(); i++) {
		out_grads = (*layers)[i].feed_back(out_grads);
	}
};

}
