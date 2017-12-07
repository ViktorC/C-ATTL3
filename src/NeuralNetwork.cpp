/*
 * NeuralNetwork.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include <NeuralNetwork.h>

#include <assert.h>
#include <detail/matrix_def.hpp>
#include <stddef.h>
#include <tools/entry_proxy.hpp>
#include <algorithm>
#include <random>
#include <vector>


namespace cppnn {

NeuralNetwork::NeuralNetwork(int input_size, std::vector<Layer*>* layers) :
		input_size(input_size),
		layers(layers) {
	if (input_size <= 0)
		throw std::invalid_argument("input size must be greater than 0.");
	if (layers == NULL || layers->size() < 0)
		throw std::invalid_argument("layers cannot be null and must contain at least 1 element.");
};
NeuralNetwork::~NeuralNetwork() {
	for (unsigned i = 0; i < layers->size(); i++) {
		delete (*layers)[i];
	}
	delete layers;
};
int NeuralNetwork::get_input_size() {
	return input_size;
}
std::vector<Layer*>& NeuralNetwork::get_layers() {
	std::vector<Layer*>& layers_ref = *layers;
	return layers_ref;
}
std::vector<double>* NeuralNetwork::feed_forward(std::vector<double>& input) {
	viennacl::matrix<double> matrix(0, input.size());
	for (unsigned i = 0; i < input.size(); i++) {
		matrix(0,i) = input[i];
	}
	viennacl::matrix<double>* input_ptr = &matrix;
	for (unsigned i = 0; i < layers->size(); i++) {
		viennacl::matrix<double>& input_ref = *input_ptr;
		input_ptr = &((*layers)[i]->feed_forward(input_ref));
	}
	std::vector<double> out = new std::vector<double>(input_ptr->size2());
	for (unsigned i = 0; i < input_ptr->size2(); i++) {
		out[i] = (*input_ptr)(0,i);
	}
	return out;
};
void NeuralNetwork::feed_back(std::vector<double>& out_grads) {
	viennacl::matrix<double> matrix(0, out_grads.size());
	for (unsigned i = 0; i < out_grads.size(); i++) {
		matrix(0,i) = out_grads[i];
	}
	viennacl::matrix<double>* out_grads_ptr = &matrix;
	for (int i = (int) layers->size() - 1; i >= 0; i--) {
		viennacl::matrix<double>& out_grads_ref = *out_grads_ptr;
		out_grads_ptr = &((*layers)[i]->feed_back(out_grads_ref));
	}
};
std::string NeuralNetwork::to_string() {
	std::string str = "<NN>\n";
	for (unsigned i = 0; i < layers->size(); i++) {
		str += "Layer " + std::to_string(i) + "--------------------\n";
		str += (*layers)[i]->to_string();
	}
	str += "</NN>\n";
	return str;
};

}
