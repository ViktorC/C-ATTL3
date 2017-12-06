/*
 * Layer.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include <random>
#include <algorithm>
#include <string>
#include <stdexcept>
#include "Layer.h"
#include "Activation.h"
#include "vector.hpp"
#include "matrix.hpp"
#include "linalg/prod.hpp"

namespace cppnn {

Layer::Layer(int nodes, int prev_nodes, Activation* act) :
		nodes(nodes),
		prev_nodes(prev_nodes),
		act(act) {
	if (act == NULL)
		throw std::invalid_argument("act cannot be null.");
	prev_out = new viennacl::matrix<double>(1, prev_nodes + 1);
	// Bias trick.
	(*prev_out)(0,prev_nodes) = 1;
	prev_out_grads = new viennacl::matrix<double>(1, nodes);
	in = new viennacl::matrix<double>(1, nodes);
	in_grads = new viennacl::matrix<double>(1, nodes);
	out = new viennacl::matrix<double>(1, nodes);
	weights = new viennacl::matrix<double>(prev_nodes + 1, nodes);
	weight_grads = new viennacl::matrix<double>(prev_nodes + 1, nodes);
};
Layer::~Layer() {
	delete prev_out;
	delete prev_out_grads;
	delete in;
	delete in_grads;
	delete out;
	delete weights;
	delete weight_grads;
	delete act;
};
viennacl::matrix<double>* Layer::get_weights() {
	return weights;
};
viennacl::matrix<double>* Layer::get_weight_grads() {
	return weight_grads;
};
viennacl::matrix<double>* Layer::feed_forward(viennacl::matrix<double>* prev_out) {
	viennacl::vector<double> frst_row_src = viennacl::row(*prev_out, 0);
	viennacl::vector<double> frst_row_dst = viennacl::row(*this->prev_out, 0);
	// Compute the neuron inputs by multiplying the output of the previous layer by the weights.
	viennacl::copy(frst_row_src.begin(), frst_row_src.end(), frst_row_dst.begin());
	*in = viennacl::linalg::prod(*weights, *this->prev_out);
	// Activate the neurons.
	for (int i = 0; i < nodes; i++) {
		(*out)(0,i) = act->function((*in)(0,i));
	}
	return out;
};
viennacl::matrix<double>* Layer::feed_back(viennacl::matrix<double>* out_grads) {
	// Compute the gradients of the outputs with respect to the weighted inputs.
	for (int i = 0; i < nodes; i++) {
		(*in_grads)(0,i) = (*out_grads)(0,i) * act->d_function((*in)(0,i), (*out)(0,i));
	}
	*weight_grads = viennacl::linalg::prod(trans(*prev_out), *in_grads);
	*prev_out_grads = viennacl::linalg::prod(trans(*weights), *in_grads);
	return prev_out_grads;
};
std::string Layer::to_string() {
	std::string str;
	for (int i = 0; i < prev_nodes; i++) {
		for (int j = 0; j < nodes; j++) {
			double w = (*weights)(i,j);
			str += "Weight[" + std::to_string(i) + "," + std::to_string(j) +
					"]: " + std::to_string(w) + "\n";
		}
	}
	return str;
};

}
