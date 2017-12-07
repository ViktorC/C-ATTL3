/*
 * Layer.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include <Layer.h>

#include <linalg/prod.hpp>
#include <matrix.hpp>
#include <tools/entry_proxy.hpp>
#include <vector.hpp>
#include <stdexcept>
#include <string>


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
	prev_out_grads = new viennacl::matrix<double>(1, prev_nodes + 1);
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
viennacl::matrix<double>& Layer::get_weights() {
	viennacl::matrix<double>& weight_o = *weights;
	return weight_o;
};
viennacl::matrix<double>& Layer::get_weight_grads() {
	viennacl::matrix<double>& weight_grads_o = *weight_grads;
	return weight_grads_o;
};
viennacl::matrix<double>& Layer::feed_forward(viennacl::matrix<double>& prev_out) {
	for (int i = 0; i < prev_nodes; i++) {
		(*(this->prev_out))(0,i) = prev_out(0,i);
	}
	// Compute the neuron inputs by multiplying the output of the previous layer by the weights.
	*in = viennacl::linalg::prod(*(this->prev_out), *weights);
	// Activate the neurons.
	for (int i = 0; i < nodes; i++) {
		(*out)(0,i) = act->function((*in)(0,i));
	}
	viennacl::matrix<double>& out_ref = *out;
	return out_ref;
};
viennacl::matrix<double>& Layer::feed_back(viennacl::matrix<double>& out_grads) {
	// Compute the gradients of the outputs with respect to the weighted inputs.
	for (int i = 0; i < nodes; i++) {
		(*in_grads)(0,i) = out_grads(0,i) * act->d_function((*in)(0,i), (*out)(0,i));
	}
	*weight_grads = viennacl::linalg::prod(trans(*prev_out), *in_grads);
	*prev_out_grads = viennacl::linalg::prod(*in_grads, trans(*weights));
	viennacl::matrix<double>& prev_out_grads_ref = *prev_out_grads;
	return prev_out_grads_ref;
};
std::string Layer::to_string() {
	std::string str;
	for (unsigned i = 0; i < weights->size1(); i++) {
		for (unsigned j = 0; j < weights->size2(); j++) {
			double w = (*weights)(i,j);
			str += "Weight[" + std::to_string(i) + "," + std::to_string(j) +
					"]: " + std::to_string(w) + "\n";
		}
	}
	return str;
};

}
