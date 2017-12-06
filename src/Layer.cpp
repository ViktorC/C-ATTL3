/*
 * Layer.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include <random>
#include <algorithm>
#include <string>
#include <iostream>
#include "Layer.h"
#include "Activation.h"
#include "vector.hpp"
#include "matrix.hpp"
#include "linalg/prod.hpp"

Layer::Layer(int nodes, int in_nodes, Activation* act) :
		nodes(nodes),
		in_nodes(in_nodes),
		bias(0),
		bias_grad(0),
		act(act) {
	weights = new viennacl::matrix<double>(in_nodes, nodes);
	weight_grads = new viennacl::matrix<double>(in_nodes, nodes);
	in = NULL;
	weighted_in = new viennacl::vector<double>(nodes);
	weighted_in_grads = new viennacl::vector<double>(nodes);
	out = new viennacl::vector<double>(nodes);
};
Layer::~Layer() {
	delete weights;
	delete weight_grads;
	delete in;
	delete weighted_in;
	delete weighted_in_grads;
	delete out;
	delete act;
};
double Layer::get_bias() {
	return bias;
};
double Layer::get_bias_grad() {
	return bias_grad;
};
viennacl::matrix<double>* Layer::get_weights() {
	return weights;
};
viennacl::matrix<double>* Layer::get_weight_grads() {
	return weight_grads;
};
viennacl::vector<double>* Layer::get_weighted_in_grads() {
	return weighted_in_grads;
};
viennacl::vector<double>* Layer::get_out() {
	return out;
};
void Layer::initialize_weights() {
	std::default_random_engine gen;
	double const abs_dist_Range = INIT_WEIGHT_ABS_MAX / in_nodes;
	double const sd = abs_dist_Range * .34;
	std::normal_distribution<> normal_distribution(0, sd);
	for (int i = 0; i < in_nodes; i++) {
		for (int j = 0; j < nodes; j++) {
			double const rand_weight = normal_distribution(gen);
			double init_weight = rand_weight >= .0 ? std::max(INIT_WEIGHT_ABS_MIN, rand_weight) :
					std::min(-INIT_WEIGHT_ABS_MIN, rand_weight);
			(*weights)(i,j) = init_weight;
		}
	}
}
void Layer::compute_weighted_input(viennacl::vector<double>* prev_out) {
	in = prev_out;
	*weighted_in = viennacl::linalg::prod(*weights, *in);
	for (int i = 0; i < nodes; i++) {
		(*weighted_in)[i] += bias;
	}
};
void Layer::activate() {
	for (int i = 0; i < nodes; i++)
		(*out)(i) = act->function((*weighted_in)(i));
};
void Layer::compute_gradients(viennacl::matrix<double>* subs_weights,
		viennacl::vector<double>* subs_weighted_in_gradss) {
	viennacl::vector<double> out_grads = viennacl::linalg::prod(*subs_weights, *subs_weighted_in_gradss);
	for (int i = 0; i < nodes; i++) {
		(*weighted_in_grads)[i] = out_grads[i] * act->d_function((*weighted_in)[i], (*out)[i]);
	}

};
std::string Layer::to_string() {
	std::string str = "Bias: " + std::to_string(bias);
	for (int i = 0; i < in_nodes; i++) {
		for (int j = 0; j < nodes; j++) {
			double w = (*weights)(i,j);
			str += "Weight[" + std::to_string(i) + "," + std::to_string(j) +
					"]: " + std::to_string(w) + "\n";
		}
	}
	return str;
}

int main(int argc, char* argv[]) {
	Layer *l = new Layer(3, 3, get_activation(Activations::Sigmoid));
	std::cout << l->to_string();
	l->initialize_weights();
	std::cout << l->to_string();
	viennacl::vector<double> in(3);
	in[0] = 7.12;
	in[1] = -2.003;
	in[2] = 0.034;
	l->compute_weighted_input(&in);
	std::cout << l->to_string();
	l->activate();
	std::cout << l->to_string();
}
