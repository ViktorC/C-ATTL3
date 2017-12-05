/*
 * Layer.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include <random>
#include <algorithm>
#include "Layer.h"
#include "Activation.h"
#include "vector.hpp"
#include "matrix.hpp"
#include "linalg/prod.hpp"

Layer::Layer(int nodes, int prev_nodes, Activation* act, Layer* prev, Layer* next) :
		nodes(nodes), prev_nodes(prev_nodes), bias(0), act(act), prev(prev), next(next) {
	weights = new viennacl::matrix<double>(prev_nodes, nodes);
	deltas = new viennacl::matrix<double>(prev_nodes, nodes);
	w_in = new viennacl::vector<double>(nodes);
	out = new viennacl::vector<double>(nodes);
};
Layer::~Layer() {
	delete weights;
	delete deltas;
	delete w_in;
	delete out;
	delete act;
};
void Layer::initialize_weights() {
	std::default_random_engine gen;
	double const abs_dist_Range = INIT_WEIGHT_ABS_MAX / prev_nodes;
	double const sd = abs_dist_Range * 2 / 6;
	std::normal_distribution<> normal_distribution(0, sd);
	for (int i = 0; i < prev_nodes; i++) {
		for (int j = 0; j < nodes; j++) {
			double const rand_weight = normal_distribution(gen);
			double init_weight = std::max(INIT_WEIGHT_ABS_MIN, rand_weight);
			weights(i,j) = init_weight;
		}
	}
}
void Layer::compute_weighted_input(viennacl::vector<double>* prev_out) {
	*w_in = viennacl::linalg::prod(*weights, *prev_out);
};
void Layer::activate() {
	for (int i = 0; i < nodes; i++)
		out[i] = act->function((double) w_in[i]);
};
void Layer::feed_forward(viennacl::vector<double>* in) {
	Layer* layer = this;
	while (layer != NULL) {
		layer->compute_weighted_input(in);
		layer->activate();
		layer = layer->next;
	}
};

int main(int argc, char* argv[]) {
	Layer *l = new Layer(3, 3, get_activation(Activations::Identity));
	l.initialize_weights();
	for (int i = 0; i < l.weights->internal_size1_; i++) {
		for (int j = 0; j < l.weights->internal_size2_; j++) {
			std::cout << "Weight[" << i << "," << j << "]: " << l.weights[i,j] << std::endl;
		}
	}
}
