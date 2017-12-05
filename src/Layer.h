/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "Activation.h"
#include "vector.hpp"
#include "matrix.hpp"

static const double INIT_WEIGHT_ABS_MIN = 5e-3;
static const double INIT_WEIGHT_ABS_MAX = 3e-1;

class Layer {
protected:
	int nodes;
	int prev_nodes;
	double bias;
	viennacl::matrix<double>* weights;
	viennacl::matrix<double>* deltas;
	viennacl::vector<double>* w_in;
	viennacl::vector<double>* out;
	Activation* act;
public:
	Layer* prev;
	Layer* next;
	Layer(int nodes, int prev_nodes, Activation* act, Layer* prev = NULL, Layer* next = NULL);
	virtual ~Layer();
	virtual void initialize_weights();
	virtual void compute_weighted_input(viennacl::vector<double>* prev_out);
	virtual void activate();
	virtual void feed_forward(viennacl::vector<double>* in);
};

enum class Layers {
	Hidden, Output
};

#endif /* LAYER_H_ */
