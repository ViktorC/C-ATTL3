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

static const double INIT_WEIGHT_ABS_MIN = 1e-5;
static const double INIT_WEIGHT_ABS_MAX = 2e-1;

class Layer {
protected:
	int nodes;
	int in_nodes;
	double bias;
	double bias_grad;
	viennacl::matrix<double>* weights;
	viennacl::matrix<double>* weight_grads;
	viennacl::vector<double>* in;
	viennacl::vector<double>* weighted_in;
	viennacl::vector<double>* weighted_in_grads;
	viennacl::vector<double>* out;
	Activation* act;
public:
	Layer(int nodes, int in_nodes, Activation* act);
	virtual ~Layer();
	virtual double get_bias();
	virtual double get_bias_grad();
	virtual viennacl::matrix<double>* get_weights();
	virtual viennacl::matrix<double>* get_weight_grads();
	virtual viennacl::vector<double>* get_weighted_in_grads();
	virtual viennacl::vector<double>* get_out();
	virtual void initialize_weights();
	virtual void compute_weighted_input(viennacl::vector<double>* prev_out);
	virtual void activate();
	virtual void compute_gradients(viennacl::matrix<double>* subs_weights,
			viennacl::vector<double>* subs_weighted_in_grads);
	virtual std::string to_string();
};

#endif /* LAYER_H_ */
