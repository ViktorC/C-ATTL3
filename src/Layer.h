/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <Activation.h>

#include <matrix.hpp>
#include <string>

namespace cppnn {

class Layer {
protected:
	int nodes;
	int prev_nodes;
	viennacl::matrix<double>* prev_out;
	viennacl::matrix<double>* prev_out_grads;
	viennacl::matrix<double>* in;
	viennacl::matrix<double>* in_grads;
	viennacl::matrix<double>* out;
	viennacl::matrix<double>* weights;
	viennacl::matrix<double>* weight_grads;
	Activation* act;
public:
	Layer(int nodes, int prev_nodes, Activation* act);
	virtual ~Layer();
	virtual viennacl::matrix<double>& get_weights();
	virtual viennacl::matrix<double>& get_weight_grads();
	virtual viennacl::matrix<double>& feed_forward(viennacl::matrix<double>& prev_out);
	virtual viennacl::matrix<double>& feed_back(viennacl::matrix<double>& out_grads);
	virtual std::string to_string();
};

}

#endif /* LAYER_H_ */
