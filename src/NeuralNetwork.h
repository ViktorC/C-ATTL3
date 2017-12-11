/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <Layer.h>
#include <string>
#include <vector>

namespace cppnn {

template <typename Scalar>
class NeuralNetwork {
public:
	NeuralNetwork(int input_size, std::vector<Layer<Scalar>> layers) :
			input_size(input_size),
			layers(layers) {
		assert(input_size > 0 && "input size must be greater than 0");
		assert(layers.size() > 0 && "layers must contain at least 1 element");
		int _input_size = input_size;
		for (unsigned i = 0; i < layers.size(); i++) {
			assert(_input_size == layers[i].get_prev_nodes() && "incompatible layer dimensions");
			_input_size = layers[i].get_nodes();
		}
	};
	virtual ~NeuralNetwork() { };
	int get_input_size() const {
		return input_size;
	}
	std::vector<Layer<Scalar>>& get_layers() {
		std::vector<Layer<Scalar>>& ref = layers;
		return ref;
	}
	virtual std::vector<Scalar> feed_forward(std::vector<Scalar>& input) {
	//	viennacl::matrix<double> input_mtrx(0, input.size());
	//	for (unsigned i = 0; i < input.size(); i++) {
	//		input_mtrx(0,i) = input(i);
	//	}
	//	viennacl::matrix<double>* input_mtrx_ptr = &input_mtrx;
	//	for (unsigned i = 0; i < layers.size(); i++) {
	//		viennacl::matrix<double>& input_mtrx_ref = *input_mtrx_ptr;
	//		input_mtrx_ptr = &(layers[i].feed_forward(input_mtrx_ref));
	//	}
	//	return viennacl::row(*input_mtrx_ptr, 0);
		return std::vector<Scalar>(0);
	};
	virtual void feed_back(std::vector<Scalar>& out_grads) {
	//	viennacl::matrix<double> out_grads_mtrx(0, out_grads.size());
	//	for (unsigned i = 0; i < out_grads.size(); i++) {
	//		out_grads_mtrx(0,i) = out_grads(i);
	//	}
	//	viennacl::matrix<double>* out_grads_mtrx_ptr = &out_grads_mtrx;
	//	for (int i = (int) layers.size() - 1; i >= 0; i--) {
	//		viennacl::matrix<double>& out_grads_mtrx_ref = *out_grads_mtrx_ptr;
	//		out_grads_mtrx_ptr = &(layers[i].feed_back(out_grads_mtrx_ref));
	//	}
	};
	virtual std::string to_string() const {
	//	std::string str = "<NN>\n";
	//	for (unsigned i = 0; i < layers.size(); i++) {
	//		str += "Layer " + std::to_string(i) + "--------------------\n";
	//		str += layers[i].to_string();
	//	}
	//	str += "</NN>\n";
	//	return str;
		return NULL;
	};
protected:
	int input_size;
	std::vector<Layer<Scalar>>& layers;
};

}

#endif /* NEURALNETWORK_H_ */
