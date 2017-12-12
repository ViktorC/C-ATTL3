/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <cassert>
#include <Layer.h>
#include <string>
#include <vector>
#include <Vector.h>
#include <iostream>

namespace cppnn {

template <typename Scalar>
class NeuralNetwork {
public:
	NeuralNetwork(std::vector<Layer<Scalar>> layers) :
			layers(layers) {
		assert(layers.size() > 0 && "layers must contain at least 1 element");
		unsigned input_size = layers[0].get_nodes();
		for (unsigned i = 1; i < layers.size(); i++) {
			assert(input_size == layers[i].get_prev_nodes() &&
					"incompatible layer dimensions");
			input_size = layers[i].get_nodes();
		}
	};
	virtual ~NeuralNetwork() = default;
	std::vector<Layer<Scalar>>& get_layers() const {
		std::vector<Layer<Scalar>>& ref = layers;
		return ref;
	}
	virtual std::vector<Scalar> feed_forward(std::vector<Scalar>& input) {
		assert(layers[0].get_prev_nodes() == input.size() &&
				"wrong neural network input size");
		Vector<Scalar> input_vctr(input.size());
		for (unsigned i = 0; i < input.size(); i++) {
			input_vctr(i) = input[i];
		}
		Vector<Scalar>* input_vctr_ptr = &input_vctr;
		for (unsigned i = 0; i < layers.size(); i++) {
			Vector<Scalar>& input_vctr_ref = *input_vctr_ptr;
			input_vctr_ptr = &(layers[i].feed_forward(input_vctr_ref));
		}
		std::vector<Scalar> out(input_vctr_ptr->cols());
		for (unsigned i = 0; i < out.size(); i++) {
			out[i] = (*input_vctr_ptr)(i);
		}
		return out;
	};
	virtual void feed_back(std::vector<Scalar>& out_grads) {
		assert(layers[layers.size() - 1].get_nodes() == out_grads.size() &&
				"wrong neural network output gradient size");
		Vector<Scalar> out_grads_vctr(out_grads.size());
		for (unsigned i = 0; i < out_grads.size(); i++) {
			out_grads_vctr(i) = out_grads[i];
		}
		Vector<Scalar>* out_grads_vctr_ptr = &out_grads_vctr;
		for (unsigned i = layers.size() - 1; i >= 0; i--) {
			Vector<Scalar>& out_grads_vctr_ref = *out_grads_vctr_ptr;
			std::cout << layers[i].get_nodes() << std::endl;
			std::cout << out_grads_vctr_ref.cols() << std::endl;
			out_grads_vctr_ptr = &(layers[i].feed_back(out_grads_vctr_ref));
		}
	};
	virtual std::string to_string() {
		std::string str = "<NN>\n";
		for (unsigned i = 0; i < layers.size(); i++) {
			str += "Layer " + std::to_string(i) + "--------------------\n";
			Matrix<Scalar> weights = layers[i].get_weights();
			for (int j = 0; j < weights.rows(); j++) {
				str += "[";
				for (int k = 0; k < weights.cols(); k++) {
					str += std::to_string(weights(j,k));
					if (k != weights.cols()) {
						str += ", ";
					}
				}
				str += "]\n";
			}
		}
		str += "</NN>\n";
		return str;
	};
protected:
	std::vector<Layer<Scalar>> layers;
};

}

#endif /* NEURALNETWORK_H_ */
