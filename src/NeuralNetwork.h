/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <cassert>
#include <iomanip>
#include <Layer.h>
#include <sstream>
#include <string>
#include <vector>
#include <Vector.h>

namespace cppnn {

// Forward declaration to Optimizer so it can be friended.
template<typename Scalar>
class Optimizer;

/**
 * A neural network class that consists of a vector of neuron layers. It allows
 * for the calculation of the output of the network given an input vector and
 * it provides a member function for back-propagating the derivative of a loss
 * function with respect to the activated output of the network's last layer.
 * This back-propagation sets the deltas of the weights associated with each
 * layer.
 */
template <typename Scalar>
class NeuralNetwork {
	friend class Optimizer<Scalar>;
public:
	/**
	 * Constructs the network using the provided vector of layers. The object
	 * takes ownership of the pointers held in the vector and deletes them when
	 * its destructor is invoked.
	 *
	 * @param layers A vector of Layer pointers to compose the neural network.
	 * The vector must contain at least 1 element and no null pointers. The
	 * the prev_nodes attributes of the Layer objects pointed to by the pointers
	 * must match the nodes attribute of the Layer objects pointed to by their
	 * pointers directly preceding them in the vector.
	 */
	NeuralNetwork(std::vector<Layer<Scalar>*> layers) :
			layers(layers) {
		assert(layers.size() > 0 && "layers must contain at least 1 element");
		unsigned input_size = layers[0]->nodes;
		for (unsigned i = 1; i < layers.size(); i++) {
			assert(layers[i] != nullptr && "layers contains null pointers");
			assert(input_size == layers[i]->prev_nodes &&
					"incompatible layer dimensions");
			input_size = layers[i]->nodes;
		}
	};
	// Copy constructor.
	NeuralNetwork(const NeuralNetwork<Scalar>& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); i++) {
			layers[i] = network.layers[i]->clone();
		}
	};
	// Move constructor.
	NeuralNetwork(NeuralNetwork<Scalar>&& network) {
		swap(*this, network);
	};
	/* The assignment uses the move or copy constructor to pass the parameter
	 * based on whether it is an lvalue or an rvalue. */
	NeuralNetwork<Scalar>& operator=(NeuralNetwork<Scalar> network) {
		swap(*this, network);
		return *this;
	};
	// Take ownership of layer pointers.
	virtual ~NeuralNetwork() {
		for (unsigned i = 0; i < layers.size(); i++) {
			delete layers[i];
		}
	};
	// For the copy-and-swap idiom.
	friend void swap(NeuralNetwork<Scalar>& network1,
			NeuralNetwork<Scalar>& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
	};
	virtual RowVector<Scalar> feed_forward(RowVector<Scalar> input) {
		assert(layers[0]->prev_nodes == (unsigned) input.cols() &&
				"wrong neural network input size");
		for (unsigned i = 0; i < layers.size(); i++) {
			input = layers[i]->feed_forward(input);
		}
		return input;
	};
	virtual void feed_back(RowVector<Scalar> out_grads) {
		assert(layers[layers.size() - 1]->nodes == (unsigned)out_grads.cols() &&
				"wrong neural network output gradient size");
		for (int i = layers.size() - 1; i >= 0; i--) {
			out_grads = layers[i]->feed_back(out_grads);
		}
	};
	virtual std::string to_string() {
		std::stringstream strm;
		strm << "Neural Net " << this << std::endl;
		for (unsigned i = 0; i < layers.size(); i++) {
			strm << "\tLayer " << std::setw(2) << std::to_string(i + 1) <<
					"----------------------------" << std::endl;
			Matrix<Scalar>& weights = layers[i]->weights;
			for (int j = 0; j < weights.rows(); j++) {
				strm << "\t\t[ ";
				for (int k = 0; k < weights.cols(); k++) {
					strm << std::setw(11) << std::setprecision(4) <<
							weights(j,k);
					if (k != weights.cols() - 1) {
						strm << ", ";
					}
				}
				strm << " ]" << std::endl;
			}
		}
		return strm.str();
	};
protected:
	std::vector<Layer<Scalar>*> layers;
};

} /* namespace cppnn */

#endif /* NEURALNETWORK_H_ */
