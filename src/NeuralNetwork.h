/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <Layer.h>
#include <cassert>
#include <iomanip>
#include <Matrix.h>
#include <sstream>
#include <string>
#include <vector>
#include <Vector.h>
#include <WeightInitialization.h>

namespace cppnn {

// Forward declaration to Optimizer so it can be friended.
template<typename Scalar>
class Optimizer;

/**
 * A neural network class that consists of a vector of neuron layers. It allows
 * for the calculation of the output of the network given an input vector and
 * it provides a member function for back-propagating the derivative of a loss
 * function with respect to the activated output of the network's last layer.
 * This back-propagation sets the gradients of the parameters associated with each
 * layer.
 */
template<typename Scalar>
class NeuralNetwork {
	friend class Optimizer<Scalar>;
public:
	virtual ~NeuralNetwork() = default;
	virtual unsigned get_input_size() const = 0;
	virtual unsigned get_output_size() const = 0;
	virtual void init() {
		std::vector<Layer<Scalar>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i]->init();
	};
	virtual Matrix<Scalar> infer(Matrix<Scalar> input) {
		return propagate(input, false);
	};
	virtual std::string to_string() {
		std::stringstream strm;
		strm << "Neural Net " << this << std::endl;
		for (unsigned i = 0; i < get_layers().size(); i++) {
			Layer<Scalar>* layer = get_layers()[i];
			strm << "\tLayer " << std::setw(3) << std::to_string(i + 1) <<
					"----------------------------" << std::endl;
			strm << "\t\tinput size: " << layer->get_input_size() << std::endl;
			strm << "\t\toutput size: " << layer->get_output_size() << std::endl;
			if (layer->is_parametric()) {
				strm << "\t\tparams:" << std::endl;
				Matrix<Scalar>& params = layer->get_params();
				for (int j = 0; j < params.rows(); j++) {
					strm << "\t\t[ ";
					for (int k = 0; k < params.cols(); k++) {
						strm << std::setw(11) << std::setprecision(4) << params(j,k);
						if (k != params.cols() - 1) {
							strm << ", ";
						}
					}
					strm << " ]" << std::endl;
				}
			}
		}
		return strm.str();
	};
protected:
	virtual std::vector<Layer<Scalar>*>& get_layers() = 0;
	virtual Matrix<Scalar> propagate(Matrix<Scalar> input, bool training) = 0;
	virtual void backpropagate(Matrix<Scalar> out_grads) = 0;
	static Matrix<Scalar> pass_forward(Layer<Scalar>& layer, Matrix<Scalar> prev_out, bool training) {
		return layer.pass_forward(prev_out, training);
	};
	static Matrix<Scalar> pass_back(Layer<Scalar>& layer, Matrix<Scalar> out_grads) {
		return layer.pass_back(out_grads);
	};
};

template<typename Scalar>
class FFNeuralNetwork : public NeuralNetwork<Scalar> {
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
	FFNeuralNetwork(std::vector<Layer<Scalar>*> layers) : // TODO use smart Layer pointers
			layers(layers) {
		assert(layers.size() > 0 && "layers must contain at least 1 element");
		input_size = layers[0]->get_input_size();
		output_size = layers[layers.size() - 1]->get_output_size();
		unsigned prev_size = layers[0]->get_output_size();
		for (unsigned i = 1; i < layers.size(); i++) {
			assert(layers[i] != nullptr && "layers contains null pointers");
			assert(prev_size == layers[i]->get_input_size() && "incompatible layer dimensions");
			prev_size = layers[i]->get_output_size();
		}
	};
	// Copy constructor.
	FFNeuralNetwork(const FFNeuralNetwork<Scalar>& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); i++) {
			layers[i] = network.layers[i]->clone();
		}
		input_size = network.input_size;
		output_size = network.output_size;
	};
	// Move constructor.
	FFNeuralNetwork(FFNeuralNetwork<Scalar>&& network) {
		swap(*this, network);
	};
	// Take ownership of layer pointers.
	~FFNeuralNetwork() {
		for (unsigned i = 0; i < layers.size(); i++) {
			delete layers[i];
		}
	};
	/* The assignment uses the move or copy constructor to pass the parameter
	 * based on whether it is an lvalue or an rvalue. */
	FFNeuralNetwork<Scalar>& operator=(FFNeuralNetwork<Scalar> network) {
		swap(*this, network);
		return *this;
	};
	unsigned get_input_size() const {
		return input_size;
	};
	unsigned get_output_size() const {
		return output_size;
	};
protected:
	// For the copy-and-swap idiom.
	friend void swap(FFNeuralNetwork<Scalar>& network1,
			FFNeuralNetwork<Scalar>& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
		swap(network1.input_size, network2.input_size);
		swap(network1.output_size, network2.output_size);
	};
	std::vector<Layer<Scalar>*>& get_layers() {
		return layers;
	};
	Matrix<Scalar> propagate(Matrix<Scalar> input, bool training) {
		assert(input_size == (unsigned) input.cols() &&
				"wrong neural network input size");
		assert(input.rows() >= 0 && input.cols() >= 0 && "empty feed forward input");
		for (unsigned i = 0; i < layers.size(); i++) {
			input = NeuralNetwork<Scalar>::pass_forward(*(layers[i]), input, training);
		}
		return input;
	};
	void backpropagate(Matrix<Scalar> out_grads) {
		assert(output_size == (unsigned) out_grads.cols() &&
				"wrong neural network output gradient size");
		for (int i = layers.size() - 1; i >= 0; i--) {
			out_grads = NeuralNetwork<Scalar>::pass_back(*(layers[i]), out_grads);
		}
	};
	std::vector<Layer<Scalar>*> layers;
	unsigned input_size;
	unsigned output_size;
};

template<typename Scalar>
class ResNeuralNetwork : public FFNeuralNetwork<Scalar> {
public:
		ResNeuralNetwork(std::vector<Layer<Scalar>*> layers) :
				FFNeuralNetwork<Scalar>::FFNeuralNetwork(layers) { };
		ResNeuralNetwork(const ResNeuralNetwork<Scalar>& network) :
				FFNeuralNetwork<Scalar>::FFNeuralNetwork(network) { };
		ResNeuralNetwork(ResNeuralNetwork<Scalar>&& network) :
				FFNeuralNetwork<Scalar>::FFNeuralNetwork(network) { };
		ResNeuralNetwork<Scalar>& operator=(ResNeuralNetwork<Scalar> network) {
			FFNeuralNetwork<Scalar>::swap(*this, network);
			return *this;
		};
protected:
		Matrix<Scalar> propagate(Matrix<Scalar> input, bool training) {
			assert(FFNeuralNetwork<Scalar>::input_size == (unsigned) input.cols() &&
					"wrong neural network input size");
			assert(input.rows() >= 0 && input.cols() >= 0 && "empty feed forward input");
			std::vector<Layer<Scalar>*> layers = FFNeuralNetwork<Scalar>::layers;
			for (unsigned i = 0; i < layers.size(); i++) {
				input = NeuralNetwork<Scalar>::pass_forward(*(layers[i]), input, training);
			}
			return input;
		};
		void backpropagate(Matrix<Scalar> out_grads) {
			assert(FFNeuralNetwork<Scalar>::output_size == (unsigned) out_grads.cols() &&
					"wrong neural network output gradient size");
			std::vector<Layer<Scalar>*> layers = FFNeuralNetwork<Scalar>::layers;
			for (int i = layers.size() - 1; i >= 0; i--) {
				out_grads = NeuralNetwork<Scalar>::pass_back(*(layers[i]), out_grads);
			}
		};
};

} /* namespace cppnn */

#endif /* NEURALNETWORK_H_ */
