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
	virtual Matrix<Scalar> infer(Matrix<Scalar> input) {
		return propagate(input, false);
	};
	virtual std::string to_string() {
		std::stringstream strm;
		strm << "Neural Net " << this << std::endl;
		for (unsigned i = 0; i < get_layers().size(); i++) {
			Layer<Scalar>* layer = get_layers()[i];
			strm << "\tLayer " << std::setw(2) << std::to_string(i + 1) <<
					"----------------------------" << std::endl;
			strm << "\t\tprev nodes: " << layer->get_prev_nodes() << std::endl;
			strm << "\t\tnodes: " << layer->get_nodes() << std::endl;
			strm << "\t\tdropout: " << std::to_string(layer->get_dropout()) << std::endl;
			strm << "\t\tbatch norm: " << std::to_string(layer->get_batch_norm()) << std::endl;
			if (layer->get_batch_norm()) {
				strm << "\t\tnorm stats momentum: " <<
						std::to_string(layer->get_norm_stats_momentum()) << std::endl;
			}
			strm << "\t\tinit: " << layer->get_init().to_string() << std::endl;
			strm << "\t\tactivation: " << layer->get_act().to_string() << std::endl;
			if (layer->get_batch_norm()) {
				RowVector<Scalar>& gammas = layer->get_gammas();
				strm << "\t\tgammas:" << std::endl << "\t\t[ ";
				for (int j = 0; j < gammas.cols(); j++) {
					strm << std::setw(11) << std::setprecision(4) << gammas(j);
					if (j != gammas.cols() - 1) {
						strm << ", ";
					}
				}
				strm << " ]" << std::endl;
				RowVector<Scalar>& betas = layer->get_betas();
				strm << "\t\tbetas:" << std::endl << "\t\t[ ";
				for (int j = 0; j < betas.cols(); j++) {
					strm << std::setw(11) << std::setprecision(4) << betas(j);
					if (j != betas.cols() - 1) {
						strm << ", ";
					}
				}
				strm << " ]" << std::endl;
				const RowVector<Scalar>& moving_means = layer->get_moving_means();
				strm << "\t\tmoving means:" << std::endl << "\t\t[ ";
				for (int j = 0; j < moving_means.cols(); j++) {
					strm << std::setw(11) << std::setprecision(4) << moving_means(j);
					if (j != moving_means.cols() - 1) {
						strm << ", ";
					}
				}
				strm << " ]" << std::endl;
				const RowVector<Scalar>& moving_vars = layer->get_moving_vars();
				strm << "\t\tmoving vars:" << std::endl << "\t\t[ ";
				for (int j = 0; j < moving_vars.cols(); j++) {
					strm << std::setw(11) << std::setprecision(4) << moving_vars(j);
					if (j != moving_vars.cols() - 1) {
						strm << ", ";
					}
				}
				strm << " ]" << std::endl;
			}
			strm << "\t\tweights:" << std::endl;
			Matrix<Scalar>& weights = layer->get_weights();
			for (int j = 0; j < weights.rows(); j++) {
				strm << "\t\t[ ";
				for (int k = 0; k < weights.cols(); k++) {
					strm << std::setw(11) << std::setprecision(4) << weights(j,k);
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
	virtual Matrix<Scalar> pass_forward(Layer<Scalar>& layer, Matrix<Scalar> prev_out, bool training) {
		return layer.pass_forward(prev_out, training);
	};
	virtual Matrix<Scalar> pass_back(Layer<Scalar>& layer, Matrix<Scalar> out_grads) {
		return layer.pass_back(out_grads);
	};
	virtual std::vector<Layer<Scalar>*>& get_layers() = 0;
	virtual Matrix<Scalar> propagate(Matrix<Scalar> input, bool train) = 0;
	virtual void backpropagate(Matrix<Scalar> out_grads) = 0;
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
		unsigned input_size = layers[0]->get_nodes();
		for (unsigned i = 1; i < layers.size(); i++) {
			assert(layers[i] != nullptr && "layers contains null pointers");
			assert(input_size == layers[i]->get_prev_nodes() &&
					"incompatible layer dimensions");
			input_size = layers[i]->get_nodes();
		}
	};
	// Copy constructor.
	FFNeuralNetwork(const FFNeuralNetwork<Scalar>& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); i++) {
			layers[i] = network.layers[i]->clone();
		}
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
protected:
	// For the copy-and-swap idiom.
	friend void swap(FFNeuralNetwork<Scalar>& network1,
			FFNeuralNetwork<Scalar>& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
	};
	std::vector<Layer<Scalar>*>& get_layers() {
		return layers;
	};
	Matrix<Scalar> propagate(Matrix<Scalar> input, bool train) {
		assert(layers[0]->get_prev_nodes() == (unsigned) input.cols() &&
				"wrong neural network input size");
		assert(input.rows() >= 0 && input.cols() >= 0 && "empty feed forward input");
		for (unsigned i = 0; i < layers.size(); i++) {
			input = NeuralNetwork<Scalar>::pass_forward(*(layers[i]), input, true);
		}
		return input;
	};
	void backpropagate(Matrix<Scalar> out_grads) {
		assert(layers[layers.size() - 1]->get_nodes() == (unsigned) out_grads.cols() &&
				"wrong neural network output gradient size");
		for (int i = layers.size() - 1; i >= 0; i--) {
			out_grads = NeuralNetwork<Scalar>::pass_back(*(layers[i]), out_grads);
		}
	};
private:
	std::vector<Layer<Scalar>*> layers;
};

} /* namespace cppnn */

#endif /* NEURALNETWORK_H_ */
