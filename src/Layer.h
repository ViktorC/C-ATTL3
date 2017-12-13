/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <Activation.h>
#include <cassert>
#include <Matrix.h>
#include <string>
#include <utility>
#include <Vector.h>

namespace cppnn {

// Forward declarations to NeuralNetwork and Optimizer so they can be friended.
template<typename Scalar>
class NeuralNetwork;
template<typename Scalar>
class Optimizer;

/**
 * A class template for a fully connected layer of a neural network.
 *
 * The layer representation has its weights before its neurons. This less
 * intuitive, reverse implementation allows for a more convenient
 * definition of neural network architectures as the input layer is not
 * normally activated, while the output layer often is.
 */
template<typename Scalar>
class Layer {
	friend class NeuralNetwork<Scalar>;
	friend class Optimizer<Scalar>;
public:
	Layer(unsigned prev_nodes, unsigned nodes, const Activation<Scalar>& act) :
			prev_nodes(prev_nodes),
			nodes(nodes),
			prev_out(prev_nodes),
			weights(prev_nodes + 1, nodes), // Bias trick.
			weight_grads(prev_nodes + 1, nodes),
			in(nodes),
			out(nodes),
			act(act) {
		assert(prev_nodes > 0 && "prev_nodes must be greater than 0");
		assert(nodes > 0 && "nodes must be greater than 0");
	};
	virtual ~Layer() = default;
	unsigned get_prev_nodes() const {
		return prev_nodes;
	};
	unsigned get_nodes() const {
		return nodes;
	};
	virtual Vector<Scalar> feed_forward(Vector<Scalar> prev_out) {
		assert((unsigned) prev_out.cols() == prev_nodes &&
				"illegal input vector size for feed forward");
		this->prev_out = std::move(prev_out);
		// Add a 1-column to the input for the bias trick.
		Vector<Scalar> biased_prev_out(this->prev_out.size() + 1);
		for (int i = 0; i < this->prev_out.cols(); i++) {
			biased_prev_out(i) = this->prev_out(i);
		}
		biased_prev_out(this->prev_out.size()) = 1;
		/* Compute the neuron inputs by multiplying the output of the
		 * previous layer by the weights. */
		in = biased_prev_out * weights;
		// Activate the neurons.
		out = act.function(in);
		return out;
	};
	virtual Vector<Scalar> feed_back(Vector<Scalar> out_grads) {
		assert((unsigned) out_grads.cols() == nodes &&
				"illegal input vector size for feed back");
		// Compute the gradients of the outputs with respect to the weighted inputs.
		Vector<Scalar> in_grads = act.d_function(in, out).cwiseProduct(out_grads);
		weight_grads = prev_out.transpose() * in_grads;
		/* Remove the bias column from the transposed weight matrix and compute the
		 * out-gradients of the previous layer. */
		return (in_grads * weights.transpose().block(0, 0, nodes, prev_nodes));
	};
protected:
	unsigned prev_nodes;
	unsigned nodes;
	/* Eigen matrices are backed by arrays allocated the heap, so these members do
	 * not burden the stack. */
	Vector<Scalar> prev_out;
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	Vector<Scalar> in;
	Vector<Scalar> out;
	const Activation<Scalar>& act;
	// Clone pattern.
	virtual Layer<Scalar>* clone() {
		return new Layer(*this);
	};
};

} /* namespace cppnn */

#endif /* LAYER_H_ */
