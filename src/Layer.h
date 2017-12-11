/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <Activation.h>
#include <Matrix.h>
#include <string>
#include <Vector.h>

namespace cppnn {

template <typename Scalar>
class Layer;

template <typename Scalar>
void swap(Layer<Scalar>& layer1, Layer<Scalar>& layer2);

template<typename Scalar>
class Layer {
public:
	Layer() = default;
	Layer(int nodes, int prev_nodes, Activation<Scalar>* act) :
			nodes(nodes),
			prev_nodes(nodes),
			prev_out(prev_nodes + 1),
			prev_out_grads(prev_nodes + 1),
			weights(prev_nodes + 1, nodes),
			weight_grads(prev_nodes + 1, nodes),
			in(nodes),
			out(nodes),
			act(act) {
		assert(nodes > 0 && "nodes must be greater than 0");
		assert(prev_nodes > 0 && "prev_nodes must be greater than 0");
		assert(act != NULL && "act cannot be null");
		// Bias trick.
		prev_out(prev_nodes) = 1;
	};
	// Copy constructor.
	Layer(const Layer<Scalar>& layer) :
			nodes(layer.nodes),
			prev_nodes(layer.prev_nodes),
			prev_out(layer.prev_out),
			prev_out_grads(layer.prev_out_grads),
			weights(layer.weights),
			weight_grads(layer.weight_grads),
			in(layer.in),
			out(layer.out),
			act(layer.act->clone()) { };
	// Move constructor.
	Layer(Layer<Scalar>&& layer) :
			Layer() {
		swap(*this, layer);
	};
	// Copy/move assignment.
	Layer<Scalar>& operator=(Layer<Scalar> layer) {
		swap(*this, layer);
		return *this;
	};
	virtual ~Layer() {
		delete act;
	};
	// For the copy-and-swap idiom.
	friend void swap(Layer<Scalar>& layer1, Layer<Scalar>& layer2) {
		using std::swap;
		swap(layer1.nodes, layer2.nodes);
		swap(layer1.prev_nodes, layer2.prev_nodes);
		swap(layer1.prev_out, layer2.prev_out);
		swap(layer1.prev_out_grads, layer2.prev_out_grads);
		swap(layer1.weights, layer2.weights);
		swap(layer1.weight_grads, layer2.weight_grads);
		swap(layer1.in, layer2.in);
		swap(layer1.out, layer2.out);
		swap(layer1.act, layer2.act);
	};
	int get_nodes() const {
		return nodes;
	};
	int get_prev_nodes() const {
		return prev_nodes;
	};
	virtual Matrix<Scalar>& get_weights() {
		Matrix<Scalar>& ref = weights;
		return ref;
	};
	virtual Matrix<Scalar>& get_weight_grads() {
		Matrix<Scalar>& ref = weight_grads;
		return ref;
	};
	virtual Vector<Scalar>& feed_forward(Vector<Scalar>& prev_out) {
		assert(prev_out.cols() == prev_nodes && "illegal input vector size for feed forward");
		for (int i = 0; i < prev_nodes; i++) {
			this->prev_out(0,i) = prev_out(0,i);
		}
		// Compute the neuron inputs by multiplying the output of the previous layer by the weights.
		in = this->prev_out * weights;
		// Activate the neurons.
		out = act->function(in);
		return out;
	};
	virtual Vector<Scalar>& feed_back(Vector<Scalar>& out_grads) {
		assert(out_grads.cols() == nodes && "illegal input vector size for feed back");
		// Compute the gradients of the outputs with respect to the weighted inputs.
		Vector<Scalar> in_grads = act->d_function(in, out).cwiseProduct(out_grads);
		weight_grads = prev_out.transpose() * in_grads;
		prev_out_grads = in_grads * weights.transpose();
		return prev_out_grads;
	};
protected:
	int nodes;
	int prev_nodes;
	// Eigen matrices are backed by arrays allocated the heap, so these members do not cause a stack overflow.
	Vector<Scalar> prev_out;
	Vector<Scalar> prev_out_grads;
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	Vector<Scalar> in;
	Vector<Scalar> out;
	Activation<Scalar>* act;
};

}

#endif /* LAYER_H_ */
