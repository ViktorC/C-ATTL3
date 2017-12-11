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

template<typename Scalar>
class Layer {
	friend class NeuralNetwork;
public:
	Layer(int nodes, int prev_nodes, Activation<Scalar> act);
	virtual ~Layer();
	virtual Matrix<Scalar>& get_weights() const;
	virtual Matrix<Scalar>& get_weight_grads() const;
	virtual Vector<Scalar>& feed_forward(Vector<Scalar>& prev_out);
	virtual Vector<Scalar>& feed_back(Vector<Scalar>& out_grads);
	virtual std::string to_string() const;
protected:
	// Eigen matrices are backed by arrays on the heap, so these members do not cause a stack overflow.
	Vector<Scalar>& prev_out;
	Vector<Scalar>& prev_out_grads;
	Matrix<Scalar>& weights;
	Matrix<Scalar>& weight_grads;
	Vector<Scalar>& in;
	Vector<Scalar>& out;
	Activation<Scalar> act;
};

#include <Layer.tpp>

}

#endif /* LAYER_H_ */
