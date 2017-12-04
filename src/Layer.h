/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "Activation.h"
#include "Neuron.h"

class Layer {
protected:
	int nodeCnt;
	Activation* act;
	Neuron* nodes;
	double bias;
public:
	Layer(int nodeCnt, Activation* act);
	virtual ~Layer();
};

enum class Layers {
	Hidden, Output
};

#endif /* LAYER_H_ */
