/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "Activation.h";
#include "Neuron.h";

class Layer {
protected:
	double bias;
	Activation act;
	Neuron nodes[];
public:
	Layer();
	virtual ~Layer();
};

#endif /* LAYER_H_ */
