/*
 * Neuron.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor
 */

#ifndef NEURON_H_
#define NEURON_H_

class Neuron {
protected:
	double in;
	double out;
	double delta;
//	double* weights;
//	Neuron* preds;
public:
	Neuron();
//	Neuron(Neuron preds[], double weights[]);
	virtual ~Neuron();
};

#endif /* NEURON_H_ */
