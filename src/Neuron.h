/*
 * Neuron.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURON_H_
#define NEURON_H_

struct Neuron {
protected:
	double in;
	double out;
	double delta;
	double weights[];
	Neuron* preds;
};

#endif /* NEURON_H_ */
