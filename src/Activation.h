/*
 * Activation.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

class Activation {
public:
	Activation();
	virtual ~Activation();
	virtual double function(double x);
	virtual double d_function(double x);
};

enum class Activations {
		Linear, Sigmoid, Tanh, ReLU
};

static Activation& get_activation(Activations type);

#endif /* ACTIVATION_H_ */
