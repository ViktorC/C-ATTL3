/*
 * Activation.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

namespace cppnn {

static const double DEF_LRELU_ALPHA = 1e-1;

class Activation {
public:
	virtual ~Activation() { };
	virtual double function(double x) = 0;
	virtual double d_function(double x, double y) = 0;
};

class IdentityActivation : public Activation {
public:
	virtual ~IdentityActivation() { };
	virtual double function(double x);
	virtual double d_function(double x, double y);
};

class BinaryStepActivation : public Activation {
public:
	virtual ~BinaryStepActivation() { };
	virtual double function(double x);
	virtual double d_function(double x, double y);
};

class SigmoidActivation : public Activation {
public:
	virtual ~SigmoidActivation() { };
	virtual double function(double x);
	virtual double d_function(double x, double y);
};

class TanhActivation : public Activation {
public:
	virtual ~TanhActivation() { };
	virtual double function(double x);
	virtual double d_function(double x, double y);
};

class ReLUActivation : public Activation {
public:
	virtual ~ReLUActivation() { };
	virtual double function(double x);
	virtual double d_function(double x, double y);
};

class LeakyReLUActivation : public Activation {
protected:
	double alpha;
public:
	LeakyReLUActivation(double alpha = DEF_LRELU_ALPHA);
	virtual ~LeakyReLUActivation();
	virtual double function(double x);
	virtual double d_function(double x, double y);
};

enum class Activations {
	Identity, BinaryStep, Sigmoid, Tanh, ReLU, LeakyReLU
};

Activation* get_activation(Activations type);

}

#endif /* ACTIVATION_H_ */
