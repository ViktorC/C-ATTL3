/*
 * Activation.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include "Activation.h"
#include "math.h"
#include <algorithm>
#include <iostream>

Activation::Activation() {};
Activation::~Activation() {};

class LinearActivation : public Activation {
protected:
	LinearActivation() {};
	virtual ~LinearActivation() {};
public:
	LinearActivation(LinearActivation const&) = delete;
	void operator=(LinearActivation const&) = delete;
	double function(double x) {
		return x;
	}
	double d_function(double x) {
		return 1;
	}
	static LinearActivation& get_instance() {
		static LinearActivation instance;
		return instance;
	}
};

class SigmoidActivation : public Activation {
protected:
	SigmoidActivation() {};
	virtual ~SigmoidActivation() {};
public:
	SigmoidActivation(SigmoidActivation const&) = delete;
	void operator=(SigmoidActivation const&) = delete;
	double function(double x) {
		return 1/(1 + exp(-x));
	}
	double d_function(double x) {
		return 1;
	}
	static SigmoidActivation& get_instance() {
		static SigmoidActivation instance;
		return instance;
	}
};

class TanhActivation : public Activation {
protected:
	TanhActivation() {};
	virtual ~TanhActivation() {};
public:
	TanhActivation(TanhActivation const&) = delete;
	void operator=(TanhActivation const&) = delete;
	double function(double x) {
		return tanh(x);
	}
	double d_function(double x) {
		double y = tanh(x);
		return 1 - y*y;
	}
	static TanhActivation& get_instance() {
		static TanhActivation instance;
		return instance;
	}
};

class ReLuActivation : public Activation {
protected:
	ReLuActivation() {};
	virtual ~ReLuActivation() {};
public:
	ReLuActivation(ReLuActivation const&) = delete;
	void operator=(ReLuActivation const&) = delete;
	double function(double x) {
		return std::max(.0, x);
	}
	double d_function(double x) {
		return x == .0 ? x : 1;
	}
	static ReLuActivation& get_instance() {
		static ReLuActivation instance;
		return instance;
	}
};

Activation& get_activation(Activations type) {
	Activation act;
	switch (type) {
		case Activations::Linear:
			act = LinearActivation::get_instance();
			break;
		case Activations::Sigmoid:
			act = SigmoidActivation::get_instance();
			break;
		case Activations::Tanh:
			act = TanhActivation::get_instance();
			break;
		case Activations::ReLU:
			act = ReLuActivation::get_instance();
			break;
		default:
			act = ReLuActivation::get_instance();
	}
	Activation& actRef = act;
	return actRef;
};

int main(int ac, char** av) {
	double in = 0.876;
	Activation& act = get_activation(Activations::Linear);
	std::cout << "y: " << act.function(in) << "; d y: " << act.d_function(in) << std::endl;
}
