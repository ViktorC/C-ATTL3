/*
 * Activation.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include "Activation.h"
#include "math.h"
#include <algorithm>
#include <stdexcept>

Activation::Activation() { };
Activation::~Activation() { };

IdentityActivation::IdentityActivation() { };
IdentityActivation::~IdentityActivation() { };
double IdentityActivation::function(double x) {
	return x;
}
double IdentityActivation::d_function(double x, double y) {
	return 1;
}

BinaryStepActivation::BinaryStepActivation() { };
BinaryStepActivation::~BinaryStepActivation() { };
double BinaryStepActivation::function(double x) {
	return x >= .0;
}
double BinaryStepActivation::d_function(double x, double y) {
	return 0;
}

SigmoidActivation::SigmoidActivation() { };
SigmoidActivation::~SigmoidActivation() { };
double SigmoidActivation::function(double x) {
	return 1 / (1 + exp(-x));
}
double SigmoidActivation::d_function(double x, double y) {
	return y * (1 - y);
}

TanhActivation::TanhActivation() { };
TanhActivation::~TanhActivation() { };
double TanhActivation::function(double x) {
	return tanh(x);
}
double TanhActivation::d_function(double x, double y) {
	return 1 - y * y;
}

ReLUActivation::ReLUActivation() { };
ReLUActivation::~ReLUActivation() { };
double ReLUActivation::function(double x) {
	return std::max(.0, x);
}
double ReLUActivation::d_function(double x, double y) {
	return x >= .0;
}

LeakyReLUActivation::LeakyReLUActivation(double alpha) : alpha(alpha) {
	if (alpha >= 1)
		throw std::invalid_argument("alpha must be less than 1.");
};
LeakyReLUActivation::~LeakyReLUActivation() { };
double LeakyReLUActivation::function(double x) {
	return std::max(x, x * alpha);
}
double LeakyReLUActivation::d_function(double x, double y) {
	return x <= .0 ? alpha : 1;
}

Activation* get_activation(Activations type) {
	Activation* act = NULL;
	switch (type) {
		case Activations::Identity: {
			act = new IdentityActivation();
			break;
		}
		case Activations::Sigmoid: {
			act = new SigmoidActivation();
			break;
		}
		case Activations::Tanh: {
			act = new TanhActivation();
			break;
		}
		case Activations::ReLU: {
			act = new ReLUActivation();
			break;
		}
		case Activations::LeakyReLU: {
			act = new LeakyReLUActivation();
			break;
		}
		default: {
			act = new ReLUActivation();
		}
	}
	return act;
};
