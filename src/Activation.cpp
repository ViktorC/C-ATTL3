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

namespace cppnn {

double IdentityActivation::function(double x) {
	return x;
};
double IdentityActivation::d_function(double x, double y) {
	return 1;
};

double BinaryStepActivation::function(double x) {
	return x >= .0;
};
double BinaryStepActivation::d_function(double x, double y) {
	return 0;
};

double SigmoidActivation::function(double x) {
	return 1 / (1 + exp(-x));
};
double SigmoidActivation::d_function(double x, double y) {
	return y * (1 - y);
};

double TanhActivation::function(double x) {
	return tanh(x);
};
double TanhActivation::d_function(double x, double y) {
	return 1 - y * y;
};

double ReLUActivation::function(double x) {
	return std::max(.0, x);
};
double ReLUActivation::d_function(double x, double y) {
	return x >= .0;
};

LeakyReLUActivation::LeakyReLUActivation(double alpha) :
		alpha(alpha) {
	if (alpha >= 1)
		throw std::invalid_argument("alpha must be less than 1.");
};
LeakyReLUActivation::~LeakyReLUActivation() { };
double LeakyReLUActivation::function(double x) {
	return std::max(x, x * alpha);
};
double LeakyReLUActivation::d_function(double x, double y) {
	return x <= .0 ? alpha : 1;
};

Activation* get_activation(Activations type) {
	switch (type) {
		case Activations::Identity: {
			return new IdentityActivation();
		}
		case Activations::Sigmoid: {
			return new SigmoidActivation();
		}
		case Activations::Tanh: {
			return new TanhActivation();
		}
		case Activations::ReLU: {
			return new ReLUActivation();
		}
		case Activations::LeakyReLU: {
			return new LeakyReLUActivation();
		}
		default: {
			return new ReLUActivation();
		}
	}
};

}
