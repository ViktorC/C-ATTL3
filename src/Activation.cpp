/*
 * Activation.cpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#include "Activation.h"

#include <math.h>
#include <algorithm>
#include <stdexcept>

namespace cppnn {

double IdentityActivation::function(double in) {
	return in;
};
double IdentityActivation::d_function(double in, double out) {
	return 1;
};

double BinaryStepActivation::function(double in) {
	return in >= .0;
};
double BinaryStepActivation::d_function(double in, double out) {
	return 0;
};

double SigmoidActivation::function(double in) {
	return 1 / (1 + exp(-in));
};
double SigmoidActivation::d_function(double in, double out) {
	return out * (1 - out);
};

double TanhActivation::function(double in) {
	return tanh(in);
};
double TanhActivation::d_function(double in, double out) {
	return 1 - out * out;
};

double ReLUActivation::function(double in) {
	return std::max(.0, in);
};
double ReLUActivation::d_function(double in, double out) {
	return in >= .0;
};

LeakyReLUActivation::LeakyReLUActivation(double alpha) :
		alpha(alpha) {
	if (alpha >= 1)
		throw std::invalid_argument("alpha must be less than 1.");
};
LeakyReLUActivation::~LeakyReLUActivation() { };
double LeakyReLUActivation::function(double in) {
	return std::max(in, in * alpha);
};
double LeakyReLUActivation::d_function(double in, double out) {
	return in <= .0 ? alpha : 1;
};

}
