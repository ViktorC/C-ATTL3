/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: viktor
 */

#include <Activation.h>
#include <Vector.h>
#include <iostream>

int main() {
	cppnn::Vector<double> in(3);
	in(0) = 1;
	in(1) = -3;
	in(2) = 5;
	std::cout << in << std::endl;
	cppnn::LeakyReLUActivation<double> act(0.5);
	cppnn::Vector<double> out = act.function(in);
	std::cout << out << std::endl;
	cppnn::Vector<double> grad = act.d_function(in, out);
	std::cout << grad << std::endl;
	return 0;
}
