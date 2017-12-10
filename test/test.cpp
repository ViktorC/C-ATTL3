/*
 * test.cpp
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor
 */

#include <Activation.h>
#include <Matrix.h>
#include <Vector.h>
#include <iostream>

int main(int argc, char *argv[]) {
	cppnn::Vector<double> vector;
	vector << 1, 3, 5;
	cppnn::SoftmaxActivation<double> softmax;
	cppnn::Vector<double> out = softmax.function(vector);
	for (int i = 0; i < out.cols(); i++) {
		std::cout << out(i) << ", ";
	}
	std::cout << std::endl;
	return 0;
}


