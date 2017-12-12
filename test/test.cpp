/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: viktor
 */

#include <Activation.h>
#include <iostream>
#include <Layer.h>
#include <NeuralNetwork.h>
#include <Vector.h>
#include <vector>

int main() {
	cppnn::Activation<double>* act1 = new cppnn::LeakyReLUActivation<double>(0.1);
	cppnn::Activation<double>* act2 = new cppnn::IdentityActivation<double>();
	cppnn::Layer<double> layer1(3, 5, act1);
	cppnn::Layer<double> layer2(5, 1, act2);
	std::vector<cppnn::Layer<double>> layers(2);
	layers[0] = layer1;
	layers[1] = layer2;
	cppnn::NeuralNetwork<double> nn(layers);
	std::vector<double> in(3);
	in[0] = 1;
	in[1] = -3;
	in[2] = 5;
	for (unsigned i = 0; i < in.size(); i++) {
		std::cout << in[i] << " ";
	}
	std::cout << std::endl;
	std::vector<double> out = nn.feed_forward(in);
	for (unsigned i = 0; i < out.size(); i++) {
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;
	std::cout << nn.to_string() << std::endl;
	nn.feed_back(out);
	std::cout << nn.to_string() << std::endl;
	return 0;
}
