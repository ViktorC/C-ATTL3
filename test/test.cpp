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
	cppnn::Vector<double> in(3);
	in(0) = 1;
	in(1) = -3;
	in(2) = 5;
	std::cout << in << std::endl;
	cppnn::Activation<double>* act = new cppnn::LeakyReLUActivation<double>(0.1);
	cppnn::Layer<double> layer(3, 3, act);
	std::vector<cppnn::Layer<double>> layers(1);
	layers[0] = layer;
	cppnn::NeuralNetwork<double> nn(3, layers);
	cppnn::Vector<double> out = layer.feed_forward(in);
	std::cout << out << std::endl;
	cppnn::Vector<double> grad = layer.feed_back(in);
	std::cout << grad << std::endl;
	return 0;
}
