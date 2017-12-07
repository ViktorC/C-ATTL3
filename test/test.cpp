/*
 * test.cpp
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor
 */

#include <Activation.h>
#include <Layer.h>
#include <NeuralNetwork.h>
#include <iostream>
#include <string>
#include <vector.hpp>

int main() {
	cppnn::Layer* l1 = new cppnn::Layer(3, 3, new cppnn::SigmoidActivation());
	cppnn::Layer* l2 = new cppnn::Layer(4, 3, new cppnn::ReLUActivation());
	viennacl::vector<cppnn::Layer*>* layers = new viennacl::vector<cppnn::Layer*>(2);
	(*layers)[0] = l1;
	(*layers)[1] = l2;
	cppnn::NeuralNetwork* network = new cppnn::NeuralNetwork(3, layers);
	std::cout << network->to_string();
	viennacl::vector<double>* in = new viennacl::vector<double>(3);
	(*in)[0] = 7.12;
	(*in)[1] = -2.003;
	(*in)[2] = 0.034;
	viennacl::vector<double>& in_ref = *in;
	network->feed_forward(in_ref);
	std::cout << network->to_string();
	viennacl::vector<double>* grads = new viennacl::vector<double>(4);
	(*grads)[0] = -0.012;
	(*grads)[1] = 0.57;
	(*grads)[2] = 0.023;
	(*grads)[3] = -1.2;
	viennacl::vector<double>& grads_ref = *grads;
	network->feed_back(grads_ref);
	std::cout << network->to_string();
	delete network;
	delete grads;
	delete in;
	return 0;
}


