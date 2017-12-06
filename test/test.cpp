/*
 * test.cpp
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor
 */

#include "Layer.h"
#include "Activation.h"
#include "NeuralNetwork.h"
#include <tools/entry_proxy.hpp>
#include <matrix.hpp>
#include <iostream>

int main() {
	cppnn::Layer* l1 = new cppnn::Layer(3, 3, cppnn::get_activation(cppnn::Activations::Sigmoid));
	cppnn::Layer* l2 = new cppnn::Layer(4, 3, cppnn::get_activation(cppnn::Activations::ReLU));
	std::vector<cppnn::Layer*>* layers = new std::vector<cppnn::Layer*>(2);
	(*layers)[0] = l1;
	(*layers)[1] = l2;
	cppnn::NeuralNetwork* network = new cppnn::NeuralNetwork(3, layers);
	std::cout << network->to_string();
	network->initialize_weights();
	std::cout << network->to_string();
	viennacl::matrix<double>* in = new viennacl::matrix<double>(1, 3);
	(*in)(0,0) = 7.12;
	(*in)(0,1) = -2.003;
	(*in)(0,2) = 0.034;
	network->feed_forward(in);
	std::cout << network->to_string();
	viennacl::matrix<double>* grads = new viennacl::matrix<double>(1, 4);
	(*grads)(0,0) = -0.012;
	(*grads)(0,1) = 0.57;
	(*grads)(0,2) = 0.023;
	(*grads)(0,3) = -1.2;
	network->feed_back(grads);
	std::cout << network->to_string();
	delete network;
	delete grads;
	delete in;
	return 0;
}


