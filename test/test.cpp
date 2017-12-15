/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: viktor
 */

#include <Activation.h>
#include <Eigen/Dense>
#include <iostream>
#include <Layer.h>
#include <Loss.h>
#include <NeuralNetwork.h>
#include <Preprocessor.h>
#include <Vector.h>
#include <vector>

int main() {
	cppnn::Matrix<double> data(4, 3);
	data(0,0) = 2;
	data(0,1) = 3;
	data(0,2) = 1;
	data(1,0) = 1;
	data(1,1) = 2;
	data(1,2) = 3;
	data(2,0) = 3;
	data(2,1) = 4;
	data(2,2) = 6;
	data(3,0) = 2;
	data(3,1) = 3;
	data(3,2) = 2;
	std::cout << data << std::endl << std::endl;
	cppnn::PCAPreprocessor<double> pca(true, true, 0.95);
	pca.fit(data);
	pca.transform(data);
	std::cout << data << std::endl;
	cppnn::LeakyReLUActivation<double> lrelu(0.1);
	cppnn::SoftmaxActivation<double> softmax;
	cppnn::Activation<double>& act1 = lrelu;
	cppnn::Activation<double>& act2 = softmax;
	cppnn::Layer<double>* layer1 = new cppnn::Layer<double>(3, 5, act1);
	cppnn::Layer<double>* layer2 = new cppnn::Layer<double>(5, 2, act2);
	std::vector<cppnn::Layer<double>*> layers(2);
	layers[0] = layer1;
	layers[1] = layer2;
	cppnn::NeuralNetwork<double> nn(layers);
	std::cout << nn.to_string() << std::endl;
	cppnn::RowVector<double> in(3);
	in[0] = 1;
	in[1] = -3;
	in[2] = 5;
	std::cout << in << std::endl;
	cppnn::RowVector<double> out = nn.feed_forward(in);
	std::cout << out << std::endl;
	cppnn::QuadraticLoss<double> loss;
	cppnn::RowVector<double> obj(2);
	obj[0] = 0;
	obj[1] = 1;
	std::cout << loss.function(out, obj) << std::endl;
	nn.feed_back(out);
	return 0;
}
