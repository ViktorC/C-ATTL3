/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: viktor Csomor
 */

#include <Activation.h>
#include <Layer.h>
#include <Eigen/Dense>
#include <Initialization.h>
#include <iostream>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <Preprocessor.h>
#include <vector>
#include <Vector.h>

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
	std::cout << data << std::endl << std::endl;
	cppnn::LeakyReLUActivation<double> lrelu(0.1);
	cppnn::SoftmaxActivation<double> softmax;
	cppnn::ReLUInitialization<double> relu_init;
	cppnn::XavierInitialization<double> xavier_init;
	cppnn::Layer<double>* layer1 = new cppnn::FCLayer<double>(data.cols(), 5, lrelu, relu_init);
	cppnn::Layer<double>* layer2 = new cppnn::FCLayer<double>(5, 2, softmax, xavier_init);
	std::vector<cppnn::Layer<double>*> layers(2);
	layers[0] = layer1;
	layers[1] = layer2;
	cppnn::NeuralNetwork<double> nn(layers);
	std::cout << nn.to_string() << std::endl << std::endl;
	cppnn::Matrix<double> out = nn.feed_forward(data);
	std::cout << out << std::endl;
//	cppnn::QuadraticLoss<double> loss;
	cppnn::Matrix<double> obj(4, 2);
	obj(0,0) = 0;
	obj(0,1) = 1;
	obj(1,0) = 0.1;
	obj(1,1) = 0.9;
	obj(2,0) = 0.25;
	obj(2,1) = 0.75;
	obj(3,0) = 0.75;
	obj(3,1) = 0.25;
//	std::cout << loss.function(out, obj) << std::endl;
	nn.feed_back(obj);
	return 0;
}
