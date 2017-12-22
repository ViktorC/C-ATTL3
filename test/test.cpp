/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <Activation.h>
#include <Layer.h>
#include <Eigen/Dense>
#include <Initialization.h>
#include <iostream>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <Optimizer.h>
#include <Preprocessor.h>
#include <Regularization.h>
#include <vector>
#include <Vector.h>

typedef double Scalar;

int main() {
	cppnn::Matrix<Scalar> data(4, 3);
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
	cppnn::PCAPreprocessor<Scalar> pca(true, true, 0.95);
	pca.fit(data);
	pca.transform(data);
	std::cout << data << std::endl << std::endl;
	cppnn::SoftmaxActivation<Scalar> act1;
	cppnn::SoftmaxActivation<Scalar> act2;
	cppnn::ReLUInitialization<Scalar> init1;
	cppnn::ReLUInitialization<Scalar> init2;
	cppnn::Layer<Scalar>* layer1 = new cppnn::FCLayer<Scalar>(data.cols(), 5, 1, act1, init1, true, 0);
	cppnn::Layer<Scalar>* layer2 = new cppnn::FCLayer<Scalar>(5, 2, 2, act2, init2, true, 0);
	std::vector<cppnn::Layer<Scalar>*> layers(2);
	layers[0] = layer1;
	layers[1] = layer2;
	cppnn::FFNeuralNetwork<Scalar> nn(layers);
	std::cout << nn.to_string() << std::endl << std::endl;
	cppnn::Matrix<Scalar> out = nn.infer(data);
	std::cout << out << std::endl << std::endl;
//	cppnn::QuadraticLoss<Scalar> loss;
	cppnn::Matrix<Scalar> obj(4, 2);
	obj(0,0) = 0;
	obj(0,1) = 1;
	obj(1,0) = 0.1;
	obj(1,1) = 0.9;
	obj(2,0) = 0.25;
	obj(2,1) = 0.75;
	obj(3,0) = 0.75;
	obj(3,1) = 0.25;
//	std::cout << loss.function(out, obj) << std::endl;
//	nn.backpropagate(obj);
	cppnn::QuadraticLoss<Scalar> loss;
	cppnn::L2Regularization<Scalar> reg;
	cppnn::SGDOptimizer<Scalar> opt(reg, loss);
	std::cout << opt.validate_gradients(nn, data, obj) << std::endl;
	return 0;
}
