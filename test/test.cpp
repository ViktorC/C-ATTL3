/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <cmath>
#include <Dimensions.h>
#include <Eigen/Dense>
#include <iostream>
#include <Layer.h>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <Optimizer.h>
#include <Preprocessor.h>
#include <RegularizationPenalty.h>
#include <Tensor.h>
#include <vector>
#include <Vector.h>
#include <WeightInitialization.h>

typedef double Scalar;

int main() {
	std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;
	cppnn::Tensor4D<Scalar> data(1000, 1, 1, 3);
	cppnn::Tensor4D<Scalar> obj(1000, 1, 1, 1);
	unsigned i = 0;
	for (Scalar v = -5.0; i < 1000; v += .01, i++) {
		for (int j = 0; j < 3; j++)
			data(i,0,0,j) = v + pow(-1, j);
		obj(i,0,0,0) = 5 * sin((data(i,0,0,0) + data(i,0,0,1) + data(i,0,0,2)) / 3);
	}
	cppnn::PCAPreprocessor<Scalar> preproc(true, true, 0.99);
	preproc.fit(data);
	preproc.transform(data);
	cppnn::HeWeightInitialization<Scalar> init;
	std::vector<cppnn::Layer<Scalar>*> layers(7);
	layers[0] = new cppnn::DenseLayer<Scalar>(cppnn::Dimensions(1, 1, 3), 30, init);
	layers[1] = new cppnn::ReLUActivationLayer<Scalar>(layers[0]->get_output_dims());
	layers[2] = new cppnn::DenseLayer<Scalar>(layers[1]->get_output_dims(), 20, init);
	layers[3] = new cppnn::ReLUActivationLayer<Scalar>(layers[2]->get_output_dims());
	layers[4] = new cppnn::DenseLayer<Scalar>(layers[3]->get_output_dims(), 10, init);
	layers[5] = new cppnn::ReLUActivationLayer<Scalar>(layers[4]->get_output_dims());
	layers[6] = new cppnn::DenseLayer<Scalar>(layers[5]->get_output_dims(), 1,
			cppnn::GlorotWeightInitialization<Scalar>());
	cppnn::FFNeuralNetwork<Scalar> nn(layers);
	nn.init();
	cppnn::QuadraticLoss<Scalar> loss;
	cppnn::ElasticNetRegularizationPenalty<Scalar> reg(5e-5, 1e-4);
	cppnn::NadamOptimizer<Scalar> opt(loss, reg, 32);
	std::cout << nn.to_string() << std::endl << std::endl;
//	std::cout << opt.verify_gradients(nn, data, obj) << std::endl;
	opt.optimize(nn, data, obj, 500);
	cppnn::Tensor4D<Scalar> in(1, 1, 1, 3);
	in(0,0,0,0) = 2.2154498;
	in(0,0,0,1) = in(0,0,0,0) - 1;
	in(0,0,0,2) = in(0,0,0,0);
	cppnn::Tensor4D<Scalar> out(1, 1, 1, 1);
	out(0,0,0,0) = 5 * sin((in(0,0,0,0) + in(0,0,0,1) + in(0,0,0,2)) / 3);
	preproc.transform(in);
	cppnn::Tensor4D<Scalar> estimate = nn.infer(in);
	std::cout << std::endl << "Estimate: " << estimate << std::endl;
	std::cout << "Actual value: " << out << std::endl;
	return 0;
};
