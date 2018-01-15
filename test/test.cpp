/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <cmath>
#include <DataPreprocessor.h>
#include <Dimensions.h>
#include <Eigen/Dense>
#include <iostream>
#include <Layer.h>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <Optimizer.h>
#include <RegularizationPenalty.h>
#include <Tensor.h>
#include <vector>
#include <Vector.h>
#include <WeightInitialization.h>

typedef double Scalar;

int main() {
	std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;
	cppnn::Tensor4D<Scalar> data(1000, 1, 1, 1);
	cppnn::Tensor4D<Scalar> obj(1000, 1, 1, 1);
	unsigned i = 0;
	for (Scalar v = -5.0; i < 1000; v += .01, i++) {
		data(i,0,0,0) = v;
		obj(i,0,0,0) = 5 * sin(v);
	}
//	cppnn::PCADataPreprocessor<Scalar> preproc(true, true, 0.99);
//	preproc.fit(data);
//	preproc.transform(data);
	cppnn::HeWeightInitialization<Scalar> init;
	cppnn::GlorotWeightInitialization<Scalar> f_init;
	std::vector<cppnn::Layer<Scalar>*> layers(11);
	layers[0] = new cppnn::DenseLayer<Scalar>(cppnn::Dimensions(1, 1, 1), 30, init);
	layers[1] = new cppnn::PReLUActivationLayer<Scalar>(layers[0]->get_output_dims());
//	layers[2] = new cppnn::BatchNormLayer<Scalar>(30);
	layers[2] = new cppnn::DenseLayer<Scalar>(layers[1]->get_output_dims(), 20, init);
	layers[3] = new cppnn::PReLUActivationLayer<Scalar>(layers[2]->get_output_dims());
//	layers[2] = new cppnn::BatchNormLayer<Scalar>(20);
	layers[4] = new cppnn::DenseLayer<Scalar>(layers[3]->get_output_dims(), 10, init);
	layers[5] = new cppnn::PReLUActivationLayer<Scalar>(layers[4]->get_output_dims());
//	layers[2] = new cppnn::BatchNormLayer<Scalar>(10);
	layers[6] = new cppnn::DenseLayer<Scalar>(layers[5]->get_output_dims(), 10, init);
	layers[7] = new cppnn::PReLUActivationLayer<Scalar>(layers[6]->get_output_dims());
//	layers[2] = new cppnn::BatchNormLayer<Scalar>(10);
	layers[8] = new cppnn::DenseLayer<Scalar>(layers[7]->get_output_dims(), 10, init);
	layers[9] = new cppnn::PReLUActivationLayer<Scalar>(layers[8]->get_output_dims());
//	layers[2] = new cppnn::BatchNormLayer<Scalar>(10);
	layers[10] = new cppnn::DenseLayer<Scalar>(layers[9]->get_output_dims(), 1, f_init);
	cppnn::FFNeuralNetwork<Scalar> nn(layers);
	nn.init();
	cppnn::QuadraticLoss<Scalar> loss;
	cppnn::ElasticNetRegularizationPenalty<Scalar> reg(5e-5, 1e-4);
	cppnn::NadamOptimizer<Scalar> opt(loss, reg, 32);
	std::cout << nn.to_string() << std::endl << std::endl;
//	std::cout << opt.verify_gradients(nn, data, obj) << std::endl;
	opt.optimize(nn, data, obj, 500);
	cppnn::Tensor4D<Scalar> in(1, 1, 1, 1);
	in(0,0,0,0) = 1.2154498;
	Scalar out = 5 * sin(in(0,0,0,0));
//	preproc.transform(in);
	std::cout << std::endl << "Estimate: " << nn.infer(in) << std::endl;
	std::cout << "Actual value: " << out << std::endl;
	return 0;
};
