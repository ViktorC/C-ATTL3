/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <Activation.h>
#include <cmath>
#include <DataPreprocessor.h>
#include <Eigen/Dense>
#include <iostream>
#include <Layer.h>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <Optimizer.h>
#include <RegularizationPenalty.h>
#include <vector>
#include <Vector.h>
#include <WeightInitialization.h>

typedef double Scalar;

int main() {
	std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;
	cppnn::Matrix<Scalar> data(1000, 1);
	cppnn::Matrix<Scalar> obj(1000, 1);
	unsigned i = 0;
	for (Scalar v = -5.0; i < 1000; v += .01, i++) {
		data(i,0) = v;
		obj(i,0) = 5 * sin(v);
	}
	cppnn::PCADataPreprocessor<Scalar> preproc(true, true, 0.99);
	preproc.fit(data);
	preproc.transform(data);
	cppnn::HeWeightInitialization<Scalar> init;
	cppnn::GlorotWeightInitialization<Scalar> f_init;
	cppnn::ReLUActivation<Scalar> act;
	cppnn::IdentityActivation<Scalar> f_act;
	std::vector<cppnn::Layer<Scalar>*> layers(4);
	layers[0] = new cppnn::FCLayer<Scalar>(data.cols(), 30, init, act);
	layers[1] = new cppnn::FCLayer<Scalar>(30, 20, init, act);
	layers[2] = new cppnn::FCLayer<Scalar>(20, 10, init, act);
	layers[3] = new cppnn::FCLayer<Scalar>(10, 1, f_init, f_act);
	cppnn::FFNeuralNetwork<Scalar> nn(layers);
	nn.init();
	cppnn::QuadraticLoss<Scalar> loss;
	cppnn::ElasticNetRegularizationPenalty<Scalar> reg(5e-5, 1e-4);
	cppnn::NadamOptimizer<Scalar> opt(loss, reg, 32);
	std::cout << nn.to_string() << std::endl << std::endl;
//	std::cout << opt.verify_gradients(nn, data, obj) << std::endl;
	opt.train(nn, data, obj, 2000);
	cppnn::Matrix<Scalar> in(1, 1);
	in(0,0) = 1.2154498;
	cppnn::Matrix<Scalar> out = 5 * in.array().sin();
	preproc.transform(in);
	std::cout << std::endl << "Estimate: " << nn.infer(in) << std::endl;
	std::cout << "Actual value: " << out << std::endl;
	return 0;
};
