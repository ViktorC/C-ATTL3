/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <cmath>
#include <Dimensions.h>
#include <iostream>
#include <Layer.h>
#include <Loss.h>
#include <NeuralNetwork.h>
#include <Optimizer.h>
#include <Preprocessor.h>
#include <RegularizationPenalty.h>
#include <utility>
#include <vector>
#include <WeightInitialization.h>

typedef double Scalar;

int main() {
	std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;
	cppnn::Tensor4<Scalar> data(100, 32, 32, 3);
	cppnn::Tensor4<Scalar> obj(100, 1, 1, 1);
	data = data.setRandom();
	obj = obj.setRandom();
//	cppnn::NormalizationPreprocessor<Scalar> preproc(true);
//	preproc.fit(data);
//	preproc.transform(data);
	cppnn::HeWeightInitialization<Scalar> init;
	std::vector<std::pair<cppnn::SequentialNeuralNetwork<Scalar>,bool>> modules;
	modules.push_back(std::make_pair(cppnn::SequentialNeuralNetwork<Scalar>(new cppnn::MaxPoolingLayer<Scalar>(cppnn::Dimensions<int>(32, 32, 3))), false));
	modules.push_back(std::make_pair(cppnn::SequentialNeuralNetwork<Scalar>(std::vector<cppnn::Layer<Scalar>*>({
		new cppnn::ConvLayer<Scalar>(cppnn::Dimensions<int>(16, 16, 3), 5, init),
		new cppnn::LeakyReLUActivationLayer<Scalar>(cppnn::Dimensions<int>(16, 16, 5)),
		new cppnn::ConvLayer<Scalar>(cppnn::Dimensions<int>(16, 16, 5), 3, init),
	})), true));
	modules.push_back(std::make_pair(cppnn::SequentialNeuralNetwork<Scalar>(std::vector<cppnn::Layer<Scalar>*>({
		new cppnn::ConvLayer<Scalar>(cppnn::Dimensions<int>(16, 16, 3), 5, init),
		new cppnn::LeakyReLUActivationLayer<Scalar>(cppnn::Dimensions<int>(16, 16, 5)),
		new cppnn::ConvLayer<Scalar>(cppnn::Dimensions<int>(16, 16, 5), 3, init),
	})), true));
	modules.push_back(std::make_pair(cppnn::SequentialNeuralNetwork<Scalar>(new cppnn::FCLayer<Scalar>(cppnn::Dimensions<int>(16, 16, 3), 1, init)), false));
	cppnn::ResidualNeuralNetwork<Scalar,false> nn(modules);
	nn.init();
	cppnn::QuadraticLoss<Scalar> loss;
	cppnn::ElasticNetRegularizationPenalty<Scalar> reg(5e-5, 1e-4);
	cppnn::NadamOptimizer<Scalar> opt(loss, reg, 20);
	std::cout << nn.to_string() << std::endl << std::endl;
//	std::cout << opt.verify_gradients(nn, data, obj) << std::endl;
	opt.optimize(nn, data, obj, 500);
	return 0;
};
