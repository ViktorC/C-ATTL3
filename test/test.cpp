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
#include <memory>
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
	cppnn::Tensor4<Scalar> data(5, 32, 32, 3);
	cppnn::Tensor4<Scalar> obj(5, 1, 1, 1);
	data = data.setRandom();
	obj = obj.setRandom();
//	cppnn::PCAPreprocessor<Scalar> preproc(true, true);
//	preproc.fit(data);
//	preproc.transform(data);
	cppnn::WeightInitSharedPtr<Scalar> init(new cppnn::HeWeightInitialization<Scalar>());
	std::vector<cppnn::LayerPtr<Scalar>> layers(4);
	layers[0] = cppnn::LayerPtr<Scalar>(new cppnn::MaxPoolingLayer<Scalar>(cppnn::Utils<Scalar>::get_dims(data)));
	layers[1] = cppnn::LayerPtr<Scalar>(new cppnn::ConvLayer<Scalar>(layers[0]->get_output_dims(), 3, init));
	layers[2] = cppnn::LayerPtr<Scalar>(new cppnn::LeakyReLUActivationLayer<Scalar>(layers[1]->get_output_dims()));
	layers[3] = cppnn::LayerPtr<Scalar>(new cppnn::FCLayer<Scalar>(layers[2]->get_output_dims(), 1, init));
	cppnn::SequentialNeuralNetwork<Scalar> nn(std::move(layers));
//	cppnn::SequentialNeuralNetwork<Scalar> nn2 = nn;
//	cppnn::SequentialNeuralNetwork<Scalar> nn3(nn);
//	cppnn::SequentialNeuralNetwork<Scalar> nn4(std::move(nn));
	nn.init();
	cppnn::LossSharedPtr<Scalar> loss(new cppnn::QuadraticLoss<Scalar>());
	cppnn::RegPenSharedPtr<Scalar> reg(new cppnn::ElasticNetRegularizationPenalty<Scalar>(5e-5, 1e-4));
	cppnn::NadamOptimizer<Scalar> opt(loss, reg, 20);
	std::cout << opt.verify_gradients(nn, data, obj) << std::endl;
//	opt.optimize(nn, data, obj, 500);
	return 0;
};
