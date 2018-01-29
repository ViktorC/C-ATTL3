/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <cmath>
#include <DataProvider.h>
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
	cppnn::Tensor4Ptr<Scalar> training_obs_ptr = cppnn::Tensor4Ptr<Scalar>(new cppnn::Tensor4<Scalar>(80, 32, 32, 3));
	cppnn::Tensor4Ptr<Scalar> training_obj_ptr = cppnn::Tensor4Ptr<Scalar>(new cppnn::Tensor4<Scalar>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	cppnn::InMemoryDataProvider<Scalar> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	cppnn::Tensor4Ptr<Scalar> test_obs_ptr = cppnn::Tensor4Ptr<Scalar>(new cppnn::Tensor4<Scalar>(20, 32, 32, 3));
	cppnn::Tensor4Ptr<Scalar> test_obj_ptr = cppnn::Tensor4Ptr<Scalar>(new cppnn::Tensor4<Scalar>(20, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	cppnn::InMemoryDataProvider<Scalar> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	cppnn::PCAPreprocessor<Scalar> preproc(true, true);
//	preproc.fit(data);
//	preproc.transform(data);
	cppnn::WeightInitSharedPtr<Scalar> init(new cppnn::HeWeightInitialization<Scalar>());
	std::vector<std::pair<cppnn::ParallelNeuralNetwork<Scalar>,bool>> modules;
	modules.push_back(std::make_pair(cppnn::ParallelNeuralNetwork<Scalar>(cppnn::SequentialNeuralNetwork<Scalar>(
			cppnn::LayerPtr<Scalar>(new cppnn::ConvLayer<Scalar>(training_prov.get_obs_dims(), 4, init)))), false));
	modules.push_back(std::make_pair(cppnn::ParallelNeuralNetwork<Scalar>(cppnn::SequentialNeuralNetwork<Scalar>(
			cppnn::LayerPtr<Scalar>(new cppnn::LeakyReLUActivationLayer<Scalar>(modules[0].first.get_output_dims())))), false));
	modules.push_back(std::make_pair(cppnn::ParallelNeuralNetwork<Scalar>(cppnn::SequentialNeuralNetwork<Scalar>(
			cppnn::LayerPtr<Scalar>(new cppnn::MaxPoolingLayer<Scalar>(modules[1].first.get_output_dims())))), false));
	std::vector<cppnn::SequentialNeuralNetwork<Scalar>> lanes;
	std::vector<cppnn::LayerPtr<Scalar>> layers1(2);
	layers1[0] = cppnn::LayerPtr<Scalar>(new cppnn::ConvLayer<Scalar>(modules[2].first.get_output_dims(), 2, init));
	layers1[1] = cppnn::LayerPtr<Scalar>(new cppnn::LeakyReLUActivationLayer<Scalar>(layers1[0]->get_output_dims()));
	lanes.push_back(cppnn::SequentialNeuralNetwork<Scalar>(std::move(layers1)));
	std::vector<cppnn::LayerPtr<Scalar>> layers2(2);
	layers2[0] = cppnn::LayerPtr<Scalar>(new cppnn::ConvLayer<Scalar>(modules[2].first.get_output_dims(), 2, init));
	layers2[1] = cppnn::LayerPtr<Scalar>(new cppnn::LeakyReLUActivationLayer<Scalar>(layers2[0]->get_output_dims()));
	lanes.push_back(cppnn::SequentialNeuralNetwork<Scalar>(std::move(layers2)));
	modules.push_back(std::make_pair(cppnn::ParallelNeuralNetwork<Scalar>(lanes), true));
	modules.push_back(std::make_pair(cppnn::ParallelNeuralNetwork<Scalar>(cppnn::SequentialNeuralNetwork<Scalar>(
			cppnn::LayerPtr<Scalar>(new cppnn::ConvLayer<Scalar>(modules[3].first.get_output_dims(), 2, init)))), false));
	std::vector<cppnn::LayerPtr<Scalar>> layers3(2);
	layers3[0] = cppnn::LayerPtr<Scalar>(new cppnn::ConvLayer<Scalar>(modules[4].first.get_output_dims(), 2, init));
	layers3[1] = cppnn::LayerPtr<Scalar>(new cppnn::LeakyReLUActivationLayer<Scalar>(layers3[0]->get_output_dims()));
	modules.push_back(std::make_pair(cppnn::ParallelNeuralNetwork<Scalar>(cppnn::SequentialNeuralNetwork<Scalar>(
			std::move(layers3))), true));
	modules.push_back(std::make_pair(cppnn::ParallelNeuralNetwork<Scalar>(cppnn::SequentialNeuralNetwork<Scalar>(
				cppnn::LayerPtr<Scalar>(new cppnn::FCLayer<Scalar>(modules[5].first.get_output_dims(), 1, init)))), false));
	cppnn::ResidualNeuralNetwork<Scalar,true> nn(modules);
	nn.init();
	cppnn::LossSharedPtr<Scalar> loss(new cppnn::QuadraticLoss<Scalar>());
	cppnn::RegPenSharedPtr<Scalar> reg(new cppnn::ElasticNetRegularizationPenalty<Scalar>());
	cppnn::NadamOptimizer<Scalar> opt(loss, reg, 20);
//	std::cout << opt.verify_gradients(nn, test_prov) << std::endl;
	opt.optimize(nn, training_prov, test_prov, 500);
	return 0;
};
