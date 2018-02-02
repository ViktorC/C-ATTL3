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
	using namespace cattle;
	std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;
	Tensor4Ptr<Scalar> training_obs_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(80, 32, 32, 3));
	Tensor4Ptr<Scalar> training_obj_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	Tensor4Ptr<Scalar> test_obs_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(20, 32, 32, 3));
	Tensor4Ptr<Scalar> test_obj_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(20, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<std::pair<CompositeNeuralNetwork<Scalar>,bool>> res_modules;
	std::vector<NeuralNetPtr<Scalar>> comp_mods;
	std::vector<NeuralNetPtr<Scalar>> parallel_lanes;
	parallel_lanes.push_back(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(
			LayerPtr<Scalar>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 5, init, 1, 0)))));													// 0
	parallel_lanes.push_back(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(
			LayerPtr<Scalar>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 3, init, 5, 2)))));													// 1
	comp_mods.push_back(NeuralNetPtr<Scalar>(new ParallelNeuralNetwork<Scalar>(std::move(parallel_lanes))));
	std::vector<LayerPtr<Scalar>> layers1(7);
	layers1[0] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(comp_mods[0]->get_output_dims()));														// 2
	layers1[1] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers1[0]->get_output_dims()));													// 3
	layers1[2] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers1[1]->get_output_dims()));															// 4
	layers1[3] = LayerPtr<Scalar>(new ConvLayer<Scalar>(layers1[2]->get_output_dims(), 8, init));														// 5
	layers1[4] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(layers1[3]->get_output_dims()));															// 6
	layers1[5] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers1[4]->get_output_dims()));													// 7
	layers1[6] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers1[5]->get_output_dims()));															// 8
	comp_mods.push_back(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(std::move(layers1))));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(std::move(comp_mods)), false));
	std::vector<LayerPtr<Scalar>> layers2(3);
	layers2[0] = LayerPtr<Scalar>(new ConvLayer<Scalar>(res_modules[0].first.get_output_dims(), 8, init));												// 9
	layers2[1] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers2[0]->get_output_dims()));													// 10
	layers2[2] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers2[1]->get_output_dims()));															// 11
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(
			std::move(layers2)))), false));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(
			LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(res_modules[1].first.get_output_dims()))))), false));											// 12
	std::vector<LayerPtr<Scalar>> layers3(3);
	layers3[0] = LayerPtr<Scalar>(new ConvLayer<Scalar>(res_modules[2].first.get_output_dims(), 8, init));												// 13
	layers3[1] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers3[0]->get_output_dims()));													// 14
	layers3[2] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers3[1]->get_output_dims()));															// 15
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(
			std::move(layers3)))), false));
	std::vector<LayerPtr<Scalar>> layers4(5);
	layers4[0] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(res_modules[3].first.get_output_dims()));													// 16
	layers4[1] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers4[0]->get_output_dims()));													// 17
	layers4[2] = LayerPtr<Scalar>(new FCLayer<Scalar>(layers4[1]->get_output_dims(), 50, init));														// 18
	layers4[3] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers4[2]->get_output_dims()));													// 19
	layers4[4] = LayerPtr<Scalar>(new FCLayer<Scalar>(layers4[3]->get_output_dims(), 1, init));															// 20
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(
			std::move(layers4)))), false));
	ResidualNeuralNetwork<Scalar> nn(res_modules);
	nn.init();
	LossSharedPtr<Scalar> loss(new QuadraticLoss<Scalar>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar> opt(loss, reg, 20);
//	std::cout << opt.verify_gradients(nn, test_prov) << std::endl;
	opt.optimize(nn, training_prov, test_prov, 500);
	return 0;
};
