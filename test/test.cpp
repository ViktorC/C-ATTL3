/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <cmath>
#include <cstddef>
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
#include <Utils.h>
#include <vector>
#include <WeightInitialization.h>

typedef double Scalar;
static constexpr size_t RANK = 3;

int main() {
	using namespace cattle;
	TensorPtr<Scalar,RANK + 1> training_obs_ptr = TensorPtr<Scalar,RANK + 1>(new Tensor<Scalar,RANK + 1>(80, 32, 32, 3));
	TensorPtr<Scalar,RANK + 1> training_obj_ptr = TensorPtr<Scalar,RANK + 1>(new Tensor<Scalar,RANK + 1>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	TensorPtr<Scalar,RANK + 1> test_obs_ptr = TensorPtr<Scalar,RANK + 1>(new Tensor<Scalar,RANK + 1>(20, 32, 32, 3));
	TensorPtr<Scalar,RANK + 1> test_obj_ptr = TensorPtr<Scalar,RANK + 1>(new Tensor<Scalar,RANK + 1>(20, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<std::pair<CompositeNeuralNetwork<Scalar,RANK>,bool>> res_modules;
	std::vector<NeuralNetPtr<Scalar,RANK>> comp_mods;
	std::vector<NeuralNetPtr<Scalar,RANK>> parallel_lanes;
	parallel_lanes.push_back(NeuralNetPtr<Scalar,RANK>(new SequentialNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 5, init, 1, 0)))));				// 0
	parallel_lanes.push_back(NeuralNetPtr<Scalar,RANK>(new SequentialNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 3, init, 5, 2)))));				// 1
	comp_mods.push_back(NeuralNetPtr<Scalar,RANK>(new ParallelNeuralNetwork<Scalar>(std::move(parallel_lanes))));
	std::vector<LayerPtr<Scalar,RANK>> layers1(7);
	layers1[0] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(comp_mods[0]->get_output_dims()));					// 2
	layers1[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers1[0]->get_output_dims()));		// 3
	layers1[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers1[1]->get_output_dims()));					// 4
	layers1[3] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(layers1[2]->get_output_dims(), 8, init));					// 5
	layers1[4] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(layers1[3]->get_output_dims()));						// 6
	layers1[5] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers1[4]->get_output_dims()));		// 7
	layers1[6] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers1[5]->get_output_dims()));					// 8
	comp_mods.push_back(NeuralNetPtr<Scalar,RANK>(new SequentialNeuralNetwork<Scalar,RANK>(std::move(layers1))));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK>(std::move(comp_mods)), false));
	std::vector<LayerPtr<Scalar,RANK>> layers2(3);
	layers2[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(res_modules[0].first.get_output_dims(), 8, init));			// 9
	layers2[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers2[0]->get_output_dims()));		// 10
	layers2[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers2[1]->get_output_dims()));					// 11
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK>(NeuralNetPtr<Scalar,RANK>(
			new SequentialNeuralNetwork<Scalar,RANK>(std::move(layers2)))), false));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK>(NeuralNetPtr<Scalar,RANK>(
			new SequentialNeuralNetwork<Scalar,RANK>(LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(
					res_modules[1].first.get_output_dims()))))), false));												// 12
	std::vector<LayerPtr<Scalar,RANK>> layers3(3);
	layers3[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(res_modules[2].first.get_output_dims(), 8, init));			// 13
	layers3[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers3[0]->get_output_dims()));		// 14
	layers3[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers3[1]->get_output_dims()));					// 15
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK>(NeuralNetPtr<Scalar,RANK>(
			new SequentialNeuralNetwork<Scalar,RANK>(std::move(layers3)))), false));
	std::vector<LayerPtr<Scalar,RANK>> layers4(5);
	layers4[0] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(res_modules[3].first.get_output_dims()));			// 16
	layers4[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers4[0]->get_output_dims()));		// 17
	layers4[2] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers4[1]->get_output_dims(), 50, init));				// 18
	layers4[3] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers4[2]->get_output_dims()));		// 19
	layers4[4] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers4[3]->get_output_dims(), 1, init));				// 20
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK>(NeuralNetPtr<Scalar,RANK>(
			new SequentialNeuralNetwork<Scalar,RANK>(std::move(layers4)))), false));
	ResidualNeuralNetwork<Scalar,RANK> nn(res_modules);
	nn.init();
	LossSharedPtr<Scalar,RANK> loss(new QuadraticLoss<Scalar,RANK>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK> opt(loss, reg, 20);
//	std::cout << opt.verify_gradients(nn, test_prov) << std::endl;
	opt.optimize(nn, training_prov, test_prov, 500);
	return 0;
};
