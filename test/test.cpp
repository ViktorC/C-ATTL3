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

static int test_rnn() {
	using namespace cattle;
	typedef double Scalar;
	const int RANK = 3;
	const bool SEQ = true;
	TensorPtr<Scalar,RANK,SEQ> training_obs_ptr = TensorPtr<Scalar,RANK,SEQ>(new Tensor<Scalar,RANK + SEQ + 1>(80, 5, 16, 16, 3));
	TensorPtr<Scalar,RANK,SEQ> training_obj_ptr = TensorPtr<Scalar,RANK,SEQ>(new Tensor<Scalar,RANK + SEQ + 1>(80, 3, 2, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,SEQ> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	TensorPtr<Scalar,RANK,SEQ> test_obs_ptr = TensorPtr<Scalar,RANK,SEQ>(new Tensor<Scalar,RANK + SEQ + 1>(20, 5, 16, 16, 3));
	TensorPtr<Scalar,RANK,SEQ> test_obj_ptr = TensorPtr<Scalar,RANK,SEQ>(new Tensor<Scalar,RANK + SEQ + 1>(20, 3, 2, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,SEQ> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new OrthogonalWeightInitialization<Scalar>());
	KernelPtr<Scalar,RANK> u_kernel = KernelPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 5, init));
	KernelPtr<Scalar,RANK> v_kernel = KernelPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(u_kernel->get_output_dims(), 2, init));
	KernelPtr<Scalar,RANK> w_kernel = KernelPtr<Scalar,RANK>(new ConvLayer<Scalar>(u_kernel->get_output_dims(), 5, init));
	ActivationPtr<Scalar,RANK> state_act = ActivationPtr<Scalar,RANK>(
			new SigmoidActivationLayer<Scalar,RANK>(u_kernel->get_output_dims()));
	ActivationPtr<Scalar,RANK> output_act = ActivationPtr<Scalar,RANK>(
			new IdentityActivationLayer<Scalar,RANK>(v_kernel->get_output_dims()));
	RecurrentNeuralNetwork<Scalar,RANK> rnn(std::move(u_kernel), std::move(v_kernel), std::move(w_kernel), std::move(state_act),
			std::move(output_act), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); });
	rnn.init();
	LossSharedPtr<Scalar,RANK,SEQ> loss(new QuadraticLoss<Scalar,RANK,SEQ>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,SEQ> opt(loss, reg, 20);
	std::cout << opt.verify_gradients(rnn, test_prov) << std::endl;
//	opt.optimize(nn, training_prov, test_prov, 500);
	return 0;
}

static int test_cnn() {
	using namespace cattle;
	typedef double Scalar;
	const int RANK = 3;
	const bool SEQ = false;
	TensorPtr<Scalar,RANK,SEQ> training_obs_ptr = TensorPtr<Scalar,RANK,SEQ>(new Tensor<Scalar,RANK + 1>(80, 32, 32, 3));
	TensorPtr<Scalar,RANK,SEQ> training_obj_ptr = TensorPtr<Scalar,RANK,SEQ>(new Tensor<Scalar,RANK + 1>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,SEQ> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	TensorPtr<Scalar,RANK,SEQ> test_obs_ptr = TensorPtr<Scalar,RANK,SEQ>(new Tensor<Scalar,RANK + 1>(5, 32, 32, 3));
	TensorPtr<Scalar,RANK,SEQ> test_obj_ptr = TensorPtr<Scalar,RANK,SEQ>(new Tensor<Scalar,RANK + 1>(5, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,SEQ> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<std::pair<CompositeNeuralNetwork<Scalar,RANK,SEQ>,bool>> res_modules;
	std::vector<NeuralNetPtr<Scalar,RANK,SEQ>> comp_mods;
	std::vector<NeuralNetPtr<Scalar,RANK,SEQ>> parallel_lanes;
	parallel_lanes.push_back(NeuralNetPtr<Scalar,RANK,SEQ>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 5, init, 1, 0)))));				// 0
	parallel_lanes.push_back(NeuralNetPtr<Scalar,RANK,SEQ>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 3, init, 5, 2)))));				// 1
	comp_mods.push_back(NeuralNetPtr<Scalar,RANK,SEQ>(new ParallelNeuralNetwork<Scalar>(std::move(parallel_lanes))));
	std::vector<LayerPtr<Scalar,RANK>> layers1(7);
	layers1[0] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(comp_mods[0]->get_output_dims()));					// 2
	layers1[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers1[0]->get_output_dims()));		// 3
	layers1[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers1[1]->get_output_dims()));					// 4
	layers1[3] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(layers1[2]->get_output_dims(), 8, init));					// 5
	layers1[4] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(layers1[3]->get_output_dims()));						// 6
	layers1[5] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers1[4]->get_output_dims()));		// 7
	layers1[6] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers1[5]->get_output_dims()));					// 8
	comp_mods.push_back(NeuralNetPtr<Scalar,RANK,SEQ>(new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers1))));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,SEQ>(std::move(comp_mods)), false));
	std::vector<LayerPtr<Scalar,RANK>> layers2(3);
	layers2[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(res_modules[0].first.get_output_dims(), 8, init));			// 9
	layers2[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers2[0]->get_output_dims()));		// 10
	layers2[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers2[1]->get_output_dims()));					// 11
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,SEQ>(NeuralNetPtr<Scalar,RANK,SEQ>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers2)))), false));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,SEQ>(NeuralNetPtr<Scalar,RANK,SEQ>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(
					res_modules[1].first.get_output_dims()))))), false));												// 12
	std::vector<LayerPtr<Scalar,RANK>> layers3(3);
	layers3[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(res_modules[2].first.get_output_dims(), 8, init));			// 13
	layers3[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers3[0]->get_output_dims()));		// 14
	layers3[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers3[1]->get_output_dims()));					// 15
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,SEQ>(NeuralNetPtr<Scalar,RANK,SEQ>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers3)))), false));
	std::vector<LayerPtr<Scalar,RANK>> layers4(5);
	layers4[0] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(res_modules[3].first.get_output_dims()));			// 16
	layers4[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers4[0]->get_output_dims()));		// 17
	layers4[2] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers4[1]->get_output_dims(), 50, init));				// 18
	layers4[3] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers4[2]->get_output_dims()));		// 19
	layers4[4] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers4[3]->get_output_dims(), 1, init));				// 20
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,SEQ>(NeuralNetPtr<Scalar,RANK,SEQ>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers4)))), false));
	ResidualNeuralNetwork<Scalar,RANK> nn(res_modules);
	nn.init();
	LossSharedPtr<Scalar,RANK,SEQ> loss(new QuadraticLoss<Scalar,RANK,SEQ>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,SEQ> opt(loss, reg, 20);
	std::cout << opt.verify_gradients(nn, test_prov) << std::endl;
//	opt.optimize(nn, training_prov, test_prov, 500);
	return 0;
}

int main() {
	return test_rnn();
}
