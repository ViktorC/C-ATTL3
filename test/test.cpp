/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "DataProvider.h"
#include "Dimensions.h"
#include "Layer.h"
#include "Loss.h"
#include "NeuralNetwork.h"
#include "Optimizer.h"
#include "Preprocessor.h"
#include "RegularizationPenalty.h"
#include "Utils.h"
#include "WeightInitialization.h"

using namespace cattle;
typedef double Scalar;

static int test_parallel() {
	const std::size_t RANK = 3;
	TensorPtr<Scalar,RANK + 1> test_obs_ptr(new Tensor<Scalar,RANK + 1>(5u, 8u, 8u, 3u));
	TensorPtr<Scalar,RANK + 1> test_obj_ptr(new Tensor<Scalar,RANK + 1>(5u, 10u, 1u, 1u));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	MemoryDataProvider<Scalar,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<NeuralNetPtr<Scalar,RANK,false>> lanes(2);
	std::vector<LayerPtr<Scalar,RANK>> layers1(2);
	layers1[0] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(test_prov.get_obs_dims(), 5, init));
	layers1[1] = LayerPtr<Scalar,RANK>(new TanhActivationLayer<Scalar,RANK>(layers1[0]->get_output_dims()));
	lanes[0] = NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers1)));
	std::vector<LayerPtr<Scalar,RANK>> layers2(2);
	layers2[0] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(test_prov.get_obs_dims(), 5, init));
	layers2[1] = LayerPtr<Scalar,RANK>(new TanhActivationLayer<Scalar,RANK>(layers2[0]->get_output_dims()));
	lanes[1] = NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers2)));
	ParallelNeuralNetwork<Scalar,RANK,CONCAT_LO_RANK> pnn(std::move(lanes));
	pnn.init();
	LossSharedPtr<Scalar,RANK,false> loss(new QuadraticLoss<Scalar,RANK,false>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,false> opt(loss, reg, 20);
	return opt.verify_gradients(pnn, test_prov);
}

static int test_residual() {
	const std::size_t RANK = 3;
	TensorPtr<Scalar,RANK + 1> test_obs_ptr(new Tensor<Scalar,RANK + 1>(5u, 32u, 32u, 3u));
	TensorPtr<Scalar,RANK + 1> test_obj_ptr(new Tensor<Scalar,RANK + 1>(5u, 1u, 1u, 1u));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	MemoryDataProvider<Scalar,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<std::pair<CompositeNeuralNetwork<Scalar,RANK,false>,bool>> res_modules;
	std::vector<NeuralNetPtr<Scalar,RANK,false>> comp_mods;
	std::vector<NeuralNetPtr<Scalar,RANK,false>> parallel_lanes;
	parallel_lanes.push_back(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(test_prov.get_obs_dims(), 5, init, 1, 0)))));
	parallel_lanes.push_back(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(test_prov.get_obs_dims(), 3, init, 5, 2)))));
	comp_mods.push_back(NeuralNetPtr<Scalar,RANK,false>(
			new ParallelNeuralNetwork<Scalar,RANK>(std::move(parallel_lanes))));
	std::vector<LayerPtr<Scalar,RANK>> layers1(7);
	layers1[0] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(comp_mods[0]->get_output_dims()));
	layers1[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers1[0]->get_output_dims()));
	layers1[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers1[1]->get_output_dims()));
	layers1[3] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(layers1[2]->get_output_dims(), 8, init));
	layers1[4] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(layers1[3]->get_output_dims()));
	layers1[5] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers1[4]->get_output_dims()));
	layers1[6] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers1[5]->get_output_dims()));
	comp_mods.push_back(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers1))));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(std::move(comp_mods)), false));
	std::vector<LayerPtr<Scalar,RANK>> layers2(3);
	layers2[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(res_modules[0].first.get_output_dims(), 8, init));
	layers2[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers2[0]->get_output_dims()));
	layers2[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers2[1]->get_output_dims()));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers2)))), false));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(
					res_modules[1].first.get_output_dims()))))), false));
	std::vector<LayerPtr<Scalar,RANK>> layers3(3);
	layers3[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(res_modules[2].first.get_output_dims(), 8, init));
	layers3[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers3[0]->get_output_dims()));
	layers3[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers3[1]->get_output_dims()));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers3)))), false));
	std::vector<LayerPtr<Scalar,RANK>> layers4(5);
	layers4[0] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(res_modules[3].first.get_output_dims()));
	layers4[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers4[0]->get_output_dims()));
	layers4[2] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers4[1]->get_output_dims(), 50, init));
	layers4[3] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers4[2]->get_output_dims()));
	layers4[4] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers4[3]->get_output_dims(), 1, init));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers4)))), false));
	ResidualNeuralNetwork<Scalar,RANK> nn(res_modules);
	nn.init();
	LossSharedPtr<Scalar,RANK,false> loss(new QuadraticLoss<Scalar,RANK,false>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,false> opt(loss, reg, 20);
	return opt.verify_gradients(nn, test_prov, 1e-5, 1e-4, 1e-1);
}

static int test_dense() {
	const std::size_t RANK = 3;
	TensorPtr<Scalar,RANK + 1> test_obs_ptr(new Tensor<Scalar,RANK + 1>(5u, 8u, 8u, 2u));
	TensorPtr<Scalar,RANK + 1> test_obj_ptr(new Tensor<Scalar,RANK + 1>(5u, 64u, 8u, 2u));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	MemoryDataProvider<Scalar,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<CompositeNeuralNetwork<Scalar,RANK,false>> modules;
	modules.push_back(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(test_prov.get_obs_dims(), 2, init))))));
	modules.push_back(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(modules[0].get_input_dims().add_along_rank(modules[0].get_output_dims(), 0), 2, init))))));
	modules.push_back(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(modules[1].get_input_dims().add_along_rank(modules[1].get_output_dims(), 0), 2, init))))));
	DenseNeuralNetwork<Scalar,RANK,LOWEST_RANK> dnn(std::move(modules));
	dnn.init();
	LossSharedPtr<Scalar,RANK,false> loss(new QuadraticLoss<Scalar,RANK,false>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,false> opt(loss, reg, 20);
	return opt.verify_gradients(dnn, test_prov);
}

static int test_seqnn() {
	const std::size_t RANK = 3;
	TensorPtr<Scalar,RANK + 2> test_obs_ptr(new Tensor<Scalar,RANK + 2>(5u, 10u, 8u, 8u, 3u));
	TensorPtr<Scalar,RANK + 2> test_obj_ptr(new Tensor<Scalar,RANK + 2>(5u, 10u, 4u, 1u, 1u));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	MemoryDataProvider<Scalar,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new GlorotWeightInitialization<Scalar>());
	std::vector<LayerPtr<Scalar,RANK>> layers(5);
	layers[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(test_prov.get_obs_dims(), 2, init));
	layers[1] = LayerPtr<Scalar,RANK>(new TanhActivationLayer<Scalar,RANK>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(layers[1]->get_output_dims(), 5, init));
	layers[3] = LayerPtr<Scalar,RANK>(new TanhActivationLayer<Scalar,RANK>(layers[2]->get_output_dims()));
	layers[4] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers[3]->get_output_dims(), 4, init));
	SequentialNeuralNetwork<Scalar,RANK> seqnn(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers))));
	LossSharedPtr<Scalar,RANK,true> loss(new QuadraticLoss<Scalar,RANK,true>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,true> opt(loss, reg, 20);
	return opt.verify_gradients(seqnn, test_prov);
}

static int test_rnn() {
	const std::size_t RANK = 3;
	TensorPtr<Scalar,RANK + 2> test_obs_ptr(new Tensor<Scalar,RANK + 2>(5u, 5u, 16u, 16u, 3u));
	TensorPtr<Scalar,RANK + 2> test_obj_ptr(new Tensor<Scalar,RANK + 2>(5u, 3u, 2u, 1u, 1u));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	MemoryDataProvider<Scalar,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new OrthogonalWeightInitialization<Scalar>());
	KernelPtr<Scalar,RANK> input_kernel(new ConvLayer<Scalar>(test_prov.get_obs_dims(), 5, init));
	KernelPtr<Scalar,RANK> state_kernel(new ConvLayer<Scalar>(input_kernel->get_output_dims(), 5, init));
	KernelPtr<Scalar,RANK> output_kernel(new FCLayer<Scalar,RANK>(input_kernel->get_output_dims(), 2, init));
	ActivationPtr<Scalar,RANK> state_act(new SigmoidActivationLayer<Scalar,RANK>(input_kernel->get_output_dims()));
	ActivationPtr<Scalar,RANK> output_act(new IdentityActivationLayer<Scalar,RANK>(output_kernel->get_output_dims()));
	RecurrentNeuralNetwork<Scalar,RANK> rnn(std::move(input_kernel), std::move(state_kernel), std::move(output_kernel), std::move(state_act),
			std::move(output_act), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); }, false, true);
	rnn.init();
	LossSharedPtr<Scalar,RANK,true> loss(new QuadraticLoss<Scalar,RANK,true>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,true> opt(loss, reg, 20);
	return opt.verify_gradients(rnn, test_prov);
}

static int test_lstm() {
	const std::size_t RANK = 1;
	TensorPtr<Scalar,RANK + 2> test_obs_ptr(new Tensor<Scalar,RANK + 2>(10u, 5u, 32u));
	TensorPtr<Scalar,RANK + 2> test_obj_ptr(new Tensor<Scalar,RANK + 2>(10u, 3u, 5u));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	MemoryDataProvider<Scalar,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new OrthogonalWeightInitialization<Scalar>());
	const Dimensions<std::size_t,RANK>& input_dims = test_prov.get_obs_dims();
	const Dimensions<std::size_t,RANK>& output_dims = test_prov.get_obj_dims();
	KernelPtr<Scalar,RANK> forget_input_kernel(new FCLayer<Scalar,RANK>(input_dims, 5, init));
	KernelPtr<Scalar,RANK> forget_output_kernel(new FCLayer<Scalar,RANK>(output_dims, 5, init));
	KernelPtr<Scalar,RANK> write_input_kernel(new FCLayer<Scalar,RANK>(input_dims, 5, init));
	KernelPtr<Scalar,RANK> write_output_kernel(new FCLayer<Scalar,RANK>(output_dims, 5, init));
	KernelPtr<Scalar,RANK> candidate_input_kernel(new FCLayer<Scalar,RANK>(input_dims, 5, init));
	KernelPtr<Scalar,RANK> candidate_output_kernel(new FCLayer<Scalar,RANK>(output_dims, 5, init));
	KernelPtr<Scalar,RANK> read_input_kernel(new FCLayer<Scalar,RANK>(input_dims, 5, init));
	KernelPtr<Scalar,RANK> read_output_kernel(new FCLayer<Scalar,RANK>(output_dims, 5, init));
	ActivationPtr<Scalar,RANK> forget_act(new SigmoidActivationLayer<Scalar,RANK>(output_dims));
	ActivationPtr<Scalar,RANK> write_act(new SigmoidActivationLayer<Scalar,RANK>(output_dims));
	ActivationPtr<Scalar,RANK> candidate_act(new TanhActivationLayer<Scalar,RANK>(output_dims));
	ActivationPtr<Scalar,RANK> state_act(new TanhActivationLayer<Scalar,RANK>(output_dims));
	ActivationPtr<Scalar,RANK> read_act(new SigmoidActivationLayer<Scalar,RANK>(output_dims));
	LSTMNeuralNetwork<Scalar,RANK> lstm(std::move(forget_input_kernel), std::move(forget_output_kernel), std::move(write_input_kernel),
			std::move(write_output_kernel), std::move(candidate_input_kernel), std::move(candidate_output_kernel), std::move(read_input_kernel),
			std::move(read_output_kernel), std::move(forget_act), std::move(write_act), std::move(candidate_act), std::move(state_act),
			std::move(read_act), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); }, false, true);
	lstm.init();
	LossSharedPtr<Scalar,RANK,true> loss(new QuadraticLoss<Scalar,RANK,true>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,true> opt(loss, reg, 20);
	return opt.verify_gradients(lstm, test_prov);
}

static int test_bdrnn() {
	const std::size_t RANK = 3;
	TensorPtr<Scalar,RANK + 2> test_obs_ptr(new Tensor<Scalar,RANK + 2>(5u, 7u, 8u, 8u, 3u));
	TensorPtr<Scalar,RANK + 2> test_obj_ptr(new Tensor<Scalar,RANK + 2>(5u, 3u, 4u, 1u, 1u));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	MemoryDataProvider<Scalar,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new OrthogonalWeightInitialization<Scalar>());
	KernelPtr<Scalar,RANK> input_kernel(new ConvLayer<Scalar>(test_prov.get_obs_dims(), 5, init));
	KernelPtr<Scalar,RANK> state_kernel(new ConvLayer<Scalar>(input_kernel->get_output_dims(), 5, init));
	KernelPtr<Scalar,RANK> output_kernel(new FCLayer<Scalar,RANK>(input_kernel->get_output_dims(), 2, init));
	ActivationPtr<Scalar,RANK> state_act(new SigmoidActivationLayer<Scalar,RANK>(input_kernel->get_output_dims()));
	ActivationPtr<Scalar,RANK> output_act(new IdentityActivationLayer<Scalar,RANK>(output_kernel->get_output_dims()));
	BidirectionalNeuralNetwork<Scalar,RANK,CONCAT_LO_RANK> bdrnn(UnidirNeuralNetPtr<Scalar,RANK>(new RecurrentNeuralNetwork<Scalar,RANK>(
			std::move(input_kernel), std::move(state_kernel), std::move(output_kernel), std::move(state_act), std::move(output_act),
			[](int input_seq_length) { return std::make_pair(3, 2); }, false, true)));
	bdrnn.init();
	LossSharedPtr<Scalar,RANK,true> loss(new QuadraticLoss<Scalar,RANK,true>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,true> opt(loss, reg, 20);
	return opt.verify_gradients(bdrnn, test_prov);
}

int main() {
	std::string mnist_folder = "C:\\Users\\A6714\\Downloads\\mnist\\";
	MNISTDataProvider<float> prov(mnist_folder + "train-images.idx3-ubyte", mnist_folder + "train-labels.idx1-ubyte");
//	MNISTDataProvider<float> test_prov(mnist_folder + "t10k-images.idx3-ubyte", mnist_folder + "t10k-labels.idx1-ubyte");
	PartitionDataProvider<float,3,false> training_prov(prov, 178, 1000);
	PartitionDataProvider<float,3,false> test_prov(prov, 7981, 500);
	WeightInitSharedPtr<float> init(new LeCunWeightInitialization<float>());
	std::vector<LayerPtr<float,3>> layers(12);
	layers[0] = LayerPtr<float,3>(new ConvLayer<float>(training_prov.get_obs_dims(), 6, init, 5, 2));
	layers[1] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<float,3>(new ConvLayer<float>(layers[2]->get_output_dims(), 16, init, 5, 0));
	layers[4] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<float,3>(new FCLayer<float,3>(layers[5]->get_output_dims(), 120, init));
	layers[7] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[6]->get_output_dims()));
	layers[8] = LayerPtr<float,3>(new FCLayer<float,3>(layers[7]->get_output_dims(), 84, init));
	layers[9] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[8]->get_output_dims()));
	layers[10] = LayerPtr<float,3>(new FCLayer<float,3>(layers[9]->get_output_dims(), 10, init));
	layers[11] = LayerPtr<float,3>(new SoftmaxActivationLayer<float,3>(layers[10]->get_output_dims()));
	FeedforwardNeuralNetwork<float,3> nn(std::move(layers));
	nn.init();
	std::cout << nn << std::endl << std::endl;
	LossSharedPtr<float,3,false> loss(new CrossEntropyLoss<float,3,false>());
	RegPenSharedPtr<float> reg(new ElasticNetRegularizationPenalty<float>(1e-4, 1e-4));
	NadamOptimizer<float,3,false> opt(loss, reg, 64);
	opt.optimize(nn, training_prov, test_prov, 500);
//	assert(test_parallel() & test_residual() & test_dense() & test_seqnn() & test_rnn() & test_lstm() & test_bdrnn());
	return 0;
}
