/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <chrono>
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
typedef double fp;

//static int test_parallel() {
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 1> test_obs_ptr(new Tensor<fp,RANK + 1>(5u, 8u, 8u, 3u));
//	TensorPtr<fp,RANK + 1> test_obj_ptr(new Tensor<fp,RANK + 1>(5u, 10u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new HeWeightInitialization<fp>());
//	std::vector<NeuralNetPtr<fp,RANK,false>> lanes(2);
//	std::vector<LayerPtr<fp,RANK>> layers1(2);
//	layers1[0] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(test_prov.get_obs_dims(), 5, init));
//	layers1[1] = LayerPtr<fp,RANK>(new TanhActivationLayer<fp,RANK>(layers1[0]->get_output_dims()));
//	lanes[0] = NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers1)));
//	std::vector<LayerPtr<fp,RANK>> layers2(2);
//	layers2[0] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(test_prov.get_obs_dims(), 5, init));
//	layers2[1] = LayerPtr<fp,RANK>(new SigmoidActivationLayer<fp,RANK>(layers2[0]->get_output_dims()));
//	lanes[1] = NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers2)));
//	ParallelNeuralNetwork<fp,RANK,CONCAT_LO_RANK> pnn(std::move(lanes));
//	pnn.init();
//	LossSharedPtr<fp,RANK,false> loss(new QuadraticLoss<fp,RANK,false>());
//	RegPenSharedPtr<fp> reg(new ElasticNetRegularizationPenalty<fp>());
//	NadamOptimizer<fp,RANK,false> opt(loss, reg, 20);
//	return opt.verify_gradients(pnn, test_prov);
//}
//
//static int test_residual() {
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 1> test_obs_ptr(new Tensor<fp,RANK + 1>(5u, 32u, 32u, 3u));
//	TensorPtr<fp,RANK + 1> test_obj_ptr(new Tensor<fp,RANK + 1>(5u, 1u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new HeWeightInitialization<fp>());
//	std::vector<std::pair<CompositeNeuralNetwork<fp,RANK,false>,bool>> res_modules;
//	std::vector<NeuralNetPtr<fp,RANK,false>> comp_mods;
//	std::vector<NeuralNetPtr<fp,RANK,false>> parallel_lanes;
//	parallel_lanes.push_back(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(test_prov.get_obs_dims(), 5, init, 1, 1, 0)))));
//	parallel_lanes.push_back(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(test_prov.get_obs_dims(), 3, init, 5, 5, 2)))));
//	comp_mods.push_back(NeuralNetPtr<fp,RANK,false>(
//			new ParallelNeuralNetwork<fp,RANK>(std::move(parallel_lanes))));
//	std::vector<LayerPtr<fp,RANK>> layers1(7);
//	layers1[0] = LayerPtr<fp,RANK>(new MaxPoolingLayer<fp>(comp_mods[0]->get_output_dims()));
//	layers1[1] = LayerPtr<fp,RANK>(new LeakyReLUActivationLayer<fp,RANK>(layers1[0]->get_output_dims()));
//	layers1[2] = LayerPtr<fp,RANK>(new BatchNormLayer<fp,RANK>(layers1[1]->get_output_dims()));
//	layers1[3] = LayerPtr<fp,RANK>(new ConvLayer<fp>(layers1[2]->get_output_dims(), 8, init));
//	layers1[4] = LayerPtr<fp,RANK>(new SumPoolingLayer<fp>(layers1[3]->get_output_dims()));
//	layers1[5] = LayerPtr<fp,RANK>(new ELUActivationLayer<fp,RANK>(layers1[4]->get_output_dims()));
//	layers1[6] = LayerPtr<fp,RANK>(new BatchNormLayer<fp,RANK>(layers1[5]->get_output_dims()));
//	comp_mods.push_back(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers1))));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(std::move(comp_mods)), false));
//	std::vector<LayerPtr<fp,RANK>> layers2(3);
//	layers2[0] = LayerPtr<fp,RANK>(new ConvLayer<fp>(res_modules[0].first.get_output_dims(), 8, init));
//	layers2[1] = LayerPtr<fp,RANK>(new PReLUActivationLayer<fp,RANK>(layers2[0]->get_output_dims()));
//	layers2[2] = LayerPtr<fp,RANK>(new BatchNormLayer<fp,RANK>(layers2[1]->get_output_dims()));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers2)))), false));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(LayerPtr<fp,RANK>(new MaxPoolingLayer<fp>(
//					res_modules[1].first.get_output_dims(), 3, 2, 2, 1))))), false));
//	std::vector<LayerPtr<fp,RANK>> layers3(3);
//	layers3[0] = LayerPtr<fp,RANK>(new ConvLayer<fp>(res_modules[2].first.get_output_dims(), 8, init));
//	layers3[1] = LayerPtr<fp,RANK>(new LeakyReLUActivationLayer<fp,RANK>(layers3[0]->get_output_dims()));
//	layers3[2] = LayerPtr<fp,RANK>(new BatchNormLayer<fp,RANK>(layers3[1]->get_output_dims()));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers3)))), false));
//	std::vector<LayerPtr<fp,RANK>> layers4(5);
//	layers4[0] = LayerPtr<fp,RANK>(new MeanPoolingLayer<fp>(res_modules[3].first.get_output_dims()));
//	layers4[1] = LayerPtr<fp,RANK>(new LeakyReLUActivationLayer<fp,RANK>(layers4[0]->get_output_dims()));
//	layers4[2] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(layers4[1]->get_output_dims(), 50, init));
//	layers4[3] = LayerPtr<fp,RANK>(new LeakyReLUActivationLayer<fp,RANK>(layers4[2]->get_output_dims()));
//	layers4[4] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(layers4[3]->get_output_dims(), 1, init));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers4)))), false));
//	ResidualNeuralNetwork<fp,RANK> nn(res_modules);
//	nn.init();
//	LossSharedPtr<fp,RANK,false> loss(new QuadraticLoss<fp,RANK,false>());
//	RegPenSharedPtr<fp> reg(new ElasticNetRegularizationPenalty<fp>());
//	NadamOptimizer<fp,RANK,false> opt(loss, reg, 20);
//	return opt.verify_gradients(nn, test_prov, 1e-5, 1e-4, 1e-1);
//}
//
//static int test_dense() {
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 1> test_obs_ptr(new Tensor<fp,RANK + 1>(5u, 8u, 8u, 2u));
//	TensorPtr<fp,RANK + 1> test_obj_ptr(new Tensor<fp,RANK + 1>(5u, 64u, 8u, 2u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new HeWeightInitialization<fp>());
//	std::vector<CompositeNeuralNetwork<fp,RANK,false>> modules;
//	modules.push_back(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(test_prov.get_obs_dims(), 2, init))))));
//	modules.push_back(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(modules[0].get_input_dims().add_along_rank(modules[0].get_output_dims(), 0), 2, init))))));
//	modules.push_back(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(modules[1].get_input_dims().add_along_rank(modules[1].get_output_dims(), 0), 2, init))))));
//	DenseNeuralNetwork<fp,RANK,LOWEST_RANK> dnn(std::move(modules));
//	dnn.init();
//	LossSharedPtr<fp,RANK,false> loss(new QuadraticLoss<fp,RANK,false>());
//	RegPenSharedPtr<fp> reg(new ElasticNetRegularizationPenalty<fp>());
//	NadamOptimizer<fp,RANK,false> opt(loss, reg, 20);
//	return opt.verify_gradients(dnn, test_prov);
//}
//
//static int test_seqnn() {
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 2> test_obs_ptr(new Tensor<fp,RANK + 2>(5u, 10u, 8u, 8u, 3u));
//	TensorPtr<fp,RANK + 2> test_obj_ptr(new Tensor<fp,RANK + 2>(5u, 10u, 4u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new GlorotWeightInitialization<fp>());
//	std::vector<LayerPtr<fp,RANK>> layers(5);
//	layers[0] = LayerPtr<fp,RANK>(new ConvLayer<fp>(test_prov.get_obs_dims(), 2, init));
//	layers[1] = LayerPtr<fp,RANK>(new TanhActivationLayer<fp,RANK>(layers[0]->get_output_dims()));
//	layers[2] = LayerPtr<fp,RANK>(new ConvLayer<fp>(layers[1]->get_output_dims(), 5, init));
//	layers[3] = LayerPtr<fp,RANK>(new TanhActivationLayer<fp,RANK>(layers[2]->get_output_dims()));
//	layers[4] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(layers[3]->get_output_dims(), 4, init));
//	SequentialNeuralNetwork<fp,RANK> seqnn(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers))));
//	LossSharedPtr<fp,RANK,true> loss(new QuadraticLoss<fp,RANK,true>());
//	RegPenSharedPtr<fp> reg(new ElasticNetRegularizationPenalty<fp>());
//	NadamOptimizer<fp,RANK,true> opt(loss, reg, 20);
//	return opt.verify_gradients(seqnn, test_prov);
//}
//
//static int test_rnn() {
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 2> test_obs_ptr(new Tensor<fp,RANK + 2>(5u, 5u, 16u, 16u, 3u));
//	TensorPtr<fp,RANK + 2> test_obj_ptr(new Tensor<fp,RANK + 2>(5u, 3u, 2u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new OrthogonalWeightInitialization<fp>());
//	KernelPtr<fp,RANK> input_kernel(new ConvLayer<fp>(test_prov.get_obs_dims(), 5, init));
//	KernelPtr<fp,RANK> state_kernel(new ConvLayer<fp>(input_kernel->get_output_dims(), 5, init));
//	KernelPtr<fp,RANK> output_kernel(new FCLayer<fp,RANK>(input_kernel->get_output_dims(), 2, init));
//	ActivationPtr<fp,RANK> state_act(new SigmoidActivationLayer<fp,RANK>(input_kernel->get_output_dims()));
//	ActivationPtr<fp,RANK> output_act(new IdentityActivationLayer<fp,RANK>(output_kernel->get_output_dims()));
//	RecurrentNeuralNetwork<fp,RANK> rnn(std::move(input_kernel), std::move(state_kernel), std::move(output_kernel), std::move(state_act),
//			std::move(output_act), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); }, false, true);
//	rnn.init();
//	LossSharedPtr<fp,RANK,true> loss(new QuadraticLoss<fp,RANK,true>());
//	RegPenSharedPtr<fp> reg(new ElasticNetRegularizationPenalty<fp>());
//	NadamOptimizer<fp,RANK,true> opt(loss, reg, 20);
//	return opt.verify_gradients(rnn, test_prov);
//}
//
//static int test_lstm() {
//	const std::size_t RANK = 1;
//	TensorPtr<fp,RANK + 2> test_obs_ptr(new Tensor<fp,RANK + 2>(10u, 5u, 32u));
//	TensorPtr<fp,RANK + 2> test_obj_ptr(new Tensor<fp,RANK + 2>(10u, 3u, 5u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new OrthogonalWeightInitialization<fp>());
//	const Dimensions<std::size_t,RANK>& input_dims = test_prov.get_obs_dims();
//	const Dimensions<std::size_t,RANK>& output_dims = test_prov.get_obj_dims();
//	KernelPtr<fp,RANK> forget_input_kernel(new FCLayer<fp,RANK>(input_dims, 5, init));
//	KernelPtr<fp,RANK> forget_output_kernel(new FCLayer<fp,RANK>(output_dims, 5, init));
//	KernelPtr<fp,RANK> write_input_kernel(new FCLayer<fp,RANK>(input_dims, 5, init));
//	KernelPtr<fp,RANK> write_output_kernel(new FCLayer<fp,RANK>(output_dims, 5, init));
//	KernelPtr<fp,RANK> candidate_input_kernel(new FCLayer<fp,RANK>(input_dims, 5, init));
//	KernelPtr<fp,RANK> candidate_output_kernel(new FCLayer<fp,RANK>(output_dims, 5, init));
//	KernelPtr<fp,RANK> read_input_kernel(new FCLayer<fp,RANK>(input_dims, 5, init));
//	KernelPtr<fp,RANK> read_output_kernel(new FCLayer<fp,RANK>(output_dims, 5, init));
//	ActivationPtr<fp,RANK> forget_act(new SigmoidActivationLayer<fp,RANK>(output_dims));
//	ActivationPtr<fp,RANK> write_act(new SigmoidActivationLayer<fp,RANK>(output_dims));
//	ActivationPtr<fp,RANK> candidate_act(new TanhActivationLayer<fp,RANK>(output_dims));
//	ActivationPtr<fp,RANK> state_act(new TanhActivationLayer<fp,RANK>(output_dims));
//	ActivationPtr<fp,RANK> read_act(new SigmoidActivationLayer<fp,RANK>(output_dims));
//	LSTMNeuralNetwork<fp,RANK> lstm(std::move(forget_input_kernel), std::move(forget_output_kernel), std::move(write_input_kernel),
//			std::move(write_output_kernel), std::move(candidate_input_kernel), std::move(candidate_output_kernel), std::move(read_input_kernel),
//			std::move(read_output_kernel), std::move(forget_act), std::move(write_act), std::move(candidate_act), std::move(state_act),
//			std::move(read_act), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); }, false, true);
//	lstm.init();
//	LossSharedPtr<fp,RANK,true> loss(new QuadraticLoss<fp,RANK,true>());
//	RegPenSharedPtr<fp> reg(new ElasticNetRegularizationPenalty<fp>());
//	NadamOptimizer<fp,RANK,true> opt(loss, reg, 20);
//	return opt.verify_gradients(lstm, test_prov);
//}
//
//static int test_bdrnn() {
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 2> test_obs_ptr(new Tensor<fp,RANK + 2>(5u, 7u, 8u, 8u, 3u));
//	TensorPtr<fp,RANK + 2> test_obj_ptr(new Tensor<fp,RANK + 2>(5u, 3u, 4u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new OrthogonalWeightInitialization<fp>());
//	KernelPtr<fp,RANK> input_kernel(new ConvLayer<fp>(test_prov.get_obs_dims(), 5, init));
//	KernelPtr<fp,RANK> state_kernel(new ConvLayer<fp>(input_kernel->get_output_dims(), 5, init));
//	KernelPtr<fp,RANK> output_kernel(new FCLayer<fp,RANK>(input_kernel->get_output_dims(), 2, init));
//	ActivationPtr<fp,RANK> state_act(new SigmoidActivationLayer<fp,RANK>(input_kernel->get_output_dims()));
//	ActivationPtr<fp,RANK> output_act(new IdentityActivationLayer<fp,RANK>(output_kernel->get_output_dims()));
//	BidirectionalNeuralNetwork<fp,RANK,CONCAT_LO_RANK> bdrnn(UnidirNeuralNetPtr<fp,RANK>(new RecurrentNeuralNetwork<fp,RANK>(
//			std::move(input_kernel), std::move(state_kernel), std::move(output_kernel), std::move(state_act), std::move(output_act),
//			[](int input_seq_length) { return std::make_pair(3, 2); }, false, true)));
//	bdrnn.init();
//	LossSharedPtr<fp,RANK,true> loss(new QuadraticLoss<fp,RANK,true>());
//	RegPenSharedPtr<fp> reg(new ElasticNetRegularizationPenalty<fp>());
//	NadamOptimizer<fp,RANK,true> opt(loss, reg, 20);
//	return opt.verify_gradients(bdrnn, test_prov);
//}

int main() {
	std::string mnist_folder = "C:\\Users\\A6714\\Downloads\\mnist\\";
	MNISTDataProvider<float> prov(mnist_folder + "train-images.idx3-ubyte", mnist_folder + "train-labels.idx1-ubyte");
	DataPair<float,3,false> data = prov.get_data(60000);
	TensorPtr<float,4> obs(new Tensor<float,4>(std::move(data.first)));
	TensorPtr<float,4> obj(new Tensor<float,4>(std::move(data.second)));
	MemoryDataProvider<float,3,false> training_prov(std::move(obs), std::move(obj));
	MNISTDataProvider<float> test_prov(mnist_folder + "t10k-images.idx3-ubyte", mnist_folder + "t10k-labels.idx1-ubyte");
	WeightInitSharedPtr<float> init(new LeCunWeightInitialization<float>());
	std::vector<LayerPtr<float,3>> layers(9);
	layers[0] = LayerPtr<float,3>(new ConvLayer<float>(training_prov.get_obs_dims(), 8, init, 5, 5, 2));
	layers[1] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<float,3>(new ConvLayer<float>(layers[2]->get_output_dims(), 8, init, 5, 5, 2));
	layers[4] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<float,3>(new FCLayer<float,3>(layers[5]->get_output_dims(), 150, init));
	layers[7] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[6]->get_output_dims()));
	layers[8] = LayerPtr<float,3>(new FCLayer<float,3>(layers[7]->get_output_dims(), 10, init));
//	layers[0] = LayerPtr<float,3>(new FCLayer<float,3>(training_prov.get_obs_dims(), 500, init));
//	layers[1] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[0]->get_output_dims()));
//	layers[2] = LayerPtr<float,3>(new FCLayer<float,3>(layers[1]->get_output_dims(), 250, init));
//	layers[3] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[2]->get_output_dims()));
//	layers[4] = LayerPtr<float,3>(new FCLayer<float,3>(layers[3]->get_output_dims(), 10, init));
	FeedforwardNeuralNetwork<float,3> nn(std::move(layers));
	nn.init();
	LossSharedPtr<float,3,false> loss(new SoftmaxCrossEntropyLoss<float,3,false>());
	RegPenSharedPtr<float> reg(new NoRegularizationPenalty<float>());
	MomentumAcceleratedSGDOptimizer<float,3,false> opt(loss, reg, 100);
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	opt.optimize(nn, training_prov, test_prov, 1);
	std::cout << "Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
			begin).count() << std::endl;
//	assert(test_parallel() & test_residual() & test_dense() & test_seqnn() & test_rnn() & test_lstm() & test_bdrnn());
	return 0;
}
