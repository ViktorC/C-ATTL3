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

using namespace cattle;
typedef double Scalar;

static int test_residual() {
	const int RANK = 3;
	TensorPtr<Scalar,RANK,false> training_obs_ptr = TensorPtr<Scalar,RANK,false>(new Tensor<Scalar,RANK + 1>(80, 32, 32, 3));
	TensorPtr<Scalar,RANK,false> training_obj_ptr = TensorPtr<Scalar,RANK,false>(new Tensor<Scalar,RANK + 1>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,false> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	TensorPtr<Scalar,RANK,false> test_obs_ptr = TensorPtr<Scalar,RANK,false>(new Tensor<Scalar,RANK + 1>(5, 32, 32, 3));
	TensorPtr<Scalar,RANK,false> test_obj_ptr = TensorPtr<Scalar,RANK,false>(new Tensor<Scalar,RANK + 1>(5, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<std::pair<CompositeNeuralNetwork<Scalar,RANK,false>,bool>> res_modules;
	std::vector<NeuralNetPtr<Scalar,RANK,false>> comp_mods;
	std::vector<NeuralNetPtr<Scalar,RANK,false>> parallel_lanes;
	parallel_lanes.push_back(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 5, init, 1, 0)))));				// 0
	parallel_lanes.push_back(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(
			LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 3, init, 5, 2)))));				// 1
	comp_mods.push_back(NeuralNetPtr<Scalar,RANK,false>(
			new ParallelNeuralNetwork<Scalar,RANK>(std::move(parallel_lanes))));
	std::vector<LayerPtr<Scalar,RANK>> layers1(7);
	layers1[0] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(comp_mods[0]->get_output_dims()));					// 2
	layers1[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers1[0]->get_output_dims()));		// 3
	layers1[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers1[1]->get_output_dims()));					// 4
	layers1[3] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(layers1[2]->get_output_dims(), 8, init));					// 5
	layers1[4] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(layers1[3]->get_output_dims()));						// 6
	layers1[5] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers1[4]->get_output_dims()));		// 7
	layers1[6] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers1[5]->get_output_dims()));					// 8
	comp_mods.push_back(NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers1))));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(std::move(comp_mods)), false));
	std::vector<LayerPtr<Scalar,RANK>> layers2(3);
	layers2[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(res_modules[0].first.get_output_dims(), 8, init));			// 9
	layers2[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers2[0]->get_output_dims()));		// 10
	layers2[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers2[1]->get_output_dims()));					// 11
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers2)))), false));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(
					res_modules[1].first.get_output_dims()))))), false));												// 12
	std::vector<LayerPtr<Scalar,RANK>> layers3(3);
	layers3[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(res_modules[2].first.get_output_dims(), 8, init));			// 13
	layers3[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers3[0]->get_output_dims()));		// 14
	layers3[2] = LayerPtr<Scalar,RANK>(new BatchNormLayer<Scalar,RANK>(layers3[1]->get_output_dims()));					// 15
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers3)))), false));
	std::vector<LayerPtr<Scalar,RANK>> layers4(5);
	layers4[0] = LayerPtr<Scalar,RANK>(new MaxPoolingLayer<Scalar>(res_modules[3].first.get_output_dims()));			// 16
	layers4[1] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers4[0]->get_output_dims()));		// 17
	layers4[2] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers4[1]->get_output_dims(), 50, init));				// 18
	layers4[3] = LayerPtr<Scalar,RANK>(new LeakyReLUActivationLayer<Scalar,RANK>(layers4[2]->get_output_dims()));		// 19
	layers4[4] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers4[3]->get_output_dims(), 1, init));				// 20
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar,RANK,false>(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers4)))), false));
	ResidualNeuralNetwork<Scalar,RANK> nn(res_modules);
	nn.init();
	LossSharedPtr<Scalar,RANK,false> loss(new QuadraticLoss<Scalar,RANK,false>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,false> opt(loss, reg, 20);
	std::cout << opt.verify_gradients(nn, test_prov) << std::endl;
//	opt.optimize(nn, training_prov, test_prov, 500);
	return 0;
}

static int test_parallel() {
	const int RANK = 3;
	TensorPtr<Scalar,RANK,false> training_obs_ptr = TensorPtr<Scalar,RANK,false>(new Tensor<Scalar,RANK + 1>(80, 8, 8, 3));
	TensorPtr<Scalar,RANK,false> training_obj_ptr = TensorPtr<Scalar,RANK,false>(new Tensor<Scalar,RANK + 1>(80, 10, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,false> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	TensorPtr<Scalar,RANK,false> test_obs_ptr = TensorPtr<Scalar,RANK,false>(new Tensor<Scalar,RANK + 1>(5, 8, 8, 3));
	TensorPtr<Scalar,RANK,false> test_obj_ptr = TensorPtr<Scalar,RANK,false>(new Tensor<Scalar,RANK + 1>(5, 10, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<NeuralNetPtr<Scalar,RANK,false>> lanes(2);
	std::vector<LayerPtr<Scalar,RANK>> layers1(2);
	layers1[0] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(training_prov.get_obs_dims(), 5, init));
	layers1[1] = LayerPtr<Scalar,RANK>(new TanhActivationLayer<Scalar,RANK>(layers1[0]->get_output_dims()));
	lanes[0] = NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers1)));
	std::vector<LayerPtr<Scalar,RANK>> layers2(2);
	layers2[0] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(training_prov.get_obs_dims(), 5, init));
	layers2[1] = LayerPtr<Scalar,RANK>(new TanhActivationLayer<Scalar,RANK>(layers2[0]->get_output_dims()));
	lanes[1] = NeuralNetPtr<Scalar,RANK,false>(new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers2)));
	ParallelNeuralNetwork<Scalar,RANK> pnn(std::move(lanes), ParallelNeuralNetwork<Scalar,RANK>::CONCAT_LO_RANK);
	pnn.init();
	LossSharedPtr<Scalar,RANK,false> loss(new QuadraticLoss<Scalar,RANK,false>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,false> opt(loss, reg, 20);
	std::cout << opt.verify_gradients(pnn, test_prov) << std::endl;
//	opt.optimize(pnn, training_prov, test_prov, 500);
	return 0;
}

static int test_rnn() {
	const int RANK = 3;
	TensorPtr<Scalar,RANK,true> training_obs_ptr = TensorPtr<Scalar,RANK,true>(new Tensor<Scalar,RANK + 2>(80, 5, 16, 16, 3));
	TensorPtr<Scalar,RANK,true> training_obj_ptr = TensorPtr<Scalar,RANK,true>(new Tensor<Scalar,RANK + 2>(80, 3, 2, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,true> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	TensorPtr<Scalar,RANK,true> test_obs_ptr = TensorPtr<Scalar,RANK,true>(new Tensor<Scalar,RANK + 2>(20, 5, 16, 16, 3));
	TensorPtr<Scalar,RANK,true> test_obj_ptr = TensorPtr<Scalar,RANK,true>(new Tensor<Scalar,RANK + 2>(20, 3, 2, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new OrthogonalWeightInitialization<Scalar>());
	KernelPtr<Scalar,RANK> u_kernel = KernelPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 5, init));
	KernelPtr<Scalar,RANK> v_kernel = KernelPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(u_kernel->get_output_dims(), 2, init));
	KernelPtr<Scalar,RANK> w_kernel = KernelPtr<Scalar,RANK>(new ConvLayer<Scalar>(u_kernel->get_output_dims(), 5, init));
	ActivationPtr<Scalar,RANK> state_act = ActivationPtr<Scalar,RANK>(
			new SigmoidActivationLayer<Scalar,RANK>(u_kernel->get_output_dims()));
	ActivationPtr<Scalar,RANK> output_act = ActivationPtr<Scalar,RANK>(
			new IdentityActivationLayer<Scalar,RANK>(v_kernel->get_output_dims()));
	RecurrentNeuralNetwork<Scalar,RANK> rnn(std::move(u_kernel), std::move(v_kernel), std::move(w_kernel), std::move(state_act),
			std::move(output_act), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); }, false, true);
	rnn.init();
	LossSharedPtr<Scalar,RANK,true> loss(new QuadraticLoss<Scalar,RANK,true>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,true> opt(loss, reg, 20);
	std::cout << opt.verify_gradients(rnn, test_prov) << std::endl;
//	opt.optimize(rnn, training_prov, test_prov, 500);
	return 0;
}

static int test_seqnn() {
	const int RANK = 3;
	TensorPtr<Scalar,RANK,true> training_obs_ptr = TensorPtr<Scalar,RANK,true>(new Tensor<Scalar,RANK + 2>(80, 10, 8, 8, 3));
	TensorPtr<Scalar,RANK,true> training_obj_ptr = TensorPtr<Scalar,RANK,true>(new Tensor<Scalar,RANK + 2>(80, 10, 2, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,true> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	TensorPtr<Scalar,RANK,true> test_obs_ptr = TensorPtr<Scalar,RANK,true>(new Tensor<Scalar,RANK + 2>(20, 10, 8, 8, 3));
	TensorPtr<Scalar,RANK,true> test_obj_ptr = TensorPtr<Scalar,RANK,true>(new Tensor<Scalar,RANK + 2>(20, 10, 2, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	WeightInitSharedPtr<Scalar> init(new GlorotWeightInitialization<Scalar>());
	std::vector<LayerPtr<Scalar,RANK>> layers(5);
	layers[0] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 2, init));
	layers[1] = LayerPtr<Scalar,RANK>(new TanhActivationLayer<Scalar,RANK>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<Scalar,RANK>(new ConvLayer<Scalar>(layers[1]->get_output_dims(), 5, init));
	layers[3] = LayerPtr<Scalar,RANK>(new TanhActivationLayer<Scalar,RANK>(layers[2]->get_output_dims()));
	layers[4] = LayerPtr<Scalar,RANK>(new FCLayer<Scalar,RANK>(layers[3]->get_output_dims(), 2, init));
	SequentialNeuralNetwork<Scalar,RANK> seqnn(NeuralNetPtr<Scalar,RANK,false>(
			new FeedforwardNeuralNetwork<Scalar,RANK>(std::move(layers))));
	LossSharedPtr<Scalar,RANK,true> loss(new QuadraticLoss<Scalar,RANK,true>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,RANK,true> opt(loss, reg, 20);
	std::cout << opt.verify_gradients(seqnn, test_prov) << std::endl;
//	opt.optimize(seqnn, training_prov, test_prov, 500);
	return 0;
}

int main() {
	return test_residual();
}
