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
#include "Eigen.h"
#include "Layer.h"
#include "Loss.h"
#include "NeuralNetwork.h"
#include "Optimizer.h"
#include "ParameterRegularization.h"
#include "Preprocessor.h"
#include "WeightInitialization.h"

using namespace cattle;
typedef double fp;

//static int test_parallel() {
//	std::cout << "PARALLEL" << std::endl;
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 1> test_obs_ptr(new Tensor<fp,RANK + 1>(5u, 8u, 8u, 3u));
//	TensorPtr<fp,RANK + 1> test_obj_ptr(new Tensor<fp,RANK + 1>(5u, 10u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	PCAPreprocessor<fp,RANK,true,true> preproc;
//	preproc.fit(*test_obs_ptr);
//	preproc.transform(*test_obs_ptr);
//	MemoryDataProvider<fp,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new HeWeightInitialization<fp>());
//	ParamRegSharedPtr<fp> reg(new L2ParameterRegularization<fp>());
//	std::vector<NeuralNetPtr<fp,RANK,false>> lanes(2);
//	std::vector<LayerPtr<fp,RANK>> layers1(2);
//	layers1[0] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(test_prov.get_obs_dims(), 5, init, reg));
//	layers1[1] = LayerPtr<fp,RANK>(new SoftplusActivationLayer<fp,RANK>(layers1[0]->get_output_dims()));
//	lanes[0] = NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers1)));
//	std::vector<LayerPtr<fp,RANK>> layers2(2);
//	layers2[0] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(test_prov.get_obs_dims(), 5, init, reg));
//	layers2[1] = LayerPtr<fp,RANK>(new SoftplusActivationLayer<fp,RANK>(layers2[0]->get_output_dims()));
//	lanes[1] = NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers2)));
//	ParallelNeuralNetwork<fp,RANK,CONCAT_LO_RANK> pnn(std::move(lanes));
//	pnn.init();
//	LossSharedPtr<fp,RANK,false> loss(new QuadraticLoss<fp,RANK,false>());
//	NadamOptimizer<fp,RANK,false> opt(loss, 20);
//	return opt.verify_gradients(pnn, test_prov);
//}
//
//static int test_residual() {
//	std::cout << "RESIDUAL" << std::endl;
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 1> test_obs_ptr(new Tensor<fp,RANK + 1>(5u, 32u, 32u, 3u));
//	TensorPtr<fp,RANK + 1> test_obj_ptr(new Tensor<fp,RANK + 1>(5u, 1u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new HeWeightInitialization<fp>());
//	ParamRegSharedPtr<fp> reg(new L2ParameterRegularization<fp>());
//	ParamRegSharedPtr<fp> reg2(new L2ParameterRegularization<fp>());
//	std::vector<std::pair<CompositeNeuralNetwork<fp,RANK,false>,bool>> res_modules;
//	std::vector<NeuralNetPtr<fp,RANK,false>> comp_mods;
//	std::vector<NeuralNetPtr<fp,RANK,false>> parallel_lanes;
//	parallel_lanes.push_back(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(test_prov.get_obs_dims(), 5, init, reg, 1, 1, 0, 0)))));
//	parallel_lanes.push_back(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(test_prov.get_obs_dims(), 3, init, reg, 5, 5, 2, 2)))));
//	comp_mods.push_back(NeuralNetPtr<fp,RANK,false>(
//			new ParallelNeuralNetwork<fp,RANK>(std::move(parallel_lanes))));
//	std::vector<LayerPtr<fp,RANK>> layers1(7);
//	layers1[0] = LayerPtr<fp,RANK>(new MaxPoolingLayer<fp>(comp_mods[0]->get_output_dims()));
//	layers1[1] = LayerPtr<fp,RANK>(new LeakyReLUActivationLayer<fp,RANK>(layers1[0]->get_output_dims()));
//	layers1[2] = LayerPtr<fp,RANK>(new BatchNormLayer<fp,RANK>(layers1[1]->get_output_dims(), reg, reg2));
//	layers1[3] = LayerPtr<fp,RANK>(new ConvLayer<fp>(layers1[2]->get_output_dims(), 8, init, reg));
//	layers1[4] = LayerPtr<fp,RANK>(new SumPoolingLayer<fp>(layers1[3]->get_output_dims()));
//	layers1[5] = LayerPtr<fp,RANK>(new ELUActivationLayer<fp,RANK>(layers1[4]->get_output_dims()));
//	layers1[6] = LayerPtr<fp,RANK>(new BatchNormLayer<fp,RANK>(layers1[5]->get_output_dims(), reg, reg2));
//	comp_mods.push_back(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers1))));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(std::move(comp_mods)), false));
//	std::vector<LayerPtr<fp,RANK>> layers2(3);
//	layers2[0] = LayerPtr<fp,RANK>(new ConvLayer<fp>(res_modules[0].first.get_output_dims(), 8, init, reg));
//	layers2[1] = LayerPtr<fp,RANK>(new PReLUActivationLayer<fp,RANK>(layers2[0]->get_output_dims(), reg));
//	layers2[2] = LayerPtr<fp,RANK>(new BatchNormLayer<fp,RANK>(layers2[1]->get_output_dims(), reg, reg2));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers2)))), false));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(LayerPtr<fp,RANK>(new MaxPoolingLayer<fp>(
//					res_modules[1].first.get_output_dims(), 3, 2, 2, 1))))), false));
//	std::vector<LayerPtr<fp,RANK>> layers3(3);
//	layers3[0] = LayerPtr<fp,RANK>(new ConvLayer<fp>(res_modules[2].first.get_output_dims(), 8, init, reg));
//	layers3[1] = LayerPtr<fp,RANK>(new LeakyReLUActivationLayer<fp,RANK>(layers3[0]->get_output_dims()));
//	layers3[2] = LayerPtr<fp,RANK>(new BatchNormLayer<fp,RANK>(layers3[1]->get_output_dims(), reg, reg2));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers3)))), false));
//	std::vector<LayerPtr<fp,RANK>> layers4(5);
//	layers4[0] = LayerPtr<fp,RANK>(new MeanPoolingLayer<fp>(res_modules[3].first.get_output_dims()));
//	layers4[1] = LayerPtr<fp,RANK>(new LeakyReLUActivationLayer<fp,RANK>(layers4[0]->get_output_dims()));
//	layers4[2] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(layers4[1]->get_output_dims(), 50, init, reg));
//	layers4[3] = LayerPtr<fp,RANK>(new LeakyReLUActivationLayer<fp,RANK>(layers4[2]->get_output_dims()));
//	layers4[4] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(layers4[3]->get_output_dims(), 1, init, reg));
//	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers4)))), false));
//	ResidualNeuralNetwork<fp,RANK> nn(res_modules);
//	nn.init();
//	LossSharedPtr<fp,RANK,false> loss(new QuadraticLoss<fp,RANK,false>());
//	NadamOptimizer<fp,RANK,false> opt(loss, 20);
//	return opt.verify_gradients(nn, test_prov, 1e-5, 1e-4, 1e-1);
//}
//
//static int test_dense() {
//	std::cout << "DENSE" << std::endl;
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 1> test_obs_ptr(new Tensor<fp,RANK + 1>(5u, 8u, 8u, 2u));
//	TensorPtr<fp,RANK + 1> test_obj_ptr(new Tensor<fp,RANK + 1>(5u, 64u, 8u, 2u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new HeWeightInitialization<fp>());
//	ParamRegSharedPtr<fp> reg(new L2ParameterRegularization<fp>());
//	std::vector<CompositeNeuralNetwork<fp,RANK,false>> modules;
//	modules.push_back(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(test_prov.get_obs_dims(), 2, init, reg))))));
//	modules.push_back(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(modules[0].get_input_dims().add_along_rank(modules[0].get_output_dims(), 0), 2, init, reg))))));
//	modules.push_back(CompositeNeuralNetwork<fp,RANK,false>(NeuralNetPtr<fp,RANK,false>(new FeedforwardNeuralNetwork<fp,RANK>(
//			LayerPtr<fp,RANK>(new ConvLayer<fp>(modules[1].get_input_dims().add_along_rank(modules[1].get_output_dims(), 0), 2, init, reg))))));
//	DenseNeuralNetwork<fp,RANK,LOWEST_RANK> dnn(std::move(modules));
//	dnn.init();
//	LossSharedPtr<fp,RANK,false> loss(new QuadraticLoss<fp,RANK,false>());
//	NadamOptimizer<fp,RANK,false> opt(loss, 20);
//	return opt.verify_gradients(dnn, test_prov);
//}
//
//static int test_seqnn() {
//	std::cout << "SEQUENTIAL" << std::endl;
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 2> test_obs_ptr(new Tensor<fp,RANK + 2>(5u, 10u, 8u, 8u, 3u));
//	TensorPtr<fp,RANK + 2> test_obj_ptr(new Tensor<fp,RANK + 2>(5u, 10u, 4u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new GlorotWeightInitialization<fp>());
//	ParamRegSharedPtr<fp> reg(new L2ParameterRegularization<fp>());
//	std::vector<LayerPtr<fp,RANK>> layers(5);
//	layers[0] = LayerPtr<fp,RANK>(new ConvLayer<fp>(test_prov.get_obs_dims(), 2, init, reg));
//	layers[1] = LayerPtr<fp,RANK>(new TanhActivationLayer<fp,RANK>(layers[0]->get_output_dims()));
//	layers[2] = LayerPtr<fp,RANK>(new ConvLayer<fp>(layers[1]->get_output_dims(), 5, init, reg));
//	layers[3] = LayerPtr<fp,RANK>(new TanhActivationLayer<fp,RANK>(layers[2]->get_output_dims()));
//	layers[4] = LayerPtr<fp,RANK>(new FCLayer<fp,RANK>(layers[3]->get_output_dims(), 4, init, reg));
//	SequentialNeuralNetwork<fp,RANK> seqnn(NeuralNetPtr<fp,RANK,false>(
//			new FeedforwardNeuralNetwork<fp,RANK>(std::move(layers))));
//	seqnn.init();
//	LossSharedPtr<fp,RANK,true> loss(new QuadraticLoss<fp,RANK,true>());
//	NadamOptimizer<fp,RANK,true> opt(loss, 20);
//	return opt.verify_gradients(seqnn, test_prov);
//}
//
//static int test_rnn() {
//	std::cout << "RECURRENT" << std::endl;
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 2> test_obs_ptr(new Tensor<fp,RANK + 2>(5u, 5u, 16u, 16u, 3u));
//	TensorPtr<fp,RANK + 2> test_obj_ptr(new Tensor<fp,RANK + 2>(5u, 3u, 2u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new OrthogonalWeightInitialization<fp>());
//	ParamRegSharedPtr<fp> reg(new L2ParameterRegularization<fp>());
//	KernelPtr<fp,RANK> input_kernel(new ConvLayer<fp>(test_prov.get_obs_dims(), 5, init, reg));
//	KernelPtr<fp,RANK> state_kernel(new ConvLayer<fp>(input_kernel->get_output_dims(), 5, init, reg));
//	KernelPtr<fp,RANK> output_kernel(new FCLayer<fp,RANK>(input_kernel->get_output_dims(), 2, init, reg));
//	ActivationPtr<fp,RANK> state_act(new SigmoidActivationLayer<fp,RANK>(input_kernel->get_output_dims()));
//	ActivationPtr<fp,RANK> output_act(new IdentityActivationLayer<fp,RANK>(output_kernel->get_output_dims()));
//	RecurrentNeuralNetwork<fp,RANK,true> rnn(std::move(input_kernel), std::move(state_kernel), std::move(output_kernel), std::move(state_act),
//			std::move(output_act), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); });
//	rnn.init();
//	LossSharedPtr<fp,RANK,true> loss(new QuadraticLoss<fp,RANK,true>());
//	NadamOptimizer<fp,RANK,true> opt(loss, 20);
//	return opt.verify_gradients(rnn, test_prov);
//}
//
//static int test_lstm() {
//	std::cout << "LSTM" << std::endl;
//	const std::size_t RANK = 1;
//	TensorPtr<fp,RANK + 2> test_obs_ptr(new Tensor<fp,RANK + 2>(10u, 5u, 32u));
//	TensorPtr<fp,RANK + 2> test_obj_ptr(new Tensor<fp,RANK + 2>(10u, 3u, 5u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new OrthogonalWeightInitialization<fp>());
//	ParamRegSharedPtr<fp> reg(new L2ParameterRegularization<fp>());
//	const Dimensions<std::size_t,RANK>& input_dims = test_prov.get_obs_dims();
//	const Dimensions<std::size_t,RANK>& output_dims = test_prov.get_obj_dims();
//	KernelPtr<fp,RANK> forget_input_kernel(new FCLayer<fp,RANK>(input_dims, 5, init, reg));
//	KernelPtr<fp,RANK> forget_output_kernel(new FCLayer<fp,RANK>(output_dims, 5, init, reg));
//	KernelPtr<fp,RANK> write_input_kernel(new FCLayer<fp,RANK>(input_dims, 5, init, reg));
//	KernelPtr<fp,RANK> write_output_kernel(new FCLayer<fp,RANK>(output_dims, 5, init, reg));
//	KernelPtr<fp,RANK> candidate_input_kernel(new FCLayer<fp,RANK>(input_dims, 5, init, reg));
//	KernelPtr<fp,RANK> candidate_output_kernel(new FCLayer<fp,RANK>(output_dims, 5, init, reg));
//	KernelPtr<fp,RANK> read_input_kernel(new FCLayer<fp,RANK>(input_dims, 5, init, reg));
//	KernelPtr<fp,RANK> read_output_kernel(new FCLayer<fp,RANK>(output_dims, 5, init, reg));
//	ActivationPtr<fp,RANK> forget_act(new SigmoidActivationLayer<fp,RANK>(output_dims));
//	ActivationPtr<fp,RANK> write_act(new SigmoidActivationLayer<fp,RANK>(output_dims));
//	ActivationPtr<fp,RANK> candidate_act(new TanhActivationLayer<fp,RANK>(output_dims));
//	ActivationPtr<fp,RANK> state_act(new TanhActivationLayer<fp,RANK>(output_dims));
//	ActivationPtr<fp,RANK> read_act(new SigmoidActivationLayer<fp,RANK>(output_dims));
//	LSTMNeuralNetwork<fp,RANK,true> lstm(std::move(forget_input_kernel), std::move(forget_output_kernel), std::move(write_input_kernel),
//			std::move(write_output_kernel), std::move(candidate_input_kernel), std::move(candidate_output_kernel), std::move(read_input_kernel),
//			std::move(read_output_kernel), std::move(forget_act), std::move(write_act), std::move(candidate_act), std::move(state_act),
//			std::move(read_act), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); });
//	lstm.init();
//	LossSharedPtr<fp,RANK,true> loss(new QuadraticLoss<fp,RANK,true>());
//	NadamOptimizer<fp,RANK,true> opt(loss, 20);
//	return opt.verify_gradients(lstm, test_prov);
//}
//
//static int test_bdrnn() {
//	std::cout << "BIDIRECTIONAL RECURRENT" << std::endl;
//	const std::size_t RANK = 3;
//	TensorPtr<fp,RANK + 2> test_obs_ptr(new Tensor<fp,RANK + 2>(5u, 7u, 8u, 8u, 3u));
//	TensorPtr<fp,RANK + 2> test_obj_ptr(new Tensor<fp,RANK + 2>(5u, 3u, 4u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	MemoryDataProvider<fp,RANK,true> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	WeightInitSharedPtr<fp> init(new OrthogonalWeightInitialization<fp>());
//	ParamRegSharedPtr<fp> reg(new L2ParameterRegularization<fp>());
//	KernelPtr<fp,RANK> input_kernel(new ConvLayer<fp>(test_prov.get_obs_dims(), 5, init, reg));
//	KernelPtr<fp,RANK> state_kernel(new ConvLayer<fp>(input_kernel->get_output_dims(), 5, init, reg));
//	KernelPtr<fp,RANK> output_kernel(new FCLayer<fp,RANK>(input_kernel->get_output_dims(), 2, init, reg));
//	ActivationPtr<fp,RANK> state_act(new SigmoidActivationLayer<fp,RANK>(input_kernel->get_output_dims()));
//	ActivationPtr<fp,RANK> output_act(new IdentityActivationLayer<fp,RANK>(output_kernel->get_output_dims()));
//	BidirectionalNeuralNetwork<fp,RANK,CONCAT_LO_RANK> bdrnn(UnidirNeuralNetPtr<fp,RANK>(new RecurrentNeuralNetwork<fp,RANK>(
//			std::move(input_kernel), std::move(state_kernel), std::move(output_kernel), std::move(state_act), std::move(output_act),
//			[](int input_seq_length) { return std::make_pair(3, 2); })));
//	bdrnn.init();
//	LossSharedPtr<fp,RANK,true> loss(new QuadraticLoss<fp,RANK,true>());
//	NadamOptimizer<fp,RANK,true> opt(loss, 20);
//	return opt.verify_gradients(bdrnn, test_prov);
//}
//
//static void std_dataset_test() {
//	std::string mnist_folder = "C:\\Users\\Viktor\\Downloads\\mnist\\";
//	MNISTDataProvider<float> file_train_prov(mnist_folder + "train-images.idx3-ubyte", mnist_folder + "train-labels.idx1-ubyte");
//	MNISTDataProvider<float> file_test_prov(mnist_folder + "t10k-images.idx3-ubyte", mnist_folder + "t10k-labels.idx1-ubyte");
////	std::string cifar_folder = "C:\\Users\\A6714\\Downloads\\cifar-10-batches-bin\\";
////	CIFARDataProvider<float> file_train_prov({ cifar_folder + "data_batch_1.bin", cifar_folder + "data_batch_2.bin", cifar_folder + "data_batch_3.bin",
////			cifar_folder + "data_batch_4.bin", cifar_folder + "data_batch_5.bin", cifar_folder + "data_batch_6.bin", });
////	CIFARDataProvider<float> file_test_prov(cifar_folder + "test_batch.bin");
//	WeightInitSharedPtr<float> conv_init(new HeWeightInitialization<float>(1e-1));
//	WeightInitSharedPtr<float> dense_init(new GlorotWeightInitialization<float>(1e-1));
//	std::vector<LayerPtr<float,3>> layers(12);
//	layers[0] = LayerPtr<float,3>(new ConvLayer<float>(file_train_prov.get_obs_dims(), 8, conv_init));
//	layers[1] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[0]->get_output_dims()));
//	layers[2] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[1]->get_output_dims()));
//	layers[3] = LayerPtr<float,3>(new ConvLayer<float>(layers[2]->get_output_dims(), 8, conv_init));
//	layers[4] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[3]->get_output_dims()));
//	layers[5] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[4]->get_output_dims()));
//	layers[6] = LayerPtr<float,3>(new DropoutLayer<float,3>(layers[5]->get_output_dims(), .25));
//	layers[7] = LayerPtr<float,3>(new FCLayer<float,3>(layers[6]->get_output_dims(), 50, dense_init));
//	layers[8] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[7]->get_output_dims()));
//	layers[9] = LayerPtr<float,3>(new DropoutLayer<float,3>(layers[8]->get_output_dims(), .5));
//	layers[10] = LayerPtr<float,3>(new FCLayer<float,3>(layers[9]->get_output_dims(), 10, dense_init));
//	layers[11] = LayerPtr<float,3>(new SoftmaxActivationLayer<float,3>(layers[10]->get_output_dims()));
//	FeedforwardNeuralNetwork<float,3> nn(std::move(layers));
//	nn.init();
//	LossSharedPtr<float,3,false> loss(new CrossEntropyLoss<float,3,false>());
//	AdadeltaOptimizer<float,3,false> opt(loss, 200);
//	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//	opt.optimize(nn, file_train_prov, file_test_prov, 10);
//	std::cout << "Training Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
//			begin).count() << std::endl;
//	begin = std::chrono::steady_clock::now();
//	DataPair<float,3,false> data = file_test_prov.get_data(10000);
//	Tensor<float,4> prediction = nn.infer(std::move(data.first));
//	std::cout << "Inference Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
//			begin).count() << std::endl;
//	unsigned wrong = 0;
//	unsigned correct = 0;
//	for (std::size_t i = 0; i < prediction.dimension(0); ++i) {
//		float max = internal::Utils<float>::MIN;
//		std::size_t max_ind = 0;
//		for (std::size_t j = 0; j < prediction.dimension(1); ++j) {
//			float val = prediction(i,j,0u,0u);
//			if (val > max) {
//				max = val;
//				max_ind = j;
//			}
//		}
//		if (data.second(i,max_ind,0u,0u) == 1)
//			correct++;
//		else
//			wrong++;
//	}
//	std::cout << "Correct: " << correct << std::endl;
//	std::cout << "Wrong: " << wrong << std::endl;
//}
//
//static void readme_test() {
//	TensorPtr<double,4> training_obs_ptr(new Tensor<double,4>(80u, 32u, 32u, 3u));
//	TensorPtr<double,4> training_obj_ptr(new Tensor<double,4>(80u, 1u, 1u, 1u));
//	training_obs_ptr->setRandom();
//	training_obj_ptr->setRandom();
//	PCAPreprocessor<double,3> preproc;
//	preproc.fit(*training_obs_ptr);
//	preproc.transform(*training_obs_ptr);
//	MemoryDataProvider<double,3,false> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
//	TensorPtr<double,4> test_obs_ptr(new Tensor<double,4>(20u, 32u, 32u, 3u));
//	TensorPtr<double,4> test_obj_ptr(new Tensor<double,4>(20u, 1u, 1u, 1u));
//	test_obs_ptr->setRandom();
//	test_obj_ptr->setRandom();
//	preproc.transform(*test_obs_ptr);
//	MemoryDataProvider<double,3,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	auto init = std::make_shared<HeWeightInitialization<double>>();
//	auto reg = std::make_shared<L2ParameterRegularization<double>>();
//	std::vector<LayerPtr<double,3>> layers(9);
//	layers[0] = LayerPtr<double,3>(new ConvLayer<double>(training_prov.get_obs_dims(), 10, init, reg, 5, 2));
//	layers[1] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[0]->get_output_dims()));
//	layers[2] = LayerPtr<double,3>(new MaxPoolingLayer<double>(layers[1]->get_output_dims()));
//	layers[3] = LayerPtr<double,3>(new ConvLayer<double>(layers[2]->get_output_dims(), 20, init, reg));
//	layers[4] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[3]->get_output_dims()));
//	layers[5] = LayerPtr<double,3>(new MaxPoolingLayer<double>(layers[4]->get_output_dims()));
//	layers[6] = LayerPtr<double,3>(new FCLayer<double,3>(layers[5]->get_output_dims(), 500, init, reg));
//	layers[7] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[6]->get_output_dims()));
//	layers[8] = LayerPtr<double,3>(new FCLayer<double,3>(layers[7]->get_output_dims(), 1, init, reg));
//	FeedforwardNeuralNetwork<double,3> nn(std::move(layers));
//	nn.init();
//	auto loss = std::make_shared<QuadraticLoss<double,3,false>>();
//	NadamOptimizer<double,3,false> opt(loss, 20);
//	opt.optimize(nn, training_prov, test_prov, 500);
//	Tensor<double,4> input(5u, 32u, 32u, 3u);
//	input.setRandom();
//	preproc.transform(input);
//	Tensor<double,4> prediction = nn.infer(input);
//}

int main() {
	std::string mnist_folder = "C:\\Users\\A6714\\Downloads\\mnist\\";
	MNISTDataProvider<float> file_train_prov(mnist_folder + "train-images.idx3-ubyte", mnist_folder + "train-labels.idx1-ubyte");
	MNISTDataProvider<float> file_test_prov(mnist_folder + "t10k-images.idx3-ubyte", mnist_folder + "t10k-labels.idx1-ubyte");
	DataPair<float,3,false> train_data = file_train_prov.get_data(60000);
	TensorPtr<float,4> train_obs(new Tensor<float,4>(std::move(train_data.first)));
	TensorPtr<float,4> train_obj(new Tensor<float,4>(std::move(train_data.second)));
	MemoryDataProvider<float,3,false> train_prov(std::move(train_obs), std::move(train_obj));
	DataPair<float,3,false> test_data = file_test_prov.get_data(1);
	TensorPtr<float,4> test_obs(new Tensor<float,4>(std::move(test_data.first)));
	TensorPtr<float,4> test_obj(new Tensor<float,4>(std::move(test_data.second)));
	MemoryDataProvider<float,3,false> test_prov(std::move(test_obs), std::move(test_obj));
	auto init = std::make_shared<GlorotWeightInitialization<float>>();
	std::vector<LayerPtr<float,3>> layers(9);
	layers[0] = LayerPtr<float,3>(new ConvLayer<float>(train_prov.get_obs_dims(), 8, init, ConvLayer<float>::NO_PARAM_REG, 5, 5, 2, 2));
	layers[1] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<float,3>(new ConvLayer<float>(layers[2]->get_output_dims(), 8, init, ConvLayer<float>::NO_PARAM_REG, 5, 5, 2, 2));
	layers[4] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<float,3>(new FCLayer<float,3>(layers[5]->get_output_dims(), 128, init));
	layers[7] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[6]->get_output_dims()));
	layers[8] = LayerPtr<float,3>(new FCLayer<float,3>(layers[7]->get_output_dims(), 10, init));
//	layers[0] = LayerPtr<float,3>(new FCLayer<float,3>(train_prov.get_obs_dims(), 500, init));
//	layers[1] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[0]->get_output_dims()));
//	layers[2] = LayerPtr<float,3>(new FCLayer<float,3>(layers[1]->get_output_dims(), 250, init));
//	layers[3] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(layers[2]->get_output_dims()));
//	layers[4] = LayerPtr<float,3>(new FCLayer<float,3>(layers[3]->get_output_dims(), 10, init));
	FeedforwardNeuralNetwork<float,3> nn(std::move(layers));
	nn.init();
	auto loss = std::make_shared<SoftmaxCrossEntropyLoss<float,3,false>>();
	VanillaSGDOptimizer<float,3,false> opt(loss, 100);
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	opt.optimize(nn, train_prov, test_prov, 1);
	std::cout << "Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - begin).count() << std::endl;
//	std_dataset_test();
//	readme_test();
//	int success = test_parallel() & test_residual() & test_dense() & test_seqnn() & test_rnn() & test_lstm() & test_bdrnn();
//	assert(success);
	return 0;
}
