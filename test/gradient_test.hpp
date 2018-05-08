/*
 * gradient_test.hpp
 *
 *  Created on: 19.04.2018
 *      Author: Viktor Csomor
 */

#ifndef GRADIENT_TEST_HPP_
#define GRADIENT_TEST_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Cattle.hpp"
#include "test_utils.hpp"


namespace cattle {
namespace test {

/**
 * Determines the verbosity of the gradient tests.
 */
extern bool verbose;

/**
 * @param name The name of the gradient test.
 * @param prov The data provider to use for gradient checking.
 * @param net The neural network whose differentiation is to be checked.
 * @param opt The optimizer to use for the gradient check.
 * @param step_size The step size for numerical differentiation.
 * @param abs_epsilon The maximum acceptable absolute difference between the analytic and
 * numerical gradients.
 * @param rel_epsilon The maximum acceptable relative difference between the analytic and
 * numerical gradients.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
inline void grad_test(std::string name, DataProvider<Scalar,Rank,Sequential>& prov,
		NeuralNetwork<Scalar,Rank,Sequential>& net, Optimizer<Scalar,Rank,Sequential>& opt,
		Scalar step_size = ScalarTraits<Scalar>::step_size, Scalar abs_epsilon = ScalarTraits<Scalar>::abs_epsilon,
		Scalar rel_epsilon = ScalarTraits<Scalar>::rel_epsilon) {
	print_test_header<Scalar,Rank,Sequential>("gradient check", name);
	EXPECT_TRUE(opt.verify_gradients(net, prov, verbose, step_size, abs_epsilon, rel_epsilon));
}

/**
 * @param name The name of the gradient test.
 * @param net The neural network to perform the gradient check on.
 * @param samples The number of samples to use.
 * @param step_size The step size for numerical differentiation.
 * @param abs_epsilon The maximum acceptable absolute difference between the analytic and
 * numerical gradients.
 * @param rel_epsilon The maximum acceptable relative difference between the analytic and
 * numerical gradients.
 */
template<typename Scalar, std::size_t Rank>
inline void nonseq_network_grad_test(std::string name, NeuralNetPtr<Scalar,Rank,false> net,
		std::size_t samples = 5, Scalar step_size = ScalarTraits<Scalar>::step_size,
		Scalar abs_epsilon = ScalarTraits<Scalar>::abs_epsilon,
		Scalar rel_epsilon = ScalarTraits<Scalar>::rel_epsilon) {
	std::array<std::size_t,Rank + 1> input_dims = net->get_input_dims().template promote<>();
	std::array<std::size_t,Rank + 1> output_dims = net->get_output_dims().template promote<>();
	input_dims[0] = samples;
	output_dims[0] = samples;
	TensorPtr<Scalar,Rank + 1> obs = random_tensor<Scalar,Rank + 1>(input_dims);
	TensorPtr<Scalar,Rank + 1> obj = random_tensor<Scalar,Rank + 1>(output_dims);
	MemoryDataProvider<Scalar,Rank,false> prov(std::move(obs), std::move(obj));
	net->init();
	auto loss = LossSharedPtr<Scalar,Rank,false>(new QuadraticLoss<Scalar,Rank,false>());
	VanillaSGDOptimizer<Scalar,Rank,false> opt(loss, samples);
	grad_test<Scalar,Rank,false>(name, prov, *net, opt, step_size, abs_epsilon, rel_epsilon);
}

/**
 * @param name The name of the gradient test.
 * @param net The neural network to perform the gradient check on.
 * @param input_seq_length The length of the input sequence.
 * @param output_seq_length The length of the output sequence.
 * @param samples The number of samples to use.
 * @param step_size The step size for numerical differentiation.
 * @param abs_epsilon The maximum acceptable absolute difference between the analytic and
 * numerical gradients.
 * @param rel_epsilon The maximum acceptable relative difference between the analytic and
 * numerical gradients.
 */
template<typename Scalar, std::size_t Rank>
inline void seq_network_grad_test(std::string name, NeuralNetPtr<Scalar,Rank,true> net,
		std::size_t input_seq_length = 5, std::size_t output_seq_length = 5, std::size_t samples = 5,
		Scalar step_size = ScalarTraits<Scalar>::step_size, Scalar abs_epsilon = ScalarTraits<Scalar>::abs_epsilon,
		Scalar rel_epsilon = ScalarTraits<Scalar>::rel_epsilon) {
	std::array<std::size_t,Rank + 2> input_dims = net->get_input_dims().template promote<2>();
	std::array<std::size_t,Rank + 2> output_dims = net->get_output_dims().template promote<2>();
	input_dims[0] = samples;
	input_dims[1] = input_seq_length;
	output_dims[0] = samples;
	output_dims[1] = output_seq_length;
	TensorPtr<Scalar,Rank + 2> obs = random_tensor<Scalar,Rank + 2>(input_dims);
	TensorPtr<Scalar,Rank + 2> obj = random_tensor<Scalar,Rank + 2>(output_dims);
	MemoryDataProvider<Scalar,Rank,true> prov(std::move(obs), std::move(obj));
	net->init();
	auto loss = LossSharedPtr<Scalar,Rank,true>(new QuadraticLoss<Scalar,Rank,true>());
	VanillaSGDOptimizer<Scalar,Rank,true> opt(loss, samples);
	grad_test<Scalar,Rank,true>(name, prov, *net, opt, step_size, abs_epsilon, rel_epsilon);
}

/**
 * @param name The name of the gradient test.
 * @param layer1 The first instance of the layer class to verify.
 * @param layer2 The second instance.
 * @param samples The number of samples to use.
 * @param step_size The step size for numerical differentiation.
 * @param abs_epsilon The maximum acceptable absolute difference between the analytic and
 * numerical gradients.
 * @param rel_epsilon The maximum acceptable relative difference between the analytic and
 * numerical gradients.
 */
template<typename Scalar, std::size_t Rank>
inline void layer_grad_test(std::string name, LayerPtr<Scalar,Rank> layer1, LayerPtr<Scalar,Rank> layer2,
		std::size_t samples = 5, Scalar step_size = ScalarTraits<Scalar>::step_size,
		Scalar abs_epsilon = ScalarTraits<Scalar>::abs_epsilon,
		Scalar rel_epsilon = ScalarTraits<Scalar>::rel_epsilon) {
	assert(layer1->get_output_dims() == layer2->get_input_dims());
	std::vector<LayerPtr<Scalar,Rank>> layers(2);
	layers[0] = std::move(layer1);
	layers[1] = std::move(layer2);
	NeuralNetPtr<Scalar,Rank,false> nn(new FeedforwardNeuralNetwork<Scalar,Rank>(std::move(layers)));
	nonseq_network_grad_test<Scalar,Rank>(name, std::move(nn), samples, step_size, abs_epsilon, rel_epsilon);
}

/************************
 * LAYER GRADIENT TESTS *
 ************************/

/**
 * Performs gradient checks on fully-connected layers.
 */
template<typename Scalar>
inline void dense_layer_grad_test() {
	auto init1 = WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>());
	auto init2 = WeightInitSharedPtr<Scalar>(new LeCunWeightInitialization<Scalar>());
	auto reg1 = ParamRegSharedPtr<Scalar>(new L1ParameterRegularization<Scalar>());
	auto reg2 = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	auto reg3 = ParamRegSharedPtr<Scalar>(new ElasticNetParameterRegularization<Scalar>());
	LayerPtr<Scalar,1> layer1_1(new DenseLayer<Scalar,1>({ 32u }, 16, init1));
	LayerPtr<Scalar,1> layer1_2(new DenseLayer<Scalar,1>(layer1_1->get_output_dims(), 1, init1, reg3));
	layer_grad_test<Scalar,1>("fc layer", std::move(layer1_1), std::move(layer1_2), 5, ScalarTraits<Scalar>::step_size,
			ScalarTraits<float>::abs_epsilon, ScalarTraits<float>::rel_epsilon);
	LayerPtr<Scalar,2> layer2_1(new DenseLayer<Scalar,2>({ 6u, 6u }, 16, init2));
	LayerPtr<Scalar,2> layer2_2(new DenseLayer<Scalar,2>(layer2_1->get_output_dims(), 2, init2, reg1));
	layer_grad_test<Scalar,2>("fc layer", std::move(layer2_1), std::move(layer2_2), 5, ScalarTraits<Scalar>::step_size,
			ScalarTraits<float>::abs_epsilon, ScalarTraits<float>::rel_epsilon);
	LayerPtr<Scalar,3> layer3_1(new DenseLayer<Scalar,3>({ 4u, 4u, 2u }, 16, init1, reg2));
	LayerPtr<Scalar,3> layer3_2(new DenseLayer<Scalar,3>(layer3_1->get_output_dims(), 4, init1, reg2));
	layer_grad_test<Scalar,3>("fc layer", std::move(layer3_1), std::move(layer3_2));
}

TEST(GradientTest, DenseLayer) {
	dense_layer_grad_test<float>();
	dense_layer_grad_test<double>();
}

/**
 * Performs gradient checks on convolutional layers.
 */
template<typename Scalar>
inline void conv_layer_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	LayerPtr<Scalar,3> layer1(new ConvolutionLayer<Scalar>({ 8u, 8u, 2u }, 5, init,
			ConvolutionLayer<Scalar>::NO_PARAM_REG, 3, 2, 1, 2, 1, 2, 1, 0));
	LayerPtr<Scalar,3> layer2(new ConvolutionLayer<Scalar>(layer1->get_output_dims(), 1, init, reg, 1, 1));
	layer_grad_test<Scalar,3>("convolution layer ", std::move(layer1), std::move(layer2));
}

TEST(GradientTest, ConvolutionLayer) {
	conv_layer_grad_test<float>();
	conv_layer_grad_test<double>();
}

/**
 * Performs gradient checks on transposed convolutional layers.
 */
template<typename Scalar>
inline void deconv_layer_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	LayerPtr<Scalar,3> layer1(new DeconvolutionLayer<Scalar>({ 2u, 3u, 2u }, 5, init,
			DeconvolutionLayer<Scalar>::NO_PARAM_REG, 5, 3, 1, 0, 1, 2, 0, 1));
	LayerPtr<Scalar,3> layer2(new DeconvolutionLayer<Scalar>(layer1->get_output_dims(), 1, init, reg, 1, 1));
	layer_grad_test<Scalar,3>("deconvolution layer ", std::move(layer1), std::move(layer2));
}

TEST(GradientTest, DeconvolutionLayer) {
	deconv_layer_grad_test<float>();
	deconv_layer_grad_test<double>();
}

/**
 * @param dims The dimensions of the layers' input tensors.
 */
template<typename Scalar, std::size_t Rank>
inline void activation_layer_grad_test(const typename std::enable_if<Rank != 3,
		Dimensions<std::size_t,Rank>>::type& dims) {
	auto init = WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>());
	LayerPtr<Scalar,Rank> layer1_1(new DenseLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_1 = layer1_1->get_output_dims();
	layer_grad_test<Scalar,Rank>("identity activation layer", std::move(layer1_1),
			LayerPtr<Scalar,Rank>(new IdentityActivationLayer<Scalar,Rank>(dims_1)));
	LayerPtr<Scalar,Rank> layer1_2(new DenseLayer<Scalar,Rank>(dims, 8, init));
	Dimensions<std::size_t,Rank> dims_2 = layer1_2->get_output_dims();
	layer_grad_test<Scalar,Rank>("scaling activation layer", std::move(layer1_2),
			LayerPtr<Scalar,Rank>(new ScaledActivationLayer<Scalar,Rank>(dims_2, 1.5)));
	LayerPtr<Scalar,Rank> layer1_3(new DenseLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_3 = layer1_3->get_output_dims();
	layer_grad_test<Scalar,Rank>("binary step activation layer", std::move(layer1_3),
			LayerPtr<Scalar,Rank>(new BinaryStepActivationLayer<Scalar,Rank>(dims_3)));
	LayerPtr<Scalar,Rank> layer1_4(new DenseLayer<Scalar,Rank>(dims, 16, init));
	Dimensions<std::size_t,Rank> dims_4 = layer1_4->get_output_dims();
	layer_grad_test<Scalar,Rank>("sigmoid activation layer rank", std::move(layer1_4),
			LayerPtr<Scalar,Rank>(new SigmoidActivationLayer<Scalar,Rank>(dims_4)));
	LayerPtr<Scalar,Rank> layer1_5(new DenseLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_5 = layer1_5->get_output_dims();
	layer_grad_test<Scalar,Rank>("tanh activation layer rank", std::move(layer1_5),
			LayerPtr<Scalar,Rank>(new TanhActivationLayer<Scalar,Rank>(dims_5)));
	LayerPtr<Scalar,Rank> layer1_6(new DenseLayer<Scalar,Rank>(dims, 9, init));
	Dimensions<std::size_t,Rank> dims_6 = layer1_6->get_output_dims();
	layer_grad_test<Scalar,Rank>("softplus activation layer rank", std::move(layer1_6),
			LayerPtr<Scalar,Rank>(new SoftplusActivationLayer<Scalar,Rank>(dims_6)));
	LayerPtr<Scalar,Rank> layer1_7(new DenseLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_7 = layer1_7->get_output_dims();
	layer_grad_test<Scalar,Rank>("softmax activation layer rank", std::move(layer1_7),
			LayerPtr<Scalar,Rank>(new SoftmaxActivationLayer<Scalar,Rank>(dims_7)));
	LayerPtr<Scalar,Rank> layer1_8(new DenseLayer<Scalar,Rank>(dims, 10, init));
	Dimensions<std::size_t,Rank> dims_8 = layer1_8->get_output_dims();
	layer_grad_test<Scalar,Rank>("relu activation layer rank", std::move(layer1_8),
			LayerPtr<Scalar,Rank>(new ReLUActivationLayer<Scalar,Rank>(dims_8)));
	LayerPtr<Scalar,Rank> layer1_9(new DenseLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_9 = layer1_9->get_output_dims();
	layer_grad_test<Scalar,Rank>("leaky relu activation layer rank", std::move(layer1_9),
			LayerPtr<Scalar,Rank>(new LeakyReLUActivationLayer<Scalar,Rank>(dims_9, 2e-1)));
	LayerPtr<Scalar,Rank> layer1_10(new DenseLayer<Scalar,Rank>(dims, 11, init));
	Dimensions<std::size_t,Rank> dims_10 = layer1_10->get_output_dims();
	layer_grad_test<Scalar,Rank>("elu activation layer rank", std::move(layer1_10),
			LayerPtr<Scalar,Rank>(new ELUActivationLayer<Scalar,Rank>(dims_10, 2e-1)));
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	layer_grad_test<Scalar,Rank>("prelu activation layer rank",
			LayerPtr<Scalar,Rank>(new PReLUActivationLayer<Scalar,Rank>(dims, reg, 2e-1)),
			LayerPtr<Scalar,Rank>(new PReLUActivationLayer<Scalar,Rank>(dims)));
}

/**
 *@param dims The dimensions of the layers' input tensors.
 */
template<typename Scalar, std::size_t Rank>
inline void activation_layer_grad_test(const typename std::enable_if<Rank == 3,
		Dimensions<std::size_t,Rank>>::type& dims) {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	LayerPtr<Scalar,3> layer1_1(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_1 = layer1_1->get_output_dims();
	layer_grad_test<Scalar,3>("identity activation layer", std::move(layer1_1),
			LayerPtr<Scalar,3>(new IdentityActivationLayer<Scalar,3>(dims_1)));
	LayerPtr<Scalar,3> layer1_2(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_2 = layer1_2->get_output_dims();
	layer_grad_test<Scalar,3>("scaling activation layer", std::move(layer1_2),
			LayerPtr<Scalar,3>(new ScaledActivationLayer<Scalar,3>(dims_2, 1.5)));
	LayerPtr<Scalar,3> layer1_3(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_3 = layer1_3->get_output_dims();
	layer_grad_test<Scalar,3>("binary step activation layer", std::move(layer1_3),
			LayerPtr<Scalar,3>(new BinaryStepActivationLayer<Scalar,3>(dims_3)));
	LayerPtr<Scalar,3> layer1_4(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_4 = layer1_4->get_output_dims();
	layer_grad_test<Scalar,3>("sigmoid activation layer", std::move(layer1_4),
			LayerPtr<Scalar,3>(new SigmoidActivationLayer<Scalar,3>(dims_4)));
	LayerPtr<Scalar,3> layer1_5(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_5 = layer1_5->get_output_dims();
	layer_grad_test<Scalar,3>("tanh activation layer", std::move(layer1_5),
			LayerPtr<Scalar,3>(new TanhActivationLayer<Scalar,3>(dims_5)));
	LayerPtr<Scalar,3> layer1_6(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_6 = layer1_6->get_output_dims();
	layer_grad_test<Scalar,3>("softplus activation layer", std::move(layer1_6),
			LayerPtr<Scalar,3>(new SoftplusActivationLayer<Scalar,3>(dims_6)));
	LayerPtr<Scalar,3> layer1_7(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_7 = layer1_7->get_output_dims();
	layer_grad_test<Scalar,3>("softmax activation layer", std::move(layer1_7),
			LayerPtr<Scalar,3>(new SoftmaxActivationLayer<Scalar,3>(dims_7)));
	LayerPtr<Scalar,3> layer1_8(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_8 = layer1_8->get_output_dims();
	layer_grad_test<Scalar,3>("relu activation layer", std::move(layer1_8),
			LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(dims_8)));
	LayerPtr<Scalar,3> layer1_9(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_9 = layer1_9->get_output_dims();
	layer_grad_test<Scalar,3>("leaky relu activation layer", std::move(layer1_9),
			LayerPtr<Scalar,3>(new LeakyReLUActivationLayer<Scalar,3>(dims_9, 2e-1)));
	LayerPtr<Scalar,3> layer1_10(new ConvolutionLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_10 = layer1_10->get_output_dims();
	layer_grad_test<Scalar,3>("elu activation layer", std::move(layer1_10),
			LayerPtr<Scalar,3>(new ELUActivationLayer<Scalar,3>(dims_10, 2e-1)));
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	layer_grad_test<Scalar,3>("prelu activation layer",
			LayerPtr<Scalar,3>(new PReLUActivationLayer<Scalar,3>(dims, reg, 2e-1)),
			LayerPtr<Scalar,3>(new PReLUActivationLayer<Scalar,3>(dims)));
}

/**
 * Performs gradient checks on all activation layers.
 */
template<typename Scalar>
inline void activation_layer_grad_test() {
	Dimensions<std::size_t,1>({ 24 });
	activation_layer_grad_test<Scalar,1>({ 24u });
	activation_layer_grad_test<Scalar,2>({ 5u, 5u });
	activation_layer_grad_test<Scalar,3>({ 4u, 3u, 2u });
}

TEST(GradientTest, ActivationLayer) {
	activation_layer_grad_test<float>();
	activation_layer_grad_test<double>();
}

/**
 * Performs gradient checks on all pooling layers.
 */
template<typename Scalar>
inline void pool_layer_grad_test() {
	Dimensions<std::size_t,3> dims({ 16u, 16u, 2u });
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	LayerPtr<Scalar,3> sum_layer1(new ConvolutionLayer<Scalar>(dims, 2, init));
	LayerPtr<Scalar,3> sum_layer2(new SumPoolLayer<Scalar>(sum_layer1->get_output_dims(), 3, 1, 1, 2, 0, 1));
	layer_grad_test<Scalar,3>("sum pool layer", std::move(sum_layer1), std::move(sum_layer2));
	LayerPtr<Scalar,3> mean_layer1(new ConvolutionLayer<Scalar>(dims, 2, init));
	LayerPtr<Scalar,3> mean_layer2(new MeanPoolLayer<Scalar>(mean_layer1->get_output_dims(), 3, 1, 1, 2, 0, 1));
	layer_grad_test<Scalar,3>("mean pool layer", std::move(mean_layer1), std::move(mean_layer2));
	LayerPtr<Scalar,3> max_layer1(new ConvolutionLayer<Scalar>(dims, 2, init));
	LayerPtr<Scalar,3> max_layer2(new MaxPoolLayer<Scalar>(max_layer1->get_output_dims(), 3, 1, 1, 2, 0, 1));
	layer_grad_test<Scalar,3>("max pool layer", std::move(max_layer1), std::move(max_layer2));
}

TEST(GradientTest, PoolLayer) {
	pool_layer_grad_test<float>();
	pool_layer_grad_test<double>();
}

/**
 * Performs gradient checks on broadcast layers.
 */
template<typename Scalar>
inline void broadcast_layer_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	LayerPtr<Scalar,1> layer1_1(new DenseLayer<Scalar,1>({ 8u }, 8, init));
	LayerPtr<Scalar,1> layer2_1(new BroadcastLayer<Scalar,1>(layer1_1->get_output_dims(), { 3u }));
	layer_grad_test<Scalar,1>("broadcast layer", std::move(layer1_1), std::move(layer2_1));
	LayerPtr<Scalar,2> layer1_2(new DenseLayer<Scalar,2>({ 6u, 6u }, 12, init));
	LayerPtr<Scalar,2> layer2_2(new BroadcastLayer<Scalar,2>(layer1_2->get_output_dims(), { 1u, 3u }));
	layer_grad_test<Scalar,2>("broadcast layer", std::move(layer1_2), std::move(layer2_2));
	LayerPtr<Scalar,3> layer1_3(new ConvolutionLayer<Scalar>({ 16u, 16u, 2u }, 2, init));
	LayerPtr<Scalar,3> layer2_3(new BroadcastLayer<Scalar,3>(layer1_3->get_output_dims(), { 2u, 2u, 2u }));
	layer_grad_test<Scalar,3>("broadcast layer", std::move(layer1_3), std::move(layer2_3));
}

TEST(GradientTest, BroadcastLayer) {
	broadcast_layer_grad_test<float>();
	broadcast_layer_grad_test<double>();
}

/**
 * Performs gradient checks on batch normalization layers.
 */
template<typename Scalar>
inline void batch_norm_layer_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	auto reg1 = ParamRegSharedPtr<Scalar>(new L1ParameterRegularization<Scalar>());
	auto reg2 = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	LayerPtr<Scalar,1> layer1_1(new DenseLayer<Scalar,1>({ 32u }, 16, init));
	LayerPtr<Scalar,1> layer1_2(new BatchNormLayer<Scalar,1>(layer1_1->get_output_dims(), reg2, reg1));
	layer_grad_test<Scalar,1>("batch norm layer", std::move(layer1_1), std::move(layer1_2), 5,
			ScalarTraits<Scalar>::step_size, ScalarTraits<float>::abs_epsilon, ScalarTraits<float>::rel_epsilon);
	LayerPtr<Scalar,2> layer2_1(new BatchNormLayer<Scalar,2>({ 6u, 6u }));
	LayerPtr<Scalar,2> layer2_2(new IdentityActivationLayer<Scalar,2>(layer2_1->get_output_dims()));
	layer_grad_test<Scalar,2>("batch norm layer", std::move(layer2_1), std::move(layer2_2));
	LayerPtr<Scalar,3> layer3_1(new ConvolutionLayer<Scalar>({ 4u, 4u, 2u }, 2, init));
	LayerPtr<Scalar,3> layer3_2(new BatchNormLayer<Scalar,3>(layer3_1->get_output_dims(), reg2, reg2));
	layer_grad_test<Scalar,3>("batch norm layer", std::move(layer3_1), std::move(layer3_2));
}

TEST(GradientTest, BatchNormLayer) {
	batch_norm_layer_grad_test<float>();
	batch_norm_layer_grad_test<double>();
}

/**
 * Performs gradient checks on reshape-layers.
 */
template<typename Scalar>
inline void reshape_layer_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	LayerPtr<Scalar,2> layer1_2(new DenseLayer<Scalar,2>({ 6u, 6u }, 12, init));
	LayerPtr<Scalar,2> layer2_2(new ReshapeLayer<Scalar,2>(layer1_2->get_output_dims(), { 4u, 3u }));
	layer_grad_test<Scalar,2>("reshape layer", std::move(layer1_2), std::move(layer2_2));
	LayerPtr<Scalar,3> layer1_3(new ConvolutionLayer<Scalar>({ 16u, 16u, 2u }, 2, init));
	LayerPtr<Scalar,3> layer2_3(new ReshapeLayer<Scalar,3>(layer1_3->get_output_dims(), { 8u, 8u, 8u }));
	layer_grad_test<Scalar,3>("reshape layer", std::move(layer1_3), std::move(layer2_3));
}

TEST(GradientTest, ReshapeLayer) {
	reshape_layer_grad_test<float>();
	reshape_layer_grad_test<double>();
}

/*********************************
 * NEURAL NETWORK GRADIENT TESTS *
 *********************************/

/**
 * Performs gradient checks on parallel neural networks.
 */
template<typename Scalar>
inline void parallel_net_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>());
	// Rank 1 with summation.
	Dimensions<std::size_t,1> dims_1({ 32u });
	std::vector<NeuralNetPtr<Scalar,1,false>> lanes1_1;
	std::vector<LayerPtr<Scalar,1>> lane1_1_1_layers(1);
	lane1_1_1_layers[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(dims_1, 6, init));
	lanes1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane1_1_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane2_1_1_layers(3);
	lane2_1_1_layers[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(dims_1, 12, init));
	lane2_1_1_layers[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(lane2_1_1_layers[0]->get_output_dims()));
	lane2_1_1_layers[2] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(lane2_1_1_layers[1]->get_output_dims(), 6, init));
	lanes1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane2_1_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane3_1_1_layers(2);
	lane3_1_1_layers[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(dims_1, 4, init));
	lane3_1_1_layers[1] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(lane3_1_1_layers[0]->get_output_dims(), 6, init));
	lanes1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane3_1_1_layers))));
	NeuralNetPtr<Scalar,1,false> parallel_net1_1(new ParallelNeuralNetwork<Scalar,1,SUM>(std::move(lanes1_1)));
	ASSERT_TRUE((parallel_net1_1->get_output_dims() == Dimensions<std::size_t,1>({ 6 })));
	nonseq_network_grad_test<Scalar,1>("parallel net with summation", std::move(parallel_net1_1));
	// Rank 1 with multiplication.
	std::vector<NeuralNetPtr<Scalar,1,false>> lanes2_1;
	std::vector<LayerPtr<Scalar,1>> lane1_2_1_layers(1);
	lane1_2_1_layers[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(dims_1, 6, init));
	lanes2_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane1_2_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane2_2_1_layers(3);
	lane2_2_1_layers[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(dims_1, 12, init));
	lane2_2_1_layers[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(lane2_2_1_layers[0]->get_output_dims()));
	lane2_2_1_layers[2] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(lane2_2_1_layers[1]->get_output_dims(), 6, init));
	lanes2_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane2_2_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane3_2_1_layers(2);
	lane3_2_1_layers[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(dims_1, 4, init));
	lane3_2_1_layers[1] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(lane3_2_1_layers[0]->get_output_dims(), 6, init));
	lanes2_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane3_2_1_layers))));
	NeuralNetPtr<Scalar,1,false> parallel_net2_1(new ParallelNeuralNetwork<Scalar,1,MUL>(std::move(lanes2_1)));
	ASSERT_TRUE((parallel_net2_1->get_output_dims() == Dimensions<std::size_t,1>({ 6 })));
	nonseq_network_grad_test<Scalar,1>("parallel net with multiplication", std::move(parallel_net2_1));
	// Rank 2 with lowest rank concatenation.
	Dimensions<std::size_t,2> dims_2({ 6u, 6u });
	std::vector<NeuralNetPtr<Scalar,2,false>> lanes_2;
	std::vector<LayerPtr<Scalar,2>> lane1_2_layers(1);
	lane1_2_layers[0] = LayerPtr<Scalar,2>(new DenseLayer<Scalar,2>(dims_2, 6, init));
	lanes_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(lane1_2_layers))));
	std::vector<LayerPtr<Scalar,2>> lane2_2_layers(1);
	lane2_2_layers[0] = LayerPtr<Scalar,2>(new DenseLayer<Scalar,2>(dims_2, 12, init));
	lanes_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(lane2_2_layers))));
	NeuralNetPtr<Scalar,2,false> parallel_net_2(new ParallelNeuralNetwork<Scalar,2,CONCAT_LO_RANK>(std::move(lanes_2)));
	ASSERT_TRUE((parallel_net_2->get_output_dims() == Dimensions<std::size_t,2>({ 18, 1 })));
	nonseq_network_grad_test<Scalar,2>("parallel net with lowest rank concatenation", std::move(parallel_net_2));
	// Rank 3 with highest rank concatenation.
	Dimensions<std::size_t,3> dims_3({ 4u, 4u, 3u });
	std::vector<NeuralNetPtr<Scalar,3,false>> lanes_3;
	std::vector<LayerPtr<Scalar,3>> lane1_3_layers(1);
	lane1_3_layers[0] = LayerPtr<Scalar,3>(new ConvolutionLayer<Scalar>(dims_3, 4, init));
	lanes_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(lane1_3_layers))));
	std::vector<LayerPtr<Scalar,3>> lane2_3_layers(1);
	lane2_3_layers[0] = LayerPtr<Scalar,3>(new ConvolutionLayer<Scalar>(dims_3, 2, init));
	lanes_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(lane2_3_layers))));
	NeuralNetPtr<Scalar,3,false> parallel_net_3(new ParallelNeuralNetwork<Scalar,3>(std::move(lanes_3)));
	ASSERT_TRUE((parallel_net_3->get_output_dims() == Dimensions<std::size_t,3>({ 4, 4, 6 })));
	nonseq_network_grad_test<Scalar,3>("parallel net with highest rank concatenation", std::move(parallel_net_3));
}

TEST(GradientTest, ParallelNet) {
	parallel_net_grad_test<float>();
	parallel_net_grad_test<double>();
}

/**
 * Performs gradient checks on residual neural networks.
 */
template<typename Scalar>
inline void residual_net_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>());
	// Rank 1.
	std::vector<NeuralNetPtr<Scalar,1,false>> modules_1;
	std::vector<NeuralNetPtr<Scalar,1,false>> sub_modules1_1;
	std::vector<LayerPtr<Scalar,1>> layers1_1(2);
	layers1_1[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>({ 32u }, 18, init));
	layers1_1[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(layers1_1[0]->get_output_dims()));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers1_1))));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new DenseLayer<Scalar,1>(sub_modules1_1[0]->get_output_dims(), 32, init)))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new StackedNeuralNetwork<Scalar,1,false>(std::move(sub_modules1_1))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new StackedNeuralNetwork<Scalar,1,false>(NeuralNetPtr<Scalar,1,false>(
			new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new DenseLayer<Scalar,1>(modules_1[0]->get_output_dims(), 32, init)))))));
	nonseq_network_grad_test<Scalar,1>("residual net", NeuralNetPtr<Scalar,1,false>(
			new ResidualNeuralNetwork<Scalar,1>(std::move(modules_1))));
	// Rank 3.
	std::vector<NeuralNetPtr<Scalar,3,false>> modules_3;
	std::vector<NeuralNetPtr<Scalar,3,false>> sub_modules1_3;
	std::vector<LayerPtr<Scalar,3>> layers1_3(2);
	layers1_3[0] = LayerPtr<Scalar,3>(new ConvolutionLayer<Scalar>({ 4u, 4u, 3u }, 4, init));
	layers1_3[1] = LayerPtr<Scalar,3>(new SigmoidActivationLayer<Scalar,3>(layers1_3[0]->get_output_dims()));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers1_3))));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvolutionLayer<Scalar>(sub_modules1_3[0]->get_output_dims(), 3, init)))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new StackedNeuralNetwork<Scalar,3,false>(std::move(sub_modules1_3))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new StackedNeuralNetwork<Scalar,3,false>(NeuralNetPtr<Scalar,3,false>(
			new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvolutionLayer<Scalar>(modules_3[0]->get_output_dims(), 3, init)))))));
	nonseq_network_grad_test<Scalar,3>("residual net", NeuralNetPtr<Scalar,3,false>(
			new ResidualNeuralNetwork<Scalar,3>(std::move(modules_3))));
}

TEST(GradientTest, ResidualNet) {
	residual_net_grad_test<float>();
	residual_net_grad_test<double>();
}

/**
 * Performs gradient checks on dense neural networks.
 */
template<typename Scalar>
inline void dense_net_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>());
	// Rank 1.
	Dimensions<std::size_t,1> dims_1({ 32u });
	std::vector<NeuralNetPtr<Scalar,1,false>> modules_1;
	std::vector<NeuralNetPtr<Scalar,1,false>> sub_modules1_1;
	std::vector<LayerPtr<Scalar,1>> layers1_1(2);
	layers1_1[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(dims_1, 18, init));
	layers1_1[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(layers1_1[0]->get_output_dims()));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers1_1))));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new DenseLayer<Scalar,1>(sub_modules1_1[0]->get_output_dims(), 32, init)))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new StackedNeuralNetwork<Scalar,1,false>(std::move(sub_modules1_1))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new StackedNeuralNetwork<Scalar,1,false>(NeuralNetPtr<Scalar,1,false>(
			new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new DenseLayer<Scalar,1>(dims_1.add_along_rank(modules_1[0]->get_output_dims(), 0), 16, init)))))));
	nonseq_network_grad_test<Scalar,1>("dense net", NeuralNetPtr<Scalar,1,false>(
			new DenseNeuralNetwork<Scalar,1>(std::move(modules_1))));
	// Rank 2.
	Dimensions<std::size_t,2> dims_2({ 6u, 6u });
	std::vector<NeuralNetPtr<Scalar,2,false>> modules_2;
	std::vector<NeuralNetPtr<Scalar,2,false>> sub_modules1_2;
	std::vector<LayerPtr<Scalar,2>> layers1_2(2);
	layers1_2[0] = LayerPtr<Scalar,2>(new DenseLayer<Scalar,2>(dims_2, 12, init));
	layers1_2[1] = LayerPtr<Scalar,2>(new SigmoidActivationLayer<Scalar,2>(layers1_2[0]->get_output_dims()));
	sub_modules1_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(layers1_2))));
	sub_modules1_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(LayerPtr<Scalar,2>(
			new DenseLayer<Scalar,2>(sub_modules1_2[0]->get_output_dims(), 6, init)))));
	modules_2.push_back(NeuralNetPtr<Scalar,2,false>(new StackedNeuralNetwork<Scalar,2,false>(std::move(sub_modules1_2))));
	modules_2.push_back(NeuralNetPtr<Scalar,2,false>(new StackedNeuralNetwork<Scalar,2,false>(NeuralNetPtr<Scalar,2,false>(
			new FeedforwardNeuralNetwork<Scalar,2>(LayerPtr<Scalar,2>(
			new DenseLayer<Scalar,2>(dims_2.add_along_rank(modules_2[0]->get_output_dims(), 1), 6, init)))))));
	nonseq_network_grad_test<Scalar,2>("dense net", NeuralNetPtr<Scalar,2,false>(
			new DenseNeuralNetwork<Scalar,2,HIGHEST_RANK>(std::move(modules_2))));
	// Rank 3.
	Dimensions<std::size_t,3> dims_3({ 4u, 4u, 3u });
	std::vector<NeuralNetPtr<Scalar,3,false>> modules_3;
	std::vector<NeuralNetPtr<Scalar,3,false>> sub_modules1_3;
	std::vector<LayerPtr<Scalar,3>> layers1_3(2);
	layers1_3[0] = LayerPtr<Scalar,3>(new ConvolutionLayer<Scalar>(dims_3, 4, init, ConvolutionLayer<Scalar>::NO_PARAM_REG, 2, 3));
	layers1_3[1] = LayerPtr<Scalar,3>(new SigmoidActivationLayer<Scalar,3>(layers1_3[0]->get_output_dims()));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers1_3))));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvolutionLayer<Scalar>(sub_modules1_3[0]->get_output_dims(), 3, init)))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new StackedNeuralNetwork<Scalar,3,false>(std::move(sub_modules1_3))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new StackedNeuralNetwork<Scalar,3,false>(NeuralNetPtr<Scalar,3,false>(
			new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvolutionLayer<Scalar>(dims_3.add_along_rank(modules_3[0]->get_output_dims(), 0), 3, init)))))));
	nonseq_network_grad_test<Scalar,3>("dense net", NeuralNetPtr<Scalar,3,false>(
			new DenseNeuralNetwork<Scalar,3,LOWEST_RANK>(std::move(modules_3))));
}

TEST(GradientTest, DenseNet) {
	dense_net_grad_test<float>();
	dense_net_grad_test<double>();
}

/**
 * Performs gradient checks on sequential neural networks.
 */
template<typename Scalar>
inline void sequential_net_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>());
	// Rank 1.
	std::vector<LayerPtr<Scalar,1>> layers_1(3);
	layers_1[0] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>({ 32u }, 16, init));
	layers_1[1] = LayerPtr<Scalar,1>(new TanhActivationLayer<Scalar,1>(layers_1[0]->get_output_dims()));
	layers_1[2] = LayerPtr<Scalar,1>(new DenseLayer<Scalar,1>(layers_1[1]->get_output_dims(), 4, init));
	seq_network_grad_test("sequential net", NeuralNetPtr<Scalar,1,true>(
			new SequentialNeuralNetwork<Scalar,1>(NeuralNetPtr<Scalar,1,false>(
					new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers_1))))));
	// Rank 3.
	std::vector<LayerPtr<Scalar,3>> layers_3(4);
	layers_3[0] = LayerPtr<Scalar,3>(new ConvolutionLayer<Scalar>({ 4u, 4u, 2u }, 4, init));
	layers_3[1] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers_3[0]->get_output_dims()));
	layers_3[2] = LayerPtr<Scalar,3>(new MaxPoolLayer<Scalar>(layers_3[1]->get_output_dims()));
	layers_3[3] = LayerPtr<Scalar,3>(new DenseLayer<Scalar,3>(layers_3[2]->get_output_dims(), 1, init));
	seq_network_grad_test("sequential net", NeuralNetPtr<Scalar,3,true>(
			new SequentialNeuralNetwork<Scalar,3>(NeuralNetPtr<Scalar,3,false>(
					new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers_3))))));
}

TEST(GradientTest, SequentialNet) {
	sequential_net_grad_test<float>();
	sequential_net_grad_test<double>();
}

/**
 * Performs gradient checks on recurrent neural networks.
 */
template<typename Scalar>
inline void recurrent_net_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new OrthogonalWeightInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	// Rank 1.
	KernelPtr<Scalar,1> input_kernel1_1(new DenseLayer<Scalar,1>({ 12u }, 12, init, reg));
	KernelPtr<Scalar,1> state_kernel1_1(new DenseLayer<Scalar,1>(input_kernel1_1->get_output_dims(), 12, init, reg));
	KernelPtr<Scalar,1> output_kernel1_1(new DenseLayer<Scalar,1>(state_kernel1_1->get_output_dims(), 4, init, reg));
	ActivationPtr<Scalar,1> state_act1_1(new SigmoidActivationLayer<Scalar,1>(state_kernel1_1->get_output_dims()));
	ActivationPtr<Scalar,1> output_act1_1(new IdentityActivationLayer<Scalar,1>(output_kernel1_1->get_output_dims()));
	KernelPtr<Scalar,1> input_kernel2_1((DenseLayer<Scalar,1>*) input_kernel1_1->clone());
	KernelPtr<Scalar,1> state_kernel2_1((DenseLayer<Scalar,1>*) input_kernel2_1->clone());
	KernelPtr<Scalar,1> output_kernel2_1((DenseLayer<Scalar,1>*) output_kernel1_1->clone());
	ActivationPtr<Scalar,1> state_act2_1((SigmoidActivationLayer<Scalar,1>*) state_act1_1->clone());
	ActivationPtr<Scalar,1> output_act2_1((IdentityActivationLayer<Scalar,1>*) output_act1_1->clone());
	seq_network_grad_test("recurrent net", NeuralNetPtr<Scalar,1,true>(
			new RecurrentNeuralNetwork<Scalar,1>(std::move(input_kernel1_1), std::move(state_kernel1_1),
					std::move(output_kernel1_1), std::move(state_act1_1), std::move(output_act1_1),
					[](int input_seq_length) { return std::make_pair(input_seq_length, 0); })), 3, 3);
	seq_network_grad_test("recurrent net with multiplicative integration", NeuralNetPtr<Scalar,1,true>(
			new RecurrentNeuralNetwork<Scalar,1,true>(std::move(input_kernel2_1), std::move(state_kernel2_1),
					std::move(output_kernel2_1), std::move(state_act2_1), std::move(output_act2_1),
					[](int input_seq_length) { return std::make_pair(1, input_seq_length - 1); })), 5, 1);
	// Rank 3.
	KernelPtr<Scalar,3> input_kernel1_3(new ConvolutionLayer<Scalar>({ 4u, 4u, 2u }, 5, init, reg));
	KernelPtr<Scalar,3> state_kernel1_3(new ConvolutionLayer<Scalar>(input_kernel1_3->get_output_dims(), 5, init, reg));
	KernelPtr<Scalar,3> output_kernel1_3(new DenseLayer<Scalar,3>(state_kernel1_3->get_output_dims(), 2, init, reg));
	ActivationPtr<Scalar,3> state_act1_3(new SoftplusActivationLayer<Scalar,3>(state_kernel1_3->get_output_dims()));
	ActivationPtr<Scalar,3> output_act1_3(new IdentityActivationLayer<Scalar,3>(output_kernel1_3->get_output_dims()));
	KernelPtr<Scalar,3> input_kernel2_3((ConvolutionLayer<Scalar>*) input_kernel1_3->clone());
	KernelPtr<Scalar,3> state_kernel2_3((ConvolutionLayer<Scalar>*) state_kernel1_3->clone());
	KernelPtr<Scalar,3> output_kernel2_3((DenseLayer<Scalar,3>*) output_kernel1_3->clone());
	ActivationPtr<Scalar,3> state_act2_3((SoftplusActivationLayer<Scalar,3>*) state_act1_3->clone());
	ActivationPtr<Scalar,3> output_act2_3((IdentityActivationLayer<Scalar,3>*) output_act1_3->clone());
	seq_network_grad_test("recurrent net", NeuralNetPtr<Scalar,3,true>(
			new RecurrentNeuralNetwork<Scalar,3>(std::move(input_kernel1_3), std::move(state_kernel1_3),
					std::move(output_kernel1_3), std::move(state_act1_3), std::move(output_act1_3),
					[](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); })), 5, 3);
	seq_network_grad_test("recurrent net with multiplicative integration", NeuralNetPtr<Scalar,3,true>(
			new RecurrentNeuralNetwork<Scalar,3,true>(std::move(input_kernel2_3), std::move(state_kernel2_3),
					std::move(output_kernel2_3), std::move(state_act2_3), std::move(output_act2_3),
					[](int input_seq_length) { return std::make_pair(2, input_seq_length); })), 3, 2);
}

TEST(GradientTest, RecurrentNet) {
	recurrent_net_grad_test<float>();
	recurrent_net_grad_test<double>();
}

/**
 * Performs gradient checks on LSTM neural networks.
 */
template<typename Scalar>
inline void lstm_net_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new OrthogonalWeightInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	// Rank 1.
	Dimensions<std::size_t,1> input_dims_1({ 32u });
	Dimensions<std::size_t,1> output_dims_1({ 5u });
	KernelPtr<Scalar,1> forget_input_kernel1_1(new DenseLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> forget_output_kernel1_1(new DenseLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> write_input_kernel1_1(new DenseLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> write_output_kernel1_1(new DenseLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> candidate_input_kernel1_1(new DenseLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> candidate_output_kernel1_1(new DenseLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> read_input_kernel1_1(new DenseLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> read_output_kernel1_1(new DenseLayer<Scalar,1>(output_dims_1, 5, init, reg));
	ActivationPtr<Scalar,1> forget_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> write_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> candidate_act1_1(new TanhActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> state_act1_1(new TanhActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> read_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	KernelPtr<Scalar,1> forget_input_kernel2_1((DenseLayer<Scalar,1>*) forget_input_kernel1_1->clone());
	KernelPtr<Scalar,1> forget_output_kernel2_1((DenseLayer<Scalar,1>*) forget_output_kernel1_1->clone());
	KernelPtr<Scalar,1> write_input_kernel2_1((DenseLayer<Scalar,1>*) write_input_kernel1_1->clone());
	KernelPtr<Scalar,1> write_output_kernel2_1((DenseLayer<Scalar,1>*) write_output_kernel1_1->clone());
	KernelPtr<Scalar,1> candidate_input_kernel2_1((DenseLayer<Scalar,1>*) candidate_input_kernel1_1->clone());
	KernelPtr<Scalar,1> candidate_output_kernel2_1((DenseLayer<Scalar,1>*) candidate_output_kernel1_1->clone());
	KernelPtr<Scalar,1> read_input_kernel2_1((DenseLayer<Scalar,1>*) read_input_kernel1_1->clone());
	KernelPtr<Scalar,1> read_output_kernel2_1((DenseLayer<Scalar,1>*) read_output_kernel1_1->clone());
	ActivationPtr<Scalar,1> forget_act2_1((SigmoidActivationLayer<Scalar,1>*) forget_act1_1->clone());
	ActivationPtr<Scalar,1> write_act2_1((SigmoidActivationLayer<Scalar,1>*) write_act1_1->clone());
	ActivationPtr<Scalar,1> candidate_act2_1((TanhActivationLayer<Scalar,1>*) candidate_act1_1->clone());
	ActivationPtr<Scalar,1> state_act2_1((TanhActivationLayer<Scalar,1>*) state_act1_1->clone());
	ActivationPtr<Scalar,1> read_act2_1((SigmoidActivationLayer<Scalar,1>*) read_act1_1->clone());
	seq_network_grad_test("lstm net", NeuralNetPtr<Scalar,1,true>(
			new LSTMNeuralNetwork<Scalar,1>(std::move(forget_input_kernel1_1), std::move(forget_output_kernel1_1),
					std::move(write_input_kernel1_1), std::move(write_output_kernel1_1), std::move(candidate_input_kernel1_1),
					std::move(candidate_output_kernel1_1), std::move(read_input_kernel1_1), std::move(read_output_kernel1_1),
					std::move(forget_act1_1), std::move(write_act1_1), std::move(candidate_act1_1), std::move(state_act1_1),
					std::move(read_act1_1), [](int input_seq_length) { return std::make_pair(input_seq_length, 0); })), 3, 3);
	seq_network_grad_test("lstm net with multiplicative integration", NeuralNetPtr<Scalar,1,true>(
			new LSTMNeuralNetwork<Scalar,1,true>(std::move(forget_input_kernel2_1), std::move(forget_output_kernel2_1),
					std::move(write_input_kernel2_1), std::move(write_output_kernel2_1), std::move(candidate_input_kernel2_1),
					std::move(candidate_output_kernel2_1), std::move(read_input_kernel2_1), std::move(read_output_kernel2_1),
					std::move(forget_act2_1), std::move(write_act2_1), std::move(candidate_act2_1), std::move(state_act2_1),
					std::move(read_act2_1), [](int input_seq_length) { return std::make_pair(1, input_seq_length - 1); })), 5, 1);
	// Rank 3.
	Dimensions<std::size_t,3> input_dims_3({ 5u, 3u, 3u });
	Dimensions<std::size_t,3> output_dims_3({ 3u, 3u, 3u });
	KernelPtr<Scalar,3> forget_input_kernel1_3(new ConvolutionLayer<Scalar>(input_dims_3, 3, init, reg, 3, 3, 0, 1));
	KernelPtr<Scalar,3> forget_output_kernel1_3(new ConvolutionLayer<Scalar>(output_dims_3, 3, init, reg));
	KernelPtr<Scalar,3> write_input_kernel1_3(new ConvolutionLayer<Scalar>(input_dims_3, 3, init, reg, 3, 3, 0, 1));
	KernelPtr<Scalar,3> write_output_kernel1_3(new ConvolutionLayer<Scalar>(output_dims_3, 3, init, reg));
	KernelPtr<Scalar,3> candidate_input_kernel1_3(new ConvolutionLayer<Scalar>(input_dims_3, 3, init, reg, 3, 3, 0, 1));
	KernelPtr<Scalar,3> candidate_output_kernel1_3(new ConvolutionLayer<Scalar>(output_dims_3, 3, init, reg));
	KernelPtr<Scalar,3> read_input_kernel1_3(new ConvolutionLayer<Scalar>(input_dims_3, 3, init, reg, 3, 3, 0, 1));
	KernelPtr<Scalar,3> read_output_kernel1_3(new ConvolutionLayer<Scalar>(output_dims_3, 3, init, reg));
	ActivationPtr<Scalar,3> forget_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> write_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> candidate_act1_3(new TanhActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> state_act1_3(new TanhActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> read_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	KernelPtr<Scalar,3> forget_input_kernel2_3((ConvolutionLayer<Scalar>*) forget_input_kernel1_3->clone());
	KernelPtr<Scalar,3> forget_output_kernel2_3((ConvolutionLayer<Scalar>*) forget_output_kernel1_3->clone());
	KernelPtr<Scalar,3> write_input_kernel2_3((ConvolutionLayer<Scalar>*) write_input_kernel1_3->clone());
	KernelPtr<Scalar,3> write_output_kernel2_3((ConvolutionLayer<Scalar>*) write_output_kernel1_3->clone());
	KernelPtr<Scalar,3> candidate_input_kernel2_3((ConvolutionLayer<Scalar>*) candidate_input_kernel1_3->clone());
	KernelPtr<Scalar,3> candidate_output_kernel2_3((ConvolutionLayer<Scalar>*) candidate_output_kernel1_3->clone());
	KernelPtr<Scalar,3> read_input_kernel2_3((ConvolutionLayer<Scalar>*) read_input_kernel1_3->clone());
	KernelPtr<Scalar,3> read_output_kernel2_3((ConvolutionLayer<Scalar>*) read_output_kernel1_3->clone());
	ActivationPtr<Scalar,3> forget_act2_3((SigmoidActivationLayer<Scalar,3>*) forget_act1_3->clone());
	ActivationPtr<Scalar,3> write_act2_3((SigmoidActivationLayer<Scalar,3>*) write_act1_3->clone());
	ActivationPtr<Scalar,3> candidate_act2_3((TanhActivationLayer<Scalar,3>*) candidate_act1_3->clone());
	ActivationPtr<Scalar,3> state_act2_3((TanhActivationLayer<Scalar,3>*) state_act1_3->clone());
	ActivationPtr<Scalar,3> read_act2_3((SigmoidActivationLayer<Scalar,3>*) read_act1_3->clone());
	seq_network_grad_test("lstm net", NeuralNetPtr<Scalar,3,true>(
			new LSTMNeuralNetwork<Scalar,3>(std::move(forget_input_kernel1_3), std::move(forget_output_kernel1_3),
					std::move(write_input_kernel1_3), std::move(write_output_kernel1_3), std::move(candidate_input_kernel1_3),
					std::move(candidate_output_kernel1_3), std::move(read_input_kernel1_3), std::move(read_output_kernel1_3),
					std::move(forget_act1_3), std::move(write_act1_3), std::move(candidate_act1_3), std::move(state_act1_3),
					std::move(read_act1_3), [](int input_seq_length) { return std::make_pair(3, input_seq_length - 3); })), 5, 3);
	seq_network_grad_test("lstm net with multiplicative integration", NeuralNetPtr<Scalar,3,true>(
			new LSTMNeuralNetwork<Scalar,3,true>(std::move(forget_input_kernel2_3), std::move(forget_output_kernel2_3),
					std::move(write_input_kernel2_3), std::move(write_output_kernel2_3), std::move(candidate_input_kernel2_3),
					std::move(candidate_output_kernel2_3), std::move(read_input_kernel2_3), std::move(read_output_kernel2_3),
					std::move(forget_act2_3), std::move(write_act2_3), std::move(candidate_act2_3), std::move(state_act2_3),
					std::move(read_act2_3), [](int input_seq_length) { return std::make_pair(2, input_seq_length); })), 3, 2);
}

TEST(GradientTest, LSTMNet) {
	lstm_net_grad_test<float>();
	lstm_net_grad_test<double>();
}

/**
 * Performs gradient checks on bidirectional recurrent and LSTM neural networks.
 */
template<typename Scalar>
inline void bidirectional_net_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new OrthogonalWeightInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	// 3rd degree RNN with highest rank concatenation.
	KernelPtr<Scalar,3> input_kernel1(new ConvolutionLayer<Scalar>({ 4u, 4u, 2u }, 5, init, reg));
	KernelPtr<Scalar,3> state_kernel1(new ConvolutionLayer<Scalar>(input_kernel1->get_output_dims(), 5, init, reg));
	KernelPtr<Scalar,3> output_kernel1(new DenseLayer<Scalar,3>(input_kernel1->get_output_dims(), 2, init, reg));
	ActivationPtr<Scalar,3> state_act1(new SigmoidActivationLayer<Scalar,3>(input_kernel1->get_output_dims()));
	ActivationPtr<Scalar,3> output_act1(new IdentityActivationLayer<Scalar,3>(output_kernel1->get_output_dims()));
	seq_network_grad_test("bidirectional recurrent net with highest rank concatenation", NeuralNetPtr<Scalar,3,true>(
			new BidirectionalNeuralNetwork<Scalar,3,CONCAT_HI_RANK>(UnidirNeuralNetPtr<Scalar,3>(
					new RecurrentNeuralNetwork<Scalar,3,true>(std::move(input_kernel1), std::move(state_kernel1),
							std::move(output_kernel1), std::move(state_act1), std::move(output_act1),
							[](int input_seq_length) { return std::make_pair(3, 2); })))), 7, 3);
	// 2nd degree LSTM with lowest rank concatenation.
	Dimensions<std::size_t,2> input_dims({ 6u, 6u });
	Dimensions<std::size_t,2> output_dims({ 6u, 1u });
	KernelPtr<Scalar,2> forget_input_kernel2(new DenseLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> forget_output_kernel2(new DenseLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> write_input_kernel2(new DenseLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> write_output_kernel2(new DenseLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> candidate_input_kernel2(new DenseLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> candidate_output_kernel2(new DenseLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> read_input_kernel2(new DenseLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> read_output_kernel2(new DenseLayer<Scalar,2>(output_dims, 6, init, reg));
	ActivationPtr<Scalar,2> forget_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> write_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> candidate_act2(new SoftplusActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> state_act2(new SoftplusActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> read_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	seq_network_grad_test("bidirectional lstm net with lowest rank concatenation", NeuralNetPtr<Scalar,2,true>(
			new BidirectionalNeuralNetwork<Scalar,2,CONCAT_LO_RANK>(UnidirNeuralNetPtr<Scalar,2>(
					new LSTMNeuralNetwork<Scalar,2,true>(std::move(forget_input_kernel2),
							std::move(forget_output_kernel2), std::move(write_input_kernel2),
							std::move(write_output_kernel2), std::move(candidate_input_kernel2),
							std::move(candidate_output_kernel2), std::move(read_input_kernel2),
							std::move(read_output_kernel2), std::move(forget_act2), std::move(write_act2),
							std::move(candidate_act2), std::move(state_act2), std::move(read_act2),
							[](int input_seq_length) { return std::make_pair(1, 2); })))), 5, 1);
	// 1st degree RNN with summation.
	KernelPtr<Scalar,1> input_kernel3(new DenseLayer<Scalar,1>({ 24u }, 8, init, reg));
	KernelPtr<Scalar,1> state_kernel3(new DenseLayer<Scalar,1>(input_kernel3->get_output_dims(), 8, init, reg));
	KernelPtr<Scalar,1> output_kernel3(new DenseLayer<Scalar,1>(input_kernel3->get_output_dims(), 1, init, reg));
	ActivationPtr<Scalar,1> state_act3(new SigmoidActivationLayer<Scalar,1>(input_kernel3->get_output_dims()));
	ActivationPtr<Scalar,1> output_act3(new SigmoidActivationLayer<Scalar,1>(output_kernel3->get_output_dims()));
	seq_network_grad_test("bidirectional recurrent net with summation", NeuralNetPtr<Scalar,1,true>(
			new BidirectionalNeuralNetwork<Scalar,1,SUM>(UnidirNeuralNetPtr<Scalar,1>(
					new RecurrentNeuralNetwork<Scalar,1>(std::move(input_kernel3), std::move(state_kernel3),
							std::move(output_kernel3), std::move(state_act3), std::move(output_act3),
							[](int input_seq_length) { return std::make_pair(5, 2); })))), 7, 5);
	// 3rd degree RNN with multiplication.
	KernelPtr<Scalar,3> input_kernel4(new ConvolutionLayer<Scalar>({ 4u, 4u, 2u }, 5, init, reg));
	KernelPtr<Scalar,3> state_kernel4(new ConvolutionLayer<Scalar>(input_kernel4->get_output_dims(), 5, init, reg));
	KernelPtr<Scalar,3> output_kernel4(new DenseLayer<Scalar,3>(input_kernel4->get_output_dims(), 2, init, reg));
	ActivationPtr<Scalar,3> state_act4(new SigmoidActivationLayer<Scalar,3>(input_kernel4->get_output_dims()));
	ActivationPtr<Scalar,3> output_act4(new IdentityActivationLayer<Scalar,3>(output_kernel4->get_output_dims()));
	seq_network_grad_test("bidirectional recurrent net with multiplication", NeuralNetPtr<Scalar,3,true>(
			new BidirectionalNeuralNetwork<Scalar,3,MUL>(UnidirNeuralNetPtr<Scalar,3>(
					new RecurrentNeuralNetwork<Scalar,3,true>(std::move(input_kernel4), std::move(state_kernel4),
							std::move(output_kernel4), std::move(state_act4), std::move(output_act4),
							[](int input_seq_length) { return std::make_pair(3, 2); })))), 7, 3);
}

TEST(GradientTest, BidirectionalNet) {
	bidirectional_net_grad_test<float>();
	bidirectional_net_grad_test<double>();
}

/***********************
 * LOSS GRADIENT TESTS *
 ***********************/

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank>
inline void absolute_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	auto net = reg_neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_tensor<Scalar,Rank + 1>(batch_out_dims));
	LossSharedPtr<Scalar,Rank,false> loss(new AbsoluteLoss<Scalar,Rank,false>());
	VanillaSGDOptimizer<Scalar,Rank,false> opt(loss, samples);
	grad_test<Scalar,Rank,false>("absolute loss", prov, *net, opt);
	// Sequential.
	SequentialNeuralNetwork<Scalar,Rank> seq_net(std::move(net));
	Dimensions<std::size_t,Rank + 2> seq_batch_dims = dims.template promote<2>();
	seq_batch_dims(0) = samples;
	seq_batch_dims(1) = time_steps;
	Dimensions<std::size_t,Rank + 2> seq_batch_out_dims = seq_net.get_output_dims().template promote<2>();
	seq_batch_out_dims(0) = samples;
	seq_batch_out_dims(1) = time_steps;
	MemoryDataProvider<Scalar,Rank,true> seq_prov(random_tensor<Scalar,Rank + 2>(seq_batch_dims),
			random_tensor<Scalar,Rank + 2>(seq_batch_out_dims));
	LossSharedPtr<Scalar,Rank,true> seq_loss(new AbsoluteLoss<Scalar,Rank,true>());
	VanillaSGDOptimizer<Scalar,Rank,true> seq_opt(seq_loss, samples);
	grad_test<Scalar,Rank,true>("absolute loss", seq_prov, seq_net, seq_opt);
}

/**
 * Performs gradient checks on the absolute loss function.
 */
template<typename Scalar>
inline void absolute_loss_grad_test() {
	absolute_loss_grad_test<Scalar,1>({ 24u });
	absolute_loss_grad_test<Scalar,2>({ 6u, 6u });
	absolute_loss_grad_test<Scalar,3>({ 4u, 4u, 2u });
}

TEST(GradientTest, AbsoluteLoss) {
	absolute_loss_grad_test<float>();
	absolute_loss_grad_test<double>();
}

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank, bool Squared>
inline void hinge_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	auto net = tanh_neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_one_hot_tensor<Scalar,Rank + 1,false>(batch_out_dims));
	LossSharedPtr<Scalar,Rank,false> loss(new HingeLoss<Scalar,Rank,false,Squared>());
	VanillaSGDOptimizer<Scalar,Rank,false> opt(loss, samples);
	std::string name = std::string(Squared ? "squared " : "") + std::string("hinge loss");
	grad_test<Scalar,Rank,false>(name, prov, *net, opt);
	// Sequential.
	SequentialNeuralNetwork<Scalar,Rank> seq_net(std::move(net));
	Dimensions<std::size_t,Rank + 2> seq_batch_dims = dims.template promote<2>();
	seq_batch_dims(0) = samples;
	seq_batch_dims(1) = time_steps;
	Dimensions<std::size_t,Rank + 2> seq_batch_out_dims = seq_net.get_output_dims().template promote<2>();
	seq_batch_out_dims(0) = samples;
	seq_batch_out_dims(1) = time_steps;
	MemoryDataProvider<Scalar,Rank,true> seq_prov(random_tensor<Scalar,Rank + 2>(seq_batch_dims),
			random_one_hot_tensor<Scalar,Rank + 2,true>(seq_batch_out_dims));
	LossSharedPtr<Scalar,Rank,true> seq_loss(new HingeLoss<Scalar,Rank,true,Squared>());
	VanillaSGDOptimizer<Scalar,Rank,true> seq_opt(seq_loss, samples);
	grad_test<Scalar,Rank,true>(name, seq_prov, seq_net, seq_opt);
}

/**
 * Performs gradient checks on the hinge loss function.
 */
template<typename Scalar>
inline void hinge_loss_grad_test() {
	Dimensions<std::size_t,1> dims_1({ 24u });
	hinge_loss_grad_test<Scalar,1,false>(dims_1);
	hinge_loss_grad_test<Scalar,1,true>(dims_1);
	Dimensions<std::size_t,2> dims_2({ 6u, 6u });
	hinge_loss_grad_test<Scalar,2,false>(dims_2);
	hinge_loss_grad_test<Scalar,2,true>(dims_2);
	Dimensions<std::size_t,3> dims_3({ 4u, 4u, 2u });
	hinge_loss_grad_test<Scalar,3,false>(dims_3);
	hinge_loss_grad_test<Scalar,3,true>(dims_3);
}

TEST(GradientTest, HingeLoss) {
	hinge_loss_grad_test<float>();
	hinge_loss_grad_test<double>();
}

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank>
inline void cross_entropy_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	auto net = softmax_neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_one_hot_tensor<Scalar,Rank + 1,false>(batch_out_dims));
	LossSharedPtr<Scalar,Rank,false> loss(new CrossEntropyLoss<Scalar,Rank,false>());
	VanillaSGDOptimizer<Scalar,Rank,false> opt(loss, samples);
	grad_test<Scalar,Rank,false>("cross entropy loss", prov, *net, opt);
	// Sequential.
	SequentialNeuralNetwork<Scalar,Rank> seq_net(std::move(net));
	Dimensions<std::size_t,Rank + 2> seq_batch_dims = dims.template promote<2>();
	seq_batch_dims(0) = samples;
	seq_batch_dims(1) = time_steps;
	Dimensions<std::size_t,Rank + 2> seq_batch_out_dims = seq_net.get_output_dims().template promote<2>();
	seq_batch_out_dims(0) = samples;
	seq_batch_out_dims(1) = time_steps;
	MemoryDataProvider<Scalar,Rank,true> seq_prov(random_tensor<Scalar,Rank + 2>(seq_batch_dims),
			random_one_hot_tensor<Scalar,Rank + 2,true>(seq_batch_out_dims));
	LossSharedPtr<Scalar,Rank,true> seq_loss(new CrossEntropyLoss<Scalar,Rank,true>());
	VanillaSGDOptimizer<Scalar,Rank,true> seq_opt(seq_loss, samples);
	grad_test<Scalar,Rank,true>("cross entropy loss", seq_prov, seq_net, seq_opt);
}

/**
 * Performs gradient checks on the cross entropy loss function.
 */
template<typename Scalar>
inline void cross_entropy_loss_grad_test() {
	cross_entropy_loss_grad_test<Scalar,1>({ 24u });
	cross_entropy_loss_grad_test<Scalar,2>({ 6u, 6u });
	cross_entropy_loss_grad_test<Scalar,3>({ 4u, 4u, 2u });
}

TEST(GradientTest, CrossEntropyLoss) {
	cross_entropy_loss_grad_test<float>();
	cross_entropy_loss_grad_test<double>();
}

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank>
inline void softmax_cross_entropy_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	auto net = reg_neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_one_hot_tensor<Scalar,Rank + 1,false>(batch_out_dims));
	LossSharedPtr<Scalar,Rank,false> loss(new SoftmaxCrossEntropyLoss<Scalar,Rank,false>());
	VanillaSGDOptimizer<Scalar,Rank,false> opt(loss, samples);
	grad_test<Scalar,Rank,false>("softmax cross entropy loss", prov, *net, opt);
	// Sequential.
	SequentialNeuralNetwork<Scalar,Rank> seq_net(std::move(net));
	Dimensions<std::size_t,Rank + 2> seq_batch_dims = dims.template promote<2>();
	seq_batch_dims(0) = samples;
	seq_batch_dims(1) = time_steps;
	Dimensions<std::size_t,Rank + 2> seq_batch_out_dims = seq_net.get_output_dims().template promote<2>();
	seq_batch_out_dims(0) = samples;
	seq_batch_out_dims(1) = time_steps;
	MemoryDataProvider<Scalar,Rank,true> seq_prov(random_tensor<Scalar,Rank + 2>(seq_batch_dims),
			random_one_hot_tensor<Scalar,Rank + 2,true>(seq_batch_out_dims));
	LossSharedPtr<Scalar,Rank,true> seq_loss(new SoftmaxCrossEntropyLoss<Scalar,Rank,true>());
	VanillaSGDOptimizer<Scalar,Rank,true> seq_opt(seq_loss, samples);
	grad_test<Scalar,Rank,true>("softmax cross entropy loss", seq_prov, seq_net, seq_opt);
}

/**
 * Performs gradient checks on the softmax-cross-entropy loss function.
 */
template<typename Scalar>
inline void softmax_cross_entropy_loss_grad_test() {
	softmax_cross_entropy_loss_grad_test<Scalar,1>({ 24u });
	softmax_cross_entropy_loss_grad_test<Scalar,2>({ 6u, 6u });
	softmax_cross_entropy_loss_grad_test<Scalar,3>({ 4u, 4u, 2u });
}

TEST(GradientTest, SoftmaxCrossEntropyLoss) {
	softmax_cross_entropy_loss_grad_test<float>();
	softmax_cross_entropy_loss_grad_test<double>();
}

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank, bool Squared>
inline void multi_label_hinge_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	auto net = tanh_neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_multi_hot_tensor<Scalar,Rank + 1>(batch_out_dims, -1));
	LossSharedPtr<Scalar,Rank,false> loss(new MultiLabelHingeLoss<Scalar,Rank,false,Squared>());
	VanillaSGDOptimizer<Scalar,Rank,false> opt(loss, samples);
	std::string name = std::string(Squared ? "squared " : "") + std::string("multi-label hinge loss");
	grad_test<Scalar,Rank,false>(name, prov, *net, opt);
	// Sequential.
	SequentialNeuralNetwork<Scalar,Rank> seq_net(std::move(net));
	Dimensions<std::size_t,Rank + 2> seq_batch_dims = dims.template promote<2>();
	seq_batch_dims(0) = samples;
	seq_batch_dims(1) = time_steps;
	Dimensions<std::size_t,Rank + 2> seq_batch_out_dims = seq_net.get_output_dims().template promote<2>();
	seq_batch_out_dims(0) = samples;
	seq_batch_out_dims(1) = time_steps;
	MemoryDataProvider<Scalar,Rank,true> seq_prov(random_tensor<Scalar,Rank + 2>(seq_batch_dims),
			random_multi_hot_tensor<Scalar,Rank + 2>(seq_batch_out_dims, -1));
	LossSharedPtr<Scalar,Rank,true> seq_loss(new MultiLabelHingeLoss<Scalar,Rank,true,Squared>());
	VanillaSGDOptimizer<Scalar,Rank,true> seq_opt(seq_loss, samples);
	grad_test<Scalar,Rank,true>(name, seq_prov, seq_net, seq_opt);
}

/**
 * Performs gradient checks on the multi-label hinge loss function.
 */
template<typename Scalar>
inline void multi_label_hinge_loss_grad_test() {
	Dimensions<std::size_t,1> dims_1({ 24u });
	multi_label_hinge_loss_grad_test<Scalar,1,false>(dims_1);
	multi_label_hinge_loss_grad_test<Scalar,1,true>(dims_1);
	Dimensions<std::size_t,2> dims_2({ 6u, 6u });
	multi_label_hinge_loss_grad_test<Scalar,2,false>(dims_2);
	multi_label_hinge_loss_grad_test<Scalar,2,true>(dims_2);
	Dimensions<std::size_t,3> dims_3({ 4u, 4u, 2u });
	multi_label_hinge_loss_grad_test<Scalar,3,false>(dims_3);
	multi_label_hinge_loss_grad_test<Scalar,3,true>(dims_3);
}

TEST(GradientTest, MultiLabelHingeLoss) {
	multi_label_hinge_loss_grad_test<float>();
	multi_label_hinge_loss_grad_test<double>();
}

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank>
inline void multi_label_log_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	auto net = sigmoid_neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_multi_hot_tensor<Scalar,Rank + 1>(batch_out_dims));
	LossSharedPtr<Scalar,Rank,false> loss(new MultiLabelLogLoss<Scalar,Rank,false>());
	VanillaSGDOptimizer<Scalar,Rank,false> opt(loss, samples);
	grad_test<Scalar,Rank,false>("multi-label log loss", prov, *net, opt);
	// Sequential.
	SequentialNeuralNetwork<Scalar,Rank> seq_net(std::move(net));
	Dimensions<std::size_t,Rank + 2> seq_batch_dims = dims.template promote<2>();
	seq_batch_dims(0) = samples;
	seq_batch_dims(1) = time_steps;
	Dimensions<std::size_t,Rank + 2> seq_batch_out_dims = seq_net.get_output_dims().template promote<2>();
	seq_batch_out_dims(0) = samples;
	seq_batch_out_dims(1) = time_steps;
	MemoryDataProvider<Scalar,Rank,true> seq_prov(random_tensor<Scalar,Rank + 2>(seq_batch_dims),
			random_multi_hot_tensor<Scalar,Rank + 2>(seq_batch_out_dims));
	LossSharedPtr<Scalar,Rank,true> seq_loss(new MultiLabelLogLoss<Scalar,Rank,true>());
	VanillaSGDOptimizer<Scalar,Rank,true> seq_opt(seq_loss, samples);
	grad_test<Scalar,Rank,true>("multi-label log loss", seq_prov, seq_net, seq_opt);
}

/**
 * Performs gradient checks on the multi-label log loss function.
 */
template<typename Scalar>
inline void multi_label_log_loss_grad_test() {
	multi_label_log_loss_grad_test<Scalar,1>({ 24u });
	multi_label_log_loss_grad_test<Scalar,2>({ 6u, 6u });
	multi_label_log_loss_grad_test<Scalar,3>({ 4u, 4u, 2u });
}

TEST(GradientTest, MultiLabelLogLoss) {
	multi_label_log_loss_grad_test<float>();
	multi_label_log_loss_grad_test<double>();
}

} /* namespace test */
} /* namespace cattle */

#endif /* GRADIENT_TEST_HPP_ */
