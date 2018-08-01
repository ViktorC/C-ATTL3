/*
 * gradient_test.cpp
 *
 *  Created on: 19.04.2018
 *      Author: Viktor Csomor
 */

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
 * @param loss The loss function to use for the gradient check.
 * @param step_size The step size for numerical differentiation.
 * @param abs_epsilon The maximum acceptable absolute difference between the analytic and
 * numerical gradients.
 * @param rel_epsilon The maximum acceptable relative difference between the analytic and
 * numerical gradients.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
inline void grad_test(std::string name, DataProvider<Scalar,Rank,Sequential>& prov,
		NeuralNetwork<Scalar,Rank,Sequential>& net, const Loss<Scalar,Rank,Sequential>& loss,
		Scalar step_size = ScalarTraits<Scalar>::step_size, Scalar abs_epsilon = ScalarTraits<Scalar>::abs_epsilon,
		Scalar rel_epsilon = ScalarTraits<Scalar>::rel_epsilon) {
	print_test_header<Scalar,Rank,Sequential>("gradient check", name);
	bool pass = GradientCheck<Scalar,Rank,Sequential>::verify_gradients(prov, net, loss, verbose, step_size,
			abs_epsilon, rel_epsilon);
	EXPECT_TRUE(pass);
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
	SquaredLoss<Scalar,Rank,false> loss;
	grad_test<Scalar,Rank,false>(name, prov, *net, loss, step_size, abs_epsilon, rel_epsilon);
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
	SquaredLoss<Scalar,Rank,true> loss;
	grad_test<Scalar,Rank,true>(name, prov, *net, loss, step_size, abs_epsilon, rel_epsilon);
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
inline void dense_kernel_layer_grad_test() {
	auto init1 = std::make_shared<GlorotParameterInitialization<Scalar>>();
	auto init2 = std::make_shared<LeCunParameterInitialization<Scalar>>();
	auto reg1 = std::make_shared<AbsoluteParameterRegularization<Scalar>>();
	auto reg2 = std::make_shared<SquaredParameterRegularization<Scalar>>();
	auto reg3 = std::make_shared<ElasticNetParameterRegularization<Scalar>>();
	LayerPtr<Scalar,1> layer1_1(new DenseKernelLayer<Scalar,1>({ 32u }, 16, init1));
	LayerPtr<Scalar,1> layer1_2(new DenseKernelLayer<Scalar,1>(layer1_1->get_output_dims(), 1, init1, reg3));
	layer_grad_test<Scalar,1>("dense kernel layer", std::move(layer1_1), std::move(layer1_2));
	LayerPtr<Scalar,2> layer2_1(new DenseKernelLayer<Scalar,2>({ 6u, 6u }, 16, init2));
	LayerPtr<Scalar,2> layer2_2(new DenseKernelLayer<Scalar,2>(layer2_1->get_output_dims(), 2, init2, reg1));
	layer_grad_test<Scalar,2>("dense kernel layer", std::move(layer2_1), std::move(layer2_2));
	LayerPtr<Scalar,3> layer3_1(new DenseKernelLayer<Scalar,3>({ 4u, 4u, 2u }, 16, init1, reg2));
	LayerPtr<Scalar,3> layer3_2(new DenseKernelLayer<Scalar,3>(layer3_1->get_output_dims(), 4, init1, reg2));
	layer_grad_test<Scalar,3>("dense kernel layer", std::move(layer3_1), std::move(layer3_2));
}

TEST(GradientTest, DenseKernelLayer) {
	dense_kernel_layer_grad_test<double>();
}

/**
 * Performs gradient checks on convolutional layers.
 */
template<typename Scalar>
inline void conv_kernel_layer_grad_test() {
	auto init = std::make_shared<HeParameterInitialization<Scalar>>();
	auto reg = std::make_shared<SquaredParameterRegularization<Scalar>>();
	LayerPtr<Scalar,1> layer1_1(new ConvKernelLayer<Scalar,1>({ 32u }, 5, init, 3, 2, 1, 1));
	LayerPtr<Scalar,1> layer2_1(new ConvKernelLayer<Scalar,1>(layer1_1->get_output_dims(), 1, init, 3, 1, 1, 0, reg));
	layer_grad_test<Scalar,1>("convolution kernel layer ", std::move(layer1_1), std::move(layer2_1));
	LayerPtr<Scalar,2> layer1_2(new ConvKernelLayer<Scalar,2>({ 10u, 10u }, 5, init, 3, 2, 1, 2, 1, 2, 1, 0));
	LayerPtr<Scalar,2> layer2_2(new ConvKernelLayer<Scalar,2>(layer1_2->get_output_dims(), 1, init, 2, 2, 1, 1,
			1, 1, 0, 0, reg));
	layer_grad_test<Scalar,2>("convolution kernel layer ", std::move(layer1_2), std::move(layer2_2));
	LayerPtr<Scalar,3> layer1_3(new ConvKernelLayer<Scalar>({ 8u, 8u, 2u }, 5, init, 3, 2, 1, 2, 1, 2, 1, 0));
	LayerPtr<Scalar,3> layer2_3(new ConvKernelLayer<Scalar>(layer1_3->get_output_dims(), 1, init, 2, 2, 1,
			1, 1, 1, 0, 0, reg));
	layer_grad_test<Scalar,3>("convolution kernel layer ", std::move(layer1_3), std::move(layer2_3));
}

TEST(GradientTest, ConvKernelLayer) {
	conv_kernel_layer_grad_test<double>();
}

/**
 * Performs gradient checks on transposed convolutional layers.
 */
template<typename Scalar>
inline void trans_conv_kernel_layer_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new HeParameterInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new SquaredParameterRegularization<Scalar>());
	LayerPtr<Scalar,1> layer1_1(new TransConvKernelLayer<Scalar,1>({ 16u }, 5, init, 4, 1, 1, 1));
	LayerPtr<Scalar,1> layer2_1(new TransConvKernelLayer<Scalar,1>(layer1_1->get_output_dims(), 1, init, 1, 1, 1, 0, reg));
	layer_grad_test<Scalar,1>("trans_convolution kernel layer ", std::move(layer1_1), std::move(layer2_1));
	LayerPtr<Scalar,2> layer1_2(new TransConvKernelLayer<Scalar,2>({ 4u, 5u }, 5, init, 5, 3, 1, 0, 1, 2, 0, 1));
	LayerPtr<Scalar,2> layer2_2(new TransConvKernelLayer<Scalar,2>(layer1_2->get_output_dims(), 1, init, 2, 2, 1, 1,
			1, 1, 0, 0, reg));
	layer_grad_test<Scalar,2>("trans_convolution kernel layer ", std::move(layer1_2), std::move(layer2_2));
	LayerPtr<Scalar,3> layer1_3(new TransConvKernelLayer<Scalar>({ 2u, 3u, 2u }, 5, init, 5, 3, 1, 0, 1, 2, 0, 1));
	LayerPtr<Scalar,3> layer2_3(new TransConvKernelLayer<Scalar>(layer1_3->get_output_dims(), 1, init, 2, 2, 1, 1,
			1, 1, 0, 0, reg));
	layer_grad_test<Scalar,3>("trans_convolution kernel layer ", std::move(layer1_3), std::move(layer2_3));
}

TEST(GradientTest, TransConvKernelLayer) {
	trans_conv_kernel_layer_grad_test<double>();
}

/**
 * @param dims The dimensions of the layers' input tensors.
 */
template<typename Scalar, std::size_t Rank>
inline void activation_layer_grad_test(const Dimensions<std::size_t,Rank>& dims) {
	auto init = ParamInitSharedPtr<Scalar>(new GlorotParameterInitialization<Scalar>());
	LayerPtr<Scalar,Rank> kernel_layer_1 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_1 = kernel_layer_1->get_output_dims();
	layer_grad_test<Scalar,Rank>("identity activation layer", std::move(kernel_layer_1),
			LayerPtr<Scalar,Rank>(new IdentityActivationLayer<Scalar,Rank>(dims_1)));
	LayerPtr<Scalar,Rank> kernel_layer_2 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_2 = kernel_layer_2->get_output_dims();
	layer_grad_test<Scalar,Rank>("scaled activation layer", std::move(kernel_layer_2),
			LayerPtr<Scalar,Rank>(new ScaledActivationLayer<Scalar,Rank>(dims_2, 1.5)));
	LayerPtr<Scalar,Rank> kernel_layer_3 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_3 = kernel_layer_3->get_output_dims();
	layer_grad_test<Scalar,Rank>("binary step activation layer", std::move(kernel_layer_3),
			LayerPtr<Scalar,Rank>(new BinaryStepActivationLayer<Scalar,Rank>(dims_3)));
	LayerPtr<Scalar,Rank> kernel_layer_4 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_4 = kernel_layer_4->get_output_dims();
	layer_grad_test<Scalar,Rank>("sigmoid activation layer", std::move(kernel_layer_4),
			LayerPtr<Scalar,Rank>(new SigmoidActivationLayer<Scalar,Rank>(dims_4)));
	LayerPtr<Scalar,Rank> kernel_layer_5 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_5 = kernel_layer_5->get_output_dims();
	layer_grad_test<Scalar,Rank>("tanh activation layer rank", std::move(kernel_layer_5),
			LayerPtr<Scalar,Rank>(new TanhActivationLayer<Scalar,Rank>(dims_5)));
	LayerPtr<Scalar,Rank> kernel_layer_6 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_6 = kernel_layer_6->get_output_dims();
	layer_grad_test<Scalar,Rank>("softsign activation layer", std::move(kernel_layer_6),
			LayerPtr<Scalar,Rank>(new SoftsignActivationLayer<Scalar,Rank>(dims_6)));
	LayerPtr<Scalar,Rank> kernel_layer_7 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_7 = kernel_layer_7->get_output_dims();
	layer_grad_test<Scalar,Rank>("softplus activation layer", std::move(kernel_layer_7),
			LayerPtr<Scalar,Rank>(new SoftplusActivationLayer<Scalar,Rank>(dims_7)));
	LayerPtr<Scalar,Rank> kernel_layer_8 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_8 = kernel_layer_8->get_output_dims();
	layer_grad_test<Scalar,Rank>("softmax activation layer", std::move(kernel_layer_8),
			LayerPtr<Scalar,Rank>(new SoftmaxActivationLayer<Scalar,Rank>(dims_8)));
	LayerPtr<Scalar,Rank> kernel_layer_9 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_9 = kernel_layer_9->get_output_dims();
	layer_grad_test<Scalar,Rank>("relu activation layer", std::move(kernel_layer_9),
			LayerPtr<Scalar,Rank>(new ReLUActivationLayer<Scalar,Rank>(dims_9)));
	LayerPtr<Scalar,Rank> kernel_layer_10 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_10 = kernel_layer_10->get_output_dims();
	layer_grad_test<Scalar,Rank>("leaky relu activation layer", std::move(kernel_layer_10),
			LayerPtr<Scalar,Rank>(new LeakyReLUActivationLayer<Scalar,Rank>(dims_10, 2e-1)));
	LayerPtr<Scalar,Rank> kernel_layer_11 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_11 = kernel_layer_11->get_output_dims();
	layer_grad_test<Scalar,Rank>("elu activation layer", std::move(kernel_layer_11),
			LayerPtr<Scalar,Rank>(new ELUActivationLayer<Scalar,Rank>(dims_11, 2e-1)));
	auto reg = ParamRegSharedPtr<Scalar>(new SquaredParameterRegularization<Scalar>());
	layer_grad_test<Scalar,Rank>("prelu activation layer rank",
			LayerPtr<Scalar,Rank>(new PReLUActivationLayer<Scalar,Rank>(dims, 2e-1, reg)),
			LayerPtr<Scalar,Rank>(new PReLUActivationLayer<Scalar,Rank>(dims)));
	LayerPtr<Scalar,Rank> kernel_layer_12 = kernel_layer<Scalar,Rank>(dims);
	Dimensions<std::size_t,Rank> dims_12 = kernel_layer_12->get_output_dims();
	layer_grad_test<Scalar,Rank>("swish activation layer", std::move(kernel_layer_12),
			LayerPtr<Scalar,Rank>(new SwishActivationLayer<Scalar,Rank>(dims_12, 1.2)));
	layer_grad_test<Scalar,Rank>("pswish activation layer",
			LayerPtr<Scalar,Rank>(new PSwishActivationLayer<Scalar,Rank>(dims)),
			LayerPtr<Scalar,Rank>(new PSwishActivationLayer<Scalar,Rank>(dims, 1.2, reg)));
}

/**
 * Performs gradient checks on all activation layers.
 */
template<typename Scalar>
inline void activation_layer_grad_test() {
	activation_layer_grad_test<Scalar,1>({ 16u });
	activation_layer_grad_test<Scalar,2>({ 4u, 4u });
	activation_layer_grad_test<Scalar,3>({ 3u, 2u, 2u });
}

TEST(GradientTest, ActivationLayer) {
	activation_layer_grad_test<double>();
}

/**
 * Performs gradient checks on all pooling layers.
 */
template<typename Scalar>
inline void pool_layer_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new HeParameterInitialization<Scalar>());
	Dimensions<std::size_t,1> dims_1({ 32u });
	LayerPtr<Scalar,1> mean_layer1_1(new ConvKernelLayer<Scalar,1>(dims_1, 1, init));
	LayerPtr<Scalar,1> mean_layer2_1(new MeanPoolLayer<Scalar,1>(mean_layer1_1->get_output_dims(), 2, 2));
	layer_grad_test<Scalar,1>("mean pool layer", std::move(mean_layer1_1), std::move(mean_layer2_1));
	LayerPtr<Scalar,1> max_layer1_1(new ConvKernelLayer<Scalar,1>(dims_1, 2, init));
	LayerPtr<Scalar,1> max_layer2_1(new MaxPoolLayer<Scalar,1>(max_layer1_1->get_output_dims(), 3, 1));
	layer_grad_test<Scalar,1>("max pool layer", std::move(max_layer1_1), std::move(max_layer2_1));
	Dimensions<std::size_t,2> dims_2({ 10u, 10u });
	LayerPtr<Scalar,2> mean_layer1_2(new ConvKernelLayer<Scalar,2>(dims_2, 2, init));
	LayerPtr<Scalar,2> mean_layer2_2(new MeanPoolLayer<Scalar,2>(mean_layer1_2->get_output_dims(), 3, 2, 1, 2));
	layer_grad_test<Scalar,2>("mean pool layer", std::move(mean_layer1_2), std::move(mean_layer2_2));
	LayerPtr<Scalar,2> max_layer1_2(new ConvKernelLayer<Scalar,2>(dims_2, 2, init));
	LayerPtr<Scalar,2> max_layer2_2(new MaxPoolLayer<Scalar,2>(max_layer1_2->get_output_dims(), 3, 2, 1, 2));
	layer_grad_test<Scalar,2>("max pool layer", std::move(max_layer1_2), std::move(max_layer2_2));
	Dimensions<std::size_t,3> dims_3({ 16u, 16u, 2u });
	LayerPtr<Scalar,3> mean_layer1_3(new ConvKernelLayer<Scalar>(dims_3, 2, init));
	LayerPtr<Scalar,3> mean_layer2_3(new MeanPoolLayer<Scalar>(mean_layer1_3->get_output_dims(), 3, 2, 1, 2));
	layer_grad_test<Scalar,3>("mean pool layer", std::move(mean_layer1_3), std::move(mean_layer2_3));
	LayerPtr<Scalar,3> max_layer1_3(new ConvKernelLayer<Scalar>(dims_3, 1, init));
	LayerPtr<Scalar,3> max_layer2_3(new MaxPoolLayer<Scalar>(max_layer1_3->get_output_dims(), 3, 2, 1, 2));
	layer_grad_test<Scalar,3>("max pool layer", std::move(max_layer1_3), std::move(max_layer2_3));
}

TEST(GradientTest, PoolLayer) {
	pool_layer_grad_test<double>();
}

/**
 * Performs gradient checks on broadcast layers.
 */
template<typename Scalar>
inline void broadcast_layer_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new HeParameterInitialization<Scalar>());
	LayerPtr<Scalar,1> layer1_1(new DenseKernelLayer<Scalar,1>({ 8u }, 8, init));
	LayerPtr<Scalar,1> layer2_1(new BroadcastLayer<Scalar,1>(layer1_1->get_output_dims(), { 3u }));
	layer_grad_test<Scalar,1>("broadcast layer", std::move(layer1_1), std::move(layer2_1));
	LayerPtr<Scalar,2> layer1_2(new DenseKernelLayer<Scalar,2>({ 6u, 6u }, 12, init));
	LayerPtr<Scalar,2> layer2_2(new BroadcastLayer<Scalar,2>(layer1_2->get_output_dims(), { 1u, 3u }));
	layer_grad_test<Scalar,2>("broadcast layer", std::move(layer1_2), std::move(layer2_2));
	LayerPtr<Scalar,3> layer1_3(new ConvKernelLayer<Scalar>({ 16u, 16u, 2u }, 2, init));
	LayerPtr<Scalar,3> layer2_3(new BroadcastLayer<Scalar,3>(layer1_3->get_output_dims(), { 2u, 2u, 2u }));
	layer_grad_test<Scalar,3>("broadcast layer", std::move(layer1_3), std::move(layer2_3));
}

TEST(GradientTest, BroadcastLayer) {
	broadcast_layer_grad_test<double>();
}

/**
 * Performs gradient checks on batch normalization layers.
 */
template<typename Scalar>
inline void batch_norm_layer_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new HeParameterInitialization<Scalar>());
	auto reg1 = ParamRegSharedPtr<Scalar>(new AbsoluteParameterRegularization<Scalar>());
	auto reg2 = ParamRegSharedPtr<Scalar>(new SquaredParameterRegularization<Scalar>());
	LayerPtr<Scalar,1> layer1_1(new DenseKernelLayer<Scalar,1>({ 32u }, 16, init));
	LayerPtr<Scalar,1> layer1_2(new BatchNormLayer<Scalar,1>(layer1_1->get_output_dims(), .2,
			NumericUtils<Scalar>::EPSILON2, reg2, 0, 0, 0, 0, 0, 0, reg1));
	layer_grad_test<Scalar,1>("per-activation batch norm layer", std::move(layer1_1), std::move(layer1_2), 5,
			ScalarTraits<Scalar>::step_size, ScalarTraits<float>::abs_epsilon, ScalarTraits<float>::rel_epsilon);
	LayerPtr<Scalar,2> layer2_1(new ConvKernelLayer<Scalar,2>({ 6u, 6u }, 2, init));
	LayerPtr<Scalar,2> layer2_2(new BatchNormLayer<Scalar,2,true>(layer2_1->get_output_dims()));
	layer_grad_test<Scalar,2>("per-rank batch norm layer", std::move(layer2_1), std::move(layer2_2));
	LayerPtr<Scalar,2> layer2_3(new BatchNormLayer<Scalar,2>({ 6u, 6u }));
	LayerPtr<Scalar,2> layer2_4(new IdentityActivationLayer<Scalar,2>(layer2_3->get_output_dims()));
	layer_grad_test<Scalar,2>("per-activation batch norm layer", std::move(layer2_3), std::move(layer2_4));
	LayerPtr<Scalar,3> layer3_1(new ConvKernelLayer<Scalar>({ 4u, 4u, 2u }, 2, init));
	LayerPtr<Scalar,3> layer3_2(new BatchNormLayer<Scalar,3>(layer3_1->get_output_dims(), .1,
			NumericUtils<Scalar>::EPSILON2, reg2, 0, 0, 0, 0, 0, 0, reg2));
	layer_grad_test<Scalar,3>("per-rank batch norm layer", std::move(layer3_1), std::move(layer3_2));
	LayerPtr<Scalar,3> layer3_3(new ConvKernelLayer<Scalar>({ 4u, 4u, 2u }, 2, init));
	LayerPtr<Scalar,3> layer3_4(new BatchNormLayer<Scalar,3,false>(layer3_3->get_output_dims(), .05,
			NumericUtils<Scalar>::EPSILON2, reg2, 0, 0, 0, 0, 0, 0, reg2));
	layer_grad_test<Scalar,3>("per-activation batch norm layer", std::move(layer3_3), std::move(layer3_4));
}

TEST(GradientTest, BatchNormLayer) {
	batch_norm_layer_grad_test<double>();
}

/**
 * Performs gradient checks on reshape-layers.
 */
template<typename Scalar>
inline void reshape_layer_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new HeParameterInitialization<Scalar>());
	LayerPtr<Scalar,2> layer1_2(new DenseKernelLayer<Scalar,2>({ 6u, 6u }, 12, init));
	LayerPtr<Scalar,2> layer2_2(new ReshapeLayer<Scalar,2>(layer1_2->get_output_dims(), { 4u, 3u }));
	layer_grad_test<Scalar,2>("reshape layer", std::move(layer1_2), std::move(layer2_2));
	LayerPtr<Scalar,3> layer1_3(new ConvKernelLayer<Scalar>({ 16u, 16u, 2u }, 2, init));
	LayerPtr<Scalar,3> layer2_3(new ReshapeLayer<Scalar,3>(layer1_3->get_output_dims(), { 8u, 8u, 8u }));
	layer_grad_test<Scalar,3>("reshape layer", std::move(layer1_3), std::move(layer2_3));
}

TEST(GradientTest, ReshapeLayer) {
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
	auto init = ParamInitSharedPtr<Scalar>(new GlorotParameterInitialization<Scalar>());
	// Rank 1 with summation.
	Dimensions<std::size_t,1> dims_1({ 32u });
	std::vector<NeuralNetPtr<Scalar,1,false>> lanes1_1;
	std::vector<LayerPtr<Scalar,1>> lane1_1_1_layers(1);
	lane1_1_1_layers[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(dims_1, 6, init));
	lanes1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane1_1_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane2_1_1_layers(3);
	lane2_1_1_layers[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(dims_1, 12, init));
	lane2_1_1_layers[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(lane2_1_1_layers[0]->get_output_dims()));
	lane2_1_1_layers[2] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(lane2_1_1_layers[1]->get_output_dims(), 6, init));
	lanes1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane2_1_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane3_1_1_layers(2);
	lane3_1_1_layers[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(dims_1, 4, init));
	lane3_1_1_layers[1] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(lane3_1_1_layers[0]->get_output_dims(), 6, init));
	lanes1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane3_1_1_layers))));
	NeuralNetPtr<Scalar,1,false> parallel_net1_1(new ParallelNeuralNetwork<Scalar,1,PARALLEL_SUM>(std::move(lanes1_1)));
	ASSERT_TRUE((parallel_net1_1->get_output_dims() == Dimensions<std::size_t,1>({ 6 })));
	nonseq_network_grad_test<Scalar,1>("parallel net with summation", std::move(parallel_net1_1));
	// Rank 1 with multiplication.
	std::vector<NeuralNetPtr<Scalar,1,false>> lanes2_1;
	std::vector<LayerPtr<Scalar,1>> lane1_2_1_layers(1);
	lane1_2_1_layers[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(dims_1, 6, init));
	lanes2_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane1_2_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane2_2_1_layers(3);
	lane2_2_1_layers[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(dims_1, 12, init));
	lane2_2_1_layers[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(lane2_2_1_layers[0]->get_output_dims()));
	lane2_2_1_layers[2] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(lane2_2_1_layers[1]->get_output_dims(), 6, init));
	lanes2_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane2_2_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane3_2_1_layers(2);
	lane3_2_1_layers[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(dims_1, 4, init));
	lane3_2_1_layers[1] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(lane3_2_1_layers[0]->get_output_dims(), 6, init));
	lanes2_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane3_2_1_layers))));
	NeuralNetPtr<Scalar,1,false> parallel_net2_1(new ParallelNeuralNetwork<Scalar,1,PARALLEL_MUL>(std::move(lanes2_1)));
	ASSERT_TRUE((parallel_net2_1->get_output_dims() == Dimensions<std::size_t,1>({ 6 })));
	nonseq_network_grad_test<Scalar,1>("parallel net with multiplication", std::move(parallel_net2_1));
	// Rank 2 with lowest rank concatenation.
	Dimensions<std::size_t,2> dims_2({ 6u, 6u });
	std::vector<NeuralNetPtr<Scalar,2,false>> lanes_2;
	std::vector<LayerPtr<Scalar,2>> lane1_2_layers(1);
	lane1_2_layers[0] = LayerPtr<Scalar,2>(new DenseKernelLayer<Scalar,2>(dims_2, 6, init));
	lanes_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(lane1_2_layers))));
	std::vector<LayerPtr<Scalar,2>> lane2_2_layers(1);
	lane2_2_layers[0] = LayerPtr<Scalar,2>(new DenseKernelLayer<Scalar,2>(dims_2, 12, init));
	lanes_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(lane2_2_layers))));
	NeuralNetPtr<Scalar,2,false> parallel_net_2(new ParallelNeuralNetwork<Scalar,2,PARALLEL_CONCAT_LO_RANK>(std::move(lanes_2)));
	ASSERT_TRUE((parallel_net_2->get_output_dims() == Dimensions<std::size_t,2>({ 18, 1 })));
	nonseq_network_grad_test<Scalar,2>("parallel net with lowest rank concatenation", std::move(parallel_net_2));
	// Rank 3 with highest rank concatenation.
	Dimensions<std::size_t,3> dims_3({ 4u, 4u, 3u });
	std::vector<NeuralNetPtr<Scalar,3,false>> lanes_3;
	std::vector<LayerPtr<Scalar,3>> lane1_3_layers(1);
	lane1_3_layers[0] = LayerPtr<Scalar,3>(new ConvKernelLayer<Scalar>(dims_3, 4, init));
	lanes_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(lane1_3_layers))));
	std::vector<LayerPtr<Scalar,3>> lane2_3_layers(1);
	lane2_3_layers[0] = LayerPtr<Scalar,3>(new ConvKernelLayer<Scalar>(dims_3, 2, init));
	lanes_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(lane2_3_layers))));
	NeuralNetPtr<Scalar,3,false> parallel_net_3(new ParallelNeuralNetwork<Scalar,3>(std::move(lanes_3)));
	ASSERT_TRUE((parallel_net_3->get_output_dims() == Dimensions<std::size_t,3>({ 4, 4, 6 })));
	nonseq_network_grad_test<Scalar,3>("parallel net with highest rank concatenation", std::move(parallel_net_3));
}

TEST(GradientTest, ParallelNet) {
	parallel_net_grad_test<double>();
}

/**
 * Performs gradient checks on residual neural networks.
 */
template<typename Scalar>
inline void residual_net_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new GlorotParameterInitialization<Scalar>());
	// Rank 1.
	std::vector<NeuralNetPtr<Scalar,1,false>> modules_1;
	std::vector<NeuralNetPtr<Scalar,1,false>> sub_modules1_1;
	std::vector<LayerPtr<Scalar,1>> layers1_1(2);
	layers1_1[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>({ 32u }, 18, init));
	layers1_1[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(layers1_1[0]->get_output_dims()));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers1_1))));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new DenseKernelLayer<Scalar,1>(sub_modules1_1[0]->get_output_dims(), 32, init)))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new StackedNeuralNetwork<Scalar,1,false>(std::move(sub_modules1_1))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new StackedNeuralNetwork<Scalar,1,false>(NeuralNetPtr<Scalar,1,false>(
			new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new DenseKernelLayer<Scalar,1>(modules_1[0]->get_output_dims(), 32, init)))))));
	nonseq_network_grad_test<Scalar,1>("residual net", NeuralNetPtr<Scalar,1,false>(
			new ResidualNeuralNetwork<Scalar,1>(std::move(modules_1))));
	// Rank 3.
	std::vector<NeuralNetPtr<Scalar,3,false>> modules_3;
	std::vector<NeuralNetPtr<Scalar,3,false>> sub_modules1_3;
	std::vector<LayerPtr<Scalar,3>> layers1_3(2);
	layers1_3[0] = LayerPtr<Scalar,3>(new ConvKernelLayer<Scalar>({ 4u, 4u, 3u }, 4, init));
	layers1_3[1] = LayerPtr<Scalar,3>(new SigmoidActivationLayer<Scalar,3>(layers1_3[0]->get_output_dims()));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers1_3))));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvKernelLayer<Scalar>(sub_modules1_3[0]->get_output_dims(), 3, init)))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new StackedNeuralNetwork<Scalar,3,false>(std::move(sub_modules1_3))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new StackedNeuralNetwork<Scalar,3,false>(NeuralNetPtr<Scalar,3,false>(
			new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvKernelLayer<Scalar>(modules_3[0]->get_output_dims(), 3, init)))))));
	nonseq_network_grad_test<Scalar,3>("residual net", NeuralNetPtr<Scalar,3,false>(
			new ResidualNeuralNetwork<Scalar,3>(std::move(modules_3))));
}

TEST(GradientTest, ResidualNet) {
	residual_net_grad_test<double>();
}

/**
 * Performs gradient checks on dense neural networks.
 */
template<typename Scalar>
inline void dense_net_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new GlorotParameterInitialization<Scalar>());
	// Rank 1.
	Dimensions<std::size_t,1> dims_1({ 32u });
	std::vector<NeuralNetPtr<Scalar,1,false>> modules_1;
	std::vector<NeuralNetPtr<Scalar,1,false>> sub_modules1_1;
	std::vector<LayerPtr<Scalar,1>> layers1_1(2);
	layers1_1[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(dims_1, 18, init));
	layers1_1[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(layers1_1[0]->get_output_dims()));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers1_1))));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new DenseKernelLayer<Scalar,1>(sub_modules1_1[0]->get_output_dims(), 32, init)))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new StackedNeuralNetwork<Scalar,1,false>(std::move(sub_modules1_1))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new StackedNeuralNetwork<Scalar,1,false>(NeuralNetPtr<Scalar,1,false>(
			new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new DenseKernelLayer<Scalar,1>(dims_1.add_along_rank(modules_1[0]->get_output_dims(), 0), 16, init)))))));
	nonseq_network_grad_test<Scalar,1>("dense net", NeuralNetPtr<Scalar,1,false>(
			new DenseNeuralNetwork<Scalar,1>(std::move(modules_1))));
	// Rank 2.
	Dimensions<std::size_t,2> dims_2({ 6u, 6u });
	std::vector<NeuralNetPtr<Scalar,2,false>> modules_2;
	std::vector<NeuralNetPtr<Scalar,2,false>> sub_modules1_2;
	std::vector<LayerPtr<Scalar,2>> layers1_2(2);
	layers1_2[0] = LayerPtr<Scalar,2>(new DenseKernelLayer<Scalar,2>(dims_2, 12, init));
	layers1_2[1] = LayerPtr<Scalar,2>(new SigmoidActivationLayer<Scalar,2>(layers1_2[0]->get_output_dims()));
	sub_modules1_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(layers1_2))));
	sub_modules1_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(LayerPtr<Scalar,2>(
			new DenseKernelLayer<Scalar,2>(sub_modules1_2[0]->get_output_dims(), 6, init)))));
	modules_2.push_back(NeuralNetPtr<Scalar,2,false>(new StackedNeuralNetwork<Scalar,2,false>(std::move(sub_modules1_2))));
	modules_2.push_back(NeuralNetPtr<Scalar,2,false>(new StackedNeuralNetwork<Scalar,2,false>(NeuralNetPtr<Scalar,2,false>(
			new FeedforwardNeuralNetwork<Scalar,2>(LayerPtr<Scalar,2>(
			new DenseKernelLayer<Scalar,2>(dims_2.add_along_rank(modules_2[0]->get_output_dims(), 1), 6, init)))))));
	nonseq_network_grad_test<Scalar,2>("dense net", NeuralNetPtr<Scalar,2,false>(
			new DenseNeuralNetwork<Scalar,2,DENSE_HIGHEST_RANK>(std::move(modules_2))));
	// Rank 3.
	Dimensions<std::size_t,3> dims_3({ 4u, 4u, 3u });
	std::vector<NeuralNetPtr<Scalar,3,false>> modules_3;
	std::vector<NeuralNetPtr<Scalar,3,false>> sub_modules1_3;
	std::vector<LayerPtr<Scalar,3>> layers1_3(2);
	layers1_3[0] = LayerPtr<Scalar,3>(new ConvKernelLayer<Scalar>(dims_3, 4, init, 2, 3));
	layers1_3[1] = LayerPtr<Scalar,3>(new SigmoidActivationLayer<Scalar,3>(layers1_3[0]->get_output_dims()));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers1_3))));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvKernelLayer<Scalar>(sub_modules1_3[0]->get_output_dims(), 3, init)))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new StackedNeuralNetwork<Scalar,3,false>(std::move(sub_modules1_3))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new StackedNeuralNetwork<Scalar,3,false>(NeuralNetPtr<Scalar,3,false>(
			new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvKernelLayer<Scalar>(dims_3.add_along_rank(modules_3[0]->get_output_dims(), 0), 3, init)))))));
	nonseq_network_grad_test<Scalar,3>("dense net", NeuralNetPtr<Scalar,3,false>(
			new DenseNeuralNetwork<Scalar,3,DENSE_LOWEST_RANK>(std::move(modules_3))));
}

TEST(GradientTest, DenseNet) {
	dense_net_grad_test<double>();
}

/**
 * Performs gradient checks on sequential neural networks.
 */
template<typename Scalar>
inline void sequential_net_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new GlorotParameterInitialization<Scalar>());
	// Rank 1.
	std::vector<LayerPtr<Scalar,1>> layers_1(3);
	layers_1[0] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>({ 32u }, 16, init));
	layers_1[1] = LayerPtr<Scalar,1>(new TanhActivationLayer<Scalar,1>(layers_1[0]->get_output_dims()));
	layers_1[2] = LayerPtr<Scalar,1>(new DenseKernelLayer<Scalar,1>(layers_1[1]->get_output_dims(), 4, init));
	seq_network_grad_test("sequential net", NeuralNetPtr<Scalar,1,true>(
			new SequentialNeuralNetwork<Scalar,1>(NeuralNetPtr<Scalar,1,false>(
					new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers_1))))));
	// Rank 3.
	std::vector<LayerPtr<Scalar,3>> layers_3(4);
	layers_3[0] = LayerPtr<Scalar,3>(new ConvKernelLayer<Scalar>({ 4u, 4u, 2u }, 4, init));
	layers_3[1] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers_3[0]->get_output_dims()));
	layers_3[2] = LayerPtr<Scalar,3>(new MaxPoolLayer<Scalar>(layers_3[1]->get_output_dims()));
	layers_3[3] = LayerPtr<Scalar,3>(new DenseKernelLayer<Scalar,3>(layers_3[2]->get_output_dims(), 1, init));
	seq_network_grad_test("sequential net", NeuralNetPtr<Scalar,3,true>(
			new SequentialNeuralNetwork<Scalar,3>(NeuralNetPtr<Scalar,3,false>(
					new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers_3))))));
}

TEST(GradientTest, SequentialNet) {
	sequential_net_grad_test<double>();
}

/**
 * Performs gradient checks on recurrent neural networks.
 */
template<typename Scalar>
inline void recurrent_net_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new OrthogonalParameterInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new SquaredParameterRegularization<Scalar>());
	// Rank 1.
	KernelPtr<Scalar,1> input_kernel1_1(new DenseKernelLayer<Scalar,1>({ 12u }, 12, init, reg));
	KernelPtr<Scalar,1> state_kernel1_1(new DenseKernelLayer<Scalar,1>(input_kernel1_1->get_output_dims(), 12, init, reg));
	KernelPtr<Scalar,1> output_kernel1_1(new DenseKernelLayer<Scalar,1>(state_kernel1_1->get_output_dims(), 4, init, reg));
	ActivationPtr<Scalar,1> state_act1_1(new TanhActivationLayer<Scalar,1>(state_kernel1_1->get_output_dims()));
	ActivationPtr<Scalar,1> output_act1_1(new IdentityActivationLayer<Scalar,1>(output_kernel1_1->get_output_dims()));
	KernelPtr<Scalar,1> input_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(input_kernel1_1->clone()));
	KernelPtr<Scalar,1> state_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(state_kernel1_1->clone()));
	KernelPtr<Scalar,1> output_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(output_kernel1_1->clone()));
	ActivationPtr<Scalar,1> state_act2_1(static_cast<TanhActivationLayer<Scalar,1>*>(state_act1_1->clone()));
	ActivationPtr<Scalar,1> output_act2_1(static_cast<IdentityActivationLayer<Scalar,1>*>(output_act1_1->clone()));
	seq_network_grad_test("recurrent net", NeuralNetPtr<Scalar,1,true>(
			new RecurrentNeuralNetwork<Scalar,1>(std::move(input_kernel1_1), std::move(state_kernel1_1),
					std::move(output_kernel1_1), std::move(state_act1_1), std::move(output_act1_1),
					[](int input_seq_length) { return std::make_pair(input_seq_length, 0); })), 3, 3);
	seq_network_grad_test("recurrent net with multiplicative integration", NeuralNetPtr<Scalar,1,true>(
			new RecurrentNeuralNetwork<Scalar,1,true>(std::move(input_kernel2_1), std::move(state_kernel2_1),
					std::move(output_kernel2_1), std::move(state_act2_1), std::move(output_act2_1),
					[](int input_seq_length) { return std::make_pair(1, input_seq_length - 1); })), 5, 1);
	// Rank 3.
	KernelPtr<Scalar,3> input_kernel1_3(new ConvKernelLayer<Scalar>({ 4u, 4u, 2u }, 5, init, 3, 3, 1, 1, 1, 1,
			0, 0, reg));
	KernelPtr<Scalar,3> state_kernel1_3(new ConvKernelLayer<Scalar>(input_kernel1_3->get_output_dims(), 5, init,
			3, 3, 1, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> output_kernel1_3(new DenseKernelLayer<Scalar,3>(state_kernel1_3->get_output_dims(), 2, init, reg));
	ActivationPtr<Scalar,3> state_act1_3(new SoftsignActivationLayer<Scalar,3>(state_kernel1_3->get_output_dims()));
	ActivationPtr<Scalar,3> output_act1_3(new IdentityActivationLayer<Scalar,3>(output_kernel1_3->get_output_dims()));
	KernelPtr<Scalar,3> input_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(input_kernel1_3->clone()));
	KernelPtr<Scalar,3> state_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(state_kernel1_3->clone()));
	KernelPtr<Scalar,3> output_kernel2_3(static_cast<DenseKernelLayer<Scalar,3>*>(output_kernel1_3->clone()));
	ActivationPtr<Scalar,3> state_act2_3(static_cast<SoftsignActivationLayer<Scalar,3>*>(state_act1_3->clone()));
	ActivationPtr<Scalar,3> output_act2_3(static_cast<IdentityActivationLayer<Scalar,3>*>(output_act1_3->clone()));
	seq_network_grad_test("recurrent net", NeuralNetPtr<Scalar,3,true>(
			new RecurrentNeuralNetwork<Scalar,3>(std::move(input_kernel1_3), std::move(state_kernel1_3),
					std::move(output_kernel1_3), std::move(state_act1_3), std::move(output_act1_3),
					[](std::size_t input_seq_length) { return std::make_pair(3, input_seq_length - 3); })), 5, 3);
	seq_network_grad_test("recurrent net with multiplicative integration", NeuralNetPtr<Scalar,3,true>(
			new RecurrentNeuralNetwork<Scalar,3,true>(std::move(input_kernel2_3), std::move(state_kernel2_3),
					std::move(output_kernel2_3), std::move(state_act2_3), std::move(output_act2_3),
					[](std::size_t input_seq_length) { return std::make_pair(2, input_seq_length); })), 3, 2);
}

TEST(GradientTest, RecurrentNet) {
	recurrent_net_grad_test<double>();
}

/**
 * Performs gradient checks on LSTM neural networks.
 */
template<typename Scalar>
inline void lstm_net_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new OrthogonalParameterInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new SquaredParameterRegularization<Scalar>());
	// Rank 1.
	Dimensions<std::size_t,1> input_dims_1({ 32u });
	Dimensions<std::size_t,1> output_dims_1({ 5u });
	KernelPtr<Scalar,1> forget_input_kernel1_1(new DenseKernelLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> forget_output_kernel1_1(new DenseKernelLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> write_input_kernel1_1(new DenseKernelLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> write_output_kernel1_1(new DenseKernelLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> candidate_input_kernel1_1(new DenseKernelLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> candidate_output_kernel1_1(new DenseKernelLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> read_input_kernel1_1(new DenseKernelLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> read_output_kernel1_1(new DenseKernelLayer<Scalar,1>(output_dims_1, 5, init, reg));
	ActivationPtr<Scalar,1> forget_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> write_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> candidate_act1_1(new TanhActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> state_act1_1(new TanhActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> read_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	KernelPtr<Scalar,1> forget_input_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(forget_input_kernel1_1->clone()));
	KernelPtr<Scalar,1> forget_output_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(forget_output_kernel1_1->clone()));
	KernelPtr<Scalar,1> write_input_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(write_input_kernel1_1->clone()));
	KernelPtr<Scalar,1> write_output_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(write_output_kernel1_1->clone()));
	KernelPtr<Scalar,1> candidate_input_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(candidate_input_kernel1_1->clone()));
	KernelPtr<Scalar,1> candidate_output_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(candidate_output_kernel1_1->clone()));
	KernelPtr<Scalar,1> read_input_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(read_input_kernel1_1->clone()));
	KernelPtr<Scalar,1> read_output_kernel2_1(static_cast<DenseKernelLayer<Scalar,1>*>(read_output_kernel1_1->clone()));
	ActivationPtr<Scalar,1> forget_act2_1(static_cast<SigmoidActivationLayer<Scalar,1>*>(forget_act1_1->clone()));
	ActivationPtr<Scalar,1> write_act2_1(static_cast<SigmoidActivationLayer<Scalar,1>*>(write_act1_1->clone()));
	ActivationPtr<Scalar,1> candidate_act2_1(static_cast<TanhActivationLayer<Scalar,1>*>(candidate_act1_1->clone()));
	ActivationPtr<Scalar,1> state_act2_1(static_cast<TanhActivationLayer<Scalar,1>*>(state_act1_1->clone()));
	ActivationPtr<Scalar,1> read_act2_1(static_cast<SigmoidActivationLayer<Scalar,1>*>(read_act1_1->clone()));
	seq_network_grad_test("lstm net", NeuralNetPtr<Scalar,1,true>(
			new LSTMNeuralNetwork<Scalar,1>(std::move(forget_input_kernel1_1), std::move(forget_output_kernel1_1),
					std::move(write_input_kernel1_1), std::move(write_output_kernel1_1), std::move(candidate_input_kernel1_1),
					std::move(candidate_output_kernel1_1), std::move(read_input_kernel1_1), std::move(read_output_kernel1_1),
					std::move(forget_act1_1), std::move(write_act1_1), std::move(candidate_act1_1), std::move(state_act1_1),
					std::move(read_act1_1), [](std::size_t input_seq_length) { return std::make_pair(input_seq_length, 0); })), 3, 3);
	seq_network_grad_test("lstm net with multiplicative integration", NeuralNetPtr<Scalar,1,true>(
			new LSTMNeuralNetwork<Scalar,1,true>(std::move(forget_input_kernel2_1), std::move(forget_output_kernel2_1),
					std::move(write_input_kernel2_1), std::move(write_output_kernel2_1), std::move(candidate_input_kernel2_1),
					std::move(candidate_output_kernel2_1), std::move(read_input_kernel2_1), std::move(read_output_kernel2_1),
					std::move(forget_act2_1), std::move(write_act2_1), std::move(candidate_act2_1), std::move(state_act2_1),
					std::move(read_act2_1), [](std::size_t input_seq_length) { return std::make_pair(1, input_seq_length - 1); })), 5, 1);
	// Rank 3.
	Dimensions<std::size_t,3> input_dims_3({ 5u, 3u, 3u });
	Dimensions<std::size_t,3> output_dims_3({ 3u, 3u, 3u });
	KernelPtr<Scalar,3> forget_input_kernel1_3(new ConvKernelLayer<Scalar>(input_dims_3, 3, init, 3, 3, 0, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> forget_output_kernel1_3(new ConvKernelLayer<Scalar>(output_dims_3, 3, init, 3, 3, 1, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> write_input_kernel1_3(new ConvKernelLayer<Scalar>(input_dims_3, 3, init, 3, 3, 0, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> write_output_kernel1_3(new ConvKernelLayer<Scalar>(output_dims_3, 3, init, 3, 3, 1, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> candidate_input_kernel1_3(new ConvKernelLayer<Scalar>(input_dims_3, 3, init, 3, 3, 0, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> candidate_output_kernel1_3(new ConvKernelLayer<Scalar>(output_dims_3, 3, init, 3, 3, 1, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> read_input_kernel1_3(new ConvKernelLayer<Scalar>(input_dims_3, 3, init, 3, 3, 0, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> read_output_kernel1_3(new ConvKernelLayer<Scalar>(output_dims_3, 3, init, 3, 3, 1, 1, 1, 1, 0, 0, reg));
	ActivationPtr<Scalar,3> forget_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> write_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> candidate_act1_3(new TanhActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> state_act1_3(new TanhActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> read_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	KernelPtr<Scalar,3> forget_input_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(forget_input_kernel1_3->clone()));
	KernelPtr<Scalar,3> forget_output_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(forget_output_kernel1_3->clone()));
	KernelPtr<Scalar,3> write_input_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(write_input_kernel1_3->clone()));
	KernelPtr<Scalar,3> write_output_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(write_output_kernel1_3->clone()));
	KernelPtr<Scalar,3> candidate_input_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(candidate_input_kernel1_3->clone()));
	KernelPtr<Scalar,3> candidate_output_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(candidate_output_kernel1_3->clone()));
	KernelPtr<Scalar,3> read_input_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(read_input_kernel1_3->clone()));
	KernelPtr<Scalar,3> read_output_kernel2_3(static_cast<ConvKernelLayer<Scalar>*>(read_output_kernel1_3->clone()));
	ActivationPtr<Scalar,3> forget_act2_3(static_cast<SigmoidActivationLayer<Scalar,3>*>(forget_act1_3->clone()));
	ActivationPtr<Scalar,3> write_act2_3(static_cast<SigmoidActivationLayer<Scalar,3>*>(write_act1_3->clone()));
	ActivationPtr<Scalar,3> candidate_act2_3(static_cast<TanhActivationLayer<Scalar,3>*>(candidate_act1_3->clone()));
	ActivationPtr<Scalar,3> state_act2_3(static_cast<TanhActivationLayer<Scalar,3>*>(state_act1_3->clone()));
	ActivationPtr<Scalar,3> read_act2_3(static_cast<SigmoidActivationLayer<Scalar,3>*>(read_act1_3->clone()));
	seq_network_grad_test("lstm net", NeuralNetPtr<Scalar,3,true>(
			new LSTMNeuralNetwork<Scalar,3>(std::move(forget_input_kernel1_3), std::move(forget_output_kernel1_3),
					std::move(write_input_kernel1_3), std::move(write_output_kernel1_3), std::move(candidate_input_kernel1_3),
					std::move(candidate_output_kernel1_3), std::move(read_input_kernel1_3), std::move(read_output_kernel1_3),
					std::move(forget_act1_3), std::move(write_act1_3), std::move(candidate_act1_3), std::move(state_act1_3),
					std::move(read_act1_3), [](std::size_t input_seq_length) { return std::make_pair(3, input_seq_length - 3); })), 5, 3);
	seq_network_grad_test("lstm net with multiplicative integration", NeuralNetPtr<Scalar,3,true>(
			new LSTMNeuralNetwork<Scalar,3,true>(std::move(forget_input_kernel2_3), std::move(forget_output_kernel2_3),
					std::move(write_input_kernel2_3), std::move(write_output_kernel2_3), std::move(candidate_input_kernel2_3),
					std::move(candidate_output_kernel2_3), std::move(read_input_kernel2_3), std::move(read_output_kernel2_3),
					std::move(forget_act2_3), std::move(write_act2_3), std::move(candidate_act2_3), std::move(state_act2_3),
					std::move(read_act2_3), [](std::size_t input_seq_length) { return std::make_pair(2, input_seq_length); })), 3, 2);
}

TEST(GradientTest, LSTMNet) {
	lstm_net_grad_test<double>();
}

/**
 * Performs gradient checks on bidirectional recurrent and LSTM neural networks.
 */
template<typename Scalar>
inline void bidirectional_net_grad_test() {
	auto init = ParamInitSharedPtr<Scalar>(new OrthogonalParameterInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new SquaredParameterRegularization<Scalar>());
	// 3rd degree RNN with highest rank concatenation.
	KernelPtr<Scalar,3> input_kernel1(new ConvKernelLayer<Scalar>({ 4u, 4u, 2u }, 5, init, 3, 3, 1, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> state_kernel1(new ConvKernelLayer<Scalar>(input_kernel1->get_output_dims(), 5, init, 3, 3, 1, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> output_kernel1(new DenseKernelLayer<Scalar,3>(input_kernel1->get_output_dims(), 2, init, reg));
	ActivationPtr<Scalar,3> state_act1(new SigmoidActivationLayer<Scalar,3>(input_kernel1->get_output_dims()));
	ActivationPtr<Scalar,3> output_act1(new IdentityActivationLayer<Scalar,3>(output_kernel1->get_output_dims()));
	seq_network_grad_test("bidirectional recurrent net with highest rank concatenation", NeuralNetPtr<Scalar,3,true>(
			new BidirectionalNeuralNetwork<Scalar,3,BIDIRECTIONAL_CONCAT_HI_RANK>(UnidirNeuralNetPtr<Scalar,3>(
					new RecurrentNeuralNetwork<Scalar,3,true>(std::move(input_kernel1), std::move(state_kernel1),
							std::move(output_kernel1), std::move(state_act1), std::move(output_act1),
							[](std::size_t input_seq_length) { return std::make_pair(3, 2); })))), 7, 3);
	// 2nd degree LSTM with lowest rank concatenation.
	Dimensions<std::size_t,2> input_dims({ 6u, 6u });
	Dimensions<std::size_t,2> output_dims({ 6u, 1u });
	KernelPtr<Scalar,2> forget_input_kernel2(new DenseKernelLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> forget_output_kernel2(new DenseKernelLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> write_input_kernel2(new DenseKernelLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> write_output_kernel2(new DenseKernelLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> candidate_input_kernel2(new DenseKernelLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> candidate_output_kernel2(new DenseKernelLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> read_input_kernel2(new DenseKernelLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> read_output_kernel2(new DenseKernelLayer<Scalar,2>(output_dims, 6, init, reg));
	ActivationPtr<Scalar,2> forget_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> write_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> candidate_act2(new SoftplusActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> state_act2(new SoftplusActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> read_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	seq_network_grad_test("bidirectional lstm net with lowest rank concatenation", NeuralNetPtr<Scalar,2,true>(
			new BidirectionalNeuralNetwork<Scalar,2,BIDIRECTIONAL_CONCAT_LO_RANK>(UnidirNeuralNetPtr<Scalar,2>(
					new LSTMNeuralNetwork<Scalar,2,true>(std::move(forget_input_kernel2),
							std::move(forget_output_kernel2), std::move(write_input_kernel2),
							std::move(write_output_kernel2), std::move(candidate_input_kernel2),
							std::move(candidate_output_kernel2), std::move(read_input_kernel2),
							std::move(read_output_kernel2), std::move(forget_act2), std::move(write_act2),
							std::move(candidate_act2), std::move(state_act2), std::move(read_act2),
							[](std::size_t input_seq_length) { return std::make_pair(1, 2); })))), 5, 1);
	// 1st degree RNN with summation.
	KernelPtr<Scalar,1> input_kernel3(new DenseKernelLayer<Scalar,1>({ 24u }, 8, init, reg));
	KernelPtr<Scalar,1> state_kernel3(new DenseKernelLayer<Scalar,1>(input_kernel3->get_output_dims(), 8, init, reg));
	KernelPtr<Scalar,1> output_kernel3(new DenseKernelLayer<Scalar,1>(input_kernel3->get_output_dims(), 1, init, reg));
	ActivationPtr<Scalar,1> state_act3(new SigmoidActivationLayer<Scalar,1>(input_kernel3->get_output_dims()));
	ActivationPtr<Scalar,1> output_act3(new SigmoidActivationLayer<Scalar,1>(output_kernel3->get_output_dims()));
	seq_network_grad_test("bidirectional recurrent net with summation", NeuralNetPtr<Scalar,1,true>(
			new BidirectionalNeuralNetwork<Scalar,1,BIDIRECTIONAL_SUM>(UnidirNeuralNetPtr<Scalar,1>(
					new RecurrentNeuralNetwork<Scalar,1>(std::move(input_kernel3), std::move(state_kernel3),
							std::move(output_kernel3), std::move(state_act3), std::move(output_act3),
							[](std::size_t input_seq_length) { return std::make_pair(5, 2); })))), 7, 5);
	// 3rd degree RNN with multiplication.
	KernelPtr<Scalar,3> input_kernel4(new ConvKernelLayer<Scalar>({ 4u, 4u, 2u }, 5, init, 3, 3, 1, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> state_kernel4(new ConvKernelLayer<Scalar>(input_kernel4->get_output_dims(), 5, init, 3, 3, 1, 1, 1, 1, 0, 0, reg));
	KernelPtr<Scalar,3> output_kernel4(new DenseKernelLayer<Scalar,3>(input_kernel4->get_output_dims(), 2, init, reg));
	ActivationPtr<Scalar,3> state_act4(new SigmoidActivationLayer<Scalar,3>(input_kernel4->get_output_dims()));
	ActivationPtr<Scalar,3> output_act4(new IdentityActivationLayer<Scalar,3>(output_kernel4->get_output_dims()));
	seq_network_grad_test("bidirectional recurrent net with multiplication", NeuralNetPtr<Scalar,3,true>(
			new BidirectionalNeuralNetwork<Scalar,3,BIDIRECTIONAL_MUL>(UnidirNeuralNetPtr<Scalar,3>(
					new RecurrentNeuralNetwork<Scalar,3,true>(std::move(input_kernel4), std::move(state_kernel4),
							std::move(output_kernel4), std::move(state_act4), std::move(output_act4),
							[](std::size_t input_seq_length) { return std::make_pair(3, 2); })))), 7, 3);
}

TEST(GradientTest, BidirectionalNet) {
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
inline void negated_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	auto net = neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_tensor<Scalar,Rank + 1>(batch_out_dims));
	auto loss = std::make_shared<SquaredLoss<Scalar,Rank,false>>();
	NegatedLoss<Scalar,Rank,false> neg_loss(loss);
	grad_test<Scalar,Rank,false>("negated quadratic loss", prov, *net, neg_loss);
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
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,Rank,true>>();
	NegatedLoss<Scalar,Rank,true> seq_neg_loss(seq_loss);
	grad_test<Scalar,Rank,true>("negated quadratic loss", seq_prov, seq_net, seq_neg_loss);
}

/**
 * Performs gradient checks on the negated squared loss function.
 */
template<typename Scalar>
inline void negated_loss_grad_test() {
	negated_loss_grad_test<Scalar,1>({ 24u });
	negated_loss_grad_test<Scalar,2>({ 6u, 6u });
	negated_loss_grad_test<Scalar,3>({ 4u, 4u, 2u });
}

TEST(GradientTest, NegatedLoss) {
	negated_loss_grad_test<double>();
}

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank>
inline void absolute_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	auto net = neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_tensor<Scalar,Rank + 1>(batch_out_dims));
	AbsoluteLoss<Scalar,Rank,false> loss;
	grad_test<Scalar,Rank,false>("absolute loss", prov, *net, loss);
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
	AbsoluteLoss<Scalar,Rank,true> seq_loss;
	grad_test<Scalar,Rank,true>("absolute loss", seq_prov, seq_net, seq_loss);
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
	HingeLoss<Scalar,Rank,false> loss;
	std::string name = std::string(Squared ? "squared " : "") + std::string("hinge loss");
	grad_test<Scalar,Rank,false>(name, prov, *net, loss);
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
	HingeLoss<Scalar,Rank,true> seq_loss;
	grad_test<Scalar,Rank,true>(name, seq_prov, seq_net, seq_loss);
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
	hinge_loss_grad_test<double>();
}

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank>
inline void binary_cross_entropy_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
		std::size_t time_steps = 3) {
	// Non-sequential.
	std::vector<LayerPtr<Scalar,Rank>> layers(2);
	layers[0] = LayerPtr<Scalar,Rank>(new DenseKernelLayer<Scalar,Rank>(dims, 1,
			ParamInitSharedPtr<Scalar>(new GlorotParameterInitialization<Scalar>())));
	layers[1] = LayerPtr<Scalar,Rank>(new SigmoidActivationLayer<Scalar,Rank>(layers[0]->get_output_dims()));
	auto net = NeuralNetPtr<Scalar,Rank,false>(new FeedforwardNeuralNetwork<Scalar,Rank>(std::move(layers)));
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	auto obj_tensor = random_tensor<Scalar,Rank + 1>(batch_out_dims);
	*obj_tensor = obj_tensor->unaryExpr([](Scalar i) { return (Scalar) (i >= 0 ? 1 : 0); });
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			std::move(obj_tensor));
	BinaryCrossEntropyLoss<Scalar,Rank,false> loss;
	grad_test<Scalar,Rank,false>("binary cross entropy loss", prov, *net, loss);
	// Sequential.
	SequentialNeuralNetwork<Scalar,Rank> seq_net(std::move(net));
	Dimensions<std::size_t,Rank + 2> seq_batch_dims = dims.template promote<2>();
	seq_batch_dims(0) = samples;
	seq_batch_dims(1) = time_steps;
	Dimensions<std::size_t,Rank + 2> seq_batch_out_dims = seq_net.get_output_dims().template promote<2>();
	seq_batch_out_dims(0) = samples;
	seq_batch_out_dims(1) = time_steps;
	auto seq_obj_tensor = random_tensor<Scalar,Rank + 2>(seq_batch_out_dims);
	*seq_obj_tensor = seq_obj_tensor->unaryExpr([](Scalar i) { return (Scalar) (i >= 0 ? 1 : 0); });
	MemoryDataProvider<Scalar,Rank,true> seq_prov(random_tensor<Scalar,Rank + 2>(seq_batch_dims),
			std::move(seq_obj_tensor));
	BinaryCrossEntropyLoss<Scalar,Rank,true> seq_loss;
	grad_test<Scalar,Rank,true>("binary cross entropy loss", seq_prov, seq_net, seq_loss);
}

/**
 * Performs gradient checks on the binary cross entropy loss function.
 */
template<typename Scalar>
inline void binary_cross_entropy_loss_grad_test() {
	binary_cross_entropy_loss_grad_test<Scalar,1>({ 24u });
	binary_cross_entropy_loss_grad_test<Scalar,2>({ 6u, 6u });
	binary_cross_entropy_loss_grad_test<Scalar,3>({ 4u, 4u, 2u });
}

TEST(GradientTest, BinaryCrossEntropyLoss) {
	binary_cross_entropy_loss_grad_test<double>();
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
	CrossEntropyLoss<Scalar,Rank,false> loss;
	grad_test<Scalar,Rank,false>("cross entropy loss", prov, *net, loss);
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
	CrossEntropyLoss<Scalar,Rank,true> seq_loss;
	grad_test<Scalar,Rank,true>("cross entropy loss", seq_prov, seq_net, seq_loss);
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
	auto net = neural_net<Scalar,Rank>(dims);
	net->init();
	Dimensions<std::size_t,Rank + 1> batch_dims = dims.template promote<>();
	batch_dims(0) = samples;
	Dimensions<std::size_t,Rank + 1> batch_out_dims = net->get_output_dims().template promote<>();
	batch_out_dims(0) = samples;
	MemoryDataProvider<Scalar,Rank,false> prov(random_tensor<Scalar,Rank + 1>(batch_dims),
			random_one_hot_tensor<Scalar,Rank + 1,false>(batch_out_dims));
	SoftmaxCrossEntropyLoss<Scalar,Rank,false> loss;
	grad_test<Scalar,Rank,false>("softmax cross entropy loss", prov, *net, loss);
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
	SoftmaxCrossEntropyLoss<Scalar,Rank,true> seq_loss;
	grad_test<Scalar,Rank,true>("softmax cross entropy loss", seq_prov, seq_net, seq_loss);
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
	softmax_cross_entropy_loss_grad_test<double>();
}

/**
 * @param dims The dimensions of the input data.
 * @param samples The number of samples in the batch.
 * @param time_steps The number of time steps for sequential loss gradient testing.
 */
template<typename Scalar, std::size_t Rank>
inline void kullback_leibler_loss_grad_test(const Dimensions<std::size_t,Rank>& dims, std::size_t samples = 5,
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
	KullbackLeiblerLoss<Scalar,Rank,false> loss;
	grad_test<Scalar,Rank,false>("kullback-leibler loss", prov, *net, loss);
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
	KullbackLeiblerLoss<Scalar,Rank,true> seq_loss;
	grad_test<Scalar,Rank,true>("kullback-leibler loss", seq_prov, seq_net, seq_loss);
}

/**
 * Performs gradient checks on the Kullback-Leibler divergence loss function.
 */
template<typename Scalar>
inline void kullback_leibler_loss_grad_test() {
	kullback_leibler_loss_grad_test<Scalar,1>({ 24u });
	kullback_leibler_loss_grad_test<Scalar,2>({ 6u, 6u });
	kullback_leibler_loss_grad_test<Scalar,3>({ 4u, 4u, 2u });
}

TEST(GradientTest, KullbackLieblerLoss) {
	kullback_leibler_loss_grad_test<double>();
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
	MultiLabelHingeLoss<Scalar,Rank,false> loss;
	std::string name = std::string(Squared ? "squared " : "") + std::string("multi-label hinge loss");
	grad_test<Scalar,Rank,false>(name, prov, *net, loss);
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
	MultiLabelHingeLoss<Scalar,Rank,true> seq_loss;
	grad_test<Scalar,Rank,true>(name, seq_prov, seq_net, seq_loss);
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
	MultiLabelLogLoss<Scalar,Rank,false> loss;
	grad_test<Scalar,Rank,false>("multi-label log loss", prov, *net, loss);
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
	MultiLabelLogLoss<Scalar,Rank,true> seq_loss;
	grad_test<Scalar,Rank,true>("multi-label log loss", seq_prov, seq_net, seq_loss);
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
	multi_label_log_loss_grad_test<double>();
}

} /* namespace test */
} /* namespace cattle */
