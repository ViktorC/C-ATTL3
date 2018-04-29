/*
 * gradient_test.hpp
 *
 *  Created on: 19.04.2018
 *      Author: Viktor Csomor
 */

#ifndef GRADIENT_TEST_HPP_
#define GRADIENT_TEST_HPP_

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cattle/Cattle.hpp"

namespace cattle {

/**
 * A namespace for C-ATTL3's test functions.
 */
namespace test {

/**
 * A trait struct for the name of a scalar type and the default numeric constants used for gradient
 * verification depending on the scalar type.
 */
template<typename Scalar>
struct ScalarTraits {
	static constexpr Scalar step_size = 1e-5;
	static constexpr Scalar abs_epsilon = 1e-2;
	static constexpr Scalar rel_epsilon = 1e-2;
	inline static std::string name() {
		return "double";
	}
};

/**
 * Template specialization for single precision floating point scalars.
 */
template<>
struct ScalarTraits<float> {
	static constexpr float step_size = 5e-4;
	static constexpr float abs_epsilon = 1.5e-1;
	static constexpr float rel_epsilon = 1e-1;
	inline static std::string name() {
		return "float";
	}
};

/**
 * @param dims The dimensions of the random tensor to create.
 * @return A tensor of the specified dimensions filled with random values in the range of
 * -1 to 1.
 */
template<typename Scalar, std::size_t Rank>
inline TensorPtr<Scalar,Rank> random_tensor(const std::array<std::size_t,Rank>& dims) {
	TensorPtr<Scalar,Rank> tensor_ptr(new Tensor<Scalar,Rank>(dims));
	tensor_ptr->setRandom();
	return tensor_ptr;
}

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
		Scalar step_size, Scalar abs_epsilon, Scalar rel_epsilon) {
	std::transform(name.begin(), name.end(), name.begin(), ::toupper);
	std::string header = "*   GRADIENT CHECK: " + name + "; SCALAR TYPE: " +
			ScalarTraits<Scalar>::name() + "; RANK: " + std::to_string(Rank) +
			"; SEQ: " + std::to_string(Sequential) + "   *";
	std::size_t header_length = header.length();
	std::string header_border = std::string(header_length, '*');
	std::string header_padding = "*" + std::string(header_length - 2, ' ') + "*";
	std::cout << std::endl << header_border << std::endl << header_padding << std::endl <<
			header << std::endl << header_padding << std::endl << header_border << std::endl;
	EXPECT_TRUE(opt.verify_gradients(net, prov, step_size, abs_epsilon, rel_epsilon));
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
inline void fc_layer_grad_test() {
	auto init1 = WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>());
	auto init2 = WeightInitSharedPtr<Scalar>(new LeCunWeightInitialization<Scalar>());
	auto reg1 = ParamRegSharedPtr<Scalar>(new L1ParameterRegularization<Scalar>());
	auto reg2 = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	auto reg3 = ParamRegSharedPtr<Scalar>(new ElasticNetParameterRegularization<Scalar>());
	LayerPtr<Scalar,1> layer1_1(new FCLayer<Scalar,1>(Dimensions<std::size_t,1>({ 32u }), 16, init1));
	LayerPtr<Scalar,1> layer1_2(new FCLayer<Scalar,1>(layer1_1->get_output_dims(), 1, init1, reg3));
	layer_grad_test<Scalar,1>("fc layer", std::move(layer1_1), std::move(layer1_2), 5, ScalarTraits<Scalar>::step_size,
			ScalarTraits<float>::abs_epsilon, ScalarTraits<float>::rel_epsilon);
	LayerPtr<Scalar,2> layer2_1(new FCLayer<Scalar,2>(Dimensions<std::size_t,2>({ 6u, 6u }), 16, init2));
	LayerPtr<Scalar,2> layer2_2(new FCLayer<Scalar,2>(layer2_1->get_output_dims(), 2, init2, reg1));
	layer_grad_test<Scalar,2>("fc layer", std::move(layer2_1), std::move(layer2_2), 5, ScalarTraits<Scalar>::step_size,
			ScalarTraits<float>::abs_epsilon, ScalarTraits<float>::rel_epsilon);
	LayerPtr<Scalar,3> layer3_1(new FCLayer<Scalar,3>(Dimensions<std::size_t,3>({ 4u, 4u, 2u }), 16, init1, reg2));
	LayerPtr<Scalar,3> layer3_2(new FCLayer<Scalar,3>(layer3_1->get_output_dims(), 4, init1, reg2));
	layer_grad_test<Scalar,3>("fc layer", std::move(layer3_1), std::move(layer3_2));
}

TEST(GradientTest, FCLayer) {
	fc_layer_grad_test<float>();
	fc_layer_grad_test<double>();
}

/**
 * Performs gradient checks on convolutional layers.
 */
template<typename Scalar>
inline void conv_layer_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	LayerPtr<Scalar,3> layer1(new ConvLayer<Scalar>(Dimensions<std::size_t,3>({ 8u, 8u, 2u }), 5, init,
			ConvLayer<Scalar>::NO_PARAM_REG, 3, 2, 1, 2, 1, 2, 1, 0));
	LayerPtr<Scalar,3> layer2(new ConvLayer<Scalar>(layer1->get_output_dims(), 1, init, reg, 1, 1));
	layer_grad_test<Scalar,3>("convolutional layer ", std::move(layer1), std::move(layer2));
}

TEST(GradientTest, ConvLayer) {
	conv_layer_grad_test<float>();
	conv_layer_grad_test<double>();
}

/**
 * @param dims The dimensions of the layers' input tensors.
 */
template<typename Scalar, std::size_t Rank>
inline void activation_layer_grad_test(const Dimensions<std::size_t,Rank>& dims) {
	auto init = WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>());
	LayerPtr<Scalar,Rank> layer1_1(new FCLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_1 = layer1_1->get_output_dims();
	layer_grad_test<Scalar,Rank>("identity activation layer", std::move(layer1_1),
			LayerPtr<Scalar,Rank>(new IdentityActivationLayer<Scalar,Rank>(dims_1)));
	LayerPtr<Scalar,Rank> layer1_2(new FCLayer<Scalar,Rank>(dims, 8, init));
	Dimensions<std::size_t,Rank> dims_2 = layer1_2->get_output_dims();
	layer_grad_test<Scalar,Rank>("scaling activation layer", std::move(layer1_2),
			LayerPtr<Scalar,Rank>(new ScalingActivationLayer<Scalar,Rank>(dims_2, 1.5)));
	LayerPtr<Scalar,Rank> layer1_3(new FCLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_3 = layer1_3->get_output_dims();
	layer_grad_test<Scalar,Rank>("binary step activation layer", std::move(layer1_3),
			LayerPtr<Scalar,Rank>(new BinaryStepActivationLayer<Scalar,Rank>(dims_3)));
	LayerPtr<Scalar,Rank> layer1_4(new FCLayer<Scalar,Rank>(dims, 16, init));
	Dimensions<std::size_t,Rank> dims_4 = layer1_4->get_output_dims();
	layer_grad_test<Scalar,Rank>("sigmoid activation layer rank", std::move(layer1_4),
			LayerPtr<Scalar,Rank>(new SigmoidActivationLayer<Scalar,Rank>(dims_4)));
	LayerPtr<Scalar,Rank> layer1_5(new FCLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_5 = layer1_5->get_output_dims();
	layer_grad_test<Scalar,Rank>("tanh activation layer rank", std::move(layer1_5),
			LayerPtr<Scalar,Rank>(new TanhActivationLayer<Scalar,Rank>(dims_5)));
	LayerPtr<Scalar,Rank> layer1_6(new FCLayer<Scalar,Rank>(dims, 9, init));
	Dimensions<std::size_t,Rank> dims_6 = layer1_6->get_output_dims();
	layer_grad_test<Scalar,Rank>("softplus activation layer rank", std::move(layer1_6),
			LayerPtr<Scalar,Rank>(new SoftplusActivationLayer<Scalar,Rank>(dims_6)));
	LayerPtr<Scalar,Rank> layer1_7(new FCLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_7 = layer1_7->get_output_dims();
	layer_grad_test<Scalar,Rank>("softmax activation layer rank", std::move(layer1_7),
			LayerPtr<Scalar,Rank>(new SoftmaxActivationLayer<Scalar,Rank>(dims_7)));
	LayerPtr<Scalar,Rank> layer1_8(new FCLayer<Scalar,Rank>(dims, 10, init));
	Dimensions<std::size_t,Rank> dims_8 = layer1_8->get_output_dims();
	layer_grad_test<Scalar,Rank>("relu activation layer rank", std::move(layer1_8),
			LayerPtr<Scalar,Rank>(new ReLUActivationLayer<Scalar,Rank>(dims_8)));
	LayerPtr<Scalar,Rank> layer1_9(new FCLayer<Scalar,Rank>(dims, 12, init));
	Dimensions<std::size_t,Rank> dims_9 = layer1_9->get_output_dims();
	layer_grad_test<Scalar,Rank>("leaky relu activation layer rank", std::move(layer1_9),
			LayerPtr<Scalar,Rank>(new LeakyReLUActivationLayer<Scalar,Rank>(dims_9, 2e-1)));
	LayerPtr<Scalar,Rank> layer1_10(new FCLayer<Scalar,Rank>(dims, 11, init));
	Dimensions<std::size_t,Rank> dims_10 = layer1_10->get_output_dims();
	layer_grad_test<Scalar,Rank>("elu activation layer rank", std::move(layer1_10),
			LayerPtr<Scalar,Rank>(new ELUActivationLayer<Scalar,Rank>(dims_10, 2e-1)));
	auto reg = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	layer_grad_test<Scalar,Rank>("prelu activation layer rank",
			LayerPtr<Scalar,Rank>(new PReLUActivationLayer<Scalar,Rank>(dims, reg, 2e-1)),
			LayerPtr<Scalar,Rank>(new PReLUActivationLayer<Scalar,Rank>(dims)));
}

/**
 * Template specialization.
 *
 * @param dims The dimensions of the layers' input tensors.
 */
template<typename Scalar>
inline void activation_layer_grad_test(const Dimensions<std::size_t,3>& dims) {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	LayerPtr<Scalar,3> layer1_1(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_1 = layer1_1->get_output_dims();
	layer_grad_test<Scalar,3>("identity activation layer", std::move(layer1_1),
			LayerPtr<Scalar,3>(new IdentityActivationLayer<Scalar,3>(dims_1)));
	LayerPtr<Scalar,3> layer1_2(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_2 = layer1_2->get_output_dims();
	layer_grad_test<Scalar,3>("scaling activation layer", std::move(layer1_2),
			LayerPtr<Scalar,3>(new ScalingActivationLayer<Scalar,3>(dims_2, 1.5)));
	LayerPtr<Scalar,3> layer1_3(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_3 = layer1_3->get_output_dims();
	layer_grad_test<Scalar,3>("binary step activation layer", std::move(layer1_3),
			LayerPtr<Scalar,3>(new BinaryStepActivationLayer<Scalar,3>(dims_3)));
	LayerPtr<Scalar,3> layer1_4(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_4 = layer1_4->get_output_dims();
	layer_grad_test<Scalar,3>("sigmoid activation layer", std::move(layer1_4),
			LayerPtr<Scalar,3>(new SigmoidActivationLayer<Scalar,3>(dims_4)));
	LayerPtr<Scalar,3> layer1_5(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_5 = layer1_5->get_output_dims();
	layer_grad_test<Scalar,3>("tanh activation layer", std::move(layer1_5),
			LayerPtr<Scalar,3>(new TanhActivationLayer<Scalar,3>(dims_5)));
	LayerPtr<Scalar,3> layer1_6(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_6 = layer1_6->get_output_dims();
	layer_grad_test<Scalar,3>("softplus activation layer", std::move(layer1_6),
			LayerPtr<Scalar,3>(new SoftplusActivationLayer<Scalar,3>(dims_6)));
	LayerPtr<Scalar,3> layer1_7(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_7 = layer1_7->get_output_dims();
	layer_grad_test<Scalar,3>("softmax activation layer", std::move(layer1_7),
			LayerPtr<Scalar,3>(new SoftmaxActivationLayer<Scalar,3>(dims_7)));
	LayerPtr<Scalar,3> layer1_8(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_8 = layer1_8->get_output_dims();
	layer_grad_test<Scalar,3>("relu activation layer", std::move(layer1_8),
			LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(dims_8)));
	LayerPtr<Scalar,3> layer1_9(new ConvLayer<Scalar>(dims, 2, init));
	Dimensions<std::size_t,3> dims_9 = layer1_9->get_output_dims();
	layer_grad_test<Scalar,3>("leaky relu activation layer", std::move(layer1_9),
			LayerPtr<Scalar,3>(new LeakyReLUActivationLayer<Scalar,3>(dims_9, 2e-1)));
	LayerPtr<Scalar,3> layer1_10(new ConvLayer<Scalar>(dims, 2, init));
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
	activation_layer_grad_test<Scalar,1>(Dimensions<std::size_t,1>({ 24u }));
	activation_layer_grad_test<Scalar,2>(Dimensions<std::size_t,2>({ 5u, 5u }));
	activation_layer_grad_test<Scalar,3>(Dimensions<std::size_t,3>({ 4u, 3u, 2u }));
}

TEST(GradientTest, ActivationLayer) {
	activation_layer_grad_test<float>();
	activation_layer_grad_test<double>();
}

/**
 * Performs gradient checks on all pooling layers.
 */
template<typename Scalar>
inline void pooling_layer_grad_test() {
	Dimensions<std::size_t,3> dims({ 16u, 16u, 2u });
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	LayerPtr<Scalar,3> sum_layer1(new ConvLayer<Scalar>(dims, 2, init));
	LayerPtr<Scalar,3> sum_layer2(new SumPoolingLayer<Scalar>(sum_layer1->get_output_dims(), 3, 1, 1, 2, 0, 1));
	layer_grad_test<Scalar,3>("sum pooling layer", std::move(sum_layer1), std::move(sum_layer2));
	LayerPtr<Scalar,3> mean_layer1(new ConvLayer<Scalar>(dims, 2, init));
	LayerPtr<Scalar,3> mean_layer2(new MeanPoolingLayer<Scalar>(mean_layer1->get_output_dims(), 3, 1, 1, 2, 0, 1));
	layer_grad_test<Scalar,3>("mean pooling layer", std::move(mean_layer1), std::move(mean_layer2));
	LayerPtr<Scalar,3> max_layer1(new ConvLayer<Scalar>(dims, 2, init));
	LayerPtr<Scalar,3> max_layer2(new MaxPoolingLayer<Scalar>(max_layer1->get_output_dims(), 3, 1, 1, 2, 0, 1));
	layer_grad_test<Scalar,3>("max pooling layer", std::move(max_layer1), std::move(max_layer2));
}

TEST(GradientTest, PoolingLayer) {
	pooling_layer_grad_test<float>();
	pooling_layer_grad_test<double>();
}

/**
 * Performs gradient checks on batch normalization layers.
 */
template<typename Scalar>
inline void batch_norm_layer_grad_test() {
	auto init = WeightInitSharedPtr<Scalar>(new HeWeightInitialization<Scalar>());
	auto reg1 = ParamRegSharedPtr<Scalar>(new L1ParameterRegularization<Scalar>());
	auto reg2 = ParamRegSharedPtr<Scalar>(new L2ParameterRegularization<Scalar>());
	LayerPtr<Scalar,1> layer1_1(new FCLayer<Scalar,1>(Dimensions<std::size_t,1>({ 32u }), 16, init));
	LayerPtr<Scalar,1> layer1_2(new BatchNormLayer<Scalar,1>(layer1_1->get_output_dims(), reg2, reg1));
	layer_grad_test<Scalar,1>("batch norm layer", std::move(layer1_1), std::move(layer1_2), 5,
			ScalarTraits<Scalar>::step_size, ScalarTraits<float>::abs_epsilon, ScalarTraits<float>::rel_epsilon);
	LayerPtr<Scalar,2> layer2_1(new BatchNormLayer<Scalar,2>(Dimensions<std::size_t,2>({ 6u, 6u })));
	LayerPtr<Scalar,2> layer2_2(new IdentityActivationLayer<Scalar,2>(layer2_1->get_output_dims()));
	layer_grad_test<Scalar,2>("batch norm layer", std::move(layer2_1), std::move(layer2_2));
	LayerPtr<Scalar,3> layer3_1(new ConvLayer<Scalar>(Dimensions<std::size_t,3>({ 4u, 4u, 2u }), 2, init));
	LayerPtr<Scalar,3> layer3_2(new BatchNormLayer<Scalar,3>(layer3_1->get_output_dims(), reg2, reg2));
	layer_grad_test<Scalar,3>("batch norm layer", std::move(layer3_1), std::move(layer3_2));
}

TEST(GradientTest, BatchNormLayer) {
	batch_norm_layer_grad_test<float>();
	batch_norm_layer_grad_test<double>();
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
	// Rank 1.
	Dimensions<std::size_t,1> dims_1({ 32u });
	std::vector<NeuralNetPtr<Scalar,1,false>> lanes_1;
	std::vector<LayerPtr<Scalar,1>> lane1_1_layers(1);
	lane1_1_layers[0] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(dims_1, 6, init));
	lanes_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane1_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane2_1_layers(3);
	lane2_1_layers[0] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(dims_1, 12, init));
	lane2_1_layers[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(lane2_1_layers[0]->get_output_dims()));
	lane2_1_layers[2] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(lane2_1_layers[1]->get_output_dims(), 6, init));
	lanes_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane2_1_layers))));
	std::vector<LayerPtr<Scalar,1>> lane3_1_layers(2);
	lane3_1_layers[0] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(dims_1, 4, init));
	lane3_1_layers[1] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(lane3_1_layers[0]->get_output_dims(), 6, init));
	lanes_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(lane3_1_layers))));
	NeuralNetPtr<Scalar,1,false> parallel_net_1(new ParallelNeuralNetwork<Scalar,1,SUM>(std::move(lanes_1)));
	ASSERT_TRUE((parallel_net_1->get_output_dims() == Dimensions<std::size_t,1>({ 6 })));
	nonseq_network_grad_test<Scalar,1>("parallel net", std::move(parallel_net_1));
	// Rank 2.
	Dimensions<std::size_t,2> dims_2({ 6u, 6u });
	std::vector<NeuralNetPtr<Scalar,2,false>> lanes_2;
	std::vector<LayerPtr<Scalar,2>> lane1_2_layers(1);
	lane1_2_layers[0] = LayerPtr<Scalar,2>(new FCLayer<Scalar,2>(dims_2, 6, init));
	lanes_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(lane1_2_layers))));
	std::vector<LayerPtr<Scalar,2>> lane2_2_layers(1);
	lane2_2_layers[0] = LayerPtr<Scalar,2>(new FCLayer<Scalar,2>(dims_2, 12, init));
	lanes_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(lane2_2_layers))));
	NeuralNetPtr<Scalar,2,false> parallel_net_2(new ParallelNeuralNetwork<Scalar,2,CONCAT_LO_RANK>(std::move(lanes_2)));
	ASSERT_TRUE((parallel_net_2->get_output_dims() == Dimensions<std::size_t,2>({ 18, 1 })));
	nonseq_network_grad_test<Scalar,2>("parallel net", std::move(parallel_net_2));
	// Rank 3.
	Dimensions<std::size_t,3> dims_3({ 4u, 4u, 3u });
	std::vector<NeuralNetPtr<Scalar,3,false>> lanes_3;
	std::vector<LayerPtr<Scalar,3>> lane1_3_layers(1);
	lane1_3_layers[0] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(dims_3, 4, init));
	lanes_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(lane1_3_layers))));
	std::vector<LayerPtr<Scalar,3>> lane2_3_layers(1);
	lane2_3_layers[0] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(dims_3, 2, init));
	lanes_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(lane2_3_layers))));
	NeuralNetPtr<Scalar,3,false> parallel_net_3(new ParallelNeuralNetwork<Scalar,3>(std::move(lanes_3)));
	ASSERT_TRUE((parallel_net_3->get_output_dims() == Dimensions<std::size_t,3>({ 4, 4, 6 })));
	nonseq_network_grad_test<Scalar,3>("parallel net", std::move(parallel_net_3));
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
	layers1_1[0] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(Dimensions<std::size_t,1>({ 32u }), 18, init));
	layers1_1[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(layers1_1[0]->get_output_dims()));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers1_1))));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new FCLayer<Scalar,1>(sub_modules1_1[0]->get_output_dims(), 32, init)))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new CompositeNeuralNetwork<Scalar,1,false>(std::move(sub_modules1_1))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new CompositeNeuralNetwork<Scalar,1,false>(NeuralNetPtr<Scalar,1,false>(
			new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new FCLayer<Scalar,1>(modules_1[0]->get_output_dims(), 32, init)))))));
	nonseq_network_grad_test<Scalar,1>("residual net", NeuralNetPtr<Scalar,1,false>(
			new ResidualNeuralNetwork<Scalar,1>(std::move(modules_1))));
	// Rank 3.
	std::vector<NeuralNetPtr<Scalar,3,false>> modules_3;
	std::vector<NeuralNetPtr<Scalar,3,false>> sub_modules1_3;
	std::vector<LayerPtr<Scalar,3>> layers1_3(2);
	layers1_3[0] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(Dimensions<std::size_t,3>({ 4u, 4u, 3u }), 4, init));
	layers1_3[1] = LayerPtr<Scalar,3>(new SigmoidActivationLayer<Scalar,3>(layers1_3[0]->get_output_dims()));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers1_3))));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvLayer<Scalar>(sub_modules1_3[0]->get_output_dims(), 3, init)))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new CompositeNeuralNetwork<Scalar,3,false>(std::move(sub_modules1_3))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new CompositeNeuralNetwork<Scalar,3,false>(NeuralNetPtr<Scalar,3,false>(
			new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvLayer<Scalar>(modules_3[0]->get_output_dims(), 3, init)))))));
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
	layers1_1[0] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(dims_1, 18, init));
	layers1_1[1] = LayerPtr<Scalar,1>(new SigmoidActivationLayer<Scalar,1>(layers1_1[0]->get_output_dims()));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers1_1))));
	sub_modules1_1.push_back(NeuralNetPtr<Scalar,1,false>(new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new FCLayer<Scalar,1>(sub_modules1_1[0]->get_output_dims(), 32, init)))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new CompositeNeuralNetwork<Scalar,1,false>(std::move(sub_modules1_1))));
	modules_1.push_back(NeuralNetPtr<Scalar,1,false>(new CompositeNeuralNetwork<Scalar,1,false>(NeuralNetPtr<Scalar,1,false>(
			new FeedforwardNeuralNetwork<Scalar,1>(LayerPtr<Scalar,1>(
			new FCLayer<Scalar,1>(dims_1.add_along_rank(modules_1[0]->get_output_dims(), 0), 16, init)))))));
	nonseq_network_grad_test<Scalar,1>("dense net", NeuralNetPtr<Scalar,1,false>(
			new DenseNeuralNetwork<Scalar,1>(std::move(modules_1))));
	// Rank 2.
	Dimensions<std::size_t,2> dims_2({ 6u, 6u });
	std::vector<NeuralNetPtr<Scalar,2,false>> modules_2;
	std::vector<NeuralNetPtr<Scalar,2,false>> sub_modules1_2;
	std::vector<LayerPtr<Scalar,2>> layers1_2(2);
	layers1_2[0] = LayerPtr<Scalar,2>(new FCLayer<Scalar,2>(dims_2, 12, init));
	layers1_2[1] = LayerPtr<Scalar,2>(new SigmoidActivationLayer<Scalar,2>(layers1_2[0]->get_output_dims()));
	sub_modules1_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(std::move(layers1_2))));
	sub_modules1_2.push_back(NeuralNetPtr<Scalar,2,false>(new FeedforwardNeuralNetwork<Scalar,2>(LayerPtr<Scalar,2>(
			new FCLayer<Scalar,2>(sub_modules1_2[0]->get_output_dims(), 6, init)))));
	modules_2.push_back(NeuralNetPtr<Scalar,2,false>(new CompositeNeuralNetwork<Scalar,2,false>(std::move(sub_modules1_2))));
	modules_2.push_back(NeuralNetPtr<Scalar,2,false>(new CompositeNeuralNetwork<Scalar,2,false>(NeuralNetPtr<Scalar,2,false>(
			new FeedforwardNeuralNetwork<Scalar,2>(LayerPtr<Scalar,2>(
			new FCLayer<Scalar,2>(dims_2.add_along_rank(modules_2[0]->get_output_dims(), 1), 6, init)))))));
	nonseq_network_grad_test<Scalar,2>("dense net", NeuralNetPtr<Scalar,2,false>(
			new DenseNeuralNetwork<Scalar,2,HIGHEST_RANK>(std::move(modules_2))));
	// Rank 3.
	Dimensions<std::size_t,3> dims_3({ 4u, 4u, 3u });
	std::vector<NeuralNetPtr<Scalar,3,false>> modules_3;
	std::vector<NeuralNetPtr<Scalar,3,false>> sub_modules1_3;
	std::vector<LayerPtr<Scalar,3>> layers1_3(2);
	layers1_3[0] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(dims_3, 4, init, ConvLayer<Scalar>::NO_PARAM_REG, 2, 3));
	layers1_3[1] = LayerPtr<Scalar,3>(new SigmoidActivationLayer<Scalar,3>(layers1_3[0]->get_output_dims()));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers1_3))));
	sub_modules1_3.push_back(NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvLayer<Scalar>(sub_modules1_3[0]->get_output_dims(), 3, init)))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new CompositeNeuralNetwork<Scalar,3,false>(std::move(sub_modules1_3))));
	modules_3.push_back(NeuralNetPtr<Scalar,3,false>(new CompositeNeuralNetwork<Scalar,3,false>(NeuralNetPtr<Scalar,3,false>(
			new FeedforwardNeuralNetwork<Scalar,3>(LayerPtr<Scalar,3>(
			new ConvLayer<Scalar>(dims_3.add_along_rank(modules_3[0]->get_output_dims(), 0), 3, init)))))));
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
	layers_1[0] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(Dimensions<std::size_t,1>({ 32u }), 16, init));
	layers_1[1] = LayerPtr<Scalar,1>(new TanhActivationLayer<Scalar,1>(layers_1[0]->get_output_dims()));
	layers_1[2] = LayerPtr<Scalar,1>(new FCLayer<Scalar,1>(layers_1[1]->get_output_dims(), 4, init));
	seq_network_grad_test("sequential net", NeuralNetPtr<Scalar,1,true>(
			new SequentialNeuralNetwork<Scalar,1>(NeuralNetPtr<Scalar,1,false>(
					new FeedforwardNeuralNetwork<Scalar,1>(std::move(layers_1))))));
	// Rank 3.
	std::vector<LayerPtr<Scalar,3>> layers_3(4);
	layers_3[0] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(Dimensions<std::size_t,3>({ 4u, 4u, 2u }), 4, init));
	layers_3[1] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers_3[0]->get_output_dims()));
	layers_3[2] = LayerPtr<Scalar,3>(new MaxPoolingLayer<Scalar>(layers_3[1]->get_output_dims()));
	layers_3[3] = LayerPtr<Scalar,3>(new FCLayer<Scalar,3>(layers_3[2]->get_output_dims(), 1, init));
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
	KernelPtr<Scalar,1> input_kernel1_1(new FCLayer<Scalar,1>(Dimensions<std::size_t,1>({ 12u }), 12, init, reg));
	KernelPtr<Scalar,1> state_kernel1_1(new FCLayer<Scalar,1>(input_kernel1_1->get_output_dims(), 12, init, reg));
	KernelPtr<Scalar,1> output_kernel1_1(new FCLayer<Scalar,1>(state_kernel1_1->get_output_dims(), 4, init, reg));
	ActivationPtr<Scalar,1> state_act1_1(new SigmoidActivationLayer<Scalar,1>(state_kernel1_1->get_output_dims()));
	ActivationPtr<Scalar,1> output_act1_1(new IdentityActivationLayer<Scalar,1>(output_kernel1_1->get_output_dims()));
	KernelPtr<Scalar,1> input_kernel2_1((FCLayer<Scalar,1>*) input_kernel1_1->clone());
	KernelPtr<Scalar,1> state_kernel2_1((FCLayer<Scalar,1>*) input_kernel2_1->clone());
	KernelPtr<Scalar,1> output_kernel2_1((FCLayer<Scalar,1>*) output_kernel1_1->clone());
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
	KernelPtr<Scalar,3> input_kernel1_3(new ConvLayer<Scalar>(Dimensions<std::size_t,3>({ 4u, 4u, 2u }), 5, init, reg));
	KernelPtr<Scalar,3> state_kernel1_3(new ConvLayer<Scalar>(input_kernel1_3->get_output_dims(), 5, init, reg));
	KernelPtr<Scalar,3> output_kernel1_3(new FCLayer<Scalar,3>(state_kernel1_3->get_output_dims(), 2, init, reg));
	ActivationPtr<Scalar,3> state_act1_3(new SoftplusActivationLayer<Scalar,3>(state_kernel1_3->get_output_dims()));
	ActivationPtr<Scalar,3> output_act1_3(new IdentityActivationLayer<Scalar,3>(output_kernel1_3->get_output_dims()));
	KernelPtr<Scalar,3> input_kernel2_3((ConvLayer<Scalar>*) input_kernel1_3->clone());
	KernelPtr<Scalar,3> state_kernel2_3((ConvLayer<Scalar>*) state_kernel1_3->clone());
	KernelPtr<Scalar,3> output_kernel2_3((FCLayer<Scalar,3>*) output_kernel1_3->clone());
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
	KernelPtr<Scalar,1> forget_input_kernel1_1(new FCLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> forget_output_kernel1_1(new FCLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> write_input_kernel1_1(new FCLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> write_output_kernel1_1(new FCLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> candidate_input_kernel1_1(new FCLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> candidate_output_kernel1_1(new FCLayer<Scalar,1>(output_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> read_input_kernel1_1(new FCLayer<Scalar,1>(input_dims_1, 5, init, reg));
	KernelPtr<Scalar,1> read_output_kernel1_1(new FCLayer<Scalar,1>(output_dims_1, 5, init, reg));
	ActivationPtr<Scalar,1> forget_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> write_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> candidate_act1_1(new TanhActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> state_act1_1(new TanhActivationLayer<Scalar,1>(output_dims_1));
	ActivationPtr<Scalar,1> read_act1_1(new SigmoidActivationLayer<Scalar,1>(output_dims_1));
	KernelPtr<Scalar,1> forget_input_kernel2_1((FCLayer<Scalar,1>*) forget_input_kernel1_1->clone());
	KernelPtr<Scalar,1> forget_output_kernel2_1((FCLayer<Scalar,1>*) forget_output_kernel1_1->clone());
	KernelPtr<Scalar,1> write_input_kernel2_1((FCLayer<Scalar,1>*) write_input_kernel1_1->clone());
	KernelPtr<Scalar,1> write_output_kernel2_1((FCLayer<Scalar,1>*) write_output_kernel1_1->clone());
	KernelPtr<Scalar,1> candidate_input_kernel2_1((FCLayer<Scalar,1>*) candidate_input_kernel1_1->clone());
	KernelPtr<Scalar,1> candidate_output_kernel2_1((FCLayer<Scalar,1>*) candidate_output_kernel1_1->clone());
	KernelPtr<Scalar,1> read_input_kernel2_1((FCLayer<Scalar,1>*) read_input_kernel1_1->clone());
	KernelPtr<Scalar,1> read_output_kernel2_1((FCLayer<Scalar,1>*) read_output_kernel1_1->clone());
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
	KernelPtr<Scalar,3> forget_input_kernel1_3(new ConvLayer<Scalar>(input_dims_3, 3, init, reg, 3, 3, 0, 1));
	KernelPtr<Scalar,3> forget_output_kernel1_3(new ConvLayer<Scalar>(output_dims_3, 3, init, reg));
	KernelPtr<Scalar,3> write_input_kernel1_3(new ConvLayer<Scalar>(input_dims_3, 3, init, reg, 3, 3, 0, 1));
	KernelPtr<Scalar,3> write_output_kernel1_3(new ConvLayer<Scalar>(output_dims_3, 3, init, reg));
	KernelPtr<Scalar,3> candidate_input_kernel1_3(new ConvLayer<Scalar>(input_dims_3, 3, init, reg, 3, 3, 0, 1));
	KernelPtr<Scalar,3> candidate_output_kernel1_3(new ConvLayer<Scalar>(output_dims_3, 3, init, reg));
	KernelPtr<Scalar,3> read_input_kernel1_3(new ConvLayer<Scalar>(input_dims_3, 3, init, reg, 3, 3, 0, 1));
	KernelPtr<Scalar,3> read_output_kernel1_3(new ConvLayer<Scalar>(output_dims_3, 3, init, reg));
	ActivationPtr<Scalar,3> forget_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> write_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> candidate_act1_3(new TanhActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> state_act1_3(new TanhActivationLayer<Scalar,3>(output_dims_3));
	ActivationPtr<Scalar,3> read_act1_3(new SigmoidActivationLayer<Scalar,3>(output_dims_3));
	KernelPtr<Scalar,3> forget_input_kernel2_3((ConvLayer<Scalar>*) forget_input_kernel1_3->clone());
	KernelPtr<Scalar,3> forget_output_kernel2_3((ConvLayer<Scalar>*) forget_output_kernel1_3->clone());
	KernelPtr<Scalar,3> write_input_kernel2_3((ConvLayer<Scalar>*) write_input_kernel1_3->clone());
	KernelPtr<Scalar,3> write_output_kernel2_3((ConvLayer<Scalar>*) write_output_kernel1_3->clone());
	KernelPtr<Scalar,3> candidate_input_kernel2_3((ConvLayer<Scalar>*) candidate_input_kernel1_3->clone());
	KernelPtr<Scalar,3> candidate_output_kernel2_3((ConvLayer<Scalar>*) candidate_output_kernel1_3->clone());
	KernelPtr<Scalar,3> read_input_kernel2_3((ConvLayer<Scalar>*) read_input_kernel1_3->clone());
	KernelPtr<Scalar,3> read_output_kernel2_3((ConvLayer<Scalar>*) read_output_kernel1_3->clone());
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
	KernelPtr<Scalar,3> input_kernel1(new ConvLayer<Scalar>(Dimensions<std::size_t,3>({ 4u, 4u, 2u }), 5, init, reg));
	KernelPtr<Scalar,3> state_kernel1(new ConvLayer<Scalar>(input_kernel1->get_output_dims(), 5, init, reg));
	KernelPtr<Scalar,3> output_kernel1(new FCLayer<Scalar,3>(input_kernel1->get_output_dims(), 2, init, reg));
	ActivationPtr<Scalar,3> state_act1(new SigmoidActivationLayer<Scalar,3>(input_kernel1->get_output_dims()));
	ActivationPtr<Scalar,3> output_act1(new IdentityActivationLayer<Scalar,3>(output_kernel1->get_output_dims()));
	seq_network_grad_test("bidirectional recurrent net", NeuralNetPtr<Scalar,3,true>(
			new BidirectionalNeuralNetwork<Scalar,3,CONCAT_HI_RANK>(UnidirNeuralNetPtr<Scalar,3>(
					new RecurrentNeuralNetwork<Scalar,3,true>(std::move(input_kernel1), std::move(state_kernel1),
							std::move(output_kernel1), std::move(state_act1), std::move(output_act1),
							[](int input_seq_length) { return std::make_pair(3, 2); })))), 7, 3);
	// 2nd degree LSTM with lowest rank concatenation.
	Dimensions<std::size_t,2> input_dims({ 6u, 6u });
	Dimensions<std::size_t,2> output_dims({ 6u, 1u });
	KernelPtr<Scalar,2> forget_input_kernel2(new FCLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> forget_output_kernel2(new FCLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> write_input_kernel2(new FCLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> write_output_kernel2(new FCLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> candidate_input_kernel2(new FCLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> candidate_output_kernel2(new FCLayer<Scalar,2>(output_dims, 6, init, reg));
	KernelPtr<Scalar,2> read_input_kernel2(new FCLayer<Scalar,2>(input_dims, 6, init, reg));
	KernelPtr<Scalar,2> read_output_kernel2(new FCLayer<Scalar,2>(output_dims, 6, init, reg));
	ActivationPtr<Scalar,2> forget_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> write_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> candidate_act2(new SoftplusActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> state_act2(new SoftplusActivationLayer<Scalar,2>(output_dims));
	ActivationPtr<Scalar,2> read_act2(new SigmoidActivationLayer<Scalar,2>(output_dims));
	seq_network_grad_test("bidirectional lstm net", NeuralNetPtr<Scalar,2,true>(
			new BidirectionalNeuralNetwork<Scalar,2,CONCAT_LO_RANK>(UnidirNeuralNetPtr<Scalar,2>(
					new LSTMNeuralNetwork<Scalar,2,true>(std::move(forget_input_kernel2),
							std::move(forget_output_kernel2), std::move(write_input_kernel2),
							std::move(write_output_kernel2), std::move(candidate_input_kernel2),
							std::move(candidate_output_kernel2), std::move(read_input_kernel2),
							std::move(read_output_kernel2), std::move(forget_act2), std::move(write_act2),
							std::move(candidate_act2), std::move(state_act2), std::move(read_act2),
							[](int input_seq_length) { return std::make_pair(1, 2); })))), 5, 1);
	// 1st degree RNN with summation.
	KernelPtr<Scalar,1> input_kernel3(new FCLayer<Scalar,1>(Dimensions<std::size_t,1>({ 24u }), 8, init, reg));
	KernelPtr<Scalar,1> state_kernel3(new FCLayer<Scalar,1>(input_kernel3->get_output_dims(), 8, init, reg));
	KernelPtr<Scalar,1> output_kernel3(new FCLayer<Scalar,1>(input_kernel3->get_output_dims(), 1, init, reg));
	ActivationPtr<Scalar,1> state_act3(new SigmoidActivationLayer<Scalar,1>(input_kernel3->get_output_dims()));
	ActivationPtr<Scalar,1> output_act3(new SigmoidActivationLayer<Scalar,1>(output_kernel3->get_output_dims()));
	seq_network_grad_test("bidirectional recurrent net", NeuralNetPtr<Scalar,1,true>(
			new BidirectionalNeuralNetwork<Scalar,1,SUM>(UnidirNeuralNetPtr<Scalar,1>(
					new RecurrentNeuralNetwork<Scalar,1>(std::move(input_kernel3), std::move(state_kernel3),
							std::move(output_kernel3), std::move(state_act3), std::move(output_act3),
							[](int input_seq_length) { return std::make_pair(5, 2); })))), 7, 5);
}

TEST(GradientTest, BidirectionalNet) {
	bidirectional_net_grad_test<float>();
	bidirectional_net_grad_test<double>();
}

/***********************
 * LOSS GRADIENT TESTS *
 ***********************/

} /* namespace test */

} /* namespace cattle */

#endif /* GRADIENT_TEST_HPP_ */
