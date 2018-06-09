/*
 * training_test.hpp
 *
 *  Created on: 06.05.2018
 *      Author: Viktor Csomor
 */

#ifndef TRAINING_TEST_HPP_
#define TRAINING_TEST_HPP_

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
 * Determines the verbosity of the training tests.
 */
extern bool verbose;

/**
 * @param name The name of the training test.
 * @param train_prov The training data provider.
 * @param net The neural network to train.
 * @param opt The optimizer to use to train the network.
 * @param epochs The number of epochs for whicht the network is to be trained.
 * @param epsilon The maximum acceptable absolute difference between 0 and
 * the final training loss.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
inline void train_test(std::string name, DataProvider<Scalar,Rank,Sequential>& train_prov,
		NeuralNetwork<Scalar,Rank,Sequential>& net, Optimizer<Scalar,Rank,Sequential>& opt,
		unsigned epochs, Scalar epsilon) {
	print_test_header<Scalar,Rank,Sequential>("training test", name);
	net.init();
	opt.fit(net);
	Scalar loss = opt.train(net, train_prov, epochs, 0, epsilon, verbose);
	EXPECT_TRUE(internal::NumericUtils<Scalar>::almost_equal((Scalar) 0, loss, epsilon));
}

/**
 * @param name The name of the training test.
 * @param opt The optimizer to use to train the feed-forward network.
 * @param input_dims The input dimensions of the network.
 * @param samples The total number of data samples to include in the training data.
 * @param epochs The number of epochs for which the network is to be trained.
 * @param epsilon The maximum acceptable absolute difference between 0 and
 * the final training loss.
 */
template<typename Scalar, std::size_t Rank>
inline void ffnn_train_test(std::string name, OptimizerPtr<Scalar,Rank,false> opt,
		const Dimensions<std::size_t,Rank> input_dims, unsigned samples,
		unsigned epochs, Scalar epsilon = ScalarTraits<Scalar>::abs_epsilon) {
	NeuralNetPtr<Scalar,Rank,false> net = single_output_net<Scalar,Rank>(tanh_neural_net<Scalar,Rank>(input_dims));
	std::array<std::size_t,Rank + 1> input_batch_dims = input_dims.template promote<>();
	input_batch_dims[0] = samples;
	TensorPtr<Scalar,Rank + 1> input_tensor(new Tensor<Scalar,Rank + 1>(input_batch_dims));
	input_tensor->setRandom();
	std::array<std::size_t,Rank + 1> output_batch_dims = net->get_output_dims().template promote<>();
	output_batch_dims[0] = samples;
	TensorPtr<Scalar,Rank + 1> output_tensor(new Tensor<Scalar,Rank + 1>(output_batch_dims));
	output_tensor->setRandom();
	MemoryDataProvider<Scalar,Rank,false> prov(std::move(input_tensor), std::move(output_tensor));
	train_test(name, prov, *net, *opt, epochs, epsilon);
}

/**
 * @param name The name of the training test.
 * @param opt The optimizer to use to train the network.
 * @param input_dims The input dimensions of the network.
 * @param samples The total number of data samples to include in the training data.
 * @param time_steps The number of time steps each sample is to contain.
 * @param epochs The number of epochs for which the network is to be trained.
 * @param epsilon The maximum acceptable absolute difference between 0 and
 * the final training loss.
 */
template<typename Scalar, std::size_t Rank>
inline void rnn_train_test(std::string name, OptimizerPtr<Scalar,Rank,true> opt,
		const Dimensions<std::size_t,Rank> input_dims, unsigned samples, unsigned time_steps,
		unsigned epochs, Scalar epsilon = ScalarTraits<Scalar>::abs_epsilon) {
	NeuralNetPtr<Scalar,Rank,true> net = single_output_recurrent_neural_net<Scalar,Rank>(input_dims,
			[](int input_seq_length) { return std::make_pair(1, input_seq_length - 1); });
	std::array<std::size_t,Rank + 2> input_batch_dims = input_dims.template promote<2>();
	input_batch_dims[0] = samples;
	input_batch_dims[1] = time_steps;
	TensorPtr<Scalar,Rank + 2> input_tensor(new Tensor<Scalar,Rank + 2>(input_batch_dims));
	input_tensor->setRandom();
	std::array<std::size_t,Rank + 2> output_batch_dims = net->get_output_dims().template promote<2>();
	output_batch_dims[0] = samples;
	output_batch_dims[1] = 1;
	TensorPtr<Scalar,Rank + 2> output_tensor(new Tensor<Scalar,Rank + 2>(output_batch_dims));
	output_tensor->setRandom();
	MemoryDataProvider<Scalar,Rank,true> prov(std::move(input_tensor), std::move(output_tensor));
	train_test(name, prov, *net, *opt, epochs, epsilon);
}

/**
 * Performs training tests using a vanilla stochastic gradient descent optimizer.
 */
template<typename Scalar>
inline void vanilla_sgd_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .2;
	const Scalar seq_epsilon = .25;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("vanilla sgd batch",
			OptimizerPtr<Scalar,3,false>(new VanillaSGDOptimizer<Scalar,3,false>(loss, samples, 2e-3)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("vanilla sgd mini-batch",
			OptimizerPtr<Scalar,3,false>(new VanillaSGDOptimizer<Scalar,3,false>(loss, 10, 5e-4)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("vanilla sgd online",
			OptimizerPtr<Scalar,3,false>(new VanillaSGDOptimizer<Scalar,3,false>(loss, 1, 1e-4)),
			dims, samples, epochs, epsilon);
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
	rnn_train_test<Scalar,3>("vanilla sgd batch",
			OptimizerPtr<Scalar,3,true>(new VanillaSGDOptimizer<Scalar,3,true>(seq_loss, samples, 2e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("vanilla sgd mini-batch",
			OptimizerPtr<Scalar,3,true>(new VanillaSGDOptimizer<Scalar,3,true>(seq_loss, 10, 5e-4)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("vanilla sgd online",
			OptimizerPtr<Scalar,3,true>(new VanillaSGDOptimizer<Scalar,3,true>(seq_loss, 1, 1e-4)),
			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, VanillaSGD) {
	vanilla_sgd_train_test<float>();
	vanilla_sgd_train_test<double>();
}

/**
 * Performs training tests using a momentum accelerated stochastic gradient descent optimizer.
 */
template<typename Scalar>
inline void momentum_accelerated_sgd_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .18;
	const Scalar seq_epsilon = .22;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("momentum accelerated sgd batch",
			OptimizerPtr<Scalar,3,false>(new MomentumAcceleratedSGDOptimizer<Scalar,3,false>(loss, samples, 2e-3)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("momentum accelerated sgd mini-batch",
			OptimizerPtr<Scalar,3,false>(new MomentumAcceleratedSGDOptimizer<Scalar,3,false>(loss, 10, 1e-3, 1e-2, .99)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("momentum accelerated sgd online",
			OptimizerPtr<Scalar,3,false>(new MomentumAcceleratedSGDOptimizer<Scalar,3,false>(loss, 1, 5e-4)),
			dims, samples, epochs, epsilon);
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
	rnn_train_test<Scalar,3>("momentum accelerated sgd batch",
			OptimizerPtr<Scalar,3,true>(new MomentumAcceleratedSGDOptimizer<Scalar,3,true>(seq_loss, samples, 2e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("momentum accelerated sgd mini-batch",
			OptimizerPtr<Scalar,3,true>(new MomentumAcceleratedSGDOptimizer<Scalar,3,true>(seq_loss, 10, 1e-3, 1e-2, .99)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("momentum accelerated sgd online",
			OptimizerPtr<Scalar,3,true>(new MomentumAcceleratedSGDOptimizer<Scalar,3,true>(seq_loss, 1, 5e-4)),
			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, MomentumAcceleratedSGD) {
	momentum_accelerated_sgd_train_test<float>();
	momentum_accelerated_sgd_train_test<double>();
}

/**
 * Performs training tests using a nesterov momentum accelerated stochastic gradient descent
 * optimizer.
 */
template<typename Scalar>
inline void nesterov_momentum_accelerated_sgd_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .18;
	const Scalar seq_epsilon = .22;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("nesterov momentum accelerated sgd batch",
			OptimizerPtr<Scalar,3,false>(new NesterovMomentumAcceleratedSGDOptimizer<Scalar,3,false>(loss, samples)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("nesterov momentum accelerated sgd mini-batch",
			OptimizerPtr<Scalar,3,false>(new NesterovMomentumAcceleratedSGDOptimizer<Scalar,3,false>(loss, 10, 1e-3, 1e-2, .99)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("nesterov momentum accelerated sgd online",
			OptimizerPtr<Scalar,3,false>(new NesterovMomentumAcceleratedSGDOptimizer<Scalar,3,false>(loss, 1, 5e-4)),
			dims, samples, epochs, epsilon);
//	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
//	rnn_train_test<Scalar,3>("nesterov momentum accelerated sgd batch",
//			OptimizerPtr<Scalar,3,true>(new NesterovMomentumAcceleratedSGDOptimizer<Scalar,3,true>(seq_loss, samples)),
//			dims, samples, time_steps, epochs, seq_epsilon);
//	rnn_train_test<Scalar,3>("nesterov momentum accelerated sgd mini-batch",
//			OptimizerPtr<Scalar,3,true>(new NesterovMomentumAcceleratedSGDOptimizer<Scalar,3,true>(seq_loss, 10, 1e-3, 1e-2, .99)),
//			dims, samples, time_steps, epochs, seq_epsilon);
//	rnn_train_test<Scalar,3>("nesterov momentum accelerated sgd online",
//			OptimizerPtr<Scalar,3,true>(new NesterovMomentumAcceleratedSGDOptimizer<Scalar,3,true>(seq_loss, 1, 5e-4)),
//			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, NesterovMomentumAcceleratedSGD) {
	nesterov_momentum_accelerated_sgd_train_test<float>();
	nesterov_momentum_accelerated_sgd_train_test<double>();
}

/**
 * Performs training tests using an AdaGrad optimizer.
 */
template<typename Scalar>
inline void adagrad_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .02;
	const Scalar seq_epsilon = .08;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("adagrad batch",
			OptimizerPtr<Scalar,3,false>(new AdaGradOptimizer<Scalar,3,false>(loss, samples, 1e-2)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("adagrad mini-batch",
			OptimizerPtr<Scalar,3,false>(new AdaGradOptimizer<Scalar,3,false>(loss, 10, 5e-3)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("adagrad online",
			OptimizerPtr<Scalar,3,false>(new AdaGradOptimizer<Scalar,3,false>(loss)),
			dims, samples, epochs, epsilon);
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
	rnn_train_test<Scalar,3>("adagrad batch",
			OptimizerPtr<Scalar,3,true>(new AdaGradOptimizer<Scalar,3,true>(seq_loss, samples, 1e-2)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("adagrad mini-batch",
			OptimizerPtr<Scalar,3,true>(new AdaGradOptimizer<Scalar,3,true>(seq_loss, 10, 5e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("adagrad online",
			OptimizerPtr<Scalar,3,true>(new AdaGradOptimizer<Scalar,3,true>(seq_loss)),
			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, AdaGrad) {
	adagrad_train_test<float>();
	adagrad_train_test<double>();
}

/**
 * Performs training tests using an RMSprop optimizer.
 */
template<typename Scalar>
inline void rmsprop_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .02;
	const Scalar seq_epsilon = .08;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("rmsprop batch",
			OptimizerPtr<Scalar,3,false>(new RMSPropOptimizer<Scalar,3,false>(loss, samples, 1e-2)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("rmsprop mini-batch",
			OptimizerPtr<Scalar,3,false>(new RMSPropOptimizer<Scalar,3,false>(loss, 10, 5e-3, 1e-2)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("rmsprop online",
			OptimizerPtr<Scalar,3,false>(new RMSPropOptimizer<Scalar,3,false>(loss)),
			dims, samples, epochs, epsilon);
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
	rnn_train_test<Scalar,3>("rmsprop batch",
			OptimizerPtr<Scalar,3,true>(new RMSPropOptimizer<Scalar,3,true>(seq_loss, samples, 1e-2)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("rmsprop mini-batch",
			OptimizerPtr<Scalar,3,true>(new RMSPropOptimizer<Scalar,3,true>(seq_loss, 10, 5e-3, 1e-2)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("rmsprop online",
			OptimizerPtr<Scalar,3,true>(new RMSPropOptimizer<Scalar,3,true>(seq_loss)),
			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, RMSProp) {
	rmsprop_train_test<float>();
	rmsprop_train_test<double>();
}

/**
 * Performs training tests using an AdaDelta optimizer.
 */
template<typename Scalar>
inline void adadelta_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .02;
	const Scalar seq_epsilon = .1;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("adadelta batch",
			OptimizerPtr<Scalar,3,false>(new AdaDeltaOptimizer<Scalar,3,false>(loss, samples)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("adadelta mini-batch",
			OptimizerPtr<Scalar,3,false>(new AdaDeltaOptimizer<Scalar,3,false>(loss, 10, 1e-2)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("adadelta online",
			OptimizerPtr<Scalar,3,false>(new AdaDeltaOptimizer<Scalar,3,false>(loss, 1, 1e-1)),
			dims, samples, epochs, epsilon);
//	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
//	rnn_train_test<Scalar,3>("adadelta batch",
//			OptimizerPtr<Scalar,3,true>(new AdaDeltaOptimizer<Scalar,3,true>(seq_loss, samples, 5e-2)),
//			dims, samples, time_steps, epochs, seq_epsilon);
//	rnn_train_test<Scalar,3>("adadelta mini-batch",
//			OptimizerPtr<Scalar,3,true>(new AdaDeltaOptimizer<Scalar,3,true>(seq_loss, 10, 1e-2)),
//			dims, samples, time_steps, epochs, seq_epsilon);
//	rnn_train_test<Scalar,3>("adadelta online",
//			OptimizerPtr<Scalar,3,true>(new AdaDeltaOptimizer<Scalar,3,true>(seq_loss, 1, 1e-1)),
//			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, AdaDelta) {
	adadelta_train_test<float>();
	adadelta_train_test<double>();
}

/**
 * Performs training tests using an Adam optimizer.
 */
template<typename Scalar>
inline void adam_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .015;
	const Scalar seq_epsilon = .1;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("adam batch",
			OptimizerPtr<Scalar,3,false>(new AdamOptimizer<Scalar,3,false>(loss, samples, 1e-2)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("adam mini-batch",
			OptimizerPtr<Scalar,3,false>(new AdamOptimizer<Scalar,3,false>(loss, 10, 1e-3, 5e-2, 5e-3)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("adam online",
			OptimizerPtr<Scalar,3,false>(new AdamOptimizer<Scalar,3,false>(loss)),
			dims, samples, epochs, epsilon);
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
	rnn_train_test<Scalar,3>("adam batch",
			OptimizerPtr<Scalar,3,true>(new AdamOptimizer<Scalar,3,true>(seq_loss, samples, 5e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("adam mini-batch",
			OptimizerPtr<Scalar,3,true>(new AdamOptimizer<Scalar,3,true>(seq_loss, 10, 1e-3, 5e-2, 5e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("adam online",
			OptimizerPtr<Scalar,3,true>(new AdamOptimizer<Scalar,3,true>(seq_loss)),
			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, Adam) {
	adam_train_test<float>();
	adam_train_test<double>();
}

/**
 * Performs training tests using an AdaMax optimizer.
 */
template<typename Scalar>
inline void adamax_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .015;
	const Scalar seq_epsilon = .1;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("adamax batch",
			OptimizerPtr<Scalar,3,false>(new AdaMaxOptimizer<Scalar,3,false>(loss, samples, 1e-2)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("adamax mini-batch",
			OptimizerPtr<Scalar,3,false>(new AdaMaxOptimizer<Scalar,3,false>(loss, 10, 1e-3, 5e-2, 5e-3)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("adamax online",
			OptimizerPtr<Scalar,3,false>(new AdaMaxOptimizer<Scalar,3,false>(loss)),
			dims, samples, epochs, epsilon);
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
	rnn_train_test<Scalar,3>("adamax batch",
			OptimizerPtr<Scalar,3,true>(new AdaMaxOptimizer<Scalar,3,true>(seq_loss, samples, 5e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("adamax mini-batch",
			OptimizerPtr<Scalar,3,true>(new AdaMaxOptimizer<Scalar,3,true>(seq_loss, 10, 1e-3, 5e-2, 5e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("adamax online",
			OptimizerPtr<Scalar,3,true>(new AdaMaxOptimizer<Scalar,3,true>(seq_loss)),
			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, AdaMax) {
	adamax_train_test<float>();
	adamax_train_test<double>();
}

/**
 * Performs training tests using an Nadam optimizer.
 */
template<typename Scalar>
inline void nadam_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .01;
	const Scalar seq_epsilon = .06;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("nadam batch",
			OptimizerPtr<Scalar,3,false>(new NadamOptimizer<Scalar,3,false>(loss, samples, 1e-2)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("nadam mini-batch",
			OptimizerPtr<Scalar,3,false>(new NadamOptimizer<Scalar,3,false>(loss, 10, 1e-3, 5e-2, 5e-3)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("nadam online",
			OptimizerPtr<Scalar,3,false>(new NadamOptimizer<Scalar,3,false>(loss)),
			dims, samples, epochs, epsilon);
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
	rnn_train_test<Scalar,3>("nadam batch",
			OptimizerPtr<Scalar,3,true>(new NadamOptimizer<Scalar,3,true>(seq_loss, samples, 5e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("nadam mini-batch",
			OptimizerPtr<Scalar,3,true>(new NadamOptimizer<Scalar,3,true>(seq_loss, 10, 1e-3, 5e-2, 5e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("nadam online",
			OptimizerPtr<Scalar,3,true>(new NadamOptimizer<Scalar,3,true>(seq_loss)),
			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, Nadam) {
	nadam_train_test<float>();
	nadam_train_test<double>();
}

/**
 * Performs training tests using an AMSGrad optimizer.
 */
template<typename Scalar>
inline void amsgrad_train_test() {
	const unsigned samples = 20;
	const unsigned time_steps = 3;
	const unsigned epochs = 500;
	const Scalar epsilon = .015;
	const Scalar seq_epsilon = .12;
	const Dimensions<std::size_t,3> dims({ 6u, 6u, 2u });
	auto loss = std::make_shared<SquaredLoss<Scalar,3,false>>();
	ffnn_train_test<Scalar,3>("amsgrad batch",
			OptimizerPtr<Scalar,3,false>(new AMSGradOptimizer<Scalar,3,false>(loss, samples, 1e-2)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("amsgrad mini-batch",
			OptimizerPtr<Scalar,3,false>(new AMSGradOptimizer<Scalar,3,false>(loss, 10, 1e-3)),
			dims, samples, epochs, epsilon);
	ffnn_train_test<Scalar,3>("amsgrad online",
			OptimizerPtr<Scalar,3,false>(new AMSGradOptimizer<Scalar,3,false>(loss)),
			dims, samples, epochs, epsilon);
	auto seq_loss = std::make_shared<SquaredLoss<Scalar,3,true>>();
	rnn_train_test<Scalar,3>("amsgrad batch",
			OptimizerPtr<Scalar,3,true>(new AMSGradOptimizer<Scalar,3,true>(seq_loss, samples, 5e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("amsgrad mini-batch",
			OptimizerPtr<Scalar,3,true>(new AMSGradOptimizer<Scalar,3,true>(seq_loss, 10, 1e-3)),
			dims, samples, time_steps, epochs, seq_epsilon);
	rnn_train_test<Scalar,3>("amsgrad online",
			OptimizerPtr<Scalar,3,true>(new AMSGradOptimizer<Scalar,3,true>(seq_loss)),
			dims, samples, time_steps, epochs, seq_epsilon);
}

TEST(TrainingTest, AMSGrad) {
	amsgrad_train_test<float>();
	amsgrad_train_test<double>();
}

} /* namespace test */
} /* namespace cattle */

#endif /* TRAINING_TEST_HPP_ */
