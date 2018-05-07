/*
 * optimization_test.hpp
 *
 *  Created on: 06.05.2018
 *      Author: Viktor Csomor
 */

#ifndef OPTIMIZATION_TEST_HPP_
#define OPTIMIZATION_TEST_HPP_

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

template<typename Scalar, std::size_t Rank, bool Sequential>
inline void opt_test(std::string name, DataProvider<Scalar,Rank,Sequential>& train_prov,
		DataProvider<Scalar,Rank,Sequential>& test_prov, NeuralNetwork<Scalar,Rank,Sequential>& net,
		Optimizer<Scalar,Rank,Sequential> opt, unsigned epochs, unsigned early_stop = 0) {
	print_test_header<Scalar,Rank,Sequential>("optimization test", name);
	net.init();
	Scalar orig_loss = opt.optimize(net, train_prov, test_prov, 0, early_stop, verbose);
	Scalar opt_loss = opt.optimize(net, train_prov, test_prov, epochs, early_stop, verbose);
	EXPECT_LT(opt_loss, orig_loss);
}

/**
 * @param input_dims The input dimensions of the network.
 * @return A fully-connected neural network for rank 3 non-sequential data with a size-10
 * vector output.
 */
template<typename Scalar>
inline NeuralNetPtr<Scalar,3,false> fc_net(const Dimensions<std::size_t,3>& input_dims) {
	WeightInitSharedPtr<Scalar> init(new GlorotWeightInitialization<Scalar>(1e-1));
	std::vector<LayerPtr<Scalar,3>> layers(5);
	layers[0] = LayerPtr<Scalar,3>(new FCLayer<Scalar,3>(input_dims, 200, init));
	layers[1] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<Scalar,3>(new FCLayer<Scalar,3>(layers[1]->get_output_dims(), 100, init));
	layers[3] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers[2]->get_output_dims()));
	layers[4] = LayerPtr<Scalar,3>(new FCLayer<Scalar,3>(layers[3]->get_output_dims(), 10, init));
	return NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers)));
}

/**
 * @param input_dims The input dimensions of the network.
 * @return A convolutional neural network for rank 3 non-sequential data with a size-10 vector
 * output.
 */
template<typename Scalar>
inline NeuralNetPtr<Scalar,3,false> conv_net(const Dimensions<std::size_t,3>& input_dims) {
	WeightInitSharedPtr<Scalar> conv_init(new HeWeightInitialization<Scalar>(1e-1));
	WeightInitSharedPtr<Scalar> dense_init(new GlorotWeightInitialization<Scalar>(1e-1));
	std::vector<LayerPtr<Scalar,3>> layers(11);
	layers[0] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(input_dims, 4, conv_init));
	layers[1] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<Scalar,3>(new MaxPoolingLayer<Scalar>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(layers[2]->get_output_dims(), 2, conv_init));
	layers[4] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<Scalar,3>(new MaxPoolingLayer<Scalar>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<Scalar,3>(new DropoutLayer<Scalar,3>(layers[5]->get_output_dims(), .25));
	layers[7] = LayerPtr<Scalar,3>(new FCLayer<Scalar,3>(layers[6]->get_output_dims(), 50, dense_init));
	layers[8] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers[7]->get_output_dims()));
	layers[9] = LayerPtr<Scalar,3>(new DropoutLayer<Scalar,3>(layers[8]->get_output_dims(), .5));
	layers[10] = LayerPtr<Scalar,3>(new FCLayer<Scalar,3>(layers[9]->get_output_dims(), 10, dense_init));
	return NeuralNetPtr<Scalar,3,false>(new FeedforwardNeuralNetwork<Scalar,3>(std::move(layers)));
}

} /* namespace test */
} /* namespace cattle */

#endif /* OPTIMIZATION_TEST_HPP_ */
