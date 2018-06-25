/*
 * simple_convnet.cpp
 *
 *  Created on: 18 Apr 2018
 *      Author: Viktor Csomor
 */

#include <memory>
#include <utility>
#include <vector>

#include "Cattle.hpp"

int main() {
	using namespace cattle;
	// Create random observation and objective tensors for training.
	TensorPtr<double,4> training_obs_ptr(new Tensor<double,4>(80u, 32u, 32u, 3u));
	TensorPtr<double,4> training_obj_ptr(new Tensor<double,4>(80u, 1u, 1u, 1u));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	// Create and fit a PCA preprocessor.
	PCAPreprocessor<double,3> preproc;
	preproc.fit(*training_obs_ptr);
	preproc.transform(*training_obs_ptr);
	// Create random training data and observations.
	MemoryDataProvider<double,3,false> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	TensorPtr<double,4> test_obs_ptr(new Tensor<double,4>(20u, 32u, 32u, 3u));
	TensorPtr<double,4> test_obj_ptr(new Tensor<double,4>(20u, 1u, 1u, 1u));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	preproc.transform(*test_obs_ptr);
	MemoryDataProvider<double,3,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
	// Create weight initialization and parameter regularization instances.
	auto init = std::make_shared<HeWeightInitialization<double>>();
	auto reg = std::make_shared<SquaredParameterRegularization<double>>();
	// Create the convolutional network.
	std::vector<LayerPtr<double,3>> layers(9);
	layers[0] = LayerPtr<double,3>(new ConvKernelLayer<double>(training_prov.get_obs_dims(), 10, init, reg, 5, 2));
	layers[1] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<double,3>(new MaxPoolLayer<double>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<double,3>(new ConvKernelLayer<double>(layers[2]->get_output_dims(), 20, init, reg));
	layers[4] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<double,3>(new MaxPoolLayer<double>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<double,3>(new DenseKernelLayer<double,3>(layers[5]->get_output_dims(), 500, init, reg));
	layers[7] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[6]->get_output_dims()));
	layers[8] = LayerPtr<double,3>(new DenseKernelLayer<double,3>(layers[7]->get_output_dims(), 1, init, reg));
	FeedforwardNeuralNetwork<double,3> nn(std::move(layers));
	// Initialize the network.
	nn.init();
	// Create the loss and the optimizer.
	auto loss = std::make_shared<SquaredLoss<double,3,false>>();
	NadamOptimizer<double,3,false> opt(loss, 20);
	// Optimize the network.
	opt.optimize(nn, training_prov, test_prov, 500);
	// Test inference.
	Tensor<double,4> input(5u, 32u, 32u, 3u);
	input.setRandom();
	preproc.transform(input);
	Tensor<double,4> prediction = nn.infer(input);
}
