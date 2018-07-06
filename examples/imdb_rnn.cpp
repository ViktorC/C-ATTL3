/*
 * imdb_rnn.cpp
 *
 *  Created on: 5 Jul 2018
 *      Author: Viktor Csomor
 */

#include "Cattle.hpp"

int main() {
	using namespace cattle;
	// Set up the IMDB data set provider.
	std::string imdb_folder = "data/imdb/";
	VocabSharedPtr vocab = IMDBDataProvider<float>::build_vocab(imdb_folder + "imdb.vocab");
	IMDBDataProvider<float> train_prov(imdb_folder + "train/pos", imdb_folder + "train/neg", vocab);
	IMDBDataProvider<float> test_prov(imdb_folder + "test/pos", imdb_folder + "test/neg", vocab);
	// Set up the recurrent neural network.
	auto init = std::make_shared<GlorotWeightInitialization<float>>();
	auto reg = std::make_shared<SquaredParameterRegularization<float>>();
	KernelPtr<float,1> input_kernel(new DenseKernelLayer<float>(train_prov.get_obs_dims(), 100, init, reg));
	KernelPtr<float,1> state_kernel(new DenseKernelLayer<float>(input_kernel->get_output_dims(), 100, init));
	KernelPtr<float,1> output_kernel(new DenseKernelLayer<float>(input_kernel->get_output_dims(), 1, init));
	ActivationPtr<float,1> state_act(new TanhActivationLayer<float,1>(state_kernel->get_output_dims()));
	ActivationPtr<float,1> output_act(new SigmoidActivationLayer<float,1>(output_kernel->get_output_dims()));
	RecurrentNeuralNetwork<float,1> rnn(std::move(input_kernel), std::move(state_kernel), std::move(output_kernel),
			std::move(state_act), std::move(output_act), [](std::size_t seq_length) { return std::make_pair(1,
					seq_length - 1); });
	// Initialize the network.
	rnn.init();
	// Create the loss and the optimizer.
	auto loss = std::make_shared<CrossEntropyLoss<float,1,true>>();
	NadamOptimizer<float,1,true> opt(loss, 20);
	// Optimize the network.
	opt.optimize(rnn, train_prov, test_prov, 100);
}
