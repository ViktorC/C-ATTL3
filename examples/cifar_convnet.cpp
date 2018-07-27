/*
 * cifar_convnet.cpp
 *
 *  Created on: 18 Apr 2018
 *      Author: Viktor Csomor
 */

#include <chrono>

#include "Cattle.hpp"

int main() {
	using namespace cattle;
	// Create a CIFAR-10 data provider by specifying the paths to the training files.
	std::string cifar_folder = "data/cifar-10/";
	CIFARDataProvider<float> file_train_prov({ cifar_folder + "data_batch_1.bin", cifar_folder + "data_batch_2.bin",
			cifar_folder + "data_batch_3.bin", cifar_folder + "data_batch_4.bin", cifar_folder + "data_batch_5.bin" });
	// Create a data provider for the test data as well.
	CIFARDataProvider<float> file_test_prov(cifar_folder + "test_batch.bin");
	// Specify the weight initializations.
	auto conv_init = std::make_shared<HeWeightInitialization<float>>(1e-1);
	auto dense_init = std::make_shared<GlorotWeightInitialization<float>>(1e-1);
	// Create the network.
	std::vector<LayerPtr<float,3>> layers(12);
	layers[0] = LayerPtr<float,3>(new ConvKernelLayer<float>(file_train_prov.get_obs_dims(), 8, conv_init));
	layers[1] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<float,3>(new MaxPoolLayer<float>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<float,3>(new ConvKernelLayer<float>(layers[2]->get_output_dims(), 8, conv_init));
	layers[4] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<float,3>(new MaxPoolLayer<float>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<float,3>(new DropoutLayer<float,3>(layers[5]->get_output_dims(), .25));
	layers[7] = LayerPtr<float,3>(new DenseKernelLayer<float,3>(layers[6]->get_output_dims(), 50, dense_init));
	layers[8] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[7]->get_output_dims()));
	layers[9] = LayerPtr<float,3>(new DropoutLayer<float,3>(layers[8]->get_output_dims(), .5));
	layers[10] = LayerPtr<float,3>(new DenseKernelLayer<float,3>(layers[9]->get_output_dims(), 10, dense_init));
	layers[11] = LayerPtr<float,3>(new SoftmaxActivationLayer<float,3>(layers[10]->get_output_dims()));
	FeedforwardNeuralNetwork<float,3> nn(std::move(layers));
	// Initialize.
	nn.init();
	// Specify the loss and the optimizer.
	auto loss = std::make_shared<CrossEntropyLoss<float,3,false>>();
	AdaDeltaOptimizer<float,3,false> opt(loss, 200);
	// Optimize the network and measure how long it takes.
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	opt.optimize(nn, file_train_prov, file_test_prov, 10);
	std::cout << "Training Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - begin).count() << std::endl;
	// Test how many images the network classifies correctly and measure the inference duration.
	begin = std::chrono::steady_clock::now();
	DataPair<float,3,false> data = file_test_prov.get_data(10000);
	Tensor<float,4> prediction = nn.infer(std::move(data.first));
	std::cout << "Inference Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - begin).count() << std::endl;
	unsigned wrong = 0;
	unsigned correct = 0;
	for (std::size_t i = 0; i < prediction.dimension(0); ++i) {
		float max = NumericUtils<float>::MIN;
		std::size_t max_ind = 0;
		for (std::size_t j = 0; j < prediction.dimension(1); ++j) {
			float val = prediction(i,j,0u,0u);
			if (val > max) {
				max = val;
				max_ind = j;
			}
		}
		if (data.second(i,max_ind,0u,0u) == 1)
			correct++;
		else
			wrong++;
	}
	std::cout << "Correct: " << correct << std::endl;
	std::cout << "Wrong: " << wrong << std::endl;
}
