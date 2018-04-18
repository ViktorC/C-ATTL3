/*
 * std_dataset.cpp
 *
 *  Created on: 18 Apr 2018
 *      Author: Viktor
 */

#include <chrono>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "Cattle.hpp"

int main() {
//	std::string mnist_folder = "../data/mnist/";
//	MNISTDataProvider<float> file_train_prov(mnist_folder + "train-images.idx3-ubyte", mnist_folder + "train-labels.idx1-ubyte");
//	MNISTDataProvider<float> file_test_prov(mnist_folder + "t10k-images.idx3-ubyte", mnist_folder + "t10k-labels.idx1-ubyte");
	std::string cifar_folder = "../data/cifar10/";
	CIFARDataProvider<float> file_train_prov({ cifar_folder + "data_batch_1.bin", cifar_folder + "data_batch_2.bin",
			cifar_folder + "data_batch_3.bin", cifar_folder + "data_batch_4.bin", cifar_folder + "data_batch_5.bin",
			cifar_folder + "data_batch_6.bin", });
	CIFARDataProvider<float> file_test_prov(cifar_folder + "test_batch.bin");
	WeightInitSharedPtr<float> conv_init(new HeWeightInitialization<float>(1e-1));
	WeightInitSharedPtr<float> dense_init(new GlorotWeightInitialization<float>(1e-1));
	std::vector<LayerPtr<float,3>> layers(12);
	layers[0] = LayerPtr<float,3>(new ConvLayer<float>(file_train_prov.get_obs_dims(), 8, conv_init));
	layers[1] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<float,3>(new ConvLayer<float>(layers[2]->get_output_dims(), 8, conv_init));
	layers[4] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<float,3>(new DropoutLayer<float,3>(layers[5]->get_output_dims(), .25));
	layers[7] = LayerPtr<float,3>(new FCLayer<float,3>(layers[6]->get_output_dims(), 50, dense_init));
	layers[8] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[7]->get_output_dims()));
	layers[9] = LayerPtr<float,3>(new DropoutLayer<float,3>(layers[8]->get_output_dims(), .5));
	layers[10] = LayerPtr<float,3>(new FCLayer<float,3>(layers[9]->get_output_dims(), 10, dense_init));
	layers[11] = LayerPtr<float,3>(new SoftmaxActivationLayer<float,3>(layers[10]->get_output_dims()));
	FeedforwardNeuralNetwork<float,3> nn(std::move(layers));
	nn.init();
	LossSharedPtr<float,3,false> loss(new CrossEntropyLoss<float,3,false>());
	AdadeltaOptimizer<float,3,false> opt(loss, 200);
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	opt.optimize(nn, file_train_prov, file_test_prov, 10);
	std::cout << "Training Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - begin).count() << std::endl;
	begin = std::chrono::steady_clock::now();
	DataPair<float,3,false> data = file_test_prov.get_data(10000);
	Tensor<float,4> prediction = nn.infer(std::move(data.first));
	std::cout << "Inference Duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - begin).count() << std::endl;
	unsigned wrong = 0;
	unsigned correct = 0;
	for (std::size_t i = 0; i < prediction.dimension(0); ++i) {
		float max = internal::Utils<float>::MIN;
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
