/*
 * mnist_autoencoder.cpp
 *
 *  Created on: 11 May 2018
 *      Author: Viktor Csomor
 */

#include <memory>
#include <utility>
#include <vector>

#include "Cattle.hpp"

int main() {
	using namespace cattle;
	// Load the MNIST data set into memory.
	std::string mnist_folder = "data/mnist/";
	MNISTDataProvider<float> file_train_prov(mnist_folder + "train-images.idx3-ubyte", mnist_folder + "train-labels.idx1-ubyte");
	MNISTDataProvider<float> file_test_prov(mnist_folder + "t10k-images.idx3-ubyte", mnist_folder + "t10k-labels.idx1-ubyte");
	TensorPtr<float,4> train_data(new Tensor<float,4>(file_train_prov.get_data(60000).first));
	// The objectives are the data instances themselves.
	TensorPtr<float,4> train_label(new Tensor<float,4>(*train_data));
	MemoryDataProvider<float,3,false> train_prov(std::move(train_data), std::move(train_label));
	// Do the same with the test data as well.
	TensorPtr<float,4> test_data(new Tensor<float,4>(file_test_prov.get_data(10000).first));
	TensorPtr<float,4> test_label(new Tensor<float,4>(*test_data));
	MemoryDataProvider<float,3,false> test_prov(std::move(test_data), std::move(test_label));
	// Create the auto-encoder.
	auto init = std::make_shared<HeWeightInitialization<float>>(1e-1);
	std::vector<LayerPtr<float,3>> encoder_layers(5);
	encoder_layers[0] = LayerPtr<float,3>(new ConvKernelLayer<float>(train_prov.get_obs_dims(), 3, init,
			ConvKernelLayer<float>::NO_PARAM_REG, 4, 4, 0, 0, 2, 2));
	encoder_layers[1] = LayerPtr<float,3>(new SoftplusActivationLayer<float,3>(encoder_layers[0]->get_output_dims()));
	encoder_layers[2] = LayerPtr<float,3>(new ConvKernelLayer<float>(encoder_layers[1]->get_output_dims(), 3, init,
			ConvKernelLayer<float>::NO_PARAM_REG, 4, 4, 0, 0, 1, 1));
	encoder_layers[3] = LayerPtr<float,3>(new SoftplusActivationLayer<float,3>(encoder_layers[2]->get_output_dims()));
	encoder_layers[4] = LayerPtr<float,3>(new DenseKernelLayer<float,3>(encoder_layers[2]->get_output_dims(), 100, init));
	NeuralNetPtr<float,3,false> encoder(new FeedforwardNeuralNetwork<float,3>(std::move(encoder_layers)));
	std::vector<LayerPtr<float,3>> decoder_layers(6);
	decoder_layers[0] = LayerPtr<float,3>(new DenseKernelLayer<float,3>(encoder->get_output_dims(), 300, init));
	decoder_layers[1] = LayerPtr<float,3>(new SoftplusActivationLayer<float,3>(decoder_layers[0]->get_output_dims()));
	decoder_layers[2] = LayerPtr<float,3>(new ReshapeLayer<float,3>(decoder_layers[1]->get_output_dims(), { 10u, 10u, 3u }));
	decoder_layers[3] = LayerPtr<float,3>(new DeconvKernelLayer<float>(decoder_layers[2]->get_output_dims(), 3, init,
			DeconvKernelLayer<float>::NO_PARAM_REG, 4, 4, 0, 0, 1, 1));
	decoder_layers[4] = LayerPtr<float,3>(new SoftplusActivationLayer<float,3>(decoder_layers[3]->get_output_dims()));
	decoder_layers[5] = LayerPtr<float,3>(new DeconvKernelLayer<float>(decoder_layers[4]->get_output_dims(), 1, init,
			DeconvKernelLayer<float>::NO_PARAM_REG, 4, 4, 0, 0, 2, 2));
	NeuralNetPtr<float,3,false> decoder(new FeedforwardNeuralNetwork<float,3>(std::move(decoder_layers)));
	std::vector<NeuralNetPtr<float,3,false>> modules;
	modules.push_back(std::move(encoder));
	modules.push_back(std::move(decoder));
	StackedNeuralNetwork<float,3,false> autoencoder(std::move(modules));
	autoencoder.init();
	// Specify the loss and the optimizer.
	auto loss = std::make_shared<SquaredLoss<float,3,false>>();
	NadamOptimizer<float,3,false> opt(loss, 1000);
	// Optimize.
	opt.optimize(autoencoder, train_prov, test_prov, 30);
	// Output some test image reconstructions.
	PPMCodec<float,P5> ppm_codec;
	std::vector<NeuralNetwork<float,3,false>*> module_ptrs = autoencoder.get_modules();
	for (int i = 0; i < 10; ++i) {
		Tensor<float,4> image = test_prov.get_data(1).first;
		ppm_codec.encode(TensorMap<float,3>(image.data(), { 28u, 28u, 1u }), std::string("image") + std::to_string(i) +
				std::string(".ppm"));
		Tensor<float,4> encoded_image = module_ptrs[0]->infer(std::move(image));
		Tensor<float,4> decoded_image = module_ptrs[1]->infer(std::move(encoded_image));
		ppm_codec.encode(TensorMap<float,3>(decoded_image.data(), { 28u, 28u, 1u }), std::string("image") + std::to_string(i) +
				std::string("_decoded.ppm"));
	}
}
