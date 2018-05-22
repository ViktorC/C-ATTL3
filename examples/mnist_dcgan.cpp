/*
 * mnist_dcgan.cpp
 *
 *  Created on: 11 May 2018
 *      Author: Viktor Csomor
 */

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Cattle.hpp"

int main() {
	using namespace cattle;
	// Define the GAN training hyper-parameters.
	const unsigned epochs = 100;
	const unsigned k = 1;
	const unsigned m = 100;
	// Load the MNIST data labelled as 'real' into memory so that it can be pre-processed and shuffled.
	std::string mnist_folder = "data/mnist/";
	MNISTDataProvider<float> real_file_train_prov(mnist_folder + "train-images.idx3-ubyte", mnist_folder + "train-labels.idx1-ubyte");
	auto mnist_train_data = real_file_train_prov.get_data(60000).first;
	mnist_train_data = (mnist_train_data - mnist_train_data.constant(127.5)) / mnist_train_data.constant(127.5);
	Tensor<float,4> mnist_train_label = Tensor<float,4>(60000u, 1u, 1u, 1u).random() * .25f;
	mnist_train_label += mnist_train_label.constant(.95);
	MemoryDataProvider<float,3,false> real_train_prov(
			TensorPtr<float,4>(new Tensor<float,4>(std::move(mnist_train_data))),
			TensorPtr<float,4>(new Tensor<float,4>(std::move(mnist_train_label))));
	MNISTDataProvider<float> real_file_test_prov(mnist_folder + "t10k-images.idx3-ubyte", mnist_folder + "t10k-labels.idx1-ubyte");
	auto mnist_test_data = real_file_test_prov.get_data(10000).first;
	mnist_test_data = (mnist_test_data - mnist_test_data.constant(127.5)) / mnist_test_data.constant(127.5);
	Tensor<float,4> mnist_test_label = Tensor<float,4>(10000u, 1u, 1u, 1u).random() * .25f;
	mnist_test_label += mnist_test_label.constant(.95);
	MemoryDataProvider<float,3,false> real_test_prov(
			TensorPtr<float,4>(new Tensor<float,4>(std::move(mnist_test_data))),
			TensorPtr<float,4>(new Tensor<float,4>(std::move(mnist_test_label))));
	// Create the GAN.
	auto init = std::make_shared<HeWeightInitialization<float>>(1e-2);
	auto reg = std::make_shared<SquaredParameterRegularization<float>>();
	std::vector<LayerPtr<float,3>> generator_layers(14);
	generator_layers[0] = LayerPtr<float,3>(new DeconvolutionLayer<float>({ 1u, 1u, 100u }, 32, init, reg, 4, 4, 0, 0));
	generator_layers[1] = LayerPtr<float,3>(new PReLUActivationLayer<float,3>(generator_layers[0]->get_output_dims()));
	generator_layers[2] = LayerPtr<float,3>(new DropoutLayer<float,3>(generator_layers[1]->get_output_dims(), .2));
	generator_layers[3] = LayerPtr<float,3>(new DeconvolutionLayer<float>(generator_layers[2]->get_output_dims(), 16, init, reg, 4, 4, 1, 1, 2, 2));
	generator_layers[4] = LayerPtr<float,3>(new PReLUActivationLayer<float,3>(generator_layers[3]->get_output_dims()));
	generator_layers[5] = LayerPtr<float,3>(new DropoutLayer<float,3>(generator_layers[4]->get_output_dims(), .2));
	generator_layers[6] = LayerPtr<float,3>(new DeconvolutionLayer<float>(generator_layers[5]->get_output_dims(), 8, init, reg, 4, 4, 2, 2, 2, 2));
	generator_layers[7] = LayerPtr<float,3>(new PReLUActivationLayer<float,3>(generator_layers[6]->get_output_dims()));
	generator_layers[8] = LayerPtr<float,3>(new DropoutLayer<float,3>(generator_layers[7]->get_output_dims(), .2));
	generator_layers[9] = LayerPtr<float,3>(new DeconvolutionLayer<float>(generator_layers[8]->get_output_dims(), 4, init, reg, 4, 4, 1, 1, 2, 2));
	generator_layers[10] = LayerPtr<float,3>(new PReLUActivationLayer<float,3>(generator_layers[9]->get_output_dims()));
	generator_layers[11] = LayerPtr<float,3>(new DropoutLayer<float,3>(generator_layers[10]->get_output_dims(), .2));
	generator_layers[12] = LayerPtr<float,3>(new DeconvolutionLayer<float>(generator_layers[11]->get_output_dims(), 1, init));
	generator_layers[13] = LayerPtr<float,3>(new TanhActivationLayer<float,3>(generator_layers[12]->get_output_dims()));
	NeuralNetPtr<float,3,false> generator(new FeedforwardNeuralNetwork<float,3>(std::move(generator_layers)));
	std::vector<LayerPtr<float,3>> discriminator_layers(14);
	discriminator_layers[0] = LayerPtr<float,3>(new ConvolutionLayer<float>(generator->get_output_dims(), 4, init, reg, 4, 4, 1, 1, 2, 2));
	discriminator_layers[1] = LayerPtr<float,3>(new PReLUActivationLayer<float,3>(discriminator_layers[0]->get_output_dims()));
	discriminator_layers[2] = LayerPtr<float,3>(new BatchNormLayer<float,3>(discriminator_layers[1]->get_output_dims()));
	discriminator_layers[3] = LayerPtr<float,3>(new ConvolutionLayer<float>(discriminator_layers[2]->get_output_dims(), 8, init, reg, 4, 4, 2, 2, 2, 2));
	discriminator_layers[4] = LayerPtr<float,3>(new PReLUActivationLayer<float,3>(discriminator_layers[3]->get_output_dims()));
	discriminator_layers[5] = LayerPtr<float,3>(new BatchNormLayer<float,3>(discriminator_layers[4]->get_output_dims()));
	discriminator_layers[6] = LayerPtr<float,3>(new ConvolutionLayer<float>(discriminator_layers[5]->get_output_dims(), 16, init, reg, 4, 4, 1, 1, 2, 2));
	discriminator_layers[7] = LayerPtr<float,3>(new PReLUActivationLayer<float,3>(discriminator_layers[6]->get_output_dims()));
	discriminator_layers[8] = LayerPtr<float,3>(new BatchNormLayer<float,3>(discriminator_layers[7]->get_output_dims()));
	discriminator_layers[9] = LayerPtr<float,3>(new ConvolutionLayer<float>(discriminator_layers[8]->get_output_dims(), 32, init, reg, 4, 4, 0, 0));
	discriminator_layers[10] = LayerPtr<float,3>(new PReLUActivationLayer<float,3>(discriminator_layers[9]->get_output_dims()));
	discriminator_layers[11] = LayerPtr<float,3>(new BatchNormLayer<float,3>(discriminator_layers[10]->get_output_dims()));
	discriminator_layers[12] = LayerPtr<float,3>(new DenseLayer<float,3>(discriminator_layers[11]->get_output_dims(), 1, init));
	discriminator_layers[13] = LayerPtr<float,3>(new SigmoidActivationLayer<float,3>(discriminator_layers[12]->get_output_dims()));
	NeuralNetPtr<float,3,false> discriminator(new FeedforwardNeuralNetwork<float,3>(std::move(discriminator_layers)));
	std::vector<NeuralNetPtr<float,3,false>> modules(2);
	modules[0] = std::move(generator);
	modules[1] = std::move(discriminator);
	StackedNeuralNetwork<float,3,false> gan(std::move(modules));
	gan.init();
	// Specify the loss and the two optimizers.
	auto loss = std::make_shared<BinaryCrossEntropyLoss<float,3,false>>();
	auto neg_loss = std::make_shared<NegatedLoss<float,3,false>>(loss);
	AdamOptimizer<float,3,false> disc_opt(loss, m, 2e-4, .5);
	AdamOptimizer<float,3,false> gen_opt(neg_loss, m, 2e-4, .5);
	PPMCodec<float,P2> ppm_codec;
	// Execute the GAN optimization algorithm.
	for (unsigned i = 0; i <= epochs; ++i) {
		std::string epoch_header = "*   GAN Epoch " + std::to_string(i) + "   *";
		std::cout << std::endl << std::string(epoch_header.length(), '*') << std::endl <<
				"*" << std::string(epoch_header.length() - 2, ' ') << "*" << std::endl <<
				epoch_header << std::endl <<
				"*" << std::string(epoch_header.length() - 2, ' ') << "*" << std::endl <<
				std::string(epoch_header.length(), '*') << std::endl;
		float disc_train_loss = 0;
		float gen_train_loss = 0;
		std::size_t total_m = 0;
		real_train_prov.reset();
		while (real_train_prov.has_more()) {
			// First optimize the discriminator by using a batch of generated and a batch of real images.
			auto real_train_data = real_train_prov.get_data(k * m);
			std::size_t actual_m = real_train_data.first.dimension(0);
			total_m += actual_m;
			Tensor<float,4> train_data = gan.get_modules()[0]->infer(Tensor<float,4>(actual_m, 1u, 1u, 100u).template random<NormalRandGen<float>>())
					.concatenate(std::move(real_train_data.first), 0);
			Tensor<float,4> train_label = Tensor<float,4>(actual_m, 1u, 1u, 1u).random() * .15f;
			train_label += train_label.constant(.15);
			train_label = Tensor<float,4>(train_label.concatenate(std::move(real_train_data.second), 0));
			MemoryDataProvider<float,3,false,false> disc_train_prov(
					TensorPtr<float,4>(new Tensor<float,4>(std::move(train_label))),
					TensorPtr<float,4>(new Tensor<float,4>(std::move(train_label))));
			disc_train_loss += disc_opt.train(*gan.get_modules()[1], disc_train_prov, 1, false) * actual_m;
			// Then optimize the generator by maximizing the GAN's loss w.r.t. the parameters of the generator.
			MemoryDataProvider<float,3,false,false> gen_train_prov(
					TensorPtr<float,4>(new Tensor<float,4>(Tensor<float,4>(actual_m, 1u, 1u, 100u).template random<NormalRandGen<float>>())),
					TensorPtr<float,4>(new Tensor<float,4>(Tensor<float,4>(actual_m, 1u, 1u, 1u).constant(0))));
			gan.get_modules()[1]->set_frozen(true);
			gen_train_loss += gen_opt.train(gan, gen_train_prov, 1, false) * actual_m;
			gan.get_modules()[1]->set_frozen(false);
		}
		std::cout << "\tdiscriminator training loss: " << (disc_train_loss / total_m) << std::endl;
		std::cout << "\tgenerator training loss: " << (gen_train_loss / total_m) << std::endl;
		// Perform tests on the GAN.
		real_test_prov.reset();
		auto real_test_data = real_test_prov.get_data();
		std::size_t actual_m = real_test_data.first.dimension(0);
		Tensor<float,4> test_data = gan.get_modules()[0]->infer(Tensor<float,4>(actual_m, 1u, 1u, 100u).template random<NormalRandGen<float>>())
				.concatenate(std::move(real_test_data.first), 0);
		Tensor<float,4> test_label = Tensor<float,4>(actual_m, 1u, 1u, 1u).random() * .15f;
		test_label += test_label.constant(.15);
		test_label = Tensor<float,4>(test_label.concatenate(std::move(real_test_data.second), 0));
		MemoryDataProvider<float,3,false> disc_test_prov(
				TensorPtr<float,4>(new Tensor<float,4>(std::move(test_data))),
				TensorPtr<float,4>(new Tensor<float,4>(std::move(test_label))));
		float disc_test_loss = disc_opt.test(*gan.get_modules()[1], disc_test_prov, false);
		std::cout << "\tdiscriminator test loss: " << disc_test_loss << std::endl;
		MemoryDataProvider<float,3,false,false> gen_test_prov(
				TensorPtr<float,4>(new Tensor<float,4>(Tensor<float,4>(actual_m, 1u, 1u, 100u).template random<NormalRandGen<float>>())),
				TensorPtr<float,4>(new Tensor<float,4>(Tensor<float,4>(actual_m, 1u, 1u, 1u).constant(0))));
		float gen_test_loss = gen_opt.test(gan, gen_test_prov, false);
		std::cout << "\tgenerator test loss: " << gen_test_loss << std::endl;
		// Generate some fake image samples every once in a while to track progress.
		for (unsigned j = 0; j < 10; ++j) {
			Tensor<float,4> fake_image_sample = gan.get_modules()[0]->infer(Tensor<float,4>(1u, 1u, 1u, 100u).template random<NormalRandGen<float>>());
			fake_image_sample = (fake_image_sample + fake_image_sample.constant(1)) * fake_image_sample.constant(127.5);
			ppm_codec.encode(TensorMap<float,3>(fake_image_sample.data(), { 28u, 28u, 1u }), std::string("image") +
					std::to_string(i) + std::string("_") + std::to_string(j) + std::string(".ppm"));
		}
	}
}
