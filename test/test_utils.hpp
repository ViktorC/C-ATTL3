/*
 * test_utils.hpp
 *
 *  Created on: 06.05.2018
 *      Author: Viktor Csomor
 */

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "Cattle.hpp"


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
 * @param dims The dimensions of the prospective tensor.
 * @return The number of samples.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
inline std::size_t get_rows(const typename std::enable_if<!Sequential,
		std::array<std::size_t,Rank>>::type& dims) {
	return dims[0];
}

/**
 * @param dims The dimensions of the prospective tensor.
 * @return The number of samples multiplied by the number of time steps.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
inline std::size_t get_rows(const typename std::enable_if<Sequential,
		std::array<std::size_t,Rank>>::type& dims) {
	return dims[0] * dims[1];
}

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
 * @param dims The dimensions of the random tensor to create.
 * @param non_label The value of the non-label elements of the objective tensor (0 for cross
 * entropy, -1 for hinge loss).
 * @return A one-hot tensor of the specified dimensions in which only one element per sample
 * and time step is 1 while all others equal non_label.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
inline TensorPtr<Scalar,Rank> random_one_hot_tensor(const std::array<std::size_t,Rank>& dims,
		Scalar non_label = 0) {
	TensorPtr<Scalar,Rank> tensor_ptr(new Tensor<Scalar,Rank>(dims));
	tensor_ptr->setConstant(non_label);
	int rows = get_rows<Scalar,Rank,Sequential>(dims);
	MatrixMap<Scalar> mat_map(tensor_ptr->data(), rows, tensor_ptr->size() / rows);
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_int_distribution<int> dist(0, mat_map.cols() - 1);
	for (int i = 0; i < mat_map.rows(); ++i)
		mat_map(i,dist(rng)) = 1;
	return tensor_ptr;
}

/**
 * @param dims The dimensions of the random tensor to create.
 * @param non_label The value of the non-label elements of the objective tensor (0 for cross
 * entropy, -1 for hinge loss).
 * @return A multi-label objective tensor whose elements equal either 1 or non_label.
 */
template<typename Scalar, std::size_t Rank>
inline TensorPtr<Scalar,Rank> random_multi_hot_tensor(const std::array<std::size_t,Rank>& dims,
		Scalar non_label = 0) {
	auto tensor_ptr = random_tensor<Scalar,Rank>(dims);
	Tensor<bool,Rank> if_tensor = (*tensor_ptr) > tensor_ptr->constant((Scalar) 0);
	Tensor<Scalar,Rank> then_tensor = tensor_ptr->constant((Scalar) 1);
	Tensor<Scalar,Rank> else_tensor = tensor_ptr->constant(non_label);
	*tensor_ptr = if_tensor.select(then_tensor, else_tensor);
	return tensor_ptr;
}

/**
 * An alias for a unique pointer to a data provider.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using DataProviderPtr = std::unique_ptr<DataProvider<Scalar,Rank,Sequential>>;

/**
 * @return A pair of data providers for the MNIST training and test data respectively.
 */
template<typename Scalar>
inline std::pair<DataProviderPtr<Scalar,3,false>,DataProviderPtr<Scalar,3,false>> mnist_provs() {
	std::string mnist_folder = "test/data/mnist/";
	DataProviderPtr<Scalar,3,false> train_prov(new MNISTDataProvider<Scalar>(mnist_folder + "train-images.idx3-ubyte",
			mnist_folder + "train-labels.idx1-ubyte"));
	DataProviderPtr<Scalar,3,false> test_prov(new MNISTDataProvider<Scalar>(mnist_folder + "t10k-images.idx3-ubyte",
			mnist_folder + "t10k-labels.idx1-ubyte"));
	return std::pair<DataProviderPtr<Scalar,3,false>,DataProviderPtr<Scalar,3,false>>(std::move(train_prov),
			std::move(test_prov));
}

/**
 * @return A pair of data providers for the CIFAR-10 training and test data respectively.
 */
template<typename Scalar>
inline std::pair<DataProviderPtr<Scalar,3,false>,DataProviderPtr<Scalar,3,false>> cifar10_provs() {
	std::string cifar_folder = "test/data/cifar10/";
	DataProviderPtr<Scalar,3,false> train_prov(new CIFARDataProvider<Scalar>({ cifar_folder + "data_batch_1.bin",
			cifar_folder + "data_batch_2.bin", cifar_folder + "data_batch_3.bin", cifar_folder + "data_batch_4.bin",
			cifar_folder + "data_batch_5.bin", cifar_folder + "data_batch_6.bin", }));
	DataProviderPtr<Scalar,3,false> test_prov(new CIFARDataProvider<Scalar>(cifar_folder + "test_batch.bin"));
	return std::pair<DataProviderPtr<Scalar,3,false>,DataProviderPtr<Scalar,3,false>>(std::move(train_prov),
			std::move(test_prov));
}

/**
 * @param input_dims The input dimensions of the layer.
 * @return A fully connected kernel layer.
 */
template<typename Scalar, std::size_t Rank>
inline KernelPtr<Scalar,Rank> kernel_layer(const typename std::enable_if<Rank != 3,
		Dimensions<std::size_t,Rank>>::type& input_dims) {
	return KernelPtr<Scalar,Rank>(new FCLayer<Scalar,Rank>(input_dims, 4,
			WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>())));
}

/**
 * @param input_dims The input dimensions of the layer.
 * @return A convolutional kernel layer.
 */
template<typename Scalar, std::size_t Rank>
inline KernelPtr<Scalar,Rank> kernel_layer(const typename std::enable_if<Rank == 3,
		Dimensions<std::size_t,Rank>>::type& input_dims) {
	return KernelPtr<Scalar,Rank>(new ConvLayer<Scalar>(input_dims, input_dims(2),
			WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>())));
}

/**
 * @param input_dims The input dimensions of the neural network.
 * @return A simple feed-forward neural network with unrestricted output values.
 */
template<typename Scalar, std::size_t Rank>
inline NeuralNetPtr<Scalar,Rank,false> reg_neural_net(const Dimensions<std::size_t,Rank>& input_dims) {
	std::vector<LayerPtr<Scalar,Rank>> layers(1);
	layers[0] = kernel_layer<Scalar,Rank>(input_dims);
	return NeuralNetPtr<Scalar,Rank,false>(new FeedforwardNeuralNetwork<Scalar,Rank>(std::move(layers)));
}

/**
 * @param input_dims The input dimensions of the neural network.
 * @return A simple feed-forward neural network with a sigmoid activation function and hence output values
 * between 0 and 1.
 */
template<typename Scalar, std::size_t Rank>
inline NeuralNetPtr<Scalar,Rank,false> sigmoid_neural_net(const Dimensions<std::size_t,Rank>& input_dims) {
	std::vector<LayerPtr<Scalar,Rank>> layers(2);
	layers[0] = kernel_layer<Scalar,Rank>(input_dims);
	layers[1] = LayerPtr<Scalar,Rank>(new SigmoidActivationLayer<Scalar,Rank>(layers[0]->get_output_dims()));
	return NeuralNetPtr<Scalar,Rank,false>(new FeedforwardNeuralNetwork<Scalar,Rank>(std::move(layers)));
}

/**
 * @param input_dims The input dimensions of the neural network.
 * @return A simple feed-forward neural network with a tanh activation function and hence output values
 * between -1 and 1.
 */
template<typename Scalar, std::size_t Rank>
inline NeuralNetPtr<Scalar,Rank,false> tanh_neural_net(const Dimensions<std::size_t,Rank>& input_dims) {
	std::vector<LayerPtr<Scalar,Rank>> layers(2);
	layers[0] = kernel_layer<Scalar,Rank>(input_dims);
	layers[1] = LayerPtr<Scalar,Rank>(new TanhActivationLayer<Scalar,Rank>(layers[0]->get_output_dims()));
	return NeuralNetPtr<Scalar,Rank,false>(new FeedforwardNeuralNetwork<Scalar,Rank>(std::move(layers)));
}

/**
 * @param input_dims The input dimensions of the neural network.
 * @return A simple feed-forward neural network with a softmax activation function and hence output values
 * between 0 and 1.
 */
template<typename Scalar, std::size_t Rank>
inline NeuralNetPtr<Scalar,Rank,false> softmax_neural_net(const Dimensions<std::size_t,Rank>& input_dims) {
	std::vector<LayerPtr<Scalar,Rank>> layers(2);
	layers[0] = kernel_layer<Scalar,Rank>(input_dims);
	layers[1] = LayerPtr<Scalar,Rank>(new SoftmaxActivationLayer<Scalar,Rank>(layers[0]->get_output_dims()));
	return NeuralNetPtr<Scalar,Rank,false>(new FeedforwardNeuralNetwork<Scalar,Rank>(std::move(layers)));
}

/**
 * @param test_case_name The name of the test case/suite.
 * @param test_name The name of the test.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
inline void print_test_header(std::string test_case_name, std::string test_name) {
	std::transform(test_case_name.begin(), test_case_name.end(), test_case_name.begin(), ::toupper);
	std::transform(test_name.begin(), test_name.end(), test_name.begin(), ::toupper);
	std::string header = "|   " + test_case_name + ": " + test_name + "; SCALAR TYPE: " +
			ScalarTraits<Scalar>::name() + "; RANK: " + std::to_string(Rank) +
			"; SEQ: " + std::to_string(Sequential) + "   |";
	std::size_t header_length = header.length();
	int padding_content_length = header_length - 2;
	std::string header_border = " " + std::string(padding_content_length, '-') + " ";
	std::string upper_header_padding = "/" + std::string(padding_content_length, ' ') + "\\";
	std::string lower_header_padding = "\\" + std::string(padding_content_length, ' ') + "/";
	std::cout << std::endl << header_border << std::endl << upper_header_padding << std::endl <<
			header << std::endl << lower_header_padding << std::endl << header_border << std::endl;
}

} /* namespace test */

} /* namespace cattle */

#endif /* TEST_UTILS_HPP_ */
