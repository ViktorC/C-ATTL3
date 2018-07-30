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
 * An alias for a unique pointer to an optimizer.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using OptimizerPtr = std::unique_ptr<Optimizer<Scalar,Rank,Sequential>>;

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
 * @param input_dims The input dimensions of the layer.
 * @return A fully connected kernel layer.
 */
template<typename Scalar, std::size_t Rank>
inline KernelPtr<Scalar,Rank> kernel_layer(const typename std::enable_if<Rank != 1,
		Dimensions<std::size_t,Rank>>::type& input_dims) {
	return KernelPtr<Scalar,Rank>(new ConvKernelLayer<Scalar,Rank>(
			input_dims, input_dims.template extend<3 - Rank>()(2),
			std::make_shared<HeParameterInitialization<Scalar>>()));
}

/**
 * @param input_dims The input dimensions of the layer.
 * @return A convolutional kernel layer.
 */
template<typename Scalar, std::size_t Rank>
inline KernelPtr<Scalar,Rank> kernel_layer(const typename std::enable_if<Rank == 1,
		Dimensions<std::size_t,Rank>>::type& input_dims) {
	return KernelPtr<Scalar,Rank>(new DenseKernelLayer<Scalar,Rank>(
			input_dims, input_dims.get_volume(),
			std::make_shared<GlorotParameterInitialization<Scalar>>()));
}

/**
 * @param input_dims The input dimensions of the neural network.
 * @return A simple feed-forward neural network with unrestricted output values.
 */
template<typename Scalar, std::size_t Rank>
inline NeuralNetPtr<Scalar,Rank,false> neural_net(const Dimensions<std::size_t,Rank>& input_dims) {
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
 * @param input_dims The input dimensions of the neural network.
 * @return A simple recurrent neural network without an identity output activation function and with a
 * single output time step.
 */
template<typename Scalar, std::size_t Rank>
inline NeuralNetPtr<Scalar,Rank,true> recurrent_neural_net(const Dimensions<std::size_t,Rank>& input_dims,
		std::function<std::pair<std::size_t,std::size_t>(std::size_t)> seq_schedule_func) {
	return NeuralNetPtr<Scalar,Rank,true>(new RecurrentNeuralNetwork<Scalar,Rank>(kernel_layer<Scalar,Rank>(input_dims),
			kernel_layer<Scalar,Rank>(input_dims), kernel_layer<Scalar,Rank>(input_dims),
			ActivationPtr<Scalar,Rank>(new TanhActivationLayer<Scalar,Rank>(input_dims)),
			ActivationPtr<Scalar,Rank>(new IdentityActivationLayer<Scalar,Rank>(input_dims)), seq_schedule_func));
}

/**
 * @param input_dims The input dimensions of the neural network.
 * @return A simple single-output recurrent neural network without an identity output activation function
 * and with a single output time step.
 */
template<typename Scalar, std::size_t Rank>
inline NeuralNetPtr<Scalar,Rank,true> single_output_recurrent_neural_net(const Dimensions<std::size_t,Rank>& input_dims,
		std::function<std::pair<std::size_t,std::size_t>(std::size_t)> seq_schedule_func) {
	auto init = std::make_shared<GlorotParameterInitialization<Scalar>>();
	return NeuralNetPtr<Scalar,Rank,true>(new RecurrentNeuralNetwork<Scalar,Rank>(kernel_layer<Scalar,Rank>(input_dims),
			kernel_layer<Scalar,Rank>(input_dims), KernelPtr<Scalar,Rank>(new DenseKernelLayer<Scalar,Rank>(input_dims, 1, init)),
			ActivationPtr<Scalar,Rank>(new TanhActivationLayer<Scalar,Rank>(input_dims)),
			ActivationPtr<Scalar,Rank>(new IdentityActivationLayer<Scalar,Rank>(Dimensions<std::size_t,Rank>())),
			seq_schedule_func));
}

/**
 * @param net The first module of the composite single-output network.
 * @return A stacked network with an output dimensionality of one.
 */
template<typename Scalar, std::size_t Rank>
inline NeuralNetPtr<Scalar,Rank,false> single_output_net(NeuralNetPtr<Scalar,Rank,false> net) {
	auto init = std::make_shared<GlorotParameterInitialization<Scalar>>();
	std::vector<NeuralNetPtr<Scalar,Rank,false>> modules(2);
	modules[0] = std::move(net);
	modules[1] = NeuralNetPtr<Scalar,Rank,false>(new FeedforwardNeuralNetwork<Scalar,Rank>(
			LayerPtr<Scalar,Rank>(new DenseKernelLayer<Scalar,Rank>(modules[0]->get_output_dims(), 1, init))));
	return NeuralNetPtr<Scalar,Rank,false>(new StackedNeuralNetwork<Scalar,Rank,false>(std::move(modules)));
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
