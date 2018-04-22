/*
 * gradient_test.hpp
 *
 *  Created on: 19.04.2018
 *      Author: Viktor Csomor
 */

#ifndef GRADIENT_TEST_HPP_
#define GRADIENT_TEST_HPP_

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

#include "cattle/Cattle.hpp"

namespace cattle {
namespace test {

template<typename Scalar, std::size_t Rank, bool Sequential>
void grad_test(DataProvider<Scalar,Rank,Sequential>& prov, NeuralNetwork<Scalar,Rank,Sequential>& net,
		Optimizer<Scalar,Rank,Sequential>& opt, Scalar step_size, Scalar abs_epsilon, Scalar rel_epsilon) {
	ASSERT_TRUE(opt.verify_gradients(net, prov, step_size, abs_epsilon, rel_epsilon));
}

using namespace internal;

template<typename Scalar>
struct DefaultNumericValues {
	static constexpr Scalar step_size = (NumericUtils<Scalar>::EPSILON2 +
			NumericUtils<Scalar>::EPSILON3) / 2;
	static constexpr Scalar abs_epsilon = NumericUtils<Scalar>::EPSILON2;
	static constexpr Scalar rel_epsilon = NumericUtils<Scalar>::EPSILON3;
};

template<>
struct DefaultNumericValues<float> {
	static constexpr float step_size = 1e-2;
	static constexpr float abs_epsilon = 1e-2;
	static constexpr float rel_epsilon = 1e-2;
};

template<typename Scalar, std::size_t Rank>
TensorPtr<Scalar,Rank> random_tensor(const std::array<std::size_t,Rank>& dims) {
	TensorPtr<Scalar,Rank> tensor_ptr(new Tensor<Scalar,Rank>(dims));
	tensor_ptr->setRandom();
	return tensor_ptr;
}

template<typename Scalar, std::size_t Rank>
void layer_grad_test(LayerPtr<Scalar,Rank> layer1, LayerPtr<Scalar,Rank> layer2, std::size_t samples = 5,
		Scalar step_size = DefaultNumericValues<Scalar>::step_size,
		Scalar abs_epsilon = DefaultNumericValues<Scalar>::abs_epsilon,
		Scalar rel_epsilon = DefaultNumericValues<Scalar>::rel_epsilon) {
	assert(layer1->get_output_dims() == layer2->get_input_dims());
	std::array<std::size_t,Rank + 1> input_dims = layer1->get_input_dims().template promote<>();
	std::array<std::size_t,Rank + 1> output_dims = layer2->get_output_dims().template promote<>();
	TensorPtr<Scalar,Rank + 1> obs = random_tensor<Scalar,Rank + 1>(input_dims);
	TensorPtr<Scalar,Rank + 1> obj = random_tensor<Scalar,Rank + 1>(output_dims);
	MemoryDataProvider<Scalar,Rank,false> prov(std::move(obs), std::move(obj));
	std::vector<LayerPtr<Scalar,Rank>> layers(4);
	layers[0] = LayerPtr<Scalar,Rank>(new IdentityActivationLayer<Scalar,Rank>(layer1->get_input_dims()));
	layers[1] = std::move(layer1);
	layers[2] = LayerPtr<Scalar,Rank>(new IdentityActivationLayer<Scalar,Rank>(layers[1]->get_output_dims()));
	layers[3] = std::move(layer2);
	FeedforwardNeuralNetwork<Scalar,Rank> nn(std::move(layers));
	nn.init();
	auto loss = LossSharedPtr<Scalar,Rank,false>(new QuadraticLoss<Scalar,Rank,false>());
	VanillaSGDOptimizer<Scalar,Rank,false> opt(loss, samples);
	grad_test<Scalar,Rank,false>(prov, nn, opt, step_size, abs_epsilon, rel_epsilon);
}


TEST(gradient_test, layer_gradient_test) {
	WeightInitSharedPtr<float> init(new GlorotWeightInitialization<float>());
	LayerPtr<float,3> layer1(new FCLayer<float,3>(Dimensions<std::size_t,3>({ 4, 4, 2}), 16, init));
	LayerPtr<float,3> layer2(new FCLayer<float,3>(layer1->get_output_dims(), 4, init));
	layer_grad_test<float,3>(std::move(layer1), std::move(layer2));
}

} /* namespace test */
} /* namespace cattle */

#endif /* GRADIENT_TEST_HPP_ */
