/*
 * GradientTest.h
 *
 *  Created on: 19.04.2018
 *      Author: Viktor Csomor
 */

#ifndef GRADIENTTEST_HPP_
#define GRADIENTTEST_HPP_

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>

#include "cattle/Cattle.hpp"

namespace cattle {
namespace test {

template<typename Scalar, std::size_t Rank, bool Sequential>
using DataProviderPtr = std::unique_ptr<DataProvider<Scalar,Rank,Sequential>>;
template<typename Scalar, std::size_t Rank, bool Sequential>
using NeuralNetPtr = std::unique_ptr<NeuralNetwork<Scalar,Rank,Sequential>>;
template<typename Scalar, std::size_t Rank, bool Sequential>
using OptimizerPtr = std::unique_ptr<Optimizer<Scalar,Rank,Sequential>>;

template<typename Scalar, std::size_t Rank>
TensorPtr<Scalar,Rank> random_tensor(const std::array<std::size_t,Rank>& dims) {
	TensorPtr<Scalar,Rank> tensor_ptr(new Tensor<Scalar,Rank>(dims));
	tensor_ptr->setRandom();
	return tensor_ptr;
}

template<typename Scalar, std::size_t Rank, bool Sequential>
class GradientTest : public ::testing::Test {
	typedef internal::NumericUtils<Scalar> NumUtils;
public:
	virtual ~GradientTest() = default;
	void perform(Scalar step_size = (NumUtils::EPSILON2 + NumUtils::EPSILON3) / 2,
			Scalar abs_epsilon = NumUtils::EPSILON2, Scalar rel_epsilon = NumUtils::EPSILON3) {
		ASSERT_TRUE(opt->verify_gradients(*net, *prov, step_size, abs_epsilon, rel_epsilon));
	}
protected:
	DataProviderPtr<Scalar,Rank,Sequential> prov;
	NeuralNetPtr<Scalar,Rank,Sequential> net;
	OptimizerPtr<Scalar,Rank,Sequential> opt;
};

template<typename Scalar, std::size_t Rank>
class LayerGradientTest :
		public GradientTest<Scalar,Rank,false>,
		public ::testing::WithParamInterface<LayerPtr<Scalar,Rank>> {
protected:
	virtual void setUp() {

	}
};

} /* namespace internal */
} /* namespace test */

#endif /* GRADIENTTEST_HPP_ */
