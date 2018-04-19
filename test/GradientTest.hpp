/*
 * GradientTest.h
 *
 *  Created on: 19.04.2018
 *      Author: Viktor Csomor
 */

#ifndef GRADIENTTEST_HPP_
#define GRADIENTTEST_HPP_

#include <gtest/gtest.h>
#include <memory>

#include "Cattle.hpp"

namespace cattle {
namespace internal {

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
	typedef std::unique_ptr<DataProvider<Scalar,Rank,Sequential>> DataProviderPtr;
	typedef std::unique_ptr<NeuralNetwork<Scalar,Rank,Sequential>> NeuralNetPtr;
	typedef std::unique_ptr<Optimizer<Scalar,Rank,Sequential>> OptimizerPtr;
	DataProviderPtr prov;
	NeuralNetPtr net;
	OptimizerPtr opt;
};

} /* namespace internal */
} /* namespace cattle */

#endif /* GRADIENTTEST_HPP_ */
