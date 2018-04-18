/*
 * test_framework.h
 *
 *  Created on: 05.04.2018
 *      Author: Viktor Csomor
 */

#ifndef TEST_FRAMEWORK_H_
#define TEST_FRAMEWORK_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>

#include "Cattle.hpp"

using namespace cattle;

template<typename Scalar, std::size_t Rank, bool Sequential>
using DataProviderPtr = std::unique_ptr<DataProvider<Scalar,Rank,Sequential>>;

template<typename Scalar, std::size_t Rank, bool Sequential>
using OptimizerPtr = std::unique_ptr<Optimizer<Scalar,Rank,Sequential>>;

template<typename Scalar, std::size_t Rank, bool Sequential>
using ProviderBuilder = DataProviderPtr<Scalar,Rank,Sequential> (*)(const Dimensions<std::size_t,Rank>&,
		const Dimensions<std::size_t,Rank>&);

template<typename Scalar, std::size_t Rank, bool Sequential>
using OptimizerBuilder = OptimizerPtr<Scalar,Rank,Sequential> (*)(LossSharedPtr<Scalar,Rank,Sequential>);

template<typename Scalar, std::size_t Rank, bool Sequential>
static void test_gradients(ProviderBuilder<Scalar,Rank,Sequential> prov_builder,
		OptimizerBuilder<Scalar,Rank,Sequential> opt_builder, NeuralNetPtr<Scalar,Rank,Sequential> net,
		LossSharedPtr<Scalar,Rank,Sequential> loss, Scalar step_size =
				(internal::NumericUtils<Scalar>::EPSILON2 + internal::NumericUtils<Scalar>::EPSILON3) / 2,
		Scalar abs_epsilon = internal::NumericUtils<Scalar>::EPSILON2,
		Scalar rel_epsilon = internal::NumericUtils<Scalar>::EPSILON3) {
	DataProviderPtr<Scalar,Rank,Sequential> prov = (*prov_builder)(net->get_input_dims(), net->get_output_dims());
	OptimizerPtr<Scalar,Rank,Sequential> opt = (*opt_builder)(loss);
	bool grad_check_pass = opt->verify_gradients(*net, *prov, step_size, abs_epsilon, rel_epsilon);
	assert(grad_check_pass && "gradient test failed");
}

template<typename Scalar, std::size_t Rank, bool Sequential>
static void test_learning(ProviderBuilder<Scalar,Rank,Sequential> train_prov_builder,
		ProviderBuilder<Scalar,Rank,Sequential> test_prov_builder, OptimizerBuilder<Scalar,Rank,Sequential> opt_builder,
		NeuralNetPtr<Scalar,Rank,Sequential> net, LossSharedPtr<Scalar,Rank,Sequential> loss,
		unsigned epochs, unsigned early_termination, Scalar max_error, Scalar min_error) {
	DataProviderPtr<Scalar,Rank,Sequential> train_prov = (*train_prov_builder)(net->get_input_dims(), net->get_output_dims());
	DataProviderPtr<Scalar,Rank,Sequential> test_prov = (*test_prov_builder)(net->get_input_dims(), net->get_output_dims());
	OptimizerPtr<Scalar,Rank,Sequential> opt = (*opt_builder)(loss);
	Scalar error = opt->optimize(*net, *train_prov, *test_prov, epochs, early_termination);
	assert(error >= min_error && error <= max_error && "optimization test failed");
}

#endif /* TEST_FRAMEWORK_H_ */
