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
#include "DataProvider.h"
#include "Dimensions.h"
#include "NeuralNetwork.h"
#include "Optimizer.h"
#include "utils/Eigen.h"
#include "utils/NumericUtils.h"

using namespace cattle;

template<typename Scalar, std::size_t Rank, bool Sequential>
using DataProviderPtr = std::unique_ptr<DataProvider<Scalar,Rank,Sequential>>;

template<typename Scalar, std::size_t Rank, bool Sequential>
using OptimizerPtr = std::unique_ptr<Optimizer<Scalar,Rank,Sequential>>;

template<typename Scalar, std::size_t Rank, bool Sequential>
using ProviderBuilder = DataProviderPtr<Scalar,Rank,Sequential> (*)(const Dimensions<std::size_t,Rank>&,
		const Dimensions<std::size_t,Rank>&);

template<typename Scalar, std::size_t Rank, bool Sequential>
using NetworkBuilder = NeuralNetPtr<Scalar,Rank,Sequential> (*)();

template<typename Scalar, std::size_t Rank, bool Sequential>
using OptimizerBuilder = OptimizerPtr<Scalar,Rank,Sequential> (*)();

template<std::size_t Rank, bool Sequential>
static std::array<std::size_t,Rank + Sequential + 1> default_dims(const Dimensions<std::size_t,Rank>& nominal_dims) {
	std::array<std::size_t,Rank + 1> obs_dims = nominal_dims.template promote<>();
	obs_dims[0] = 10;
	return obs_dims;
}

template<std::size_t Rank>
static std::array<std::size_t,Rank + true + 1> default_dims(const Dimensions<std::size_t,Rank>& nominal_dims) {
	std::array<std::size_t,Rank + 2> obs_dims = nominal_dims.template promote<2>();
	obs_dims[0] = 5;
	obs_dims[1] = 5;
	return obs_dims;
}

template<typename Scalar, std::size_t Rank, bool Sequential>
static DataProviderPtr<Scalar,Rank,Sequential> default_provider(const Dimensions<std::size_t,Rank>& nominal_obs_dims,
		const Dimensions<std::size_t,Rank>& nominal_obj_dims) {
	TensorPtr<Scalar,Rank + Sequential + 1> test_obs_ptr(new Tensor<Scalar,Rank + Sequential + 1>(default_dims<Rank,Sequential>(nominal_obs_dims)));
	TensorPtr<Scalar,Rank + Sequential + 1> test_obj_ptr(new Tensor<Scalar,Rank + Sequential + 1>(default_dims<Rank,Sequential>(nominal_obj_dims)));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	return std::unique_ptr<DataProvider<Scalar,Rank,Sequential>>(new MemoryDataProvider<Scalar,Rank,Sequential>(std::move(test_obs_ptr),
			std::move(test_obj_ptr)));
}

template<std::size_t Rank>
static Dimensions<std::size_t,Rank> default_nominal_input_dims() {
	return Dimensions<std::size_t,1>({ 50u });
}

template<>
static Dimensions<std::size_t,2> default_nominal_input_dims() {
	return Dimensions<std::size_t,2>({ 10u, 10u });
}

template<>
static Dimensions<std::size_t,3> default_nominal_input_dims() {
	return Dimensions<std::size_t,3>({ 10u, 10u, 3u });
}

template<typename Scalar, std::size_t Rank, bool Sequential>
static NeuralNetPtr<Scalar,Rank,Sequential> default_network() {
	return NeuralNetPtr<Scalar,Rank,Sequential>(new FeedforwardNeuralNetwork<Scalar,Rank>(LayerPtr<Scalar,Rank>(
			new FCLayer<Scalar,Rank>(default_nominal_input_dims<Rank>(), 1, WeightInitSharedPtr<Scalar>(new GlorotWeightInitialization<Scalar>())))));
}

template<typename Scalar, std::size_t Rank>
static NeuralNetPtr<Scalar,Rank,true> default_network() {
	return NeuralNetPtr<Scalar,Rank,true>(new SequentialNeuralNetwork<Scalar,Rank>(default_network<Scalar,Rank,false>()));
}

template<typename Scalar, std::size_t Rank, bool Sequential>
static std::unique_ptr<Optimizer<Scalar,Rank,Sequential>> default_optimizer() {
	return OptimizerPtr<Scalar,Rank,Sequential>(new VanillaSGDOptimizer<Scalar,Rank,Sequential>(LossSharedPtr<Scalar,Rank,Sequential>(
			new QuadraticLoss<Scalar,Rank,Sequential>()), ParamRegSharedPtr<Scalar>(new L1ParameterRegularization<Scalar>())));
}

template<typename Scalar, std::size_t Rank, bool Sequential>
static void check_gradient(ProviderBuilder<Scalar,Rank,Sequential> prov_builder = &default_provider<Scalar,Rank,Sequential>,
		NetworkBuilder<Scalar,Rank,Sequential> net_builder = &default_network<Scalar,Rank,Sequential>,
		OptimizerBuilder<Scalar,Rank,Sequential> opt_builder = &default_optimizer<Scalar,Rank,Sequential>,
		Scalar step_size = internal::NumericUtils<Scalar>::EPSILON2, Scalar abs_epsilon = internal::NumericUtils<Scalar>::EPSILON2,
		Scalar rel_epsilon = internal::NumericUtils<Scalar>::EPSILON3) {
	NeuralNetPtr<Scalar,Rank,Sequential> net = (*net_builder)();
	DataProviderPtr<Scalar,Rank,Sequential> prov = (*prov_builder)(net->get_input_dims(), net->get_output_dims());
	OptimizerPtr<Scalar,Rank,Sequential> opt = (*opt_builder)();
	assert(opt->verify_gradients(*net, *prov, step_size, abs_epsilon, rel_epsilon) && "gradient check failed");
}

#endif /* TEST_FRAMEWORK_H_ */
