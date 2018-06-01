/*
 * optimization_test.hpp
 *
 *  Created on: 06.05.2018
 *      Author: Viktor Csomor
 */

#ifndef OPTIMIZATION_TEST_HPP_
#define OPTIMIZATION_TEST_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Cattle.hpp"
#include "test_utils.hpp"


namespace cattle {
namespace test {

/**
 * Determines the verbosity of the gradient tests.
 */
extern bool verbose;

template<typename Scalar, std::size_t Rank, bool Sequential>
inline void opt_test(std::string name, DataProvider<Scalar,Rank,Sequential>& train_prov,
		DataProvider<Scalar,Rank,Sequential>& test_prov, NeuralNetwork<Scalar,Rank,Sequential>& net,
		Optimizer<Scalar,Rank,Sequential> opt, unsigned epochs, unsigned early_stop = 0) {
	print_test_header<Scalar,Rank,Sequential>("optimization test", name);
	net.init();
	Scalar orig_loss = opt.test(net, test_prov, verbose);
	Scalar opt_loss = opt.optimize(net, train_prov, test_prov, epochs, early_stop, verbose);
	EXPECT_LT(opt_loss, orig_loss);
}

template<typename Scalar, std::size_t Rank, bool Sequential>
inline void train_test(std::string name, DataProvider<Scalar,Rank,Sequential>& train_prov,
		NeuralNetwork<Scalar,Rank,Sequential>& net, Optimizer<Scalar,Rank,Sequential> opt,
		unsigned epochs, unsigned early_stop = 0, Scalar abs_epsilon = ScalarTraits<Scalar>::abs_epsilon,
		Scalar rel_epsilon = ScalarTraits<Scalar>::rel_epsilon) {
	print_test_header<Scalar,Rank,Sequential>("training test", name);
	net.init();
	Scalar loss = opt.train(net, train_prov, epochs, early_stop, verbose);
	EXPECT_TRUE(internal::NumericUtils::almost_equal((Scalar) 0, loss, abs_epsilon, rel_epsilon));
}

template<typename Scalar, std::size_t Rank>
inline void ff_train_test(std::string name, Optimizer<Scalar,Rank,false> opt,
		unsigned epochs, unsigned early_stop = 0) {

}



} /* namespace test */
} /* namespace cattle */

#endif /* OPTIMIZATION_TEST_HPP_ */
