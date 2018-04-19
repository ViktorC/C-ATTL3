/*
 * LossGradientTest.hpp
 *
 *  Created on: 19 Apr 2018
 *      Author: Viktor Csomor
 */

#ifndef LOSSGRADIENTTEST_HPP_
#define LOSSGRADIENTTEST_HPP_

#include <cstddef>

#include "GradientTest.hpp"

namespace cattle {
namespace internal {

template<typename Scalar, std::size_t Rank, bool Sequential>
class LossGradientTest: public GradientTest<Scalar,Rank,Sequential> {

};

} /* namespace internal */
} /* namespace cattle */

#endif /* LOSSGRADIENTTEST_HPP_ */
