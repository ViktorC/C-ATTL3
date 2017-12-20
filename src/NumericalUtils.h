/*
 * NumericUtils.h
 *
 *  Created on: 20.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NUMERICALUTILS_H_
#define NUMERICALUTILS_H_

#include <algorithm>

namespace cppnn {

template<typename Scalar>
bool almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = std::numeric_limits<Scalar>::epsilon(),
		Scalar rel_epsilon = std::numeric_limits<Scalar>::epsilon()) {
	Scalar diff = n1 - n2;
	if (diff <= abs_epsilon)
		return true;
	Scalar max = std::max(std::abs(n1), std::abs(n2));
	return diff <= max * rel_epsilon;
};
template<typename Scalar>
bool greater(Scalar n1, Scalar n2, Scalar abs_epsilon = std::numeric_limits<Scalar>::epsilon(),
		Scalar rel_epsilon = std::numeric_limits<Scalar>::epsilon()) {
	return n1 > n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
};
template<typename Scalar>
bool greater_or_almost_equal(Scalar n1, Scalar n2,
		Scalar abs_epsilon = std::numeric_limits<Scalar>::epsilon(),
		Scalar rel_epsilon = std::numeric_limits<Scalar>::epsilon()) {
	return n1 > n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
};
template<typename Scalar>
bool lesser(Scalar n1, Scalar n2, Scalar abs_epsilon = std::numeric_limits<Scalar>::epsilon(),
		Scalar rel_epsilon = std::numeric_limits<Scalar>::epsilon()) {
	return n1 < n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
};
template<typename Scalar>
bool lesser_or_almost_equal(Scalar n1, Scalar n2,
		Scalar abs_epsilon = std::numeric_limits<Scalar>::epsilon(),
		Scalar rel_epsilon = std::numeric_limits<Scalar>::epsilon()) {
	return n1 < n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
};

}

#endif /* NUMERICALUTILS_H_ */
