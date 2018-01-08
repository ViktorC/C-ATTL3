/*
 * Utils.h
 *
 *  Created on: 20.12.2017
 *      Author: Viktor Csomor
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <limits>

namespace cppnn {

template<typename Scalar>
class Utils {
private:
	Utils() { };
public:
	static constexpr Scalar EPSILON1 = std::numeric_limits<Scalar>::epsilon();
	static constexpr Scalar EPSILON2 = 1e-8;
	static constexpr Scalar EPSILON3 = 1e-4;
	static bool almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		Scalar diff = std::abs(n1 - n2);
		if (diff <= abs_epsilon)
			return true;
		Scalar max = std::max(std::abs(n1), std::abs(n2));
		return diff <= max * rel_epsilon;
	};
	static bool decidedly_greater(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	static bool greater_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	bool decidedly_lesser(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
	static bool lesser_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	};
};

}

#endif /* UTILS_H_ */
