/*
 * NumericUtils.h
 *
 *  Created on: 20.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTLE_UTILS_NUMERICUTILS_H_
#define CATTLE_UTILS_NUMERICUTILS_H_

#include <algorithm>
#include <limits>
#include <type_traits>

namespace cattle {
namespace internal {

/**
 * A utility class template containing static methods and variables to help with
 * numerical issues.
 */
template<typename Scalar>
class NumericUtils {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
private:
	NumericUtils() { }
public:
	static constexpr Scalar MIN = std::numeric_limits<Scalar>::lowest();
	static constexpr Scalar MAX = std::numeric_limits<Scalar>::max();
	static constexpr Scalar EPSILON1 = std::numeric_limits<Scalar>::epsilon();
	static constexpr Scalar EPSILON2 = 1e-8;
	static constexpr Scalar EPSILON3 = 1e-4;
	/**
	 * Returns whether the two numerals are close enough to be considered equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the two numerals can be considered equal.
	 */
	inline static bool almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		Scalar diff = std::abs(n1 - n2);
		if (diff <= abs_epsilon)
			return true;
		Scalar max = std::max(std::abs(n1), std::abs(n2));
		return diff <= max * rel_epsilon;
	}
	/**
	 * Returns whether a numeral is greater than another one by a margin great enough for
	 * them not to be considered almost equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the the first numeral is sufficiently greater than the second.
	 */
	inline static bool decidedly_greater(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	}
	/**
	 * Returns whether a numeral is not lesser than another one by a margin great enough for
	 * them not to be considered almost equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the the first numeral is sufficiently great enough not to be considered
	 * decidedly smaller than the second.
	 */
	inline static bool greater_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 > n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	}
	/**
	 * Returns whether a numeral is lesser than another one by a margin great enough for
	 * them not to be considered almost equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the the first numeral is sufficiently lesser than the second.
	 */
	inline static bool decidedly_lesser(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 && !almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	}
	/**
	 * Returns whether a numeral is not greater than another one by a margin great enough for
	 * them not to be considered almost equal.
	 *
	 * @param n1 The first numeral.
	 * @param n2 The second numeral.
	 * @param abs_epsilon The maximum absolute difference that would still allow
	 * them to be considered equal.
	 * @param rel_epsilon The maximum relative (to the greater numeral of the two)
	 * difference that would still allow them to be considered equal.
	 * @return Whether the the first numeral is sufficiently small enough not to be considered
	 * decidedly greater than the second.
	 */
	inline static bool lesser_or_almost_equal(Scalar n1, Scalar n2, Scalar abs_epsilon = EPSILON1,
			Scalar rel_epsilon = EPSILON1) {
		return n1 < n2 || almost_equal(n1, n2, abs_epsilon, rel_epsilon);
	}
};

}
}

#endif /* CATTLE_UTILS_NUMERICUTILS_H_ */
