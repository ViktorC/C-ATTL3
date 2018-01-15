/*
 * Dimensions.cpp
 *
 *  Created on: 13 Jan 2018
 *      Author: Viktor Csomor
 */

#include <cassert>
#include <Dimensions.h>
#include <string>

namespace cppnn {

Dimensions::Dimensions() :
		Dimensions(1, 1, 1) { };
Dimensions::Dimensions(unsigned dim1, unsigned dim2, unsigned dim3) :
		dim1(dim1),
		dim2(dim2),
		dim3(dim3) {
	assert(dim1 > 0 && dim2 > 0 && dim3 > 0);
};
unsigned Dimensions::get_dim1() const {
	return dim1;
};
unsigned Dimensions::get_dim2() const {
	return dim2;
};
unsigned Dimensions::get_dim3() const {
	return dim3;
};
unsigned Dimensions::get_points() const {
	return dim1 * dim2 * dim3;
};
bool Dimensions::equals(const Dimensions& dims) const {
	return dim1 == dims.dim1 && dim2 == dims.dim2 && dim3 == dims.dim3;
};
std::string Dimensions::to_string() const {
	return "[" + std::to_string(dim1) + ", " + std::to_string(dim2) + ", " +
			std::to_string(dim3) + "]";
};

} /* namespace cppnn */
