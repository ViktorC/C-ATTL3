/*
 * Dimensions.h
 *
 *  Created on: 13 Jan 2018
 *      Author: Viktor Csomor
 */

#ifndef DIMENSIONS_H_
#define DIMENSIONS_H_

#include <cassert>
#include <sstream>
#include <string>

namespace cppnn {

template<typename IndexType>
class Dimensions {
public:
	Dimensions(IndexType dim1, IndexType dim2, IndexType dim3) :
			dim1(dim1),
			dim2(dim2),
			dim3(dim3) {
		assert(dim1 > 0 && dim2 > 0 && dim3 > 0);
	};
	Dimensions(IndexType dim1, IndexType dim2) :
			Dimensions(dim1, dim2, 1) { };
	Dimensions(IndexType dim1) :
			Dimensions(dim1, 1) { };
	Dimensions() :
			Dimensions(1) { };
	IndexType get_dim1() const {
		return dim1;
	};
	IndexType get_dim2() const {
		return dim2;
	};
	IndexType get_dim3() const {
		return dim3;
	};
	IndexType get_points() const {
		return dim1 * dim2 * dim3;
	};
	bool equals(const Dimensions<IndexType>& dims) const {
		return dim1 == dims.dim1 && dim2 == dims.dim2 && dim3 == dims.dim3;
	};
	std::string to_string() const {
		return "[" + std::to_string(dim1) + ", " + std::to_string(dim2) + ", " +
				std::to_string(dim3) + "]";
	};
	friend std::ostream& operator<<(std::ostream& os, const Dimensions<IndexType>& dims) {
		return os << dims.to_string() << std::endl;
	};
private:
	IndexType dim1;
	IndexType dim2;
	IndexType dim3;
};

} /* namespace cppnn */

#endif /* DIMENSIONS_H_ */
