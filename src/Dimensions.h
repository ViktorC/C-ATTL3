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
#include <stdexcept>
#include <string>

namespace cattle {

template<typename IndexType>
class Dimensions {
public:
	Dimensions(IndexType height, IndexType width, IndexType depth) :
			height(height),
			width(width),
			depth(depth) {
		assert(height > 0 && width > 0 && depth > 0);
	};
	Dimensions(IndexType height, IndexType width) :
			Dimensions(height, width, 1) { };
	Dimensions(IndexType height) :
			Dimensions(height, 1) { };
	Dimensions() :
			Dimensions(1) { };
	inline IndexType get_height() const {
		return height;
	};
	inline IndexType get_width() const {
		return width;
	};
	inline IndexType get_depth() const {
		return depth;
	};
	inline IndexType get_volume() const {
		return height * width * depth;
	};
	inline bool equals(const Dimensions<IndexType>& dims) const {
		return height == dims.height && width == dims.width && depth == dims.depth;
	};
	inline bool equals(IndexType height, IndexType width, IndexType depth) {
		return this->height == height && this->width == width && this->depth == depth;
	};
	std::string to_string() const {
		return "[" + std::to_string(height) + ", " + std::to_string(width) + ", " +
				std::to_string(depth) + "]";
	};
	inline Dimensions<IndexType> add_along_depth(const Dimensions<IndexType> dims) {
		assert(height == dims.height && width == dims.width);
		return Dimensions<IndexType>(height, width, depth + dims.depth);
	};
	inline Dimensions<IndexType> subtract_along_depth(const Dimensions<IndexType> dims) {
		assert(height == dims.height && width == dims.width);
		return Dimensions<IndexType>(height, width, depth - dims.depth);
	};
	inline Dimensions<IndexType> operator+(const Dimensions<IndexType> dims) {
		return Dimensions<IndexType>(height + dims.height, width + dims.width, depth + dims.depth);
	};
	inline Dimensions<IndexType> operator-(const Dimensions<IndexType> dims) {
		return Dimensions<IndexType>(height - dims.height, width - dims.width, depth - dims.depth);
	};
	inline IndexType operator()(int i) {
		switch (i) {
			case 0:
				return height;
			case 1:
				return width;
			case 2:
				return depth;
			default:
				throw std::invalid_argument("illegal index value: " + i);
		}
	};
	friend std::ostream& operator<<(std::ostream& os, const Dimensions<IndexType>& dims) {
		return os << dims.to_string() << std::endl;
	};
private:
	IndexType height;
	IndexType width;
	IndexType depth;
};

} /* namespace cattle */

#endif /* DIMENSIONS_H_ */
