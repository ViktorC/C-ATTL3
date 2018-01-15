/*
 * Dimensions.h
 *
 *  Created on: 13 Jan 2018
 *      Author: Viktor Csomor
 */

#ifndef DIMENSIONS_H_
#define DIMENSIONS_H_

#include <string>

namespace cppnn {

class Dimensions {
public:
	Dimensions();
	Dimensions(unsigned dim1, unsigned dim2, unsigned dim3);
	unsigned get_dim1() const;
	unsigned get_dim2() const;
	unsigned get_dim3() const;
	unsigned get_points() const;
	bool equals(const Dimensions& dims) const;
	std::string to_string() const;
private:
	unsigned dim1;
	unsigned dim2;
	unsigned dim3;
};

} /* namespace cppnn */

#endif /* DIMENSIONS_H_ */
