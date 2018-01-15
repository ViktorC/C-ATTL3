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
	Dimensions(int dim1, int dim2, int dim3);
	int get_dim1() const;
	int get_dim2() const;
	int get_dim3() const;
	int get_points() const;
	bool equals(const Dimensions& dims) const;
	std::string to_string() const;
private:
	int dim1;
	int dim2;
	int dim3;
};

} /* namespace cppnn */

#endif /* DIMENSIONS_H_ */
