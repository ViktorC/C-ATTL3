/*
 * Cost.cpp
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor
 */

#include <assert.h>
#include <detail/matrix_def.hpp>
#include <detail/vector_def.hpp>
#include <linalg/matrix_operations.hpp>
#include <linalg/vector_operations.hpp>
#include <Loss.h>
#include <math.h>
#include <matrix.hpp>

namespace cppnn {

double QuadraticLoss::function(std::vector<double>& out, std::vector<double>& obj) {
	assert(out.size() == obj.size());
	double loss = 0;
	for (unsigned i = 0; i < out.size(); i++) {
		double diff = out[i] - obj[i];
		loss += diff * diff;
	}
	return loss/2;
};
double QuadraticLoss::d_function(std::vector<double> out, double y) {
	return 0;
};

}
