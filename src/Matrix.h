/*
 * Matrix.h
 *
 *  Created on: 9 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <Eigen/Dense>

namespace cppnn {

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor,Eigen::Dynamic,Eigen::Dynamic>;

} /* namespace cppnn */

#endif /* MATRIX_H_ */
