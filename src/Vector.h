/*
 * Vector.h
 *
 *  Created on: 9 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include <Eigen/Dense>

namespace cppnn {

template<typename Scalar>
using RowVector = Eigen::Matrix<Scalar,1,Eigen::Dynamic,Eigen::RowMajor,1,Eigen::Dynamic>;

template <typename Scalar>
using ColVector = Eigen::Matrix<Scalar,Eigen::Dynamic,1,Eigen::ColMajor,Eigen::Dynamic,1>;

} /* namespace cppnn */

#endif /* VECTOR_H_ */
