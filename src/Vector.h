/*
 * Vector.h
 *
 *  Created on: 9 Dec 2017
 *      Author: Viktor
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include <Eigen/Dense>

namespace cppnn {

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar,1,Eigen::Dynamic,Eigen::RowMajor,1,Eigen::Dynamic>;

}

#endif /* VECTOR_H_ */
