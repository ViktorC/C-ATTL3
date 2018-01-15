/*
 * Tensor.h
 *
 *  Created on: 13 Jan 2018
 *      Author: Viktor Csomor
 */

#ifndef TENSOR_H_
#define TENSOR_H_

#include <unsupported/Eigen/CXX11/Tensor>

namespace cppnn {

template<typename Scalar>
using Tensor4D = Eigen::Tensor<Scalar,4,Eigen::ColMajor,int>;

} /* namespace cppnn */

#endif /* TENSOR_H_ */
