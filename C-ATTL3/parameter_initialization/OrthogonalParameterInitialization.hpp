/*
 * OrthogonalParameterInitialization.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETER_INITIALIZATION_ORTHOGONALPARAMETERINITIALIZATION_H_
#define C_ATTL3_PARAMETER_INITIALIZATION_ORTHOGONALPARAMETERINITIALIZATION_H_

#include "GaussianParameterInitialization.hpp"

namespace cattle {

/**
 * A class template representing the orthogonal weight initialization method.
 *
 * \see https://arxiv.org/abs/1312.6120
 */
template<typename Scalar>
class OrthogonalParameterInitialization : public GaussianParameterInitialization<Scalar> {
public:
	/**
	 * @param sd The standard deviation of the normal distribution to sample from.
	 */
	inline OrthogonalParameterInitialization(Scalar sd = 1) :
			GaussianParameterInitialization<Scalar>::GaussianParameterInitialization(sd) { }
	inline void apply(Matrix<Scalar>& weights) const {
		GaussianParameterInitialization<Scalar>::apply(weights);
		int rows = weights.rows();
		int cols = weights.cols();
		bool more_rows = rows > cols;
		SVD<Scalar> svd;
		weights.block(0, 0, rows, cols) = more_rows ?
				svd.compute(weights, SVDOptions::ComputeFullU).matrixU().block(0, 0, rows, cols) :
				svd.compute(weights, SVDOptions::ComputeFullV).matrixV().block(0, 0, rows, cols);
	}
};

}

#endif /* C_ATTL3_PARAMETER_INITIALIZATION_ORTHOGONALPARAMETERINITIALIZATION_H_ */
