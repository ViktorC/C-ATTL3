/*
 * Cost.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <cassert>
#include <Matrix.h>
#include <string>
#include <Vector.h>

namespace cppnn {

static const std::string INCOMPATIBLE_DIM_ERR_MSG = "incompatible out and object matrix dimensions";

template<typename Scalar>
class Loss {
public:
	virtual ~Loss() = default;
	virtual ColVector<Scalar> function(Matrix<Scalar>& out, Matrix<Scalar>& obj) const = 0;
	virtual ColVector<Scalar> d_function(Matrix<Scalar>& out, Scalar error) const = 0;
};

template<typename Scalar>
class QuadraticLoss : public Loss<Scalar> {
public:
	virtual ~QuadraticLoss() = default;
	virtual ColVector<Scalar> function(Matrix<Scalar>& out, Matrix<Scalar>& obj) const {
		assert(out.rows() == obj.rows() && out.cols() == obj.cols() &&
				&INCOMPATIBLE_DIM_ERR_MSG);
		return (out - obj).array().square().rowwise().sum();
	};
	virtual ColVector<Scalar> d_function(Matrix<Scalar>& out, Scalar error) const {
		return 0;
	}
};

template<typename Scalar>
class CrossEntropyLoss : public Loss<Scalar> {
public:
	virtual ~CrossEntropyLoss() = default;
	virtual ColVector<Scalar> function(Matrix<Scalar>& out, Matrix<Scalar>& obj) const;
	virtual ColVector<Scalar> d_function(Matrix<Scalar>& out, Scalar error) const;
};

} /* namespace cppnn */

#endif /* LOSS_H_ */
