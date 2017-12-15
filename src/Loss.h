/*
 * Cost.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <cassert>
#include <Vector.h>

namespace cppnn {

template<typename Scalar>
class Loss {
public:
	virtual ~Loss() = default;
	virtual Scalar function(RowVector<Scalar>& out, RowVector<Scalar>& obj) const = 0;
	virtual Scalar d_function(RowVector<Scalar>& out, Scalar error) const = 0;
};

template<typename Scalar>
class QuadraticLoss : public Loss<Scalar> {
public:
	virtual ~QuadraticLoss() = default;
	virtual Scalar function(RowVector<Scalar>& out, RowVector<Scalar>& obj) const {
		assert(out.size() == obj.size());
		return (out - obj).array().square().sum()/2;
	};
	virtual Scalar d_function(RowVector<Scalar>& out, Scalar error) const {
		return 0;
	}
};

template<typename Scalar>
class CrossEntropyLoss : public Loss<Scalar> {
public:
	virtual ~CrossEntropyLoss() = default;
	virtual Scalar function(RowVector<Scalar>& out, RowVector<Scalar>& obj) const;
	virtual Scalar d_function(RowVector<Scalar>& out, Scalar error) const;
};

} /* namespace cppnn */

#endif /* LOSS_H_ */
