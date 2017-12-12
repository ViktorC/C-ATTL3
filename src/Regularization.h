/*
 * Regularization.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef REGULARIZATION_H_
#define REGULARIZATION_H_

namespace cppnn {

template<typename Scalar>
class Regularization {
public:
	Regularization();
	virtual ~Regularization() = default;
};

template<typename Scalar>
class L1Regularization : public Regularization<Scalar> {
public:
	virtual ~L1Regularization() = default;
};

template<typename Scalar>
class L2Regularization : public Regularization<Scalar> {
public:
	virtual ~L2Regularization() = default;
};

template<typename Scalar>
class MaxNormRegularization : public Regularization<Scalar> {
public:
	virtual ~MaxNormRegularization() = default;
};

template<typename Scalar>
class DropoutRegularization : public Regularization<Scalar> {
public:
	virtual ~DropoutRegularization() = default;
};

} /* namespace cppnn */

#endif /* REGULARIZATION_H_ */
