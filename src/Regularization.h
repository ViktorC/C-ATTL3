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
	virtual ~Regularization() = default;
};

template<typename Scalar>
class L1Regularization : public Regularization<Scalar> {
public:
};

template<typename Scalar>
class L2Regularization : public Regularization<Scalar> {
public:
};

template<typename Scalar>
class ElasticNetRegularization : public Regularization<Scalar> {
public:
};

} /* namespace cppnn */

#endif /* REGULARIZATION_H_ */
