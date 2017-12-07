/*
 * Regularization.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor
 */

#ifndef REGULARIZATION_H_
#define REGULARIZATION_H_

namespace cppnn {

class Regularization {
public:
	Regularization();
	virtual ~Regularization();
};

class L1Regularization : public Regularization {
public:
	L1Regularization();
	virtual ~L1Regularization();
};

class L2Regularization : public Regularization {
public:
	L2Regularization();
	virtual ~L2Regularization();
};

class MaxNormRegularization : public Regularization {
public:
	MaxNormRegularization();
	virtual ~MaxNormRegularization();
};

class DropoutRegularization : public Regularization {
public:
	DropoutRegularization();
	virtual ~DropoutRegularization();
};

}

#endif /* REGULARIZATION_H_ */
