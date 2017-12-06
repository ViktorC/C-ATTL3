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

enum class Regularizations {
	L1, L2, MaxNorm, Dropout
};

}

#endif /* REGULARIZATION_H_ */
