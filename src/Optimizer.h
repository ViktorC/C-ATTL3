/*
 * Optimizer.h
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "NeuralNetwork.h"
#include "Cost.h"
#include "Regularization.h"

namespace cppnn {

class Optimizer {
public:
	Optimizer();
	virtual ~Optimizer();
};

}

#endif /* OPTIMIZER_H_ */
