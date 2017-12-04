/*
 * Cost.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor
 */

#ifndef COST_H_
#define COST_H_

class Cost {
public:
	Cost();
	virtual ~Cost();
};

enum class Costs {
	Quadratic, CrossEntropy, SVM, Softmax
};

#endif /* COST_H_ */
