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
	virtual double compute(double objective, double out) = 0;
};

class QuadraticCost {
public:
	QuadraticCost();
	virtual ~QuadraticCost();
	virtual double compute(double objective, double out);
};

class CrossEntropyCost {
public:
	CrossEntropyCost();
	virtual ~CrossEntropyCost();
	virtual double compute(double objective, double out);
};

enum class Costs {
	Quadratic, CrossEntropy
};

Cost* get_cost(Costs type);

#endif /* COST_H_ */
