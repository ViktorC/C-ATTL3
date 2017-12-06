/*
 * Cost.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor
 */

#ifndef COST_H_
#define COST_H_

#include <vector>

namespace cppnn {

class Cost {
public:
	virtual ~Cost() { };
	virtual double function(int sample, std::vector<double> out, std::vector<double> obj) = 0;
	virtual double d_function(int sample, std::vector<double> out, double y) = 0;
};

class QuadraticCost : public Cost {
public:
	virtual ~QuadraticCost() { };
	virtual double function(int sample, std::vector<double> out, std::vector<double> obj);
	virtual double d_function(int sample, std::vector<double> out, double y);
};

class CrossEntropyCost : public Cost {
public:
	virtual ~CrossEntropyCost() { };
	virtual double function(int sample, std::vector<double> out, std::vector<double> obj);
	virtual double d_function(int sample, std::vector<double> out, double y);
};

enum class Costs {
	Quadratic, CrossEntropy
};

Cost* get_cost(Costs type);

}

#endif /* COST_H_ */
