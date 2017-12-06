/*
 * Cost.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor
 */

#ifndef COST_H_
#define COST_H_

namespace cppnn {

class Cost {
public:
	virtual ~Cost() { };
	virtual double function(int sample, double* obj, int obj_len, double* out, int out_len) = 0;
	virtual double d_function(double* out, int out_len, double y) = 0;
};

class QuadraticCost : public Cost {
public:
	virtual ~QuadraticCost() { };
	virtual double function(int sample, double* obj, int obj_len, double* out, int out_len);
	virtual double d_function(double* out, int out_len, double y);
};

class CrossEntropyCost : public Cost {
public:
	virtual ~CrossEntropyCost() { };
	virtual double function(int sample, double* obj, int obj_len, double* out, int out_len);
	virtual double d_function(double* out, int out_len, double y);
};

enum class Costs {
	Quadratic, CrossEntropy
};

Cost* get_cost(Costs type);

}

#endif /* COST_H_ */
