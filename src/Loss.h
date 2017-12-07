/*
 * Cost.h
 *
 *  Created on: 4 Dec 2017
 *      Author: Viktor
 */

#ifndef LOSS_H_
#define LOSS_H_

#include <vector>

namespace cppnn {

class Loss {
public:
	virtual ~Loss() { };
	virtual double function(std::vector<double>& out, std::vector<double>& obj) = 0;
	virtual double d_function(std::vector<double> out, double y) = 0;
};

class QuadraticLoss : public Loss {
public:
	virtual ~QuadraticLoss() { };
	virtual double function(std::vector<double>& out, std::vector<double>& obj);
	virtual double d_function(std::vector<double> out, double y);
};

class CrossEntropyLoss : public Loss {
public:
	virtual ~CrossEntropyLoss() { };
	virtual double function(std::vector<double>& out, std::vector<double>& obj);
	virtual double d_function(std::vector<double>& out, double y);
};

}

#endif /* LOSS_H_ */
