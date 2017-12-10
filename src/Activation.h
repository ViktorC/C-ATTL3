/*
 * Activation.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include <Vector.h>

namespace cppnn {

template<typename Scalar>
class Activation {
public:
	virtual ~Activation() { };
	virtual Vector<Scalar> function(Vector<Scalar> x) const = 0;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const = 0;
};

template<typename Scalar>
class IdentityActivation : public Activation<Scalar> {
public:
	virtual ~IdentityActivation();
	virtual Vector<Scalar> function(Vector<Scalar> x) const;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const;
};

template<typename Scalar>
class BinaryStepActivation : public Activation<Scalar> {
public:
	virtual ~BinaryStepActivation();
	virtual Vector<Scalar> function(Vector<Scalar> x) const;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const;
};

template<typename Scalar>
class SigmoidActivation : public Activation<Scalar> {
public:
	virtual ~SigmoidActivation();
	virtual Vector<Scalar> function(Vector<Scalar> x) const;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const;
};

template<typename Scalar>
class SoftmaxActivation : public Activation<Scalar> {
public:
	virtual ~SoftmaxActivation() { };
	virtual Vector<Scalar> function(Vector<Scalar> x) const;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const;
};

template<typename Scalar>
class TanhActivation : public Activation<Scalar> {
public:
	virtual ~TanhActivation();
	virtual Vector<Scalar> function(Vector<Scalar> x) const;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const;
};

template<typename Scalar>
class ReLUActivation : public Activation<Scalar> {
public:
	virtual ~ReLUActivation();
	virtual Vector<Scalar> function(Vector<Scalar> x) const;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const;
};

template<typename Scalar>
class LeakyReLUActivation : public ReLUActivation<Scalar> {
	static const Scalar DEF_LRELU_ALPHA = 1e-1;
protected:
	Scalar alpha;
public:
	LeakyReLUActivation(Scalar alpha = DEF_LRELU_ALPHA);
	virtual ~LeakyReLUActivation();
	virtual Vector<Scalar> function(Vector<Scalar> x) const;
	virtual Vector<Scalar> d_function(Vector<Scalar>& x,
			Vector<Scalar>& y) const;
};

}

#endif /* ACTIVATION_H_ */
