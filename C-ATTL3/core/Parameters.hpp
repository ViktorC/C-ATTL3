/*
 * Parameters.hpp
 *
 *  Created on: 20.07.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_PARAMETERS_H_
#define C_ATTL3_CORE_PARAMETERS_H_

#include "EigenProxy.hpp"

namespace cattle {

template<typename Scalar>
class Parameters {
public:
	virtual ~Parameters() = default;
	/**
	 * @return A pointer to copy of the instance on which the method is
	 * invoked.
	 */
	virtual Parameters<Scalar>* clone() const = 0;
	/**
	 * Determines whether the parameters are optimizable. Non-optimizable
	 * parameters are ignored by optimizers and do not have to maintain
	 * gradients or worry about regularization.
	 *
	 * @return Whether the parameters are learnable via gradient descent
	 * or any other optimization method.
	 */
	virtual bool are_optimizable() const = 0;
	/**
	 * @return The number of rows of the parameter matrix.
	 */
	virtual std::size_t get_rows() const = 0;
	/**
	 * @return The number of columns of the parameter matrix.
	 */
	virtual std::size_t get_cols() const = 0;
	/**
	 * It initializes the parameters.
	 */
	virtual void init() = 0;
	/**
	 * @return A constant reference to the values of the parameters.
	 */
	virtual const Matrix<Scalar>& get_values() const = 0;
	/**
	 * @param values The new values of the parameters. The matrix is expected to havey
	 * the dimensions specified by the get_rows() and get_cols() methods.
	 */
	virtual void set_values(Matrix<Scalar> values) = 0;
	/**
	 * @return A constant reference to the gradient of the parameters.
	 */
	virtual const Matrix<Scalar>& get_grad() const = 0;
	/**
	 * @param grad The values to add to the current gradient of the parameters. The matrix
	 * is expected to have the dimensions specified by the get_rows() and get_cols() methods.
	 */
	virtual void accumulate_grad(const Matrix<Scalar>& grad) = 0;
	/**
	 * It resets the gradient to all zeroes.
	 */
	virtual void reset_grad() = 0;
	/**
	 * @return The regularization penalty imposed on the parameters.
	 */
	virtual Scalar get_regularization_penalty() const = 0;
	/**
	 * It adds the derivative of the regularization function w.r.t.t values of the
	 * parameters to the gradient of the parameters.
	 */
	virtual void regularize() = 0;
	/**
	 * @return Whether the parameters should not be updated.
	 */
	virtual bool are_frozen() const = 0;
	/**
	 * @param frozen Whether the parameters are to be frozen, i.e. not to be updated.
	 */
	virtual void set_frozen(bool frozen) = 0;
};

} /* namespace cattle */

#endif /* C_ATTL3_CORE_PARAMETERS_H_ */
