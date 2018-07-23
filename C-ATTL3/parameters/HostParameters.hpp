/*
 * MatrixParameters.hpp
 *
 *  Created on: 20 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_PARAMETERS_MATRIXPARAMETERS_H_
#define C_ATTL3_PARAMETERS_MATRIXPARAMETERS_H_

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

#include "core/NumericUtils.hpp"
#include "core/ParameterInitialization.hpp"
#include "core/ParameterRegularization.hpp"
#include "core/Parameters.hpp"

namespace cattle {

/**
 * An alias for a shared pointer to a WeightInitialization implementation instance of
 * an arbitrary scalar type.
 */
template<typename Scalar>
using ParamInitSharedPtr = std::shared_ptr<ParameterInitialization<Scalar>>;

/**
 * An alias for a shared pointer to a regularization penalty of an arbitrary scalar type.
 */
template<typename Scalar>
using ParamRegSharedPtr = std::shared_ptr<ParameterRegularization<Scalar>>;

template<typename Scalar>
class HostParameters : Parameters<Scalar> {
public:
	/**
	 * @param rows The number of rows of the parameter matrix.
	 * @param cols The number of columns of the parameter matrix.
	 * @param init The parameter value initialization. It cannot be a null pointer.
	 * @param optimizable Whether the parameters are optimizable. Non-optimizable
	 * parameters do not maintain gradient information and are thus not regularizable
	 * (but can still incur a regularization penalty).
	 * @param reg The optional regularization to use on the values of the parameter
	 * matrix. If it is a null pointer, no regularization is applied.
	 * @param value_clip The maximum allowed absolute parameter value. If it is 0
	 * or less, no value clipping is performed.
	 * @param value_max_l1_norm The maximum allowed L1 parameter value norm. If it
	 * is 0 or less, no L1 max norm constraint is enforced.
	 * @param value_max_l2_norm The maximum allowed L2 parameter value norm. If it
	 * is 0 or less, no L2 max norm constraint is enforced.
	 * @param grad_clip The maximum allowed absolute parameter gradient. If it is 0
	 * or less, no gradient clipping is performed.
	 * @param grad_max_l1_norm The maximum allowed L1 parameter gradient norm. If it
	 * is 0 or less, no L1 gradient max norm constraint is enforced.
	 * @param grad_max_l2_norm The maximum allowed L2 parameter gradient norm. If it
	 * is 0 or less, no L2 gradient max norm constraint is enforced.
	 */
	HostParameters(std::size_t rows, std::size_t cols, ParamInitSharedPtr<Scalar> init, bool optimizable = true,
			ParamRegSharedPtr<Scalar> reg = nullptr, Scalar value_clip = 0, Scalar value_max_l1_norm = 0,
			Scalar value_max_l2_norm = 0, Scalar grad_clip = 0, Scalar grad_max_l1_norm = 0,
			Scalar grad_max_l2_norm = 0) :
				rows(rows),
				cols(cols),
				param_init(init),
				optimizable(optimizable),
				param_reg(reg),
				value_clip(value_clip),
				value_max_l1_norm(value_max_l1_norm),
				value_max_l2_norm(value_max_l2_norm),
				grad_clip(grad_clip),
				grad_max_l1_norm(grad_max_l1_norm),
				grad_max_l2_norm(grad_max_l2_norm) {
		assert(rows > 0 && cols > 0);
		assert(init);
	}
	inline Parameters<Scalar>* clone() const {
		return new HostParameters<Scalar>(*this);
	}
	inline bool are_optimizable() const {
		return optimizable;
	}
	inline void init() {
		values = Matrix<Scalar>(rows, cols);
		param_init->apply(values);
		reset_grad();
	}
	inline const Matrix<Scalar>& get_values() const {
		return values;
	}
	inline void update_values(const Matrix<Scalar>& values) {
		assert(values.rows() == rows && values.cols() == cols);
		this->values += values;
		enforce_clip_constraint(this->values, value_clip);
		enforce_l1_norm_constraint(this->values, value_max_l1_norm);
		enforce_l2_norm_constraint(this->values, value_max_l2_norm);
	}
	inline const Matrix<Scalar>& get_grad() const {
		return grad;
	}
	inline void update_grad(const Matrix<Scalar>& grad) {
		if (!optimizable)
			return;
		assert(grad.rows() == rows && grad.cols() == cols);
		this->grad += grad;
		enforce_clip_constraint(this->grad, grad_clip);
		enforce_l1_norm_constraint(this->grad, grad_max_l1_norm);
		enforce_l2_norm_constraint(this->grad, grad_max_l2_norm);
	}
	inline void reset_grad() {
		if (optimizable)
			grad = Matrix<Scalar>::Zero(rows, cols);
	}
	inline Scalar get_regularization_penalty() const {
		if (!param_reg)
			return 0;
		return param_reg->function(values);
	}
	inline void regularize() {
		if (optimizable && param_reg)
			grad += param_reg->d_function(values);
	}
	inline bool are_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
protected:
	/**
	 * It clips the values of the matrix falling outside the interval [-clip, clip].
	 *
	 * @param matrix The matrix on which the clip constraint is to be enforced.
	 * @param clip The clipping limit.
	 */
	inline static void enforce_clip_constraint(Matrix<Scalar>& matrix, Scalar clip) {
		if (NumericUtils<Scalar>::decidedly_greater(clip, (Scalar) 0)) {
			matrix = matrix.unaryExpr([clip](Scalar m) {
				return m > clip ? clip : (m < -clip ? - clip : m);
			});
		}
	}
	/**
	 * It limits the L1 norm of the matrix by scaling its coefficients.
	 *
	 * @param matrix The matrix whose L1 norm is to be constrained.
	 * @param max_l1_norm The maximum allowed L1 norm.
	 */
	inline static void enforce_l1_norm_constraint(Matrix<Scalar>& matrix, Scalar max_l1_norm) {
		if (NumericUtils<Scalar>::decidedly_greater(max_l1_norm, (Scalar) 0)) {
			Scalar l1_norm = matrix.norm();
			if (l1_norm > max_l1_norm)
				matrix *= (max_l1_norm / l1_norm);
		}
	}
	/**
	 * It limits the L2 norm of the matrix by scaling its coefficients.
	 *
	 * @param matrix The matrix whose L2 norm is to be constrained.
	 * @param max_l2_norm The maximum allowed L2 norm.
	 */
	inline static void enforce_l2_norm_constraint(Matrix<Scalar>& matrix, Scalar max_l2_norm) {
		if (NumericUtils<Scalar>::decidedly_greater(max_l2_norm, (Scalar) 0)) {
			Scalar l2_norm = matrix.squaredNorm();
			if (l2_norm > max_l2_norm)
				matrix *= (max_l2_norm / l2_norm);
		}
	}
	std::size_t rows, cols;
	Matrix<Scalar> values, grad;
	ParamInitSharedPtr<Scalar> param_init;
	bool optimizable;
	ParamRegSharedPtr<Scalar> param_reg;
	Scalar value_clip, value_max_l1_norm, value_max_l2_norm;
	Scalar grad_clip, grad_max_l1_norm, grad_max_l2_norm;
	bool frozen;
};

} /* namespace cattle */

#endif /* C_ATTL3_PARAMETERS_MATRIXPARAMETERS_H_ */
