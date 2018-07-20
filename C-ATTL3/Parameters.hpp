/*
 * Parameters.hpp
 *
 *  Created on: 20.07.2018
 *      Author: A6714
 */

#ifndef C_ATTL3_PARAMETERS_H_
#define C_ATTL3_PARAMETERS_H_

#include "ParameterInitialization.hpp"
#include "ParameterRegularization.hpp";
#include "utils/EigenProxy.hpp"

namespace cattle {

/**
 * An alias for a shared pointer to a WeightInitialization implementation instance of
 * an arbitrary scalar type.
 */
template<typename Scalar>
using ParamInitSharedPtr = std::shared_ptr<ParameterInitialization<Scalar>>;

template<typename Scalar>
class Parameters {
public:
	inline const Matrix<Scalar>& get_values() const {
		return values;
	}
	inline virtual void set_values(Matrix<Scalar> values) {
		this->values = values;
	}
	/**
	 * It initializes the parameters.
	 */
	inline virtual void init() {
		values = Matrix<Scalar>(rows, cols);
		init->apply(values);
	}
	/**
	 * @return Whether the parameters should not be updated.
	 */
	inline bool is_frozen() const {
		return frozen;
	}
	/**
	 * @param frozen Whether the parameters are to be frozen, i.e. not to be updated.
	 */
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
protected:
	std::size_t rows, cols;
	Matrix<Scalar> values;
	ParamInitSharedPtr init;
	bool frozen;
};

/**
 * An alias for a shared pointer to a regularization penalty of an arbitrary scalar type.
 */
template<typename Scalar>
using ParamRegSharedPtr = std::shared_ptr<ParamaterRegularization<Scalar>>;

} /* namespace cattle */

#endif /* C_ATTL3_PARAMETERS_H_ */
