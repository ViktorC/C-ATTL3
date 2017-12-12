/*
 * Preprocessor.h
 *
 *  Created on: 12.12.2017
 *      Author: A6714
 */

#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <Matrix.h>

namespace cppnn {

template<typename Scalar>
class Preprocessor {
public:
	virtual ~Preprocessor() = default;
	virtual Matrix<Scalar>& process(Matrix<Scalar>& data) = 0;
};

} /* namespace cppnn */

#endif /* PREPROCESSOR_H_ */
