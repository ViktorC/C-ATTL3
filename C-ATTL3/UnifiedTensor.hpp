/*
 * UnifiedTensor.hpp
 *
 *  Created on: 18 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_UNIFIEDTENSOR_H_
#define C_ATTL3_UNIFIEDTENSOR_H_

namespace cattle {

template<typename Scalar, std::size_t Rank>
class UnifiedTensor {
public:
	UnifiedTensor();
	virtual ~UnifiedTensor();
};

} /* namespace cattle */

#endif /* C_ATTL3_UNIFIEDTENSOR_H_ */
