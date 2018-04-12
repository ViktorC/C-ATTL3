/*
 * CUDAUtils.h
 *
 *  Created on: 12 Apr 2018
 *      Author: Viktor Csomor
 */

#ifndef UTILS_GPU_CUDAUTILS_CUH_
#define UTILS_GPU_CUDAUTILS_CUH_

namespace cattle {
namespace internal {

template<typename Scalar>
class CUDAUtils {
public:
	static void mat_mul(const Scalar* a, const Scalar* b, Scalar* c) {
		
	}
private:
	CUDAUtils() { }
};

}
} /* namespace cattle */

#endif /* UTILS_GPU_CUDAUTILS_CUH_ */
