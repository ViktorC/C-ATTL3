/*
 * CuRANDGenerator.hpp
 *
 *  Created on: 5 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_CURAND_CURANDGENERATOR_H_
#define C_ATTL3_CORE_GPU_CURAND_CURANDGENERATOR_H_

#include "core/gpu/cuda/CUDAArray.hpp"
#include "CuRANDError.hpp"

namespace cattle {
namespace gpu {

namespace {

template<typename Scalar>
using NormalGenerationRoutine = curandStatus_t (*)(curandGenerator_t, Scalar*, std::size_t, Scalar, Scalar);
template<typename Scalar>
__inline__ NormalGenerationRoutine<Scalar> normal_gen_routine() { return &curandGenerateNormalDouble; }
template<> __inline__ NormalGenerationRoutine<float> normal_gen_routine() { return &curandGenerateNormal; }

template<typename Scalar>
using UniformGenerationRoutine = curandStatus_t (*)(curandGenerator_t, Scalar*, std::size_t);
template<typename Scalar>
__inline__ UniformGenerationRoutine<Scalar> uniform_gen_routine() { return &curandGenerateUniformDouble; }
template<> __inline__ UniformGenerationRoutine<float> uniform_gen_routine() { return &curandGenerateUniform; }

}

/**
 * A template class for generating normally or uniformly distributed random numbers
 * using the cuRAND generator.
 */
template<typename Scalar>
class CuRANDGenerator {
public:
	static constexpr curandRngType_t RNG_TYPE = CURAND_RNG_PSEUDO_DEFAULT;
	inline CuRANDGenerator() :
			gen() {
		curandAssert(curandCreateGenerator(&gen, RNG_TYPE));
	}
	inline ~CuRANDGenerator() {
		curandAssert(curandDestroyGenerator(gen));
	}
	/**
	 * @param mean The mean of the distribution.
	 * @param sd The standard deviation of the distribution.
	 * @param array The device array to fill with the randomly generated numbers.
	 */
	inline void generate_normal(Scalar mean, Scalar sd, CUDAArray<Scalar>& array) const {
		NormalGenerationRoutine<Scalar> norm_gen = normal_gen_routine<Scalar>();
		curandAssert(norm_gen(gen, array.data(), array.size() * sizeof(Scalar), mean, sd));
	}
	/**
	 * @param array The device array to fill with the randomly generated numbers.
	 */
	inline void generate_uniform(CUDAArray<Scalar>& array) const {
		UniformGenerationRoutine<Scalar> uni_gen = uniform_gen_routine<Scalar>();
		curandAssert(uni_gen(gen, array.data(), array.size() * sizeof(Scalar)));
	}
private:
	curandGenerator_t gen;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CURAND_CURANDGENERATOR_H_ */
