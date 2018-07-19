/*
 * UnifiedTensor.hpp
 *
 *  Created on: 18 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_UNIFIEDTENSOR_H_
#define C_ATTL3_UNIFIEDTENSOR_H_

#include <array>
#include <cstddef>
#include <utility>

#include "EigenProxy.hpp"

#ifdef CATTL3_USE_CUDNN
#include "gpu/CuDNNTensor.hpp"
#endif

namespace cattle {

#ifndef CATTL3_USE_CUDNN
template<typename Scalar>
/**
 * An alias for a rank 4 host tensor.
 */
using UnifiedTensor = Tensor<Scalar,4>;
#else
/**
 * A template class for a tensor either backed by a cuDNN device tensor or
 * an Eigen host tensor. It allows for seamless conversion between host and
 * device tensors.
 */
template<typename Scalar>
class UnifiedTensor {
	typedef Tensor<Scalar,4> CPUTensor;
	typedef gpu::CuDNNTensor<Scalar> GPUTensor;
public:
	inline UnifiedTensor(CPUTensor cpu_tensor) :
		cpu_tensor(std::move(cpu_tensor)),
		cpu_based(true) { }
	inline UnifiedTensor(GPUTensor gpu_tensor) :
		gpu_tensor(std::move(gpu_tensor)),
		cpu_based(false) { }
	inline operator CPUTensor() const & {
		if (cpu_based)
			return cpu_tensor;
		return convert_from_gpu_to_cpu(gpu_tensor);
	}
	inline operator CPUTensor() && {
		if (cpu_based)
			return std::move(cpu_tensor);
		return convert_from_gpu_to_cpu(gpu_tensor);
	}
	inline operator GPUTensor() const & {
		if (!cpu_based)
			return gpu_tensor;
		return convert_from_cpu_to_gpu(cpu_tensor);
	}
	inline operator GPUTensor() && {
		if (!cpu_based)
			return std::move(gpu_tensor);
		return convert_from_cpu_to_gpu(cpu_tensor);
	}
private:
	inline static CPUTensor convert_from_gpu_to_cpu(const GPUTensor& gpu_tensor) {
		CPUTensor out(gpu_tensor.width(), gpu_tensor.height(),
				gpu_tensor.channels(), gpu_tensor.samples());
		gpu_tensor.copy_to_host(out.data());
		static std::array<std::size_t,4> gpu_to_cpu_layout({ 3u, 1u, 0u, 2u });
		return out.shuffle(gpu_to_cpu_layout)
	}
	inline static GPUTensor convert_from_cpu_to_gpu(const CPUTensor& cpu_tensor) {
		static std::array<std::size_t,4> cpu_to_gpu_layout({ 2u, 1u, 2u, 0u });
		CPUTensor shuffled_cpu_tensor = cpu_tensor.shuffle(cpu_to_gpu_layout);
		GPUTensor out(cpu_tensor.dimension(0), cpu_tensor.dimension(1),
				cpu_tensor.dimension(2), cpu_tensor.dimension(3));
		out.copy_from_host(shuffled_cpu_tensor.data());
		return out;
	}
	CPUTensor cpu_tensor;
	GPUTensor gpu_tensor;
	bool cpu_based;
};
#endif

} /* namespace cattle */

#endif /* C_ATTL3_UNIFIEDTENSOR_H_ */
