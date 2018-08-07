/*
 * GPUParameters.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor Csommor
 */

#ifndef C_ATTL3_CORE_GPU_GPUPARAMETERS_H_
#define C_ATTL3_CORE_GPU_GPUPARAMETERS_H_

#include "core/Parameters.hpp"
#include "cudnn/CuDNNTensor.hpp"

namespace cattle {
namespace gpu {

template<typename Scalar>
class GPUParameters : public virtual Parameters<Scalar> {
public:
	virtual std::size_t get_samples() const = 0;
	virtual std::size_t get_height() const = 0;
	virtual std::size_t get_width() const = 0;
	virtual std::size_t get_channels() const = 0;
	virtual const CuDNNTensor<Scalar>& get_gpu_values() const = 0;
	virtual void set_gpu_values(CuDNNTensor<Scalar>& values) = 0;
	virtual const CuDNNTensor<Scalar>& get_gpu_grad() const = 0;
	virtual void accumulate_gpu_grad(const CuDNNTensor<Scalar>& grad) = 0;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_GPUPARAMETERS_H_ */
