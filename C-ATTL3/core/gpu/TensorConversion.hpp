/*
 * TensorConversion.hpp
 *
 *  Created on: 4 Aug 2018
 *      Author: Viktor
 */

#ifndef C_ATTL3_CORE_GPU_TENSORCONVERSION_H_
#define C_ATTL3_CORE_GPU_TENSORCONVERSION_H_

#include "core/EigenProxy.hpp"
#include "cudnn/CuDNNTensor.hpp"

namespace cattle {
namespace gpu {

/**
 * A utility class for converting cuDNN tensors to Eigen tensors and vice versa.
 */
template<typename Scalar>
class TensorConversion {
public:
	/**
	 * @param cudnn_tensor The NCHW row-major cuDNN tensor.
	 * @return The cuDNN tensor coverted into an NHWC col-major rank-4 Eigen tensor.
	 */
	inline static Tensor<Scalar,4> convert_from_cudnn_to_eigen(const CuDNNTensor<Scalar>& cudnn_tensor) {
		Tensor<Scalar,4> out(cudnn_tensor.width(), cudnn_tensor.height(),
				cudnn_tensor.channels(), cudnn_tensor.samples());
		cudnn_tensor.copy_to_host(out.data());
		static std::array<std::size_t,4> cudnn_to_eigen_layout({ 3u, 1u, 0u, 2u });
		return out.shuffle(cudnn_to_eigen_layout);
	}
	/**
	 * @param eigen_tensor The NHWC col-major rank-4 Eigen tensor.
	 * @return The Eigen tensor converted into an NCWH col-major cuDNN tensor.
	 */
	inline static CuDNNTensor<Scalar> convert_from_eigen_to_cudnn(const Tensor<Scalar,4>& eigen_tensor) {
		static std::array<std::size_t,4> eigen_to_cudnn_layout({ 2u, 1u, 2u, 0u });
		Tensor<Scalar,4> shuffled_cpu_tensor = eigen_tensor.shuffle(eigen_to_cudnn_layout);
		CuDNNTensor<Scalar> out(eigen_tensor.dimension(0), eigen_tensor.dimension(1),
				eigen_tensor.dimension(2), eigen_tensor.dimension(3));
		out.copy_from_host(shuffled_cpu_tensor.data());
		return out;
	}
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_TENSORCONVERSION_H_ */
