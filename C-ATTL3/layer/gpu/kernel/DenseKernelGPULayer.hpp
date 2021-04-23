/*
 * DenseKernelGPULayer.hpp
 *
 *  Created on: 19 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_KERNEL_DENSEKERNELGPULAYER_H_
#define C_ATTL3_LAYER_GPU_KERNEL_DENSEKERNELGPULAYER_H_

#include "layer/gpu/KernelGPULayer.hpp"
#include "parameter_initialization/gpu/ZeroGPUParameterInitialization.hpp"
#include "parameters/gpu/StandardGPUParameters.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar, std::size_t Rank>
class DenseKernelGPULayer : public KernelGPULayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef KernelGPULayer<Scalar, Rank> Base;

 public:
  inline DenseKernelGPULayer(const typename Root::Dims& input_dims, std::size_t output_size,
                             GPUParamInitSharedPtr<Scalar> weight_init,
                             GPUParamRegSharedPtr<Scalar> weight_reg = nullptr,
                             GPUParamRegSharedPtr<Scalar> bias_reg = nullptr)
      : Base(input_dims, {output_size}, input_dims.template extend<3 - Rank>(), {output_size},
             std::make_shared<StandardGPUParameters<Scalar>>(input_dims.volume(), output_size, true, weight_init,
                                                             weight_reg),
             std::make_shared<StandardGPUParameters<Scalar>>(
                 1, output_size, true, std::make_shared<ZeroGPUParameterInitialization<Scalar>>(), bias_reg)) {}

  inline DenseKernelGPULayer(const DenseKernelGPULayer<Scalar, Rank>& layer, bool share_params = false)
      : Base(layer, share_params), in_cache(layer.in_cache) {}

  inline GPULayer<Scalar, Rank>* gpu_clone() const { return new DenseKernelGPULayer<Scalar, Rank>(*this); }

  inline GPULayer<Scalar, Rank>* gpu_clone_with_shared_params() {
    return new DenseKernelGPULayer<Scalar, Rank>(*this, true);
  }

  inline void empty_cache() { in_cache = CuDNNTensor<Scalar>(); }

  inline CuDNNTensor<Scalar> gpu_pass_forward(CuDNNTensor<Scalar> in, bool training) {
    assert(in.height() == Base::gpu_input_dims(0) && in.width() == Base::gpu_input_dims(1) &&
           in.channels() == Base::gpu_input_dims(2));
    assert(in.samples() > 0);
    in_cache = std::move(in);
    CuBLASMatrix<Scalar> in_mat(in_cache.data(), Base::input_dims.volume(), in_cache.samples());
    CuDNNTensor<Scalar> out(in_cache.samples(), Base::gpu_output_dims(0), Base::gpu_output_dims(1),
                            Base::gpu_output_dims(2));
    CuBLASMatrix<Scalar> out_mat(out.data(), Base::output_dims.volume(), out.samples());
    CuBLASMatrix<Scalar>::gemm(Base::weights.get_gpu_values(), true, in_mat, false, 1, 0, out_mat);
    out += Base::bias.get_gpu_values();
    return out;
  }

  inline CuDNNTensor<Scalar> gpu_pass_back(CuDNNTensor<Scalar> out_grad) {
    assert(out_grad.height() == Base::gpu_output_dims(0) && out_grad.width() == Base::gpu_output_dims(1) &&
           out_grad.channels() == Base::gpu_output_dims(2));
    assert(out_grad.samples() == in_cache.samples());
    StandardGPUParameters<Scalar>& weights = static_cast<StandardGPUParameters<Scalar>&>(*Base::weights);
    StandardGPUParameters<Scalar>& bias = static_cast<StandardGPUParameters<Scalar>&>(*Base::bias);
    CuDNNTensor<Scalar> weights_grad(weights.get_samples(), weights.get_height(), weights.get_width(),
                                     weights.get_channels());
    CuBLASMatrix<Scalar> weights_grad_mat(weights_grad.data(), weights.rows(), weights.cols());
    CuBLASMatrix<Scalar> out_grad_mat(out_grad.data(), out_grad.height() * out_grad.width() * out_grad.channels(),
                                      out_grad.samples());
    CuBLASMatrix<Scalar>::gemm(in_cache, false, out_grad_mat, true, 1, 0, weights_grad_mat);
    weights.accumulate_gpu_grad(weights_grad);
    bias.accumulate_gpu_grad(out_grad.sum({true, false, false, false}));
    if (Base::is_input_layer()) return CuDNNTensor<Scalar>();
    CuDNNTensor<Scalar> prev_out_grad(out_grad.samples(), Base::gpu_input_dims(0), Base::gpu_input_dims(1),
                                      Base::gpu_input_dims(2));
    CuBLASMatrix<Scalar> prev_out_grad_mat(prev_out_grad.data(),
                                           prev_out_grad.height() * prev_out_grad.width() * prev_out_grad.channels(),
                                           prev_out_grad.samples());
    CuBLASMatrix<Scalar>::gemm(out_grad_mat, true, weights.get_gpu_values(), false, 1, 0, prev_out_grad_mat);
    return prev_out_grad;
  }

 private:
  CuDNNTensor<Scalar> in_cache;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_KERNEL_DENSEKERNELGPULAYER_H_ */
