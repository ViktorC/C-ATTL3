/*
 * ConvKernelGPULayer.hpp
 *
 *  Created on: 19 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_KERNEL_CONVKERNELGPULAYER_H_
#define C_ATTL3_LAYER_GPU_KERNEL_CONVKERNELGPULAYER_H_

#include "layer/gpu/KernelGPULayer.hpp"
#include "parameters/gpu/StandardGPUParameters.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar, std::size_t Rank>
class ConvKernelGPULayer : public KernelGPULayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef KernelGPULayer<Scalar, Rank> Base;

 public:
  inline ConvKernelGPULayer(const typename Root::Dims& input_dims, std::size_t filters, std::size_t receptor_height,
                            std::size_t receptor_width, std::size_t vertical_padding, std::size_t horizontal_padding,
                            std::size_t vertical_stride, std::size_t horizontal_stride, std::size_t vertical_dilation,
                            std::size_t horizontal_dilation, GPUParamInitSharedPtr<Scalar> weight_init,
                            GPUParamRegSharedPtr<Scalar> weight_reg = nullptr,
                            GPUParamRegSharedPtr<Scalar> bias_reg = nullptr)
      : Base(input_dims,
             calculate_adjusted_output_dims(input_dims, filters, receptor_height, receptor_width, vertical_padding,
                                            horizontal_padding, vertical_stride, horizontal_stride, vertical_dilation,
                                            horizontal_dilation),
             input_dims.template extend<3 - Rank>(),
             calculate_output_dims(input_dims.template extend<3 - Rank>(), filters, receptor_height, receptor_width,
                                   vertical_padding, horizontal_padding, vertical_stride, horizontal_stride,
                                   vertical_dilation, horizontal_dilation),
             std::make_shared<StandardGPUParameters<Scalar>>(filters, receptor_height, receptor_width,
                                                             input_dims.dimension(3), true, weight_init, weight_reg),
             std::make_shared<StandardGPUParameters<Scalar>>(
                 1, filters, true, std::make_shared<ZeroParameterInitialization<Scalar>>(), bias_reg)),
        filters(filters),
        receptor_height(receptor_height),
        receptor_width(receptor_width),
        vertical_padding(vertical_padding),
        horizontal_padding(horizontal_padding),
        vertical_stride(vertical_stride),
        horizontal_stride(horizontal_stride),
        vertical_dilation(vertical_dilation),
        horizontal_dilation(horizontal_dilation),
        alpha(1),
        beta(0) {
    assert(filters > 0);
    assert(receptor_height > 0);
    assert(receptor_width > 0);
    assert(vertical_stride > 0 && horizontal_stride > 0);
    assert(Base::gpu_input_dims(0) + 2 * vertical_padding >=
               receptor_height + (receptor_height - 1) * vertical_dilation &&
           Base::gpu_input_dims(1) + 2 * horizontal_padding >=
               receptor_width + (receptor_width - 1) * horizontal_dilation);
  }
  inline ConvKernelGPULayer(const ConvKernelGPULayer<Scalar, Rank>& layer, bool share_params = false)
      : Base(layer, share_params),
        filters(layer.filters),
        receptor_height(layer.receptor_height),
        receptor_width(layer.receptor_width),
        vertical_padding(layer.vertical_padding),
        horizontal_padding(layer.horizontal_padding),
        vertical_stride(layer.vertical_stride),
        horizontal_stride(layer.horizontal_stride),
        vertical_dilation(layer.vertical_dilation),
        horizontal_dilation(layer.horizontal_dilation),
        alpha(layer.alpha),
        beta(layer.beta),
        in_cache(layer.in_cache) {}
  inline GPULayer<Scalar, Rank>* gpu_clone() const { return new ConvKernelGPULayer<Scalar, Rank>(*this); }
  inline GPULayer<Scalar, Rank>* gpu_clone_with_shared_params() {
    return new ConvKernelGPULayer<Scalar, Rank>(*this, true);
  }
  inline void empty_cache() { in_cache = CuDNNTensor<Scalar>(); }
  inline CuDNNTensor<Scalar> pass_forward(CuDNNTensor<Scalar> in, bool training) {
    assert(in.height() == Base::gpu_input_dims(0) && in.width() == Base::gpu_input_dims(1) &&
           in.channels() == Base::gpu_input_dims(2));
    assert(in.samples() > 0);
    StandardGPUParameters<Scalar>& weights = static_cast<StandardGPUParameters<Scalar>&>(*Base::weights);
    StandardGPUParameters<Scalar>& bias = static_cast<StandardGPUParameters<Scalar>&>(*Base::bias);
    in_cache = std::move(in);
    CuDNNTensor<Scalar> out(in_cache.samples(), Base::gpu_output_dims(0), Base::gpu_output_dims(1),
                            Base::gpu_output_dims(2));
    // Create and set up the convolution descriptor.
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnAssert(cudnnCreateConvolutionDescriptor(&conv_desc));
    cudnnAssert(cudnnSetConvolution2dDescriptor(conv_desc, vertical_padding, horizontal_padding, vertical_stride,
                                                horizontal_stride, vertical_dilation, horizontal_dilation,
                                                CUDNN_CROSS_CORRELATION, CuDNNTensor<Scalar>::DATA_TYPE));
    // Have cuDNN find the most performant algorithm given the convolution
    // parameters.
    int returned_algo_count;
    cudnnConvolutionFwdAlgoPerf_t conv_algo_perf;
    cudnnAssert(cudnnGetConvolutionForwardAlgorithm_v7(CuDNNHandle::get_instance(), in_cache.desc(),
                                                       weights.get_gpu_values().filter_desc(), conv_desc, out.desc(), 1,
                                                       &returned_algo_count, &conv_algo_perf));
    cudnnConvolutionFwdAlgo_t conv_algo = conv_algo_perf.algo;
    /* Have cuDNN compute the workspace memory required for the selected
     * convolution algorithm given the convolution parameters. */
    std::size_t workspace_size;
    cudnnAssert(cudnnGetConvolutionForwardWorkspaceSize(CuDNNHandle::get_instance(), in_cache.desc(),
                                                        weights.get_gpu_values().filter_desc(), conv_desc, out.desc(),
                                                        conv_algo, &workspace_size));
    // Allocate the memory for the workspace required for the convolution.
    CUDAArray<Scalar> workspace(static_cast<std::size_t>(ceil(static_cast<Scalar>(workspace_size) / sizeof(Scalar))));
    // Perform the convolution.
    cudnnAssert(cudnnConvolutionForward(CuDNNHandle::get_instance(), &alpha, in_cache.desc(), in_cache.data(),
                                        weights.get_gpu_values().filter_desc(), weights.get_values().data(), conv_desc,
                                        conv_algo, workspace.data(), workspace_size, &beta, out.desc(), out.data()));
    // Free the convolution resources.
    cudnnAssert(cudnnDestroyConvolutionDescriptor(conv_desc));
    // Apply the bias to the output tensor.
    out += bias.get_gpu_values();
    return out;
  }
  inline CuDNNTensor<Scalar> pass_back(CuDNNTensor<Scalar> out_grad) {
    assert(out_grad.height() == Base::gpu_output_dims(0) && out_grad.width() == Base::gpu_output_dims(1) &&
           out_grad.channels() == Base::gpu_output_dims(2));
    assert(out_grad.samples() == in_cache.samples());
    StandardGPUParameters<Scalar>& weights = static_cast<StandardGPUParameters<Scalar>&>(*Base::weights);
    StandardGPUParameters<Scalar>& bias = static_cast<StandardGPUParameters<Scalar>&>(*Base::bias);
    CuDNNTensor<Scalar> weights_grad(weights.samples(), weights.height(), weights.width(), weights.channels());
    CuDNNTensor<Scalar> bias_grad(bias.samples(), bias.height(), bias.width(), bias.channels());
    // Create and set up the backward convolution descriptor.
    cudnnConvolutionDescriptor_t dconv_desc;
    cudnnAssert(cudnnCreateConvolutionDescriptor(&dconv_desc));
    cudnnAssert(cudnnSetConvolution2dDescriptor(dconv_desc, vertical_padding, horizontal_padding, vertical_stride,
                                                horizontal_stride, vertical_dilation, horizontal_dilation,
                                                CUDNN_CROSS_CORRELATION, CuDNNTensor<Scalar>::DATA_TYPE));
    // Have cuDNN find the most performant algorithm given the convolution
    // parameters.
    int returned_algo_count;
    cudnnConvolutionBwdFilterAlgoPerf_t dconv_filter_algo_perf;
    cudnnAssert(cudnnGetConvolutionBackwardFilterAlgorithm_v7(CuDNNHandle::get_instance(), in_cache.desc(),
                                                              out_grad.desc(), dconv_desc, weights_grad.filter_desc(),
                                                              1, &returned_algo_count, &dconv_filter_algo_perf));
    cudnnConvolutionBwdFilterAlgo_t dconv_filter_algo = dconv_filter_algo_perf.algo;
    /* Have cuDNN compute the data_workspace memory required for the selected
     * backward convolution algorithms given the convolution parameters. */
    std::size_t filter_workspace_size;
    cudnnAssert(cudnnGetConvolutionBackwardFilterWorkspaceSize(CuDNNHandle::get_instance(), in_cache.desc(),
                                                               out_grad.desc(), dconv_desc, weights_grad.filter_desc(),
                                                               dconv_filter_algo, &filter_workspace_size));
    // Allocate the memory required for the backwards filter convolution on the
    // device.
    CUDAArray<Scalar> filter_workspace(
        static_cast<std::size_t>(ceil(static_cast<Scalar>(filter_workspace_size) / sizeof(Scalar))));
    // Perform the backwards filter convolution.
    cudnnAssert(cudnnConvolutionBackwardFilter(CuDNNHandle::get_instance(), &alpha, in_cache.desc(), in_cache.data(),
                                               out_grad.desc(), out_grad.data(), dconv_desc, dconv_filter_algo,
                                               filter_workspace.data(), filter_workspace_size, &beta,
                                               weights_grad.filter_desc(), weights_grad.data()));
    // Free up resources.
    cudnnAssert(cudnnDestroyConvolutionDescriptor(dconv_desc));
    // Perform the backwards bias convolution.
    cudnnAssert(cudnnConvolutionBackwardBias(CuDNNHandle::get_instance(), &alpha, out_grad.desc(), out_grad.data(),
                                             &beta, bias_grad.desc(), bias_grad.data()));
    weights.accumulate_grad(weights_grad);
    bias.accumulate_grad(bias_grad);
    if (Base::is_input_layer()) return CuDNNTensor<Scalar>();
    CuDNNTensor<Scalar> prev_out_grad(out_grad.samples(), Base::gpu_input_dims(0), Base::gpu_input_dims(1),
                                      Base::gpu_input_dims(2));
    // Get the backward data convolution algorithm.
    cudnnConvolutionBwdDataAlgoPerf_t dconv_data_algo_perf;
    cudnnAssert(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        CuDNNHandle::get_instance(), weights.get_gpu_values().filter_desc(), out_grad.desc(), dconv_desc,
        prev_out_grad.desc(), 1, &returned_algo_count, &dconv_data_algo_perf));
    cudnnConvolutionBwdDataAlgo_t dconv_data_algo = dconv_data_algo_perf.algo;
    // Calculate the workspace size needed.
    std::size_t data_workspace_size;
    cudnnAssert(cudnnGetConvolutionBackwardDataWorkspaceSize(
        CuDNNHandle::get_instance(), weights.get_gpu_values().filter_desc(), out_grad.desc(), dconv_desc,
        prev_out_grad.desc(), dconv_data_algo, &data_workspace_size));
    // Allocate the memory required for the backwards data convolution on the
    // device.
    CUDAArray<Scalar> data_workspace(
        static_cast<std::size_t>(ceil(static_cast<Scalar>(data_workspace_size) / sizeof(Scalar))));
    // Perform the backwards data convolution.
    cudnnAssert(cudnnConvolutionBackwardData(
        CuDNNHandle::get_instance(), &alpha, weights.get_gpu_values().filter_desc(), weights.get_gpu_values().data(),
        out_grad.desc(), out_grad.data(), dconv_desc, dconv_data_algo, data_workspace.data(), data_workspace_size,
        &beta, prev_out_grad.desc(), prev_out_grad.data()));
    return prev_out_grad;
  }

 private:
  inline static Dimensions<std::size_t, 3> calculate_output_dims(
      const Dimensions<std::size_t, 3>& input_dims, std::size_t filters, std::size_t receptor_height,
      std::size_t receptor_width, std::size_t vertical_padding, std::size_t horizontal_padding,
      std::size_t vertical_stride, std::size_t horizontal_stride, std::size_t vertical_dilation,
      std::size_t horizontal_dilation) {
    // Create and set the input tensor descriptor.
    cudnnTensorDescriptor_t input_desc;
    CuDNNTensor<Scalar>::create_tensor_descriptor(input_desc, 1, input_dims(0), input_dims(1), input_dims(2));
    // Create and set up the filter descriptor.
    cudnnFilterDescriptor_t filter_desc;
    CuDNNTensor<Scalar>::create_filter_descriptor(filter_desc, filters, receptor_height, receptor_width, input_dims(2));
    // Create and set up the convolution descriptor.
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnAssert(cudnnCreateConvolutionDescriptor(&conv_desc));
    cudnnAssert(cudnnSetConvolution2dDescriptor(conv_desc, vertical_padding, horizontal_padding, vertical_stride,
                                                horizontal_stride, vertical_dilation, horizontal_dilation,
                                                CUDNN_CROSS_CORRELATION, CuDNNTensor<Scalar>::DATA_TYPE));
    // Compute the dimensions.
    int n, h, w, c;
    cudnnAssert(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w));
    // Free the resources.
    cudnnAssert(cudnnDestroyConvolutionDescriptor(conv_desc));
    CuDNNTensor<Scalar>::destroy_tensor_descriptor(input_desc);
    CuDNNTensor<Scalar>::destroy_filter_descriptor(filter_desc);
    return {(std::size_t)h, (std::size_t)w, (std::size_t)c};
  }
  inline static Dimensions<std::size_t, Rank> calculate_adjusted_output_dims(
      const Dimensions<std::size_t, Rank>& input_dims, std::size_t filters, std::size_t receptor_height,
      std::size_t receptor_width, std::size_t vertical_padding, std::size_t horizontal_padding,
      std::size_t vertical_stride, std::size_t horizontal_stride, std::size_t vertical_dilation,
      std::size_t horizontal_dilation) {
    auto ext_input_dims = input_dims.template extend<3 - Rank>();
    auto ext_output_dims = calculate_output_dims(ext_input_dims, filters, receptor_height, receptor_width,
                                                 vertical_padding, horizontal_padding, vertical_stride,
                                                 horizontal_stride, vertical_dilation, horizontal_dilation);
    ext_output_dims(2) /= filters;
    ext_output_dims(Rank - 1) *= filters;
    return ext_output_dims.template contract<3 - Rank>();
  }
  const std::size_t filters, receptor_height, receptor_width, vertical_padding, horizontal_padding, vertical_stride,
      horizontal_stride, vertical_dilation, horizontal_dilation;
  const Scalar alpha, beta;
  CuDNNTensor<Scalar> in_cache;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_KERNEL_CONVKERNELGPULAYER_H_ */
