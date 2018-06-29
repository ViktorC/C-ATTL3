/*
 * CuDNNHandle.hpp
 *
 *  Created on: 28 May 2018
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_UTILS_CUDNNHANDLE_H_
#define CATTL3_UTILS_CUDNNHANDLE_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

#include "CuDNNError.hpp"
#include "CUDAError.hpp"

namespace cattle {
namespace internal {

/**
 * A singleton utility class providing methods for GPU accelerated deep neural network
 * operations on rank 4 data of [N,H,W,C] orientation.
 */
template<typename Scalar>
class CuDNNHandle {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	typedef std::array<std::size_t,4> Array4;
	static constexpr cudnnDataType_t DATA_TYPE = std::is_same<Scalar,float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
	static constexpr cudnnTensorFormat_t TENSOR_FORMAT = CUDNN_TENSOR_NHWC;
	static constexpr cudnnNanPropagation_t NAN_PROP = CUDNN_PROPAGATE_NAN;
	static constexpr std::size_t SCALAR_SIZE = sizeof(Scalar);
public:
	CuDNNHandle(const CuDNNHandle&) = delete;
	~CuDNNHandle() {
		// Destroy the cuDNN handle.
		cudnnAssert(cudnnDestroy(handle));
	}
	CuDNNHandle& operator=(const CuDNNHandle&) = delete;
	/**
	 * @return A reference to the only instance of the class.
	 */
	inline static CuDNNHandle& get_instance() {
		static CuDNNHandle instance;
		return instance;
	}
	/**
	 * Computes the dimensions of the output of the convolution.
	 *
	 * @param input_dims The dimensions of the input tensor.
	 * @param filters The number of convolution filters.
	 * @param receptor_height The height of the receptor.
	 * @param receptor_width The width of the receptor.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the convolution.
	 * @param horizontal_stride The horizontal stride of the convolution.
	 * @param vertical_dilation The vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The horizontal dilation to apply to the receptor.
	 * @return The output dimensions.
	 */
	inline Array4 conv2d_output_dims(const Array4& input_dims, std::size_t filters, std::size_t receptor_height,
			std::size_t receptor_width, std::size_t vertical_padding, std::size_t horizontal_padding,
			std::size_t vertical_stride, std::size_t horizontal_stride, std::size_t vertical_dilation,
			std::size_t horizontal_dilation) const {
		// Create and set the input tensor descriptor.
		cudnnTensorDescriptor_t input_desc = setup_tens_desc(input_dims);
		// Create and set up the filter descriptor.
		cudnnFilterDescriptor_t filter_desc;
		cudnnAssert(cudnnCreateFilterDescriptor(&filter_desc));
		cudnnAssert(cudnnSetFilter4dDescriptor(filter_desc, DATA_TYPE, TENSOR_FORMAT, filters, input_dims[3],
				receptor_height, receptor_width));
		// Create and set up the convolution descriptor.
		cudnnConvolutionDescriptor_t conv_desc;
		cudnnAssert(cudnnCreateConvolutionDescriptor(&conv_desc));
		cudnnAssert(cudnnSetConvolution2dDescriptor(conv_desc, vertical_padding, horizontal_padding, vertical_stride,
				horizontal_stride, vertical_dilation, horizontal_dilation, CUDNN_CROSS_CORRELATION, DATA_TYPE));
		// Compute the dimensions.
		int n, c, h, w;
		cudnnAssert(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w));
		// Free the resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(input_desc));
		cudnnAssert(cudnnDestroyFilterDescriptor(filter_desc));
		cudnnAssert(cudnnDestroyConvolutionDescriptor(conv_desc));
		return { (std::size_t) n, (std::size_t) h, (std::size_t) w, (std::size_t) c };
	}
	/**
	 * Performs a GPU accelerated 2D convolution on a rank 4 tensor of [N,H,W,C] orientation.
	 *
	 * @param input The input tensor.
	 * @param filter The convolution filter.
	 * @param bias The bias to apply to the output of the convolution.
	 * @param input_dims The dimensions of the input tensor.
	 * @param output_dims The dimensions of the output tensor of the convolution.
	 * @param receptor_height The height of the receptor.
	 * @param receptor_width The width of the receptor.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the convolution.
	 * @param horizontal_stride The horizontal stride of the convolution.
	 * @param vertical_dilation The vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The horizontal dilation to apply to the receptor.
	 * @param output The convolution output tensor with the bias applied to it.
	 */
	inline void convolution2d_fwd(Scalar* input, Scalar* filter, Scalar* bias, const Array4& input_dims, const Array4& output_dims,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_padding, std::size_t horizontal_padding,
			std::size_t vertical_stride, std::size_t horizontal_stride, std::size_t vertical_dilation, std::size_t horizontal_dilation,
			/* out */ Scalar* output) const {
		std::size_t depth = input_dims[3];
		std::size_t filters = output_dims[3];
		// Create and set up the input tensor descriptor.
		cudnnTensorDescriptor_t input_desc = setup_tens_desc(input_dims);
		// Create and set up the filter descriptor.
		cudnnFilterDescriptor_t filter_desc;
		cudnnAssert(cudnnCreateFilterDescriptor(&filter_desc));
		cudnnAssert(cudnnSetFilter4dDescriptor(filter_desc, DATA_TYPE, TENSOR_FORMAT, filters, depth,
				receptor_height, receptor_width));
		// Create and set up the bias descriptor.
		cudnnTensorDescriptor_t bias_desc = setup_tens_desc(1, 1, 1, filters);
		// Create and set up the convolution descriptor.
		cudnnConvolutionDescriptor_t conv_desc;
		cudnnAssert(cudnnCreateConvolutionDescriptor(&conv_desc));
		cudnnAssert(cudnnSetConvolution2dDescriptor(conv_desc, vertical_padding, horizontal_padding, vertical_stride,
				horizontal_stride, vertical_dilation, horizontal_dilation, CUDNN_CROSS_CORRELATION, DATA_TYPE));
		// Create and set up the output tensor descriptor.
		cudnnTensorDescriptor_t output_desc = setup_tens_desc(output_dims);
		// Have cuDNN find the most performant algorithm given the convolution parameters.
		cudnnConvolutionFwdAlgo_t conv_algo;
		cudnnAssert(cudnnGetConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc, output_desc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_algo));
		/* Have cuDNN compute the workspace memory required for the selected convolution algorithm given
		 * the convolution parameters. */
		std::size_t workspace_size;
		cudnnAssert(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc,
				conv_algo, &workspace_size));
		// Allocate the memory for the device arrays required for the convolution.
		Scalar* dev_input;
		Scalar* dev_filter;
		Scalar* dev_workspace;
		Scalar* dev_output;
		const std::size_t input_size = input_dims[0] * input_dims[1] * input_dims[2] * depth * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_input, input_size));
		const std::size_t filter_size = receptor_height * receptor_width * depth * filters * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_filter, filter_size));
		cudaAssert(cudaMalloc(&dev_workspace, workspace_size));
		const std::size_t output_size = output_dims[0] * output_dims[1] * output_dims[2] * filters * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_output, output_size));
		// Copy the contents of the input tensor and the filter to the respective device arrays.
		cudaAssert(cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_filter, filter, filter_size, cudaMemcpyHostToDevice));
		// Perform the convolution.
		cudnnAssert(cudnnConvolutionForward(handle, &alpha, input_desc, dev_input, filter_desc, dev_filter,
				conv_desc, conv_algo, dev_workspace, workspace_size, &beta, output_desc, dev_output));
		// Free the convolution resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(input_desc));
		cudnnAssert(cudnnDestroyFilterDescriptor(filter_desc));
		cudnnAssert(cudnnDestroyConvolutionDescriptor(conv_desc));
		cudaAssert(cudaFree(dev_input));
		cudaAssert(cudaFree(dev_workspace));
		cudaAssert(cudaFree(dev_filter));
		// Allocate the memory for the bias array on the device and fill it.
		Scalar* dev_bias;
		const std::size_t bias_size = filters * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_bias, bias_size));
		cudaAssert(cudaMemcpy(dev_bias, bias, bias_size, cudaMemcpyHostToDevice));
		// Apply the bias to the output tensor.
		cudnnAssert(cudnnAddTensor(handle, &alpha, bias_desc, dev_bias, &beta, output_desc, dev_output));
		// Copy the output tensor to the host.
		cudaAssert(cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost));
		// Free up the remaining resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(bias_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(output_desc));
		cudaAssert(cudaFree(dev_bias));
		cudaAssert(cudaFree(dev_output));
	}
	/**
	 * Performs a backward 2D convolution on a rank 4 tensor to compute the gradients of
	 * the output of the previous layer, the convolution filter, and the bias.
	 *
	 * @param input The input tensor.
	 * @param out_grad The gradient of the output.
	 * @param filter The convolution filter.
	 * @param bias The bias applied to the output of the convolution.
	 * @param input_dims The dimensions of the input tensor.
	 * @param output_dims The dimensions of the output tensor of the convolution.
	 * @param receptor_height The height of the receptor.
	 * @param receptor_width The width of the receptor.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the convolution.
	 * @param horizontal_stride The horizontal stride of the convolution.
	 * @param vertical_dilation The vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The horizontal dilation to apply to the receptor.
	 * @param prev_out_grad The gradient of the previous layer's output.
	 * @param filter_grad The gradient of the convolution filter.
	 * @param bias_grad The gradient of the bias.
	 */
	inline void convolution2d_bwd(Scalar* input, Scalar* out_grad, Scalar* filter, Scalar* bias, const Array4& input_dims,
			const Array4& output_dims, std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_padding,
			std::size_t horizontal_padding, std::size_t vertical_stride, std::size_t horizontal_stride,
			std::size_t vertical_dilation, std::size_t horizontal_dilation, /* out */ Scalar* prev_out_grad,
			/* out */ Scalar* filter_grad, /* out */ Scalar* bias_grad) const {
		std::size_t depth = input_dims[3];
		std::size_t filters = output_dims[3];
		// Create and set up the input tensor descriptor.
		cudnnTensorDescriptor_t input_desc = setup_tens_desc(input_dims);
		// Create and set up the output gradient descriptor.
		cudnnTensorDescriptor_t out_grad_desc = setup_tens_desc(output_dims);
		// Create and set up the filter descriptor.
		cudnnFilterDescriptor_t filter_desc;
		cudnnAssert(cudnnCreateFilterDescriptor(&filter_desc));
		cudnnAssert(cudnnSetFilter4dDescriptor(filter_desc, DATA_TYPE, TENSOR_FORMAT, filters, depth,
				receptor_height, receptor_width));
		// Create and set up the bias descriptor.
		cudnnTensorDescriptor_t bias_desc = setup_tens_desc(1, 1, 1, filters);
		// Create and set up the backward convolution descriptor.
		cudnnConvolutionDescriptor_t dconv_desc;
		cudnnAssert(cudnnCreateConvolutionDescriptor(&dconv_desc));
		cudnnAssert(cudnnSetConvolution2dDescriptor(dconv_desc, vertical_padding, horizontal_padding, vertical_stride,
				horizontal_stride, vertical_dilation, horizontal_dilation, CUDNN_CROSS_CORRELATION, DATA_TYPE));
		// Have cuDNN find the most performant algorithm given the convolution parameters.
		cudnnConvolutionBwdDataAlgo_t dconv_data_algo;
		cudnnAssert(cudnnGetConvolutionBackwardDataAlgorithm(handle, filter_desc, out_grad_desc, dconv_desc,
				input_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &dconv_data_algo));
		cudnnConvolutionBwdFilterAlgo_t dconv_filter_algo;
		cudnnAssert(cudnnGetConvolutionBackwardFilterAlgorithm(handle, input_desc, out_grad_desc, dconv_desc,
				filter_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &dconv_filter_algo));
		/* Have cuDNN compute the workspace memory required for the selected backward convolution algorithms given
		 * the convolution parameters. */
		std::size_t data_workspace_size;
		cudnnAssert(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, out_grad_desc, dconv_desc,
				input_desc, dconv_data_algo, &data_workspace_size));
		std::size_t filter_workspace_size;
		cudnnAssert(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, out_grad_desc, dconv_desc,
				filter_desc, dconv_filter_algo, &filter_workspace_size));
		// Allocate the memory required for the backwards data convolution on the device.
		Scalar* dev_filter;
		Scalar* dev_out_grad;
		Scalar* dev_data_workspace;
		Scalar* dev_prev_out_grad;
		const std::size_t filter_size = receptor_height * receptor_width * filters * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_filter, filter_size));
		const std::size_t out_grad_size = output_dims[0] * output_dims[1] * output_dims[2] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_out_grad, out_grad_size));
		cudaAssert(cudaMalloc(&dev_data_workspace, data_workspace_size));
		const std::size_t input_size = input_dims[0] * input_dims[1] * input_dims[2] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_prev_out_grad, input_size));
		// Copy the contents of the filter and the output gradient from the host to the device.
		cudaAssert(cudaMemcpy(dev_filter, filter, filter_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_out_grad, out_grad, out_grad_size, cudaMemcpyHostToDevice));
		// Perform the backwards data convolution.
		cudnnAssert(cudnnConvolutionBackwardData(handle, &alpha, filter_desc, dev_filter, out_grad_desc, dev_out_grad,
				dconv_desc, dconv_data_algo, dev_data_workspace, data_workspace_size, &beta, input_desc, dev_prev_out_grad));
		// Copy the previous layer's output gradient from the device to the host.
		cudaAssert(cudaMemcpy(prev_out_grad, dev_prev_out_grad, input_size, cudaMemcpyDeviceToHost));
		/* Free the resources not needed anymore (reuse the filter array for the gradient and the
		 * previous layer's output gradient array for the input). */
		cudaAssert(cudaFree(dev_data_workspace));
		// Allocate the memory required for the backwards filter convolution on the device.
		Scalar* dev_filter_workspace;
		cudaAssert(cudaMalloc(&dev_prev_out_grad, input_size));
		cudaAssert(cudaMalloc(&dev_filter_workspace, filter_workspace_size));
		// Populate the input array.
		cudaAssert(cudaMemcpy(dev_prev_out_grad, input, input_size, cudaMemcpyHostToDevice));
		// Perform the backwards filter convolution.
		cudnnAssert(cudnnConvolutionBackwardFilter(handle, &alpha, input_desc, dev_prev_out_grad, out_grad_desc, dev_out_grad,
				dconv_desc, dconv_filter_algo, dev_filter_workspace, filter_workspace_size, &beta, filter_desc, dev_filter));
		// Copy the filter gradient to the host.
		cudaAssert(cudaMemcpy(filter_grad, dev_filter, filter_size, cudaMemcpyDeviceToHost));
		// Free up resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(input_desc));
		cudnnAssert(cudnnDestroyFilterDescriptor(filter_desc));
		cudnnAssert(cudnnDestroyConvolutionDescriptor(dconv_desc));
		cudaAssert(cudaFree(dev_filter));
		cudaAssert(cudaFree(dev_filter_workspace));
		cudaAssert(cudaFree(dev_prev_out_grad));
		// Allocate the memory needed for the bias gradient on the device.
		Scalar* dev_bias_grad;
		const std::size_t bias_size = filters * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_bias_grad, bias_size));
		// Perform the backwards bias convolution.
		cudnnAssert(cudnnConvolutionBackwardBias(handle, &alpha, out_grad_desc, dev_out_grad, &beta, bias_desc,
				dev_bias_grad));
		// Copy the bias gradient from the device to the host.
		cudaAssert(cudaMemcpy(bias_grad, dev_bias_grad, bias_size, cudaMemcpyDeviceToHost));
		// Free up the remaining resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(out_grad_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(bias_desc));
		cudaAssert(cudaFree(dev_out_grad));
		cudaAssert(cudaFree(dev_bias_grad));
	}
	/**
	 * It applies the specified activation function the the input tensor.
	 *
	 * @param input The input tensor.
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower than 4).
	 * @param act_mode The type of activation function to use.
	 * @param coeff The activation function coefficient used by certain activation functions (e.g. ELU).
	 * @param output The activated output tensor.
	 */
	inline void activation_fwd(Scalar* input, const Array4& dims, cudnnActivationMode_t act_mode,
			Scalar coeff, /* out */ Scalar* output) const {
		// Create and set the activation descriptor.
		cudnnActivationDescriptor_t act_desc;
		cudnnAssert(cudnnCreateActivationDescriptor(&act_desc));
		cudnnAssert(cudnnSetActivationDescriptor(act_desc, act_mode, NAN_PROP, (double) coeff));
		// Create and set the input/output tensor descriptor.
		cudnnTensorDescriptor_t tens_desc = setup_tens_desc(dims);
		// Allocate the necessary device memory and copy the input tensor from the host to the device.
		Scalar* dev_input;
		const std::size_t size = dims[0] * dims[1] * dims[2] * dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_input, size));
		cudaAssert(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));
		// Perform the activation.
		cudnnAssert(cudnnActivationForward(handle, act_desc, &alpha, tens_desc, dev_input, &beta, tens_desc, dev_input));
		// Copy the output tensor from the device to the host.
		cudaAssert(cudaMemcpy(output, dev_input, size, cudaMemcpyDeviceToHost));
		// Free up the resources.
		cudnnAssert(cudnnDestroyActivationDescriptor(act_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(tens_desc));
		cudaAssert(cudaFree(dev_input));
	}
	/**
	 * It computes the gradient of the input of the activation function.
	 *
	 * @param input The input tensor.
	 * @param output The output tensor.
	 * @param out_grad The gradient of the output of the activation function.
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower than 4).
	 * @param act_mode The type of activation function used.
	 * @param coeff The activation function coefficient used by certain activation functions (e.g. ELU).
	 * @param prev_out_grad The gradient of the activation function's input.
	 */
	inline void activation_bwd(Scalar* input, Scalar* output, Scalar* out_grad, const Array4& dims,
			cudnnActivationMode_t act_mode, Scalar coeff, /* out */ Scalar* prev_out_grad) const {
		// Create and set the activation and tensor descriptors.
		cudnnActivationDescriptor_t act_desc;
		cudnnAssert(cudnnCreateActivationDescriptor(&act_desc));
		cudnnAssert(cudnnSetActivationDescriptor(act_desc, act_mode, NAN_PROP, (double) coeff));
		cudnnTensorDescriptor_t tens_desc = setup_tens_desc(dims);
		// Allocate and populate the required memory on the device.
		Scalar* dev_input;
		Scalar* dev_output;
		Scalar* dev_out_grad;
		const std::size_t size = dims[0] * dims[1] * dims[2] * dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_input, size));
		cudaAssert(cudaMalloc(&dev_output, size));
		cudaAssert(cudaMalloc(&dev_out_grad, size));
		cudaAssert(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_output, output, size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_out_grad, out_grad, size, cudaMemcpyHostToDevice));
		// Apply the backwards activation function.
		cudnnAssert(cudnnActivationBackward(handle, act_desc, &alpha, tens_desc, dev_output, tens_desc, dev_out_grad,
				tens_desc, dev_input, &beta, tens_desc, dev_out_grad));
		// Copy the result to the host.
		cudaAssert(cudaMemcpy(prev_out_grad, dev_out_grad, size, cudaMemcpyDeviceToHost));
		// Free up the resources.
		cudnnAssert(cudnnDestroyActivationDescriptor(act_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(tens_desc));
		cudaAssert(cudaFree(dev_input));
		cudaAssert(cudaFree(dev_output));
		cudaAssert(cudaFree(dev_out_grad));
	}
	/**
	 * It applies the softmax activation function the the input tensor.
	 *
	 * @param input The input tensor.
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower than 4).
	 * @param output The softmax activated output tensor.
	 */
	inline void softmax_fwd(Scalar* input, const Array4& dims, /* out */ Scalar* output) const {
		// Create and set up the input tensor descriptor.
		cudnnTensorDescriptor_t tens_desc = setup_tens_desc(dims);
		// Allocate the necessary memory and move the input tensor from the host to the device.
		Scalar* dev_input;
		const std::size_t size = dims[0] * dims[1] * dims[2] * dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_input, size));
		cudaAssert(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));
		// Perform the softmax activation.
		cudnnAssert(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
				tens_desc, dev_input, &beta, tens_desc, dev_input));
		// Copy the output tensor from the device to the host.
		cudaAssert(cudaMemcpy(output, dev_input, size, cudaMemcpyDeviceToHost));
		// Free up resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(tens_desc));
		cudaAssert(cudaFree(dev_input));
	}
	/**
	 * It computes the gradient of the input of the softmax activation function.
	 *
	 * @param output The output tensor.
	 * @param out_grad The gradient of the output of the activation function.
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower than 4).
	 * @param prev_out_grad The gradient of the softmax activation function's input.
	 */
	inline void softmax_bwd(Scalar* output, Scalar* out_grad, const Array4& dims,
			/* out */ Scalar* prev_out_grad) const {
		// Create and set up the tensor descriptor.
		cudnnTensorDescriptor_t tens_desc = setup_tens_desc(dims);
		// Allocate and populate the memory required for the softmax activation.
		Scalar* dev_output;
		Scalar* dev_out_grad;
		const std::size_t size = dims[0] * dims[1] * dims[2] * dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_output, size));
		cudaAssert(cudaMalloc(&dev_out_grad, size));
		cudaAssert(cudaMemcpy(dev_output, output, size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_out_grad, out_grad, size, cudaMemcpyHostToDevice));
		// Apply the softmax function.
		cudnnAssert(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, tens_desc,
				dev_output, tens_desc, dev_out_grad, &beta, tens_desc, dev_out_grad));
		// Copy the result to the host.
		cudaAssert(cudaMemcpy(prev_out_grad, dev_out_grad, size, cudaMemcpyDeviceToHost));
		// Free the device resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(tens_desc));
		cudaAssert(cudaFree(dev_output));
		cudaAssert(cudaFree(dev_out_grad));
	}
	/**
	 * Computes the dimensions of the output of the 2D pooling operation.
	 *
	 * @param input_dims The input dimensions.
	 * @param pool_mode The pooling mode.
	 * @param window_height The height of the pooling window.
	 * @param window_width The width of the pooling window.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the pooling.
	 * @param horizontal_stride The horizontal stride of the pooling.
	 * @return The output dimensions.
	 */
	inline Array4 pooling2d_output_dims(const Array4& input_dims, cudnnPoolingMode_t pool_mode, std::size_t window_height,
			std::size_t window_width, std::size_t vertical_padding, std::size_t horizontal_padding, std::size_t vertical_stride,
			std::size_t horizontal_stride) const {
		// Create and set the input tensor descriptor.
		cudnnTensorDescriptor_t input_desc = setup_tens_desc(input_dims);
		// Create and set the pooling descriptor.
		cudnnPoolingDescriptor_t pool_desc;
		cudnnAssert(cudnnCreatePoolingDescriptor(&pool_desc));
		cudnnAssert(cudnnSetPooling2dDescriptor(pool_desc, pool_mode, NAN_PROP, window_height, window_width,
				vertical_padding, horizontal_padding, vertical_stride, horizontal_stride));
		// Compute the dimensions.
		int n, c, h, w;
		cudnnAssert(cudnnGetPooling2dForwardOutputDim(pool_desc, input_desc, &n, &c, &h, &w));
		// Free the resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(input_desc));
		cudnnAssert(cudnnDestroyPoolingDescriptor(pool_desc));
		return { (std::size_t) n, (std::size_t) h, (std::size_t) w, (std::size_t) c };
	}
	/**
	 * It performs a 2D pooling operation on the input tensor.
	 *
	 * @param input The input tensor.
	 * @param input_dims The input dimensions.
	 * @param output_dims The output dimensions.
	 * @param pool_mode The pooling mode.
	 * @param window_height The height of the pooling window.
	 * @param window_width The width of the pooling window.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the pooling.
	 * @param horizontal_stride The horizontal stride of the pooling.
	 * @param output The output of the pooling operation.
	 */
	inline void pooling2d_fwd(Scalar* input, const Array4& input_dims, const Array4& output_dims, cudnnPoolingMode_t pool_mode,
			std::size_t window_height, std::size_t window_width, std::size_t vertical_padding, std::size_t horizontal_padding,
			std::size_t vertical_stride, std::size_t horizontal_stride, /* out */ Scalar* output) const {
		// Create and set the input tensor descriptor.
		cudnnTensorDescriptor_t input_desc = setup_tens_desc(input_dims);
		// Create and set the output tensor descriptor.
		cudnnTensorDescriptor_t output_desc = setup_tens_desc(output_dims);
		// Create and set the pooling descriptor.
		cudnnPoolingDescriptor_t pool_desc;
		cudnnAssert(cudnnCreatePoolingDescriptor(&pool_desc));
		cudnnAssert(cudnnSetPooling2dDescriptor(pool_desc, pool_mode, NAN_PROP, window_height, window_width,
				vertical_padding, horizontal_padding, vertical_stride, horizontal_stride));
		// Allocate the necessary device memory and copy the input tensor from the host to the device.
		Scalar* dev_input;
		Scalar* dev_output;
		const std::size_t input_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_input, input_size));
		const std::size_t output_size = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_output, output_size));
		cudaAssert(cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice));
		// Perform the pooling.
		cudnnAssert(cudnnPoolingForward(handle, pool_desc, &alpha, input_desc, dev_input, &beta, output_desc, dev_output));
		// Copy the output tensor from the device to the host.
		cudaAssert(cudaMemcpy(output, dev_output, output_size, cudaMemcpyDeviceToHost));
		// Free up the resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(input_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(output_desc));
		cudnnAssert(cudnnDestroyPoolingDescriptor(pool_desc));
		cudaAssert(cudaFree(dev_input));
		cudaAssert(cudaFree(dev_output));
	}
	/**
	 * Computes the gradient of the input of the pooling layer.
	 *
	 * @param input The input tensor.
	 * @param output The output tensor.
	 * @param out_grad The gradient of the output.
	 * @param input_dims The input dimensions.
	 * @param output_dims The output dimensions.
	 * @param pool_mode The pooling mode.
	 * @param window_height The height of the pooling window.
	 * @param window_width The width of the pooling window.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the pooling.
	 * @param horizontal_stride The horizontal stride of the pooling.
	 * @param prev_out_grad The gradient of the input of the pooling layer.
	 */
	inline void pooling2d_bwd(Scalar* input, Scalar* output, Scalar* out_grad, const Array4& input_dims, const Array4& output_dims,
			cudnnPoolingMode_t pool_mode, std::size_t window_height, std::size_t window_width, std::size_t vertical_padding,
			std::size_t horizontal_padding, std::size_t vertical_stride, std::size_t horizontal_stride,
			/* out */ Scalar* prev_out_grad) const {
		// Create and set the input tensor descriptor.
		cudnnTensorDescriptor_t input_desc = setup_tens_desc(input_dims);
		// Create and set the output tensor descriptor.
		cudnnTensorDescriptor_t output_desc = setup_tens_desc(output_dims);
		// Create and set the pooling descriptor.
		cudnnPoolingDescriptor_t pool_desc;
		cudnnAssert(cudnnCreatePoolingDescriptor(&pool_desc));
		cudnnAssert(cudnnSetPooling2dDescriptor(pool_desc, pool_mode, NAN_PROP, window_height, window_width,
				vertical_padding, horizontal_padding, vertical_stride, horizontal_stride));
		// Allocate the necessary device memory.
		Scalar* dev_input;
		Scalar* dev_output;
		Scalar* dev_out_grad;
		Scalar* dev_prev_out_grad;
		const std::size_t input_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_input, input_size));
		const std::size_t output_size = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_output, output_size));
		cudaAssert(cudaMalloc(&dev_out_grad, output_size));
		cudaAssert(cudaMalloc(&dev_prev_out_grad, input_size));
		// Copy the contents of the input, output, and output gradient tensors to the device.
		cudaAssert(cudaMemcpy(dev_input, input, input_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_output, output, input_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_prev_out_grad, out_grad, input_size, cudaMemcpyHostToDevice));
		// Perform the backwards pooling.
		cudnnAssert(cudnnPoolingBackward(handle, pool_desc, &alpha, output_desc, dev_output, output_desc, dev_out_grad,
				input_desc, dev_input, &beta, input_desc, dev_prev_out_grad));
		// Copy the input gradient tensor from the device to the host.
		cudaAssert(cudaMemcpy(prev_out_grad, dev_prev_out_grad, input_size, cudaMemcpyDeviceToHost));
		// Free up the resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(input_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(output_desc));
		cudnnAssert(cudnnDestroyPoolingDescriptor(pool_desc));
		cudaAssert(cudaFree(dev_input));
		cudaAssert(cudaFree(dev_output));
		cudaAssert(cudaFree(dev_out_grad));
		cudaAssert(cudaFree(dev_prev_out_grad));
	}
	/**
	 * Applies the batch normalization function to the input data and updates the running mean and
	 * variance averages.
	 *
	 * @param input The input tensor.
	 * @param gamma The gamma scaling tensor.
	 * @param beta The beta bias.
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower
	 * than 4).
	 * @param spatial Whether the batch normalization should be performed in spatial or per-activation
	 * mode.
	 * @param exp_avg_factor The exponential average factor for the running mean and variance averages.
	 * @param epsilon A small constant for numerical stability.
	 * @param means The running mean average (it gets updated by the method).
	 * @param vars The running variance average (it gets updated by the method).
	 * @param output The output tensor.
	 * @param mean_cache The cached mean for back-propagation.
	 * @param inv_var_cache The cached inverse variance for back-propagation.
	 */
	inline void batch_norm_fwd_training(Scalar* input, Scalar* gamma, Scalar* beta, const Array4& dims, bool spatial,
			Scalar exp_avg_factor, Scalar epsilon, /* in/out */ Scalar* means, /* in/out */ Scalar* vars,
			/* out */ Scalar* output, /* out */ Scalar* mean_cache, /* out */ Scalar* inv_var_cache) const {
		// Setup the tensor descriptors.
		cudnnTensorDescriptor_t in_out_tens_desc = setup_tens_desc(dims);
		cudnnTensorDescriptor_t mean_var_tens_desc = spatial ? setup_tens_desc(1, 1, 1, dims[3]) :
				setup_tens_desc(1, dims[1], dims[2], dims[3]);
		// Allocate the necessary device memory.
		Scalar* dev_input;
		Scalar* dev_output;
		Scalar* dev_gamma;
		Scalar* dev_beta;
		Scalar* dev_mean;
		Scalar* dev_var;
		Scalar* dev_mean_cache;
		Scalar* dev_inv_var_cache;
		std::size_t in_out_tens_size = dims[0] * dims[1] * dims[2] * dims[3];
		std::size_t mean_var_tens_size = spatial ? dims[3] : dims[1] * dims[2] * dims[3];
		cudaAssert(cudaMalloc(&dev_input, in_out_tens_size));
		cudaAssert(cudaMalloc(&dev_output, in_out_tens_size));
		cudaAssert(cudaMalloc(&dev_gamma, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_beta, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_mean, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_var, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_mean_cache, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_inv_var_cache, mean_var_tens_size));
		// Populate the device arrays.
		cudaAssert(cudaMemcpy(dev_input, input, in_out_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_gamma, gamma, mean_var_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_beta, beta, mean_var_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_mean, means, mean_var_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_var, vars, mean_var_tens_size, cudaMemcpyHostToDevice));
		// Perform the batch normalization training pass.
		cudnnAssert(cudnnBatchNormalizationForwardTraining(handle, spatial ? CUDNN_BATCHNORM_SPATIAL_PERSISTENT :
				CUDNN_BATCHNORM_PER_ACTIVATION, &alpha, &this->beta, in_out_tens_desc, dev_input, in_out_tens_desc,
				dev_output, mean_var_tens_desc, dev_gamma, dev_beta, exp_avg_factor, dev_mean, dev_var, epsilon,
				dev_mean_cache, dev_inv_var_cache));
		// Copy the required data back to the host.
		cudaAssert(cudaMemcpy(means, dev_mean, mean_var_tens_size, cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(vars, dev_var, mean_var_tens_size, cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(output, dev_output, in_out_tens_size, cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(mean_cache, dev_mean_cache, mean_var_tens_size, cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(inv_var_cache, dev_inv_var_cache, mean_var_tens_size, cudaMemcpyDeviceToHost));
		// Free resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(in_out_tens_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(mean_var_tens_desc));
		cudaAssert(cudaFree(dev_input));
		cudaAssert(cudaFree(dev_output));
		cudaAssert(cudaFree(dev_gamma));
		cudaAssert(cudaFree(dev_beta));
		cudaAssert(cudaFree(dev_mean));
		cudaAssert(cudaFree(dev_var));
		cudaAssert(cudaFree(dev_mean_cache));
		cudaAssert(cudaFree(dev_inv_var_cache));
	}
	/**
	 * Applies the batch normalization function to the input data for inference using the running
	 * mean and variance averages.
	 *
	 * @param input The input tensor.
	 * @param gamma The gamma scaling tensor.
	 * @param beta The beta bias.
	 * @param means The running mean average.
	 * @param vars The running variance average.
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower
	 * than 4).
	 * @param spatial Whether the batch normalization should be performed in spatial or per-activation
	 * mode.
	 * @param epsilon A small constant for numerical stability.
	 * @param output The output tensor.
	 */
	inline void batch_norm_fwd_inference(Scalar* input, Scalar* gamma, Scalar* beta, Scalar* means, Scalar* vars,
			const Array4& dims, bool spatial, Scalar epsilon, /* out */ Scalar* output) const {
		// Setup the tensor descriptors.
		cudnnTensorDescriptor_t in_out_tens_desc = setup_tens_desc(dims);
		cudnnTensorDescriptor_t mean_var_tens_desc = spatial ? setup_tens_desc(1, 1, 1, dims[3]) :
				setup_tens_desc(1, dims[1], dims[2], dims[3]);
		// Allocate the necessary device memory.
		Scalar* dev_input;
		Scalar* dev_output;
		Scalar* dev_gamma;
		Scalar* dev_beta;
		Scalar* dev_mean;
		Scalar* dev_var;
		std::size_t in_out_tens_size = dims[0] * dims[1] * dims[2] * dims[3];
		std::size_t mean_var_tens_size = spatial ? dims[3] : dims[1] * dims[2] * dims[3];
		cudaAssert(cudaMalloc(&dev_input, in_out_tens_size));
		cudaAssert(cudaMalloc(&dev_output, in_out_tens_size));
		cudaAssert(cudaMalloc(&dev_gamma, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_beta, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_mean, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_var, mean_var_tens_size));
		// Populate the device arrays.
		cudaAssert(cudaMemcpy(dev_input, input, in_out_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_gamma, gamma, mean_var_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_beta, beta, mean_var_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_mean, means, mean_var_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_var, vars, mean_var_tens_size, cudaMemcpyHostToDevice));
		// Perform the batch normalization training pass.
		cudnnAssert(cudnnBatchNormalizationForwardInference(handle, spatial ? CUDNN_BATCHNORM_SPATIAL_PERSISTENT :
				CUDNN_BATCHNORM_PER_ACTIVATION, &alpha, &this->beta, in_out_tens_desc, dev_input, in_out_tens_desc,
				dev_output, mean_var_tens_desc, dev_gamma, dev_beta, dev_mean, dev_var, epsilon));
		// Copy the required data back to the host.
		cudaAssert(cudaMemcpy(output, dev_output, in_out_tens_size, cudaMemcpyDeviceToHost));
		// Free resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(in_out_tens_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(mean_var_tens_desc));
		cudaAssert(cudaFree(dev_input));
		cudaAssert(cudaFree(dev_output));
		cudaAssert(cudaFree(dev_gamma));
		cudaAssert(cudaFree(dev_beta));
		cudaAssert(cudaFree(dev_mean));
		cudaAssert(cudaFree(dev_var));
	}
	/**
	 * Performs the backward pass of the batch normalization function and computes the gradients
	 * of the function's input, the gamma scaling tensor, and the beta bias tensor.
	 *
	 * @param input The input tensor used.
	 * @param out_grad The gradient of the output of the function.
	 * @param gamma The gamma scaling tensor.
	 * @param mean_cache The mean cached during the forward pass.
	 * @param inv_var_cache The inverse variance cached during the forward pass.
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower
	 * than 4).
	 * @param spatial Whether the batch normalization should be performed in spatial or per-activation
	 * mode.
	 * @param epsilon A small constant for numerical stability.
	 * @param prev_out_grad The gradient of the batch normalization function's input.
	 * @param gamma_grad The gradient of gamma.
	 * @param beta_grad The gradient of beta.
	 */
	inline void batch_norm_bwd(Scalar* input, Scalar* out_grad, Scalar* gamma, Scalar* mean_cache,
			Scalar* inv_var_cache, const Array4& dims, bool spatial, Scalar epsilon, /* out */ Scalar* prev_out_grad,
			/* out */ Scalar* gamma_grad, /* out */ Scalar* beta_grad) const {
		// Setup the tensor descriptors.
		cudnnTensorDescriptor_t in_out_tens_desc = setup_tens_desc(dims);
		cudnnTensorDescriptor_t mean_var_tens_desc = spatial ? setup_tens_desc(1, 1, 1, dims[3]) :
				setup_tens_desc(1, dims[1], dims[2], dims[3]);
		// Allocate the necessary device memory.
		Scalar* dev_input;
		Scalar* dev_out_grad;
		Scalar* dev_gamma;
		Scalar* dev_mean_cache;
		Scalar* dev_inv_var_cache;
		Scalar* dev_prev_out_grad;
		Scalar* dev_gamma_grad;
		Scalar* dev_beta_grad;
		std::size_t in_out_tens_size = dims[0] * dims[1] * dims[2] * dims[3];
		std::size_t mean_var_tens_size = spatial ? dims[3] : dims[1] * dims[2] * dims[3];
		cudaAssert(cudaMalloc(&dev_input, in_out_tens_size));
		cudaAssert(cudaMalloc(&dev_out_grad, in_out_tens_size));
		cudaAssert(cudaMalloc(&dev_gamma, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_mean_cache, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_inv_var_cache, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_prev_out_grad, in_out_tens_size));
		cudaAssert(cudaMalloc(&dev_gamma_grad, mean_var_tens_size));
		cudaAssert(cudaMalloc(&dev_beta_grad, mean_var_tens_size));
		// Populate the device arrays.
		cudaAssert(cudaMemcpy(dev_input, input, in_out_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_out_grad, out_grad, in_out_tens_size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_gamma, gamma, mean_var_tens_size, cudaMemcpyHostToDevice));
		// Perform the batch normalization training pass.
		cudnnAssert(cudnnBatchNormalizationBackward(handle, spatial ? CUDNN_BATCHNORM_SPATIAL_PERSISTENT :
				CUDNN_BATCHNORM_PER_ACTIVATION, &alpha, &beta, &alpha, &beta, in_out_tens_desc, dev_input,
				in_out_tens_desc, dev_out_grad, in_out_tens_desc, dev_prev_out_grad, mean_var_tens_desc,
				dev_gamma, dev_gamma_grad, dev_beta_grad, epsilon, dev_mean_cache, dev_inv_var_cache));
		// Copy the required data back to the host.
		cudaAssert(cudaMemcpy(prev_out_grad, dev_prev_out_grad, in_out_tens_size, cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(gamma_grad, dev_gamma_grad, mean_var_tens_size, cudaMemcpyDeviceToHost));
		cudaAssert(cudaMemcpy(beta_grad, dev_beta_grad, mean_var_tens_size, cudaMemcpyDeviceToHost));
		// Free resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(in_out_tens_desc));
		cudnnAssert(cudnnDestroyTensorDescriptor(mean_var_tens_desc));
		cudaAssert(cudaFree(dev_input));
		cudaAssert(cudaFree(dev_out_grad));
		cudaAssert(cudaFree(dev_gamma));
		cudaAssert(cudaFree(dev_mean_cache));
		cudaAssert(cudaFree(dev_inv_var_cache));
		cudaAssert(cudaFree(dev_prev_out_grad));
		cudaAssert(cudaFree(dev_gamma_grad));
		cudaAssert(cudaFree(dev_beta_grad));
	}
	/**
	 * It applies the dropout function to the input tensor.
	 *
	 * @param input The input tensor
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower than 4).
	 * @param dropout The average factor of elements to set to 0 in the input tensor.
	 * @param output The output tensor.
	 * @param reserve The reserve used for backpropagation.
	 */
	inline void dropout_fwd(Scalar* input, const Array4& dims, Scalar dropout, /* out */ Scalar* output,
			/* out */ std::vector<Scalar>& reserve) const {
		// Create and set the input/output/reserve tensor descriptor.
		cudnnTensorDescriptor_t tens_desc = setup_tens_desc(dims);
		// Create and set the dropout descriptor.
		cudnnDropoutDescriptor_t dropout_desc;
		cudnnAssert(cudnnCreateDropoutDescriptor(&dropout_desc));
		cudnnAssert(cudnnSetDropoutDescriptor(dropout_desc, handle, (float) dropout, nullptr, 0, 0));
		// Calculate the required reserve size.
		std::size_t reserve_size;
		cudnnAssert(cudnnDropoutGetReserveSpaceSize(tens_desc, &reserve_size));
		// Allocate the memory for the device arrays and reserve space.
		Scalar* dev_input;
		Scalar* dev_output;
		Scalar* dev_reserve;
		const std::size_t size = dims[0] * dims[1] * dims[2] * dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_input, size));
		cudaAssert(cudaMalloc(&dev_output, size));
		cudaAssert(cudaMalloc(&dev_reserve, reserve_size));
		// Copy the contents of the input tensor from the host to the device.
		cudaAssert(cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice));
		// Perform the dropout.
		cudnnAssert(cudnnDropoutForward(handle, dropout_desc, tens_desc, dev_input, tens_desc, dev_output,
				dev_reserve, reserve_size));
		// Copy the results to the host.
		cudaAssert(cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost));
		reserve = std::vector<Scalar>(reserve_size / SCALAR_SIZE);
		cudaAssert(cudaMemcpy(reserve.data(), dev_reserve, size, cudaMemcpyDeviceToHost));
		// Free resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(tens_desc));
		cudnnAssert(cudnnDestroyDropoutDescriptor(dropout_desc));
		cudaAssert(cudaFree(dev_input));
		cudaAssert(cudaFree(dev_output));
		cudaAssert(cudaFree(dev_reserve));
	}
	/**
	 * It computes the gradient of the input of the dropout function.
	 *
	 * @param out_grad The gradient of the output of the dropout function.
	 * @param reserve The reserve filled during the forward pass.
	 * @param dims The dimensions of the input/output tensor (extended by 1s for data of rank lower than 4).
	 * @param dropout The average factor of elements set to 0 in the input tensor.
	 * @param prev_out_grad The gradient of the input of the dropout function.
	 */
	inline void dropout_bwd(Scalar* out_grad, std::vector<Scalar>& reserve, const Array4& dims, Scalar dropout,
			/* out */ Scalar* prev_out_grad) const {
		// Create and set the input/output/reserve tensor descriptor.
		cudnnTensorDescriptor_t tens_desc = setup_tens_desc(dims);
		// Create and set the dropout descriptor.
		cudnnDropoutDescriptor_t dropout_desc;
		cudnnAssert(cudnnCreateDropoutDescriptor(&dropout_desc));
		cudnnAssert(cudnnSetDropoutDescriptor(dropout_desc, handle, (float) dropout, nullptr, 0, 0));
		// Allocate the memory for the device arrays and reserve space.
		Scalar* dev_out_grad;
		Scalar* dev_reserve;
		Scalar* dev_prev_out_grad;
		const std::size_t size = dims[0] * dims[1] * dims[2] * dims[3] * SCALAR_SIZE;
		cudaAssert(cudaMalloc(&dev_out_grad, size));
		cudaAssert(cudaMalloc(&dev_reserve, size));
		cudaAssert(cudaMalloc(&dev_prev_out_grad, size));
		// Copy the contents of the input tensor and the reserve vector from the host to the device.
		cudaAssert(cudaMemcpy(dev_out_grad, out_grad, size, cudaMemcpyHostToDevice));
		cudaAssert(cudaMemcpy(dev_reserve, reserve.data(), size, cudaMemcpyHostToDevice));
		// Perform the dropout backward pass.
		cudnnAssert(cudnnDropoutBackward(handle, dropout_desc, tens_desc, dev_out_grad, tens_desc, dev_prev_out_grad,
				dev_reserve, reserve.size() * SCALAR_SIZE));
		// Copy the result to the host.
		cudaAssert(cudaMemcpy(prev_out_grad, dev_prev_out_grad, size, cudaMemcpyDeviceToHost));
		// Free the device resources.
		cudnnAssert(cudnnDestroyTensorDescriptor(tens_desc));
		cudnnAssert(cudnnDestroyDropoutDescriptor(dropout_desc));
		cudaAssert(cudaFree(dev_out_grad));
		cudaAssert(cudaFree(dev_reserve));
		cudaAssert(cudaFree(dev_prev_out_grad));
	}
private:
	cudnnHandle_t handle;
	const Scalar alpha;
	const Scalar beta;
	inline CuDNNHandle() :
			handle(),
			alpha(1),
			beta(0) {
		// Create the cuBLAS handle.
		cudnnAssert(cudnnCreate(&handle));
	}
	/**
	 * Creates and sets a tensor descriptor using the specified dimensions.
	 *
	 * @param dims The dimensions of the tensor.
	 * @return The tensor descriptor.
	 */
	inline cudnnTensorDescriptor_t setup_tens_desc(const Array4& dims) const {
		cudnnTensorDescriptor_t tens_desc;
		cudnnAssert(cudnnCreateTensorDescriptor(&tens_desc));
		cudnnAssert(cudnnSetTensor4dDescriptor(tens_desc, TENSOR_FORMAT, DATA_TYPE, dims[0], dims[3],
				dims[1], dims[2]));
		return tens_desc;
	}
	/**
	 * Creates and sets a tensor descriptor using the specified dimensions.
	 *
	 * @param n The number of samples.
	 * @param h Height.
	 * @param w Width.
	 * @param c The number of color channels.
	 * @return The tensor descriptor.
	 */
	inline cudnnTensorDescriptor_t setup_tens_desc(std::size_t n, std::size_t h, std::size_t w,
			std::size_t c) const {
		cudnnTensorDescriptor_t tens_desc;
		cudnnAssert(cudnnCreateTensorDescriptor(&tens_desc));
		cudnnAssert(cudnnSetTensor4dDescriptor(tens_desc, TENSOR_FORMAT, DATA_TYPE, (int) n, (int) c,
				(int) h, (int) w));
		return tens_desc;
	}
};

}
} /* namespace cattle */

#endif /* CATTL3_UTILS_CUDNNHANDLE_H_ */
