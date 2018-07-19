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
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

#include "CuDNNError.hpp"
#include "CuDNNTensor.hpp"
#include "CUDAError.hpp"

namespace cattle {
namespace gpu {

/**
 * A singleton utility class providing methods for GPU accelerated deep neural network
 * operations on rank 4 data.
 */
template<typename Scalar>
class CuDNNHandle {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	typedef CuDNNHandle<Scalar> Self;
	static constexpr cudnnNanPropagation_t NAN_PROP = CUDNN_PROPAGATE_NAN;
public:
	CuDNNHandle(const Self&) = delete;
	~CuDNNHandle() {
		cudnnAssert(cudnnDestroy(handle));
	}
	Self& operator=(const Self&) = delete;
	/**
	 * @return A reference to the only instance of the class.
	 */
	inline static const Self& get_instance() {
		static Self instance;
		return instance;
	}
	/**
	 * It performs the specified operation on tensors a and b and saves the result in c.
	 *
	 * \f$C = op(\alpha * A, \beta * B) + \gamma * C\f$
	 *
	 * @param a The first operand.
	 * @param alpha The scaling factor of the first operand.
	 * @param b The second operand.
	 * @param beta The scaling factor of the second operand.
	 * @param op_type The operation type.
	 * @param gamma The scaling factor of the result tensor.
	 * @param c The resutl tensor.
	 */
	inline void op(const CuDNNTensor<Scalar>& a, Scalar alpha, const CuDNNTensor<Scalar>& b, Scalar beta,
			cudnnOpTensorOp_t op_type, Scalar gamma, /* in/out */ CuDNNTensor<Scalar>& c) const {
		cudnnOpTensorDescriptor_t op_desc;
		cudnnAssert(cudnnCreateOpTensorDescriptor(&op_desc));
		cudnnAssert(cudnnSetOpTensorDescriptor(op_desc, op_type, CuDNNTensor<Scalar>::DATA_TYPE,
				CUDNN_PROPAGATE_NAN));
		cudnnAssert(cudnnOpTensor(op_desc, alpha, a.desc(), a.data(), beta, b.desc(), b.data(),
				gamma, c.desc(), c.data()));
		cudnnAssert(cudnnDestroyOpTensorDescriptor(op_desc));
	}
	/**
	 * Adds a bias tensor to another tensor.
	 *
	 * @param bias The bias tensor.
	 * @param tensor The tensor to which the bias is to be added.
	 */
	inline void add_bias(const CuDNNTensor<Scalar>& bias, /* in/out */ CuDNNTensor<Scalar>& tensor) const {
		cudnnAssert(cudnnAddTensor(handle, &alpha, bias.desc(), bias.data(), &alpha,
				tensor.desc(), tensor.data()));
	}
	/**
	 * It computes the dimensions of the output tensor of the convolution.
	 *
	 * @param input_height The input height.
	 * @param input_width The input width.
	 * @param input_channels The number of input channels.
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
	 * @param output_height The output height.
	 * @param output_width The output width.
	 * @param output_channels The number of output channels.
	 */
	inline void conv2d_output_dims(std::size_t input_height, std::size_t input_width, std::size_t input_channels,
			std::size_t filters, std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_padding,
			std::size_t horizontal_padding, std::size_t vertical_stride, std::size_t horizontal_stride,
			std::size_t vertical_dilation, std::size_t horizontal_dilation, /* out */ std::size_t& output_height,
			/* out */ std::size_t& output_width, /* out */ std::size_t& output_channels) const {
		// Create and set the input tensor descriptor.
		cudnnTensorDescriptor_t input_desc;
		CuDNNTensorDescriptorManager<>::create_descriptor(input_desc, CuDNNTensor<Scalar>::DATA_TYPE,
				CuDNNTensor<Scalar>::TENSOR_FORMAT, 1, input_height, input_width, input_channels);
		// Create and set up the filter descriptor.
		cudnnFilterDescriptor_t filter_desc;
		CuDNNTensorDescriptorManager<true>::create_descriptor(filter_desc, CuDNNTensor<Scalar>::DATA_TYPE,
				CuDNNTensor<Scalar>::TENSOR_FORMAT, filters, receptor_height, receptor_width, input_channels);
		// Create and set up the convolution descriptor.
		cudnnConvolutionDescriptor_t conv_desc;
		cudnnAssert(cudnnCreateConvolutionDescriptor(&conv_desc));
		cudnnAssert(cudnnSetConvolution2dDescriptor(conv_desc, vertical_padding, horizontal_padding, vertical_stride,
				horizontal_stride, vertical_dilation, horizontal_dilation, CUDNN_CROSS_CORRELATION,
				CuDNNTensor<Scalar>::DATA_TYPE));
		// Compute the dimensions.
		int n, h, w, c;
		cudnnAssert(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w));
		// Free the resources.
		cudnnAssert(cudnnDestroyConvolutionDescriptor(conv_desc));
		CuDNNTensorDescriptorManager<>::destroy_descriptor(input_desc);
		CuDNNTensorDescriptorManager<true>::destroy_descriptor(filter_desc);
		output_height = (std::size_t) h;
		output_width = (std::size_t) w;
		output_channels = (std::size_t) c;
	}
	/**
	 * Performs a GPU accelerated 2D convolution on a rank 4 tensor.
	 *
	 * @param input The input tensor.
	 * @param filter The convolution filter.
	 * @param bias The bias tensor to apply to the output of the convolution.
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
	inline void convolution2d_fwd(const CuDNNTensor<Scalar>& input, const CuDNNTensor<Scalar,true>& filter,
			const CuDNNTensor<Scalar>& bias, std::size_t vertical_padding, std::size_t horizontal_padding,
			std::size_t vertical_stride, std::size_t horizontal_stride, std::size_t vertical_dilation,
			std::size_t horizontal_dilation, /* out */ CuDNNTensor<Scalar>& output) const {
		// Create and set up the convolution descriptor.
		cudnnConvolutionDescriptor_t conv_desc;
		cudnnAssert(cudnnCreateConvolutionDescriptor(&conv_desc));
		cudnnAssert(cudnnSetConvolution2dDescriptor(conv_desc, vertical_padding, horizontal_padding, vertical_stride,
				horizontal_stride, vertical_dilation, horizontal_dilation, CUDNN_CROSS_CORRELATION,
				CuDNNTensor<Scalar>::DATA_TYPE));
		// Have cuDNN find the most performant algorithm given the convolution parameters.
		cudnnConvolutionFwdAlgo_t conv_algo;
		cudnnAssert(cudnnGetConvolutionForwardAlgorithm(handle, input.desc(), filter.desc(), conv_desc,
				output.desc(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_algo));
		/* Have cuDNN compute the workspace memory required for the selected convolution algorithm given
		 * the convolution parameters. */
		std::size_t workspace_size;
		cudnnAssert(cudnnGetConvolutionForwardWorkspaceSize(handle, input.desc(), filter.desc(), conv_desc,
				output.desc(), conv_algo, &workspace_size));
		// Allocate the memory for the workspace required for the convolution.
		Scalar* workspace;
		cudaAssert(cudaMalloc(&workspace, workspace_size));
		// Perform the convolution.
		cudnnAssert(cudnnConvolutionForward(handle, &alpha, input.desc(), input.data(), filter.desc(),
				filter.data(), conv_desc, conv_algo, workspace, workspace_size, &beta, output.desc(),
				output.data()));
		// Free the convolution resources.
		cudnnAssert(cudnnDestroyConvolutionDescriptor(conv_desc));
		cudaAssert(cudaFree(workspace));
		// Apply the bias to the output tensor.
		add_bias(bias, output);
	}
	/**
	 * Performs a backward 2D convolution on a rank 4 tensor to compute the gradients of
	 * the output of the previous layer, the convolution filter, and the bias.
	 *
	 * @param input The input tensor.
	 * @param out_grad The gradient of the output tensor.
	 * @param filter The convolution filter.
	 * @param bias The bias tensor applied to the output of the convolution.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the convolution.
	 * @param horizontal_stride The horizontal stride of the convolution.
	 * @param vertical_dilation The vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The horizontal dilation to apply to the receptor.
	 * @param prev_out_grad The gradient of the previous layer's output tensor.
	 * @param filter_grad The gradient of the convolution filter.
	 * @param bias_grad The gradient of the bias tensor.
	 */
	inline void convolution2d_bwd(const CuDNNTensor<Scalar>& input, const CuDNNTensor<Scalar>& out_grad,
			const CuDNNTensor<Scalar,true>& filter, const CuDNNTensor<Scalar>& bias, std::size_t vertical_padding,
			std::size_t horizontal_padding, std::size_t vertical_stride, std::size_t horizontal_stride,
			std::size_t vertical_dilation, std::size_t horizontal_dilation, /* out */ CuDNNTensor<Scalar>& prev_out_grad,
			/* out */ CuDNNTensor<Scalar,true>& filter_grad, /* out */ CuDNNTensor<Scalar>& bias_grad) const {
		// Create and set up the backward convolution descriptor.
		cudnnConvolutionDescriptor_t dconv_desc;
		cudnnAssert(cudnnCreateConvolutionDescriptor(&dconv_desc));
		cudnnAssert(cudnnSetConvolution2dDescriptor(dconv_desc, vertical_padding, horizontal_padding, vertical_stride,
				horizontal_stride, vertical_dilation, horizontal_dilation, CUDNN_CROSS_CORRELATION,
				CuDNNTensor<Scalar>::DATA_TYPE));
		// Have cuDNN find the most performant algorithm given the convolution parameters.
		cudnnConvolutionBwdDataAlgo_t dconv_data_algo;
		cudnnAssert(cudnnGetConvolutionBackwardDataAlgorithm(handle, filter.desc(), out_grad.desc(), dconv_desc,
				prev_out_grad.desc(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &dconv_data_algo));
		cudnnConvolutionBwdFilterAlgo_t dconv_filter_algo;
		cudnnAssert(cudnnGetConvolutionBackwardFilterAlgorithm(handle, input.desc(), out_grad.desc(), dconv_desc,
				filter_grad.desc(), CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &dconv_filter_algo));
		/* Have cuDNN compute the data_workspace memory required for the selected backward convolution algorithms given
		 * the convolution parameters. */
		std::size_t data_workspace_size;
		cudnnAssert(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter.desc(), out_grad.desc(), dconv_desc,
				prev_out_grad.desc(), dconv_data_algo, &data_workspace_size));
		std::size_t filter_workspace_size;
		cudnnAssert(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input.desc(), out_grad.desc(), dconv_desc,
				filter_grad.desc(), dconv_filter_algo, &filter_workspace_size));
		// Allocate the memory required for the backwards data convolution on the device.
		Scalar* data_workspace;
		cudaAssert(cudaMalloc(&data_workspace, data_workspace_size));
		// Perform the backwards data convolution.
		cudnnAssert(cudnnConvolutionBackwardData(handle, &alpha, filter.desc(), filter.data(), out_grad.desc(),
				out_grad.data(), dconv_desc, dconv_data_algo, data_workspace, data_workspace_size, &beta,
				prev_out_grad.desc(), prev_out_grad.data()));
		// Free the resources.
		cudaAssert(cudaFree(data_workspace));
		// Allocate the memory required for the backwards filter convolution on the device.
		Scalar* filter_workspace;
		cudaAssert(cudaMalloc(&filter_workspace, filter_workspace_size));
		// Perform the backwards filter convolution.
		cudnnAssert(cudnnConvolutionBackwardFilter(handle, &alpha, input.desc(), input.data(), out_grad.desc(),
				out_grad.data(), dconv_desc, dconv_filter_algo, filter_workspace, filter_workspace_size, &beta,
				filter_grad.desc(), filter_grad.data()));
		// Free up resources.
		cudaAssert(cudaFree(filter_workspace));
		cudnnAssert(cudnnDestroyConvolutionDescriptor(dconv_desc));
		// Perform the backwards bias convolution.
		cudnnAssert(cudnnConvolutionBackwardBias(handle, &alpha, out_grad.desc(), out_grad.data(), &beta,
				bias_grad.desc(), bias_grad.data()));
	}
	/**
	 * It applies the specified activation function the the input tensor.
	 *
	 * @param input The input tensor.
	 * @param act_mode The type of activation function to use.
	 * @param coeff The activation function coefficient used by certain activation functions (e.g. ELU).
	 * @param output The activated output tensor.
	 */
	inline void activation_fwd(const CuDNNTensor<Scalar>& input, cudnnActivationMode_t act_mode, Scalar coeff,
			/* out */ CuDNNTensor<Scalar>& output) const {
		cudnnActivationDescriptor_t act_desc;
		cudnnAssert(cudnnCreateActivationDescriptor(&act_desc));
		cudnnAssert(cudnnSetActivationDescriptor(act_desc, act_mode, NAN_PROP, (double) coeff));
		cudnnAssert(cudnnActivationForward(handle, act_desc, &alpha, input.desc(), input.data(), &beta,
				output.desc(), output.data()));
		cudnnAssert(cudnnDestroyActivationDescriptor(act_desc));
	}
	/**
	 * It computes the gradient of the input of the activation function.
	 *
	 * @param input The input tensor.
	 * @param output The output tensor.
	 * @param act_mode The type of activation function used.
	 * @param coeff The activation function coefficient used by certain activation functions (e.g. ELU).
	 * @param in_out_grad The gradient of the output of the activation function and after the method
	 * finishes execution, the gradient of the input of the activation function.
	 */
	inline void activation_bwd(const CuDNNTensor<Scalar>& input, const CuDNNTensor<Scalar>& output,
			cudnnActivationMode_t act_mode, Scalar coeff, /* in/out */ CuDNNTensor<Scalar>& in_out_grad) const {
		cudnnActivationDescriptor_t act_desc;
		cudnnAssert(cudnnCreateActivationDescriptor(&act_desc));
		cudnnAssert(cudnnSetActivationDescriptor(act_desc, act_mode, NAN_PROP, (double) coeff));
		cudnnAssert(cudnnActivationBackward(handle, act_desc, &alpha, output.desc(), output.data(), in_out_grad.desc(),
				in_out_grad.data(), input.desc(), input.data(), &beta, in_out_grad.desc(), in_out_grad.data()));
		cudnnAssert(cudnnDestroyActivationDescriptor(act_desc));
	}
	/**
	 * It applies the softmax activation function the the input tensor.
	 *
	 * @param input The input tensor.
	 * @param output The softmax activated output tensor.
	 */
	inline void softmax_fwd(const CuDNNTensor<Scalar>& input, /* out */ CuDNNTensor<Scalar>& output) const {
		cudnnAssert(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
				input.desc(), input.data(), &beta, output.desc(), output.data()));
	}
	/**
	 * It computes the gradient of the input of the softmax activation function.
	 *
	 * @param output The output tensor.
	 * @param in_out_grad The gradient of the output of the activation function and after the method finishes
	 * execution, the gradient of the softmax activation function's input.
	 */
	inline void softmax_bwd(const CuDNNTensor<Scalar>& output, /* in/out */ CuDNNTensor<Scalar>& in_out_grad) const {
		cudnnAssert(cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
				output.desc(), output.data(), in_out_grad.desc(), in_out_grad.data(), &beta, in_out_grad.desc(),
				in_out_grad.data()));
	}
	/**
	 * Computes the dimensions of the output of the 2D pooling operation.
	 *
	 * @param input_height The input height.
	 * @param input_width The input width.
	 * @param input_channels The number of input channels.
	 * @param pool_mode The pooling mode.
	 * @param window_height The height of the pooling window.
	 * @param window_width The width of the pooling window.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the pooling.
	 * @param horizontal_stride The horizontal stride of the pooling.
	 * @param output_height The output height.
	 * @param output_width The output width.
	 * @param output_channels The number of output channels.
	 */
	inline void pool2d_output_dims(std::size_t input_height, std::size_t input_width, std::size_t input_channels,
			cudnnPoolingMode_t pool_mode, std::size_t window_height, std::size_t window_width, std::size_t vertical_padding,
			std::size_t horizontal_padding, std::size_t vertical_stride, std::size_t horizontal_stride,
			/* out */ std::size_t& output_height, /* out */ std::size_t& output_width,
			/* out */ std::size_t& output_channels) const {
		// Create and set the input tensor descriptor.
		cudnnTensorDescriptor_t input_desc;
		CuDNNTensorDescriptorManager<>::create_descriptor(input_desc, CuDNNTensor<Scalar>::DATA_TYPE,
				CuDNNTensor<Scalar>::TENSOR_FORMAT, 1, input_height, input_width, input_channels);
		// Create and set the pooling descriptor.
		cudnnPoolingDescriptor_t pool_desc;
		cudnnAssert(cudnnCreatePoolingDescriptor(&pool_desc));
		cudnnAssert(cudnnSetPooling2dDescriptor(pool_desc, pool_mode, NAN_PROP, window_height, window_width,
				vertical_padding, horizontal_padding, vertical_stride, horizontal_stride));
		// Compute the dimensions.
		int n, h, w, c;
		cudnnAssert(cudnnGetPooling2dForwardOutputDim(pool_desc, input_desc, &n, &c, &h, &w));
		// Free the resources.
		cudnnAssert(cudnnDestroyPoolingDescriptor(pool_desc));
		CuDNNTensorDescriptorManager<>::destroy_descriptor(input_desc);
		output_height = (std::size_t) h;
		output_width = (std::size_t) w;
		output_channels = (std::size_t) c;
	}
	/**
	 * It performs a 2D pooling operation on the input tensor.
	 *
	 * @param input The input tensor.
	 * @param pool_mode The pooling mode.
	 * @param window_height The height of the pooling window.
	 * @param window_width The width of the pooling window.
	 * @param vertical_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the height rank.
	 * @param horizontal_padding The padding to apply to both the top and the bottom of the input
	 * tensor along the width rank.
	 * @param vertical_stride The vertical stride of the pooling.
	 * @param horizontal_stride The horizontal stride of the pooling.
	 * @param output The output tensor of the pooling operation.
	 */
	inline void pool2d_fwd(const CuDNNTensor<Scalar>& input, cudnnPoolingMode_t pool_mode,
			std::size_t window_height, std::size_t window_width, std::size_t vertical_padding,
			std::size_t horizontal_padding, std::size_t vertical_stride, std::size_t horizontal_stride,
			/* out */ CuDNNTensor<Scalar>& output) const {
		cudnnPoolingDescriptor_t pool_desc;
		cudnnAssert(cudnnCreatePoolingDescriptor(&pool_desc));
		cudnnAssert(cudnnSetPooling2dDescriptor(pool_desc, pool_mode, NAN_PROP, window_height, window_width,
				vertical_padding, horizontal_padding, vertical_stride, horizontal_stride));
		cudnnAssert(cudnnPoolingForward(handle, pool_desc, &alpha, input.desc(), input.data(), &beta,
				output.desc(), output.data()));
		cudnnAssert(cudnnDestroyPoolingDescriptor(pool_desc));
	}
	/**
	 * Computes the gradient of the input of the pooling layer.
	 *
	 * @param input The input tensor.
	 * @param output The output tensor.
	 * @param out_grad The gradient of the output.
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
	inline void pool2d_bwd(const CuDNNTensor<Scalar>& input, const CuDNNTensor<Scalar>& output,
			const CuDNNTensor<Scalar>& out_grad, cudnnPoolingMode_t pool_mode, std::size_t window_height,
			std::size_t window_width, std::size_t vertical_padding, std::size_t horizontal_padding,
			std::size_t vertical_stride, std::size_t horizontal_stride,
			/* out */ CuDNNTensor<Scalar>& prev_out_grad) const {
		cudnnPoolingDescriptor_t pool_desc;
		cudnnAssert(cudnnCreatePoolingDescriptor(&pool_desc));
		cudnnAssert(cudnnSetPooling2dDescriptor(pool_desc, pool_mode, NAN_PROP, window_height, window_width,
				vertical_padding, horizontal_padding, vertical_stride, horizontal_stride));
		cudnnAssert(cudnnPoolingBackward(handle, pool_desc, &alpha, output.desc(), output.data(),
				out_grad.desc(), out_grad.data(), input.desc(), input.data(), &beta, prev_out_grad.desc(),
				prev_out_grad.data()));
		cudnnAssert(cudnnDestroyPoolingDescriptor(pool_desc));
	}
	/**
	 * Applies the batch normalization function to the input data and updates the running mean and
	 * variance averages.
	 *
	 * @param input The input tensor.
	 * @param gamma The gamma scaling tensor.
	 * @param beta The beta bias.
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
	inline void batch_norm_fwd_training(const CuDNNTensor<Scalar>& input, const CuDNNTensor<Scalar>& gamma,
			const CuDNNTensor<Scalar>& beta, bool spatial, Scalar exp_avg_factor, Scalar epsilon,
			/* in/out */ CuDNNTensor<Scalar>& means, /* in/out */ CuDNNTensor<Scalar>& vars,
			/* out */ CuDNNTensor<Scalar>& output, /* out */ CuDNNTensor<Scalar>& mean_cache,
			/* out */ CuDNNTensor<Scalar>& inv_var_cache) const {
		cudnnAssert(cudnnBatchNormalizationForwardTraining(handle, spatial ? CUDNN_BATCHNORM_SPATIAL_PERSISTENT :
				CUDNN_BATCHNORM_PER_ACTIVATION, &alpha, &this->beta, input.desc(), input.data(), output.desc(),
				output.data(), gamma.desc(), gamma.data(), beta.data(), exp_avg_factor, means.data(),
				vars.data(), epsilon, mean_cache.data(), inv_var_cache.data()));
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
	 * @param spatial Whether the batch normalization should be performed in spatial or per-activation
	 * mode.
	 * @param epsilon A small constant for numerical stability.
	 * @param output The output tensor.
	 */
	inline void batch_norm_fwd_inference(const CuDNNTensor<Scalar>& input, const CuDNNTensor<Scalar>& gamma,
			const CuDNNTensor<Scalar>& beta, const CuDNNTensor<Scalar>& means, const CuDNNTensor<Scalar>& vars,
			bool spatial, Scalar epsilon, /* out */ CuDNNTensor<Scalar>& output) const {
		cudnnAssert(cudnnBatchNormalizationForwardInference(handle, spatial ? CUDNN_BATCHNORM_SPATIAL_PERSISTENT :
				CUDNN_BATCHNORM_PER_ACTIVATION, &alpha, &this->beta, input.desc(), input.data(), output.desc(),
				output.data(), gamma.desc(), gamma.data(), beta.data(), means.data(),
				vars.data(), epsilon));
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
	 * @param spatial Whether the batch normalization should be performed in spatial or per-activation
	 * mode.
	 * @param epsilon A small constant for numerical stability.
	 * @param prev_out_grad The gradient of the batch normalization function's input.
	 * @param gamma_grad The gradient of gamma.
	 * @param beta_grad The gradient of beta.
	 */
	inline void batch_norm_bwd(const CuDNNTensor<Scalar>& input, const CuDNNTensor<Scalar>& out_grad,
			const CuDNNTensor<Scalar>& gamma, const CuDNNTensor<Scalar>& mean_cache,
			const CuDNNTensor<Scalar>& inv_var_cache, bool spatial, Scalar epsilon,
			/* out */ CuDNNTensor<Scalar>& prev_out_grad, /* out */ CuDNNTensor<Scalar>& gamma_grad,
			/* out */ CuDNNTensor<Scalar>& beta_grad) const {
		cudnnAssert(cudnnBatchNormalizationBackward(handle, spatial ? CUDNN_BATCHNORM_SPATIAL_PERSISTENT :
				CUDNN_BATCHNORM_PER_ACTIVATION, &alpha, &beta, &alpha, &beta, input.desc(), input.data(),
				out_grad.desc(), out_grad.data(), prev_out_grad.desc(), prev_out_grad.data(), gamma.desc(),
				gamma.data(), gamma_grad.data(), beta_grad.data(), epsilon, mean_cache.data(),
				inv_var_cache.data()));
	}
	/**
	 * It computes the necessary state size required for the RNG used by the dropout function.
	 *
	 * @param state_size The state size.
	 */
	inline void dropout_state_size(/* out */ std::size_t& state_size) const {
		cudnnAssert(cudnnDropoutGetStatesSize(handle, &state_size));
		state_size = (std::size_t) ceil(((Scalar) state_size) / sizeof(Scalar));
	}
	/**
	 * It computes the necessary reserve size for the dropout.
	 *
	 * @param input The input tensor.
	 * @param reserve_size The reserve size.
	 */
	inline void dropout_reserve_size(const CuDNNTensor<Scalar>& input, /* out */ std::size_t& reserve_size) const {
		cudnnAssert(cudnnDropoutGetReserveSpaceSize(input.desc(), &reserve_size));
		reserve_size = (std::size_t) ceil(((Scalar) reserve_size) / sizeof(Scalar));
	}
	/**
	 * It applies the dropout function to the input tensor.
	 *
	 * @param input The input tensor.
	 * @param dropout The average factor of elements to set to 0 in the input tensor.
	 * @param state The memory used by the RNG.
	 * @param reserve The reserve used for backpropagation.
	 * @param output The output tensor.
	 */
	inline void dropout_fwd(const CuDNNTensor<Scalar>& input, Scalar dropout,
			CuDNNTensor<Scalar>& state, /* in/out */ CuDNNTensor<Scalar>& reserve,
			/* out */ CuDNNTensor<Scalar>& output) const {
		cudnnDropoutDescriptor_t dropout_desc;
		cudnnAssert(cudnnCreateDropoutDescriptor(&dropout_desc));
		cudnnAssert(cudnnSetDropoutDescriptor(dropout_desc, handle, (float) dropout, state.data(),
				state.size() * sizeof(Scalar), 0));
		cudnnAssert(cudnnDropoutForward(handle, dropout_desc, input.desc(), input.data(), output.desc(),
				output.data(), reserve.data(), reserve.size() * sizeof(Scalar)));
		cudnnAssert(cudnnDestroyDropoutDescriptor(dropout_desc));
	}
	/**
	 * It computes the gradient of the input of the dropout function.
	 *
	 * @param out_grad The gradient of the output of the dropout function.
	 * @param state The memory used by the RNG.
	 * @param reserve The reserve filled during the forward pass.
	 * @param dropout The average factor of elements set to 0 in the input tensor.
	 * @param prev_out_grad The gradient of the input of the dropout function.
	 */
	inline void dropout_bwd(const CuDNNTensor<Scalar>& out_grad, Scalar dropout, CuDNNTensor<Scalar>& state,
			CuDNNTensor<Scalar>& reserve, /* out */ CuDNNTensor<Scalar>& prev_out_grad) const {
		cudnnDropoutDescriptor_t dropout_desc;
		cudnnAssert(cudnnCreateDropoutDescriptor(&dropout_desc));
		cudnnAssert(cudnnSetDropoutDescriptor(dropout_desc, handle, (float) dropout, state.data(),
				state.size() * sizeof(Scalar), 0));
		cudnnAssert(cudnnDropoutBackward(handle, dropout_desc, out_grad.desc(), out_grad.data(),
				prev_out_grad.desc(), prev_out_grad.data(), reserve.data(),
				reserve.size() * sizeof(Scalar)));
		cudnnAssert(cudnnDestroyDropoutDescriptor(dropout_desc));
	}
private:
	inline CuDNNHandle() :
			handle(),
			alpha(1),
			beta(0) {
		cudnnAssert(cudnnCreate(&handle));
	}
	cudnnHandle_t handle;
	const Scalar alpha;
	const Scalar beta;
};

// Arithmetic operator overloads.

template<typename Scalar>
inline CuDNNTensor<Scalar> operator+(const CuDNNTensor<Scalar>& a, const CuDNNTensor<Scalar>& b) {
	CuDNNTensor<Scalar> c(a.samples(), a.height(), a.width(), a.channels());
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b, 1, CUDNN_OP_TENSOR_ADD, 0, c);
	return c;
}

template<typename Scalar>
inline CuDNNTensor<Scalar> operator-(const CuDNNTensor<Scalar>& a, const CuDNNTensor<Scalar>& b) {
	CuDNNTensor<Scalar> c(a.samples(), a.height(), a.width(), a.channels());
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b, 1, CUDNN_OP_TENSOR_MIN, 0, c);
	return c;
}

template<typename Scalar>
inline CuDNNTensor<Scalar> operator*(const CuDNNTensor<Scalar>& a, const CuDNNTensor<Scalar>& b) {
	CuDNNTensor<Scalar> c(a.samples(), a.height(), a.width(), a.channels());
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b, 1, CUDNN_OP_TENSOR_MUL, 0, c);
	return c;
}

template<typename Scalar>
inline CuDNNTensor<Scalar>& operator+=(CuDNNTensor<Scalar>& a, const CuDNNTensor<Scalar>& b) {
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b, 1, CUDNN_OP_TENSOR_ADD, 1, a);
	return a;
}

template<typename Scalar>
inline CuDNNTensor<Scalar>& operator-=(CuDNNTensor<Scalar>& a, const CuDNNTensor<Scalar>& b) {
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b, 1, CUDNN_OP_TENSOR_MIN, 1, a);
	return a;
}

template<typename Scalar>
inline CuDNNTensor<Scalar>& operator*=(CuDNNTensor<Scalar>& a, const CuDNNTensor<Scalar>& b) {
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b, 1, CUDNN_OP_TENSOR_MUL, 1, a);
	return a;
}

template<typename Scalar>
inline CuDNNTensor<Scalar> operator+(const CuDNNTensor<Scalar>& a, Scalar b) {
	CuDNNTensor<Scalar> b_tensor(1u, 1u, 1u, 1u);
	b_tensor.copy_from_host(&b);
	CuDNNTensor<Scalar> c(a.samples(), a.height(), a.width(), a.channels());
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b_tensor, 1, CUDNN_OP_TENSOR_ADD, 0, c);
	return c;
}

template<typename Scalar>
inline CuDNNTensor<Scalar> operator-(const CuDNNTensor<Scalar>& a, Scalar b) {
	CuDNNTensor<Scalar> b_tensor(1u, 1u, 1u, 1u);
	b_tensor.copy_from_host(&b);
	CuDNNTensor<Scalar> c(a.samples(), a.height(), a.width(), a.channels());
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b_tensor, 1, CUDNN_OP_TENSOR_MIN, 0, c);
	return c;
}

template<typename Scalar>
inline CuDNNTensor<Scalar> operator*(const CuDNNTensor<Scalar>& a, Scalar b) {
	CuDNNTensor<Scalar> b_tensor(1u, 1u, 1u, 1u);
	b_tensor.copy_from_host(&b);
	CuDNNTensor<Scalar> c(a.samples(), a.height(), a.width(), a.channels());
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b_tensor, 1, CUDNN_OP_TENSOR_MUL, 0, c);
	return c;
}

template<typename Scalar>
inline CuDNNTensor<Scalar>& operator+=(CuDNNTensor<Scalar>& a, Scalar b) {
	CuDNNTensor<Scalar> b_tensor(1u, 1u, 1u, 1u);
	b_tensor.copy_from_host(&b);
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b_tensor, 1, CUDNN_OP_TENSOR_ADD, 1, a);
	return a;
}

template<typename Scalar>
inline CuDNNTensor<Scalar>& operator-=(CuDNNTensor<Scalar>& a, Scalar b) {
	CuDNNTensor<Scalar> b_tensor(1u, 1u, 1u, 1u);
	b_tensor.copy_from_host(&b);
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b_tensor, 1, CUDNN_OP_TENSOR_MIN, 1, a);
	return a;
}

template<typename Scalar>
inline CuDNNTensor<Scalar>& operator*=(CuDNNTensor<Scalar>& a, Scalar b) {
	CuDNNTensor<Scalar> b_tensor(1u, 1u, 1u, 1u);
	b_tensor.copy_from_host(&b);
	CuDNNHandle<Scalar>::get_instance().op(a, 1, b_tensor, 1, CUDNN_OP_TENSOR_MUL, 1, a);
	return a;
}

}
} /* namespace cattle */

#endif /* CATTL3_UTILS_CUDNNHANDLE_H_ */
