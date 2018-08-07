/*
 * TransConvKernelLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_KERNEL_TRANSCONVKERNELLAYER_H_
#define C_ATTL3_LAYER_KERNEL_TRANSCONVKERNELLAYER_H_

#include <array>
#include <cassert>

#include "layer/KernelLayer.hpp"
#include "parameter_initialization/ZeroParameterInitialization.hpp"
#include "parameters/StandardParameters.hpp"

namespace cattle {

/**
 * An abstract base class template for a transposed 2D convolutional layer.
 */
template<typename Scalar, std::size_t Rank>
class TransConvKernelLayerBase : public KernelLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef KernelLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,4> Array4;
	typedef std::array<std::pair<std::size_t,std::size_t>,4> PaddingsArray4;
public:
	inline void empty_cache() {
		in_mat_cache = Matrix<Scalar>();
	}
protected:
	inline TransConvKernelLayerBase(const typename Root::Dims& input_dims, std::size_t filters,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_padding,
			std::size_t horizontal_padding, std::size_t vertical_stride, std::size_t horizontal_stride,
			std::size_t vertical_dilation, std::size_t horizontal_dilation, ParamInitSharedPtr<Scalar> weight_init,
			ParamRegSharedPtr<Scalar> weight_reg, Scalar weight_clip, Scalar weight_max_l1_norm,
			Scalar weight_max_l2_norm, Scalar weight_grad_clip, Scalar weight_grad_max_l1_norm,
			Scalar weight_grad_max_l2_norm, ParamRegSharedPtr<Scalar> bias_reg, Scalar bias_clip,
			Scalar bias_max_l1_norm, Scalar bias_max_l2_norm, Scalar bias_grad_clip,
			Scalar bias_grad_max_l1_norm, Scalar bias_grad_max_l2_norm) :
				Base::KernelLayer(input_dims, calculate_adjusted_output_dims(input_dims, filters,
						receptor_height, receptor_width, vertical_padding, horizontal_padding,
						vertical_stride, horizontal_stride, vertical_dilation, horizontal_dilation),
						std::make_shared<StandardParameters<Scalar>>(input_dims.template extend<3 - Rank>()(2),
								receptor_height * receptor_width * filters, true, weight_init, weight_reg,
								weight_clip, weight_max_l1_norm, weight_max_l2_norm, weight_grad_clip,
								weight_grad_max_l1_norm, weight_grad_max_l2_norm),
						std::make_shared<StandardParameters<Scalar>>(1, calculate_adjusted_output_dims(input_dims,
								filters, receptor_height, receptor_width, vertical_padding, horizontal_padding,
								vertical_stride, horizontal_stride, vertical_dilation, horizontal_dilation)
								.get_volume(), true, std::make_shared<ZeroParameterInitialization<Scalar>>(),
								bias_reg, bias_clip, bias_max_l1_norm, bias_max_l2_norm, bias_grad_clip,
								bias_grad_max_l1_norm, bias_grad_max_l2_norm)),
				filters(filters),
				receptor_height(receptor_height),
				receptor_width(receptor_width),
				vertical_padding(vertical_padding),
				horizontal_padding(horizontal_padding),
				vertical_stride(vertical_stride),
				horizontal_stride(horizontal_stride),
				vertical_dilation(vertical_dilation),
				horizontal_dilation(horizontal_dilation),
				ext_input_dims(input_dims.template extend<3 - Rank>()),
				ext_output_dims(calculate_output_dims(ext_input_dims, filters, receptor_height, receptor_width,
						vertical_padding, horizontal_padding, vertical_stride, horizontal_stride,
						vertical_dilation, horizontal_dilation)),
				padded_height(ext_output_dims(0) + 2 * vertical_padding),
				padded_width(ext_output_dims(1) + 2 * horizontal_padding),
				dil_receptor_height(receptor_height + (receptor_height - 1) * vertical_dilation),
				dil_receptor_width(receptor_width + (receptor_width - 1) * horizontal_dilation),
				patches_per_sample(ext_input_dims(0) * ext_input_dims(1)),
				prev_out_conversion_dims({ 0u, ext_input_dims(0), ext_input_dims(1), ext_input_dims(2) }),
				patch_offsets({ 0u, 0u, 0u, 0u }),
				patch_extents({ 0u, dil_receptor_height, dil_receptor_width, filters }),
				dil_strides({ 1u, vertical_dilation + 1u, horizontal_dilation + 1u, 1u }),
				no_padding_offsets({ 0u, vertical_padding, horizontal_padding, 0u }),
				no_padding_extents({ 0u, ext_output_dims(0), ext_output_dims(1), ext_output_dims(2) }),
				paddings({ std::make_pair(0, 0), std::make_pair(vertical_padding, vertical_padding),
						std::make_pair(horizontal_padding, horizontal_padding), std::make_pair(0, 0) }) {
		assert(filters > 0);
		assert(receptor_height > 0);
		assert(receptor_width > 0);
		assert(vertical_stride > 0 && horizontal_stride > 0);
		assert(ext_output_dims(0) + 2 * vertical_padding >= dil_receptor_height &&
				ext_output_dims(1) + 2 * horizontal_padding >= dil_receptor_width);
	}
	inline TransConvKernelLayerBase(const TransConvKernelLayerBase<Scalar,Rank>& layer, bool share_params = false) :
			Base::KernelLayer(layer, share_params),
			filters(layer.filters),
			receptor_height(layer.receptor_height),
			receptor_width(layer.receptor_width),
			vertical_padding(layer.vertical_padding),
			horizontal_padding(layer.horizontal_padding),
			vertical_stride(layer.vertical_stride),
			horizontal_stride(layer.horizontal_stride),
			vertical_dilation(layer.vertical_dilation),
			horizontal_dilation(layer.horizontal_dilation),
			ext_input_dims(layer.ext_input_dims),
			ext_output_dims(layer.ext_output_dims),
			padded_height(layer.padded_height),
			padded_width(layer.padded_width),
			dil_receptor_height(layer.dil_receptor_height),
			dil_receptor_width(layer.dil_receptor_width),
			patches_per_sample(layer.patches_per_sample),
			prev_out_conversion_dims(layer.prev_out_conversion_dims),
			patch_offsets(layer.patch_offsets),
			patch_extents(layer.patch_extents),
			dil_strides(layer.dil_strides),
			no_padding_offsets(layer.no_padding_offsets),
			no_padding_extents(layer.no_padding_extents),
			paddings(layer.paddings),
			in_mat_cache(layer.in_mat_cache) { }
	inline Tensor<Scalar,4> _pass_forward(Tensor<Scalar,4> in, bool training) {
		std::size_t rows = in.dimension(0);
		std::size_t depth = ext_input_dims(2);
		std::size_t receptor_vol = Base::weights->get_values().cols();
		std::size_t total_patches = rows * patches_per_sample;
		in_mat_cache = MatrixMap<Scalar>(in.data(), total_patches, depth);
		Matrix<Scalar> out_conv_mat = in_mat_cache * Base::weights->get_values();
		/* Given the values of the stretched out receptor patches, accumulate them in the output tensor. */
		Tensor<Scalar,4> out(rows, padded_height, padded_width, ext_output_dims(2));
		out.setZero();
		std::size_t patch_ind = 0;
		patch_extents[0] = rows;
		for (std::size_t i = 0; i <= padded_width - dil_receptor_width; i += horizontal_stride) {
			patch_offsets[2] = i;
			for (std::size_t j = 0; j <= padded_height - dil_receptor_height; j += vertical_stride) {
				patch_offsets[1] = j;
				// Accumulate the gradients where the receptor-patch-tensors overlap.
				Matrix<Scalar> out_conv_mat_block = out_conv_mat.block(patch_ind, 0, rows, receptor_vol);
				TensorMap<Scalar,4> out_patch(out_conv_mat_block.data(), rows, receptor_height,
						receptor_width, ext_output_dims(2));
				if (vertical_dilation > 0 || horizontal_dilation > 0)
					out.slice(patch_offsets, patch_extents).stride(dil_strides) += out_patch;
				else
					out.slice(patch_offsets, patch_extents) += out_patch;
				patch_ind += rows;
			}
		}
		assert(patch_ind == total_patches);
		if (vertical_padding > 0 || horizontal_padding > 0) {
			// Cut off the padding.
			no_padding_extents[0] = rows;
			out = Tensor<Scalar,4>(out.slice(no_padding_offsets, no_padding_extents));
		}
		MatrixMap<Scalar> out_mat(out.data(), rows, ext_output_dims.get_volume());
		out_mat.rowwise() += Base::bias->get_values().row(0);
		return out;
	}
	inline Tensor<Scalar,4> _pass_back(Tensor<Scalar,4> out_grad) {
		std::size_t rows = out_grad.dimension(0);
		std::size_t depth = ext_input_dims(2);
		std::size_t receptor_vol = Base::weights->get_values().cols();
		std::size_t total_patches = rows * patches_per_sample;
		std::size_t patch_ind = 0;
		patch_extents[0] = rows;
		// Compute the gradient of the bias.
		Base::bias->accumulate_grad(MatrixMap<Scalar>(out_grad.data(), rows,
				ext_output_dims.get_volume()).colwise().sum());
		// Spatial padding.
		if (vertical_padding > 0 || horizontal_padding > 0)
			out_grad = Tensor<Scalar,4>(out_grad.pad(paddings));
		Matrix<Scalar> out_grad_conv_mat(total_patches, receptor_vol);
		for (std::size_t i = 0; i <= padded_width - dil_receptor_width; i += horizontal_stride) {
			patch_offsets[2] = i;
			for (std::size_t j = 0; j <= padded_height - dil_receptor_height; j += vertical_stride) {
				patch_offsets[1] = j;
				Tensor<Scalar,4> patch;
				// If the patch is dilated, skip the spatial gaps when flattening it into a matrix.
				if (vertical_dilation > 0 || horizontal_dilation > 0)
					patch = out_grad.slice(patch_offsets, patch_extents).stride(dil_strides);
				else
					patch = out_grad.slice(patch_offsets, patch_extents);
				out_grad_conv_mat.block(patch_ind, 0, rows, receptor_vol) = MatrixMap<Scalar>(patch.data(),
						rows, receptor_vol);
				patch_ind += rows;
			}
		}
		assert(patch_ind == total_patches);
		Base::weights->accumulate_grad(in_mat_cache.transpose() * out_grad_conv_mat);
		if (Base::is_input_layer())
			return Tensor<Scalar,4>();
		Matrix<Scalar> prev_out_grad = out_grad_conv_mat * Base::weights->get_values().transpose();
		prev_out_conversion_dims[0] = rows;
		return TensorMap<Scalar,4>(prev_out_grad.data(), prev_out_conversion_dims);
	}
	// The defining attributes of the deconvolutional layer.
	const std::size_t filters;
	const std::size_t receptor_height;
	const std::size_t receptor_width;
	const std::size_t vertical_padding;
	const std::size_t horizontal_padding;
	const std::size_t vertical_stride;
	const std::size_t horizontal_stride;
	const std::size_t vertical_dilation;
	const std::size_t horizontal_dilation;
private:
	inline static std::size_t calculate_spatial_output_dim(std::size_t input_dim, std::size_t receptor_size,
			std::size_t padding, std::size_t dilation, std::size_t stride) {
		return (input_dim - 1) * stride + receptor_size + (receptor_size - 1) * dilation - 2 * padding;
	}
	inline static Dimensions<std::size_t,3> calculate_output_dims(const Dimensions<std::size_t,3>& input_dims,
			std::size_t filters, std::size_t receptor_height, std::size_t receptor_width,
			std::size_t vertical_padding, std::size_t horizontal_padding, std::size_t vertical_stride,
			std::size_t horizontal_stride, std::size_t vertical_dilation, std::size_t horizontal_dilation) {
		return { calculate_spatial_output_dim(input_dims(0), receptor_height, vertical_padding, vertical_dilation,
						vertical_stride), calculate_spatial_output_dim(input_dims(1), receptor_width,
								horizontal_padding, horizontal_dilation,horizontal_stride), filters };
	}
	inline static Dimensions<std::size_t,Rank> calculate_adjusted_output_dims(const typename Root::Dims& input_dims,
			std::size_t filters, std::size_t receptor_height, std::size_t receptor_width,
			std::size_t vertical_padding, std::size_t horizontal_padding, std::size_t vertical_stride,
			std::size_t horizontal_stride, std::size_t vertical_dilation, std::size_t horizontal_dilation) {
		auto output_dims = calculate_output_dims(input_dims.template extend<3 - Rank>(), filters, receptor_height,
				receptor_width, vertical_padding, horizontal_padding, vertical_stride, horizontal_stride,
				vertical_dilation, horizontal_dilation);
		output_dims(2) /= filters;
		output_dims(Rank - 1) *= filters;
		return output_dims.template contract<3 - Rank>();
	}
	const Dimensions<std::size_t,3> ext_input_dims, ext_output_dims;
	// Pre-computed values to improve propagation-time performance.
	const std::size_t padded_height, padded_width, dil_receptor_height, dil_receptor_width, patches_per_sample;
	Array4 prev_out_conversion_dims, patch_offsets, patch_extents, dil_strides,
			no_padding_offsets, no_padding_extents;
	PaddingsArray4 paddings;
	// Staged computation caches
	Matrix<Scalar> in_mat_cache;
};

/**
 * A class template for a transposed 2D convolutional layer operating on rank-3 data batches (rank-4 tensors).
 * The results of the convolutions of the filters and the input tensor are concatenated along the highest (4th)
 * rank of the output tensor.
 *
 * \see https://arxiv.org/abs/1603.07285v1
 */
template<typename Scalar, std::size_t Rank = 3>
class TransConvKernelLayer : public TransConvKernelLayerBase<Scalar,Rank> {
	typedef Layer<Scalar,3> Root;
	typedef KernelLayer<Scalar,3> KernelBase;
	typedef TransConvKernelLayerBase<Scalar,3> TransConvBase;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the weights of
	 * the layer.
	 * @param receptor_height The height of the base of the receptor cuboid.
	 * @param receptor_width The width of the base of the receptor cuboid.
	 * @param vertical_padding The extent of padding to apply to the input tensor along its height (both
	 * at the top and at the bottom).
	 * @param horizontal_padding The extent of padding to apply to the input tensor along its width (both
	 * at the left and at the right).
	 * @param vertical_stride The vertical transposed convolution stride i.e. the number of elements by
	 * which the receptor is to be shifted along the height of the input tensor.
	 * @param horizontal_stride The horizonzal transposed convolution stride i.e. the number of elements
	 * by which the receptor is to be shifted along the width of the input tensor.
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor.
	 * @param weight_reg An optional regularization function to apply to the weights.
	 * @param weight_clip The maximum allowed absolute weight value. If it is 0 or less, no value
	 * clipping is performed.
	 * @param weight_max_l1_norm The maximum allowed L1 weight value norm. If it is 0 or less, no L1
	 * max norm constraint is enforced.
	 * @param weight_max_l2_norm The maximum allowed L2 weight value norm. If it is 0 or less, no L2
	 * max norm constraint is enforced.
	 * @param weight_grad_clip The maximum allowed absolute weight gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param weight_grad_max_l1_norm The maximum allowed L1 weight gradient norm. If it is 0 or less,
	 * no L1 gradient max norm constraint is enforced.
	 * @param weight_grad_max_l2_norm The maximum allowed L2 weight gradient norm. If it is 0 or less,
	 * no L2 gradient max norm constraint is enforced.
	 * @param bias_reg An optional regularization function to apply to the bias.
	 * @param bias_clip The maximum allowed absolute bias value. If it is 0 or less, no value clipping
	 * is performed.
	 * @param bias_max_l1_norm The maximum allowed L1 bias value norm. If it is 0 or less, no bias L1
	 * max norm constraint is enforced.
	 * @param bias_max_l2_norm The maximum allowed L2 bias value norm. If it is 0 or less, no bias L2
	 * max norm constraint is enforced.
	 * @param bias_grad_clip The maximum allowed absolute bias gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param bias_grad_max_l1_norm The maximum allowed L1 bias gradient norm. If it is 0 or less, no
	 * bias L1 gradient max norm constraint is enforced.
	 * @param bias_grad_max_l2_norm The maximum allowed L2 bias gradient norm. If it is 0 or less, no
	 * bias L2 gradient max norm constraint is enforced.
	 */
	inline TransConvKernelLayer(const typename Root::Dims& input_dims, std::size_t filters,
			ParamInitSharedPtr<Scalar> weight_init, std::size_t receptor_height = 3, std::size_t receptor_width = 3,
			std::size_t vertical_padding = 1, std::size_t horizontal_padding = 1, std::size_t vertical_stride = 1,
			std::size_t horizontal_stride = 1, std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0,
			ParamRegSharedPtr<Scalar> weight_reg = nullptr, Scalar weight_clip = 0, Scalar weight_max_l1_norm = 0,
			Scalar weight_max_l2_norm = 0, Scalar weight_grad_clip = 0, Scalar weight_grad_max_l1_norm = 0,
			Scalar weight_grad_max_l2_norm = 0, ParamRegSharedPtr<Scalar> bias_reg = nullptr, Scalar bias_clip = 0,
			Scalar bias_max_l1_norm = 0, Scalar bias_max_l2_norm = 0, Scalar bias_grad_clip = 0,
			Scalar bias_grad_max_l1_norm = 0, Scalar bias_grad_max_l2_norm = 0) :
				TransConvBase::TransConvKernelLayerBase(input_dims, filters, receptor_height, receptor_width,
						vertical_padding, horizontal_padding, vertical_stride, horizontal_stride, vertical_dilation,
						horizontal_dilation,weight_init,  weight_reg, weight_clip, weight_max_l1_norm,
						weight_max_l2_norm, weight_grad_clip, weight_grad_max_l1_norm, weight_grad_max_l2_norm,
						bias_reg, bias_clip, bias_max_l1_norm, bias_max_l2_norm, bias_grad_clip,
						bias_grad_max_l1_norm, bias_grad_max_l2_norm) { }
	inline TransConvKernelLayer(const TransConvKernelLayer<Scalar,3>& layer, bool share_params = false) :
			TransConvBase::TransConvKernelLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone() const {
		return new TransConvKernelLayer(*this);
	}
	inline Root* clone_with_shared_params() {
		return new TransConvKernelLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return TransConvBase::_pass_forward(std::move(in), training);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,4>(out_grad.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		return TransConvBase::_pass_back(std::move(out_grad));
	}
private:
	std::size_t batch_size;
};

/**
 * A class template for a transposed 2D convolutional layer operating on rank-2 data batches (rank-3 tensors).
 * The results of the convolutions of the filters and the input tensor are concatenated along the highest (3rd)
 * rank of the output tensor.
 *
 * \see https://arxiv.org/abs/1603.07285v1
 */
template<typename Scalar>
class TransConvKernelLayer<Scalar,2> : public TransConvKernelLayerBase<Scalar,2> {
	typedef Layer<Scalar,2> Root;
	typedef KernelLayer<Scalar,2> KernelBase;
	typedef TransConvKernelLayerBase<Scalar,2> TransConvBase;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the weights of
	 * the layer.
	 * @param receptor_height The height of the base of the receptor cuboid.
	 * @param receptor_width The width of the base of the receptor cuboid.
	 * @param vertical_padding The extent of padding to apply to the input tensor along its height (both
	 * at the top and at the bottom).
	 * @param horizontal_padding The extent of padding to apply to the input tensor along its width (both
	 * at the left and at the right).
	 * @param vertical_stride The vertical transposed convolution stride i.e. the number of elements by
	 * which the receptor is to be shifted along the height of the input tensor.
	 * @param horizontal_stride The horizonzal transposed convolution stride i.e. the number of elements
	 * by which the receptor is to be shifted along the width of the input tensor.
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor.
	 * @param weight_reg An optional regularization function to apply to the weights.
	 * @param weight_clip The maximum allowed absolute weight value. If it is 0 or less, no value
	 * clipping is performed.
	 * @param weight_max_l1_norm The maximum allowed L1 weight value norm. If it is 0 or less, no L1
	 * max norm constraint is enforced.
	 * @param weight_max_l2_norm The maximum allowed L2 weight value norm. If it is 0 or less, no L2
	 * max norm constraint is enforced.
	 * @param weight_grad_clip The maximum allowed absolute weight gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param weight_grad_max_l1_norm The maximum allowed L1 weight gradient norm. If it is 0 or less,
	 * no L1 gradient max norm constraint is enforced.
	 * @param weight_grad_max_l2_norm The maximum allowed L2 weight gradient norm. If it is 0 or less,
	 * no L2 gradient max norm constraint is enforced.
	 * @param bias_reg An optional regularization function to apply to the bias.
	 * @param bias_clip The maximum allowed absolute bias value. If it is 0 or less, no value clipping
	 * is performed.
	 * @param bias_max_l1_norm The maximum allowed L1 bias value norm. If it is 0 or less, no bias L1
	 * max norm constraint is enforced.
	 * @param bias_max_l2_norm The maximum allowed L2 bias value norm. If it is 0 or less, no bias L2
	 * max norm constraint is enforced.
	 * @param bias_grad_clip The maximum allowed absolute bias gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param bias_grad_max_l1_norm The maximum allowed L1 bias gradient norm. If it is 0 or less, no
	 * bias L1 gradient max norm constraint is enforced.
	 * @param bias_grad_max_l2_norm The maximum allowed L2 bias gradient norm. If it is 0 or less, no
	 * bias L2 gradient max norm constraint is enforced.
	 */
	inline TransConvKernelLayer(const typename Root::Dims& input_dims, std::size_t filters,
			ParamInitSharedPtr<Scalar> weight_init, std::size_t receptor_height = 3, std::size_t receptor_width = 3,
			std::size_t vertical_padding = 1, std::size_t horizontal_padding = 1, std::size_t vertical_stride = 1,
			std::size_t horizontal_stride = 1, std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0,
			ParamRegSharedPtr<Scalar> weight_reg = nullptr, Scalar weight_clip = 0, Scalar weight_max_l1_norm = 0,
			Scalar weight_max_l2_norm = 0, Scalar weight_grad_clip = 0, Scalar weight_grad_max_l1_norm = 0,
			Scalar weight_grad_max_l2_norm = 0, ParamRegSharedPtr<Scalar> bias_reg = nullptr, Scalar bias_clip = 0,
			Scalar bias_max_l1_norm = 0, Scalar bias_max_l2_norm = 0, Scalar bias_grad_clip = 0,
			Scalar bias_grad_max_l1_norm = 0, Scalar bias_grad_max_l2_norm = 0) :
				TransConvBase::TransConvKernelLayerBase(input_dims, filters, receptor_height, receptor_width,
						vertical_padding, horizontal_padding, vertical_stride, horizontal_stride, vertical_dilation,
						horizontal_dilation, weight_init, weight_reg, weight_clip, weight_max_l1_norm,
						weight_max_l2_norm, weight_grad_clip, weight_grad_max_l1_norm, weight_grad_max_l2_norm,
						bias_reg, bias_clip, bias_max_l1_norm, bias_max_l2_norm, bias_grad_clip,
						bias_grad_max_l1_norm, bias_grad_max_l2_norm) { }
	inline TransConvKernelLayer(const TransConvKernelLayer<Scalar,2>& layer, bool share_params = false) :
			TransConvBase::TransConvKernelLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone() const {
		return new TransConvKernelLayer(*this);
	}
	inline Root* clone_with_shared_params() {
		return new TransConvKernelLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,3>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return TransConvBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1),
				in.dimension(2), 1u }), training).reshape(std::array<std::size_t,3>({ batch_size,
						KernelBase::output_dims(0), KernelBase::output_dims(1) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,3>(out_grad.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		Tensor<Scalar,4> prev_out_grad = TransConvBase::_pass_back(TensorMap<Scalar,4>(out_grad.data(),
				{ batch_size, KernelBase::output_dims(0), KernelBase::output_dims(1) / TransConvBase::filters,
						TransConvBase::filters }));
		if (KernelBase::is_input_layer())
			return typename Root::Data();
		return TensorMap<Scalar,3>(prev_out_grad.data(), { batch_size, KernelBase::input_dims(0),
				KernelBase::input_dims(1) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template for a transposed 2D convolutional layer operating on rank-1 data batches (rank-2 tensors).
 * The results of the convolutions of the filters and the input tensor are concatenated along the highest (2nd)
 * rank of the output tensor.
 *
 * \see https://arxiv.org/abs/1603.07285v1
 */
template<typename Scalar>
class TransConvKernelLayer<Scalar,1> : public TransConvKernelLayerBase<Scalar,1> {
	typedef Layer<Scalar,1> Root;
	typedef KernelLayer<Scalar,1> KernelBase;
	typedef TransConvKernelLayerBase<Scalar,1> TransConvBase;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the weights of
	 * the layer.
	 * @param receptor_length The length of the receptor.
	 * @param padding The extent of padding to apply to the input tensor along its length on both ends.
	 * @param stride The convolution stride i.e. the number of elements by which the receptor is to be
	 * shifted along the length of the input tensor.
	 * @param dilation The extent of dilation to apply to the receptor.
	 * @param weight_reg An optional regularization function to apply to the weights.
	 * @param weight_clip The maximum allowed absolute weight value. If it is 0 or less, no value
	 * clipping is performed.
	 * @param weight_max_l1_norm The maximum allowed L1 weight value norm. If it is 0 or less, no L1
	 * max norm constraint is enforced.
	 * @param weight_max_l2_norm The maximum allowed L2 weight value norm. If it is 0 or less, no L2
	 * max norm constraint is enforced.
	 * @param weight_grad_clip The maximum allowed absolute weight gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param weight_grad_max_l1_norm The maximum allowed L1 weight gradient norm. If it is 0 or less,
	 * no L1 gradient max norm constraint is enforced.
	 * @param weight_grad_max_l2_norm The maximum allowed L2 weight gradient norm. If it is 0 or less,
	 * no L2 gradient max norm constraint is enforced.
	 * @param bias_reg An optional regularization function to apply to the bias.
	 * @param bias_clip The maximum allowed absolute bias value. If it is 0 or less, no value clipping
	 * is performed.
	 * @param bias_max_l1_norm The maximum allowed L1 bias value norm. If it is 0 or less, no bias L1
	 * max norm constraint is enforced.
	 * @param bias_max_l2_norm The maximum allowed L2 bias value norm. If it is 0 or less, no bias L2
	 * max norm constraint is enforced.
	 * @param bias_grad_clip The maximum allowed absolute bias gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param bias_grad_max_l1_norm The maximum allowed L1 bias gradient norm. If it is 0 or less, no
	 * bias L1 gradient max norm constraint is enforced.
	 * @param bias_grad_max_l2_norm The maximum allowed L2 bias gradient norm. If it is 0 or less, no
	 * bias L2 gradient max norm constraint is enforced.
	 */
	TransConvKernelLayer(const typename Root::Dims& input_dims, std::size_t filters,
			ParamInitSharedPtr<Scalar> weight_init, std::size_t receptor_length = 3, std::size_t padding = 1,
			std::size_t stride = 1, std::size_t dilation = 0, ParamRegSharedPtr<Scalar> weight_reg = nullptr,
			Scalar weight_clip = 0, Scalar weight_max_l1_norm = 0, Scalar weight_max_l2_norm = 0,
			Scalar weight_grad_clip = 0, Scalar weight_grad_max_l1_norm = 0, Scalar weight_grad_max_l2_norm = 0,
			ParamRegSharedPtr<Scalar> bias_reg = nullptr, Scalar bias_clip = 0, Scalar bias_max_l1_norm = 0,
			Scalar bias_max_l2_norm = 0, Scalar bias_grad_clip = 0, Scalar bias_grad_max_l1_norm = 0,
			Scalar bias_grad_max_l2_norm = 0) :
				TransConvBase::TransConvKernelLayerBase(input_dims, filters, receptor_length, 1, padding, 0,
						stride, 1, dilation, 0, weight_init, weight_reg, weight_clip, weight_max_l1_norm,
						weight_max_l2_norm, weight_grad_clip, weight_grad_max_l1_norm, weight_grad_max_l2_norm,
						bias_reg, bias_clip, bias_max_l1_norm, bias_max_l2_norm, bias_grad_clip,
						bias_grad_max_l1_norm, bias_grad_max_l2_norm) { }
	inline TransConvKernelLayer(const TransConvKernelLayer<Scalar,1>& layer, bool share_params = false) :
			TransConvBase::TransConvKernelLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone() const {
		return new TransConvKernelLayer(*this);
	}
	inline Root* clone_with_shared_params() {
		return new TransConvKernelLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,2>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return TransConvBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), 1u, 1u }),
				training).reshape(std::array<std::size_t,2>({ batch_size, KernelBase::output_dims(0) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,2>(out_grad.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		Tensor<Scalar,4> prev_out_grad = TransConvBase::_pass_back(TensorMap<Scalar,4>(out_grad.data(),
				{ batch_size, KernelBase::output_dims(0) / TransConvBase::filters, 1, TransConvBase::filters }));
		if (KernelBase::is_input_layer())
			return typename Root::Data();
		return TensorMap<Scalar,2>(prev_out_grad.data(), { batch_size, KernelBase::input_dims(0) });
	}
private:
	std::size_t batch_size;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_KERNEL_TRANSCONVKERNELLAYER_H_ */
