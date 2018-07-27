/*
 * PoolLayer.hpp
 *
 *  Created on: 24 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_POOLLAYER_H_
#define C_ATTL3_LAYER_POOLLAYER_H_

#include <array>
#include <cassert>

#include "core/Layer.hpp"

namespace cattle {

/**
 * An abstract base class template representing a pooling layer.
 */
template<typename Scalar, std::size_t Rank>
class PoolLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	inline Base* clone_with_shared_params() {
		return Base::clone();
	}
	inline const Base& get_params_owner() const {
		return *this;
	}
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
		return output_dims;
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline std::vector<const Parameters<Scalar>*> get_params() const {
		return std::vector<const Parameters<Scalar>*>(0);
	}
	inline std::vector<Parameters<Scalar>*> get_params() {
		return std::vector<Parameters<Scalar>*>(0);
	}
protected:
	typedef std::array<std::size_t,4> Array4;
	typedef std::array<std::size_t,2> ReductionRanksArray2;
	inline PoolLayer(const typename Base::Dims& input_dims, std::size_t receptor_height, std::size_t receptor_width,
			std::size_t vertical_stride, std::size_t horizontal_stride) :
				ext_input_dims(input_dims.template extend<3 - Rank>()),
				ext_output_dims(calculate_output_dims(ext_input_dims, receptor_height, receptor_width,
						vertical_stride, horizontal_stride)),
				input_dims(input_dims),
				output_dims(ext_output_dims.template contract<3 - Rank>()),
				receptor_height(receptor_height),
				receptor_width(receptor_width),
				vertical_stride(vertical_stride),
				horizontal_stride(horizontal_stride),
				height_rem(ext_input_dims(0) - receptor_height),
				width_rem(ext_input_dims(1) - receptor_width),
				input_layer(false),
				reduction_ranks({ 1u, 2u }),
				broadcast({ 1u, receptor_height, receptor_width, 1u }),
				patch_offsets({ 0u, 0u, 0u, 0u }),
				patch_extents({ 0u, receptor_height, receptor_width, ext_input_dims(2) }),
				reduced_patch_offsets({ 0u, 0u, 0u, 0u }),
				reduced_patch_extents({ 0u, 1u, 1u, ext_input_dims(2) }) {
		assert(receptor_height > 0 && receptor_width > 0);
		assert(vertical_stride > 0 && horizontal_stride > 0);
		assert(ext_input_dims(0) >= receptor_height && ext_input_dims(1) >= receptor_width);
	}
	inline Tensor<Scalar,4> _pass_forward(Tensor<Scalar,4> in, bool training) {
		std::size_t rows = in.dimension(0);
		patch_extents[0] = rows;
		reduced_patch_extents[0] = rows;
		Tensor<Scalar,4> out(rows, ext_output_dims(0), ext_output_dims(1), ext_output_dims(2));
		_init_cache();
		std::size_t patch_ind = 0;
		std::size_t out_i = 0;
		for (std::size_t i = 0; i <= width_rem; i += horizontal_stride, ++out_i) {
			patch_offsets[2] = i;
			reduced_patch_offsets[2] = out_i;
			std::size_t out_j = 0;
			for (std::size_t j = 0; j <= height_rem; j += vertical_stride, ++out_j) {
				patch_offsets[1] = j;
				reduced_patch_offsets[1] = out_j;
				Tensor<Scalar,4> patch = in.slice(patch_offsets, patch_extents);
				out.slice(reduced_patch_offsets, reduced_patch_extents) = _reduce(patch, patch_ind++);
			}
		}
		return out;
	}
	inline Tensor<Scalar,4> _pass_back(Tensor<Scalar,4> out_grad) {
		Tensor<Scalar,4> prev_out_grad(patch_extents[0], ext_input_dims(0), ext_input_dims(1),  ext_input_dims(2));
		prev_out_grad.setZero();
		std::size_t patch_ind = 0;
		std::size_t out_grad_i = 0;
		for (std::size_t i = 0; i <= width_rem; i += horizontal_stride, ++out_grad_i) {
			patch_offsets[2] = i;
			reduced_patch_offsets[2] = out_grad_i;
			std::size_t out_grad_j = 0;
			for (std::size_t j = 0; j <= height_rem; j += vertical_stride, ++out_grad_j) {
				patch_offsets[1] = j;
				reduced_patch_offsets[1] = out_grad_j;
				Tensor<Scalar,4> reduced_patch_grad = out_grad.slice(reduced_patch_offsets, reduced_patch_extents);
				// Accumulate the gradients where the patches overlap.
				prev_out_grad.slice(patch_offsets, patch_extents) += _d_reduce(reduced_patch_grad, patch_ind++);
			}
		}
		return prev_out_grad;
	}
	/**
	 * Initializes the cache required for back-propagation.
	 */
	virtual void _init_cache() = 0;
	/**
	 * Reduces the input tensor patch along the specified ranks.
	 *
	 * @param patch A tensor representing a spatial patch of the input tensor.
	 * @param patch_ind The index of the patch.
	 * @return The reduced tensor.
	 */
	virtual Tensor<Scalar,4> _reduce(const Tensor<Scalar,4>& patch, std::size_t patch_ind) = 0;
	/**
	 * Differentiates the reduction function and returns the derivative of the loss function
	 * w.r.t. the non-reduced patch.
	 *
	 * @param grad The derivative of the loss function w.r.t. the reduced patch.
	 * @param patch_ind The index of the patch.
	 * @return The derivative of the loss function w.r.t. the non-reduced patch.
	 */
	virtual Tensor<Scalar,4> _d_reduce(const Tensor<Scalar,4>& grad, std::size_t patch_ind) = 0;
	const Dimensions<std::size_t,3> ext_input_dims, ext_output_dims;
	const typename Base::Dims input_dims, output_dims;
	const std::size_t receptor_height, receptor_width, vertical_stride, horizontal_stride, height_rem, width_rem;
	// Arrays for tensor manipulation.
	ReductionRanksArray2 reduction_ranks;
	Array4 broadcast, patch_offsets, patch_extents, reduced_patch_offsets, reduced_patch_extents, dil_strides;
private:
	inline static std::size_t calculate_spatial_output_dim(std::size_t input_dim, std::size_t receptor_size,
			std::size_t stride) {
		return (input_dim - receptor_size) / stride + 1;
	}
	inline static Dimensions<std::size_t,3> calculate_output_dims(const Dimensions<std::size_t,3>& input_dims,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_stride,
			std::size_t horizontal_stride) {
		return { calculate_spatial_output_dim(input_dims(0), receptor_height, vertical_stride),
				calculate_spatial_output_dim(input_dims(1), receptor_width, horizontal_stride),
				input_dims(2) };
	}
	bool input_layer;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_POOLLAYER_H_ */
