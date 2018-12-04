/*
 * DenseKernelLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_KERNEL_DENSEKERNELLAYER_H_
#define C_ATTL3_LAYER_KERNEL_DENSEKERNELLAYER_H_

#include <array>
#include <cassert>

#include "layer/KernelLayer.hpp"
#include "parameter_initialization/ZeroParameterInitialization.hpp"
#include "parameters/StandardParameters.hpp"

namespace cattle {

/**
 * A class template representing a fully connected layer.
 */
template<typename Scalar, std::size_t Rank = 1>
class DenseKernelLayer : public KernelLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef KernelLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * @param output_size The length of the vector output for each sample.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the weights.
	 * @param weight_reg An optional regularization function to apply to the weights.
	 * @param weight_clip The maximum allowed absolute weight value. If it is 0 or less, no
	 * value clipping is performed.
	 * @param weight_max_l1_norm The maximum allowed L1 weight value norm. If it is 0 or
	 * less, no L1 max norm constraint is enforced.
	 * @param weight_max_l2_norm The maximum allowed L2 weight value norm. If it is 0 or
	 * less, no L2 max norm constraint is enforced.
	 * @param weight_grad_clip The maximum allowed absolute weight gradient. If it is 0
	 * or less, no gradient clipping is performed.
	 * @param weight_grad_max_l1_norm The maximum allowed L1 weight gradient norm. If it
	 * is 0 or less, no L1 gradient max norm constraint is enforced.
	 * @param weight_grad_max_l2_norm The maximum allowed L2 weight gradient norm. If it
	 * is 0 or less, no L2 gradient max norm constraint is enforced.
	 * @param bias_reg An optional regularization function to apply to the bias.
	 * @param bias_clip The maximum allowed absolute bias value. If it is 0 or less, no
	 * value clipping is performed.
	 * @param bias_max_l1_norm The maximum allowed L1 bias value norm. If it is 0 or less,
	 * no bias L1 max norm constraint is enforced.
	 * @param bias_max_l2_norm The maximum allowed L2 bias value norm. If it is 0 or less,
	 * no bias L2 max norm constraint is enforced.
	 * @param bias_grad_clip The maximum allowed absolute bias gradient. If it is 0 or
	 * less, no gradient clipping is performed.
	 * @param bias_grad_max_l1_norm The maximum allowed L1 bias gradient norm. If it is 0
	 * or less, no bias L1 gradient max norm constraint is enforced.
	 * @param bias_grad_max_l2_norm The maximum allowed L2 bias gradient norm. If it is 0
	 * or less, no bias L2 gradient max norm constraint is enforced.
	 */
	inline DenseKernelLayer(const typename Root::Dims& input_dims, std::size_t output_size,
			ParamInitSharedPtr<Scalar> weight_init, ParamRegSharedPtr<Scalar> weight_reg = nullptr,
			Scalar weight_clip = 0, Scalar weight_max_l1_norm = 0, Scalar weight_max_l2_norm = 0,
			Scalar weight_grad_clip = 0, Scalar weight_grad_max_l1_norm = 0, Scalar weight_grad_max_l2_norm = 0,
			ParamRegSharedPtr<Scalar> bias_reg = nullptr, Scalar bias_clip = 0, Scalar bias_max_l1_norm = 0,
			Scalar bias_max_l2_norm = 0, Scalar bias_grad_clip = 0, Scalar bias_grad_max_l1_norm = 0,
			Scalar bias_grad_max_l2_norm = 0) :
				Base(input_dims, { output_size },
						std::make_shared<StandardParameters<Scalar>>(input_dims.get_volume(), output_size, true,
								weight_init, weight_reg, weight_clip, weight_max_l1_norm, weight_max_l2_norm,
								weight_grad_clip, weight_grad_max_l1_norm, weight_grad_max_l2_norm),
						std::make_shared<StandardParameters<Scalar>>(1, output_size, true,
								std::make_shared<ZeroParameterInitialization<Scalar>>(), bias_reg, bias_clip,
								bias_max_l1_norm, bias_max_l2_norm, bias_grad_clip, bias_grad_max_l1_norm,
								bias_grad_max_l2_norm)),
				out_conversion_dims(Base::output_dims.template promote<>()),
				prev_out_conversion_dims(Base::input_dims.template promote<>()) { }
	inline DenseKernelLayer(const DenseKernelLayer<Scalar,Rank>& layer, bool share_params = false) :
				Base(layer, share_params),
				out_conversion_dims(layer.out_conversion_dims),
				prev_out_conversion_dims(layer.prev_out_conversion_dims),
				in_mat_cache(layer.in_mat_cache) { }
	inline Root* clone() const {
		return new DenseKernelLayer(*this);
	}
	inline Root* clone_with_shared_params() {
		return new DenseKernelLayer(*this, true);
	}
	inline void empty_cache() {
		in_mat_cache = Matrix<Scalar>();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(in.dimensions()).template demote<>()) == Base::input_dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		out_conversion_dims[0] = rows;
		prev_out_conversion_dims[0] = rows;
		in_mat_cache = MatrixMap<Scalar>(in.data(), in.dimension(0), Base::input_dims.get_volume());
		Matrix<Scalar> out_mat = (in_mat_cache * Base::weights->get_values()).rowwise() +
				Base::bias->get_values().row(0);
		return TensorMap<Scalar,Root::DATA_RANK>(out_mat.data(), out_conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::output_dims);
		assert(out_grad.dimension(0) > 0 && out_conversion_dims[0] == out_grad.dimension(0));
		// Compute the gradient of the outputs with respect to the weights and the bias.
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), out_grad.dimension(0), Base::output_dims.get_volume());
		Base::weights->accumulate_grad(in_mat_cache.transpose() * out_grad_mat);
		Base::bias->accumulate_grad(out_grad_mat.colwise().sum());
		if (Base::is_input_layer())
			return typename Root::Data();
		// Compute the gradient of the previous layer's output.
		Matrix<Scalar> prev_out_grad_mat = out_grad_mat * Base::weights->get_values().transpose();
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad_mat.data(), prev_out_conversion_dims);
	}
private:
	RankwiseArray out_conversion_dims, prev_out_conversion_dims;
	// Staged computation caches
	Matrix<Scalar> in_mat_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_KERNEL_DENSEKERNELLAYER_H_ */
