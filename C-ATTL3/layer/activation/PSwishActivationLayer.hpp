/*
 * PSwishActivationLayer.hpp
 *
 *  Created on: 24 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_PSWISHACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_PSWISHACTIVATIONLAYER_H_

#include <array>
#include <cassert>
#include <utility>

#include "layer/ActivationLayer.hpp"
#include "parameter_initialization/ConstantParameterInitialization.hpp"
#include "parameters/StandardParameters.hpp"

namespace cattle {

/**
 * A class template representing the parametric Swish activation function with learnable beta
 * values.
 *
 * \f$f(x) = x \sigma(\beta x)\f$
 *
 * \see https://arxiv.org/abs/1710.05941
 */
template<typename Scalar, std::size_t Rank>
class PSwishActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param init_beta The initial factor by which the input of the sigmoid factor of the
	 * Swish function is to be scaled.
	 * @param beta_reg An optional regularization function to apply to the parameters.
	 * @param beta_clip The maximum allowed absolute parameter value. If it is 0 or less, no
	 * value clipping is performed.
	 * @param beta_max_l1_norm The maximum allowed L1 beta value norm. If it is 0 or
	 * less, no L1 max norm constraint is enforced.
	 * @param beta_max_l2_norm The maximum allowed L2 beta value norm. If it is 0 or
	 * less, no L2 max norm constraint is enforced.
	 * @param beta_grad_clip The maximum allowed absolute parameter gradient. If it is 0
	 * or less, no gradient clipping is performed.
	 * @param beta_grad_max_l1_norm The maximum allowed L1 beta gradient norm. If it
	 * is 0 or less, no L1 gradient max norm constraint is enforced.
	 * @param beta_grad_max_l2_norm The maximum allowed L2 beta gradient norm. If it
	 * is 0 or less, no L2 gradient max norm constraint is enforced.
	 */
	inline PSwishActivationLayer(const typename Root::Dims& dims, Scalar init_beta = 1,
			ParamRegSharedPtr<Scalar> beta_reg = nullptr, Scalar beta_clip = 0, Scalar beta_max_l1_norm = 0,
			Scalar beta_max_l2_norm = 0, Scalar beta_grad_clip = 0, Scalar beta_grad_max_l1_norm = 0,
			Scalar beta_grad_max_l2_norm = 0) :
				Base::ActivationLayer(dims, std::make_shared<StandardParameters<Scalar>>(1, dims.get_volume(),
						true, std::make_shared<ConstantParameterInitialization<Scalar>>(init_beta),
						beta_reg, beta_clip, beta_max_l1_norm, beta_max_l2_norm, beta_grad_clip,
						beta_grad_max_l1_norm, beta_grad_max_l2_norm)),
				conversion_dims(dims.template promote<>()) { }
	inline PSwishActivationLayer(const PSwishActivationLayer<Scalar,Rank>& layer, bool share_params = false) :
			Base::ActivationLayer(layer, share_params),
			conversion_dims(layer.conversion_dims),
			in_mat_cache(layer.in_mat_cache),
			sig_out_mat_cache(layer.sig_out_mat_cache) { }
	inline Root* clone() const {
		return new PSwishActivationLayer(*this);
	}
	inline Root* clone_with_shared_params() {
		return new PSwishActivationLayer(*this, true);
	}
	inline void empty_cache() {
		in_mat_cache = Matrix<Scalar>();
		sig_out_mat_cache = Matrix<Scalar>();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		conversion_dims[0] = in.dimension(0);
		in_mat_cache = MatrixMap<Scalar>(in.data(), conversion_dims[0], in.size() / conversion_dims[0]);
		sig_out_mat_cache = ((in_mat_cache * (-Base::params->get_values()).asDiagonal()).array().exp() + 1).inverse();
		Matrix<Scalar> out_mat = in_mat_cache.cwiseProduct(sig_out_mat_cache);
		return TensorMap<Scalar,Root::DATA_RANK>(out_mat.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && conversion_dims[0] == out_grad.dimension(0));
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), conversion_dims[0], out_grad.size() / conversion_dims[0]);
		Matrix<Scalar> one_min_sig_out_mat = 1 - sig_out_mat_cache.array();
		Base::params->accumulate_grad(sig_out_mat_cache.cwiseProduct(one_min_sig_out_mat).cwiseProduct(in_mat_cache)
				.cwiseProduct(in_mat_cache).cwiseProduct(out_grad_mat).colwise().sum());
		if (Base::is_input_layer())
			return typename Root::Data();
		Matrix<Scalar> prev_out_grad_mat = sig_out_mat_cache.cwiseProduct(((one_min_sig_out_mat *
				Base::params->get_values().asDiagonal()).array() *
				in_mat_cache.array() + 1).matrix()).cwiseProduct(out_grad_mat);
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad_mat.data(), conversion_dims);
	}
private:
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in_mat_cache, sig_out_mat_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_PSWISHACTIVATIONLAYER_H_ */
