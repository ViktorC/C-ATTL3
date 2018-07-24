/*
 * SwishActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_SWISHACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_SWISHACTIVATIONLAYER_H_

#include <cassert>
#include <utility>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template representing the Swish activation function.
 *
 * \f$f(x) = x \sigma(\beta x)\f$
 *
 * \see https://arxiv.org/abs/1710.05941
 */
template<typename Scalar, std::size_t Rank>
class SwishActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param beta The factor by which the input of the sigmoid factor of the Swish
	 * function is to be scaled.
	 */
	inline SwishActivationLayer(const typename Root::Dims& dims, Scalar beta = 1) :
			Base::ActivationLayer(dims),
			beta(beta) { }
	inline Root* clone() const {
		return new SwishActivationLayer(*this);
	}
	inline void empty_cache() {
		in_cache = typename Root::Data();
		sig_out_cache = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		sig_out_cache = (-beta * in).exp() + in.constant(1).inverse();
		in_cache = std::move(in);
		return in_cache * sig_out_cache;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && sig_out_cache.dimension(0) == out_grad.dimension(0));
		if (Base::is_input_layer())
			return typename Root::Data();
		return sig_out_cache * ((sig_out_cache.constant(1) - sig_out_cache) * beta * in_cache +
				sig_out_cache.constant(1)) * out_grad;
	}
private:
	const Scalar beta;
	// Staged computation cache.
	typename Root::Data in_cache, sig_out_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_ELUACTIVATIONLAYER_H_ */
