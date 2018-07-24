/*
 * SoftsignActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_SOFTSIGNACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_SOFTSIGNACTIVATIONLAYER_H_

#include <cassert>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template representing a softsign activation function layer, an alternative to the
 * tanh layer.
 *
 * \f$f(x) = \frac{x}{1 + \left|x\right|}\f$
 */
template<typename Scalar, std::size_t Rank>
class SoftsignActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SoftsignActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			Base::ActivationLayer(dims) { }
	inline Root* clone() const {
		return new SoftsignActivationLayer(*this);
	}
	inline void empty_cache() {
		denom_cache = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		denom_cache = in.constant(1) + in.abs();
		return in / denom_cache;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && denom_cache.dimension(0) == out_grad.dimension(0));
		return denom_cache.square().inverse() * out_grad;
	}
private:
	// Staged computation cache.
	typename Root::Data denom_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_SOFTSIGNACTIVATIONLAYER_H_ */
