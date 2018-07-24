/*
 * TanhActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_TANHACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_TANHACTIVATIONLAYER_H_

#include <cassert>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template representing a hyperbolic tangent activation function layer.
 *
 * \f$f(x) = \text{tanh}(x)\f$
 */
template<typename Scalar, std::size_t Rank>
class TanhActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline TanhActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			Base::ActivationLayer(dims) { }
	inline Root* clone() const {
		return new TanhActivationLayer(*this);
	}
	inline void empty_cache() {
		out_cache = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		out_cache = in.tanh();
		return out_cache;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && out_cache.dimension(0) == out_grad.dimension(0));
		return (out.constant(1) - out_cache * out_cache) * out_grad;
	}
private:
	typename Root::Data out_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_TANHACTIVATIONLAYER_H_ */
