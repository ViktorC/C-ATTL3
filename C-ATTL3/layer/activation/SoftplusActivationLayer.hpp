/*
 * SoftplusActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_SOFTPLUSACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_SOFTPLUSACTIVATIONLAYER_H_

#include <cassert>
#include <utility>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template representing a softplus activation function layer. The softplus activation function
 * is a differentiable function that approximates the rectified linear unit function.
 *
 * \f$f(x) = \ln(1 + e^x)\f$
 */
template<typename Scalar, std::size_t Rank>
class SoftplusActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SoftplusActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			Base::ActivationLayer(dims) { }
	inline Root* clone() const {
		return new SoftplusActivationLayer(*this);
	}
	inline void empty_cache() {
		in_cache = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		in_cache = std::move(in);
		return (in_cache.exp() + in_cache.constant(1)).log();
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && in.dimension(0) == out_grad.dimension(0));
		return ((-in_cache).exp() + in_cache.constant(1)).inverse() * out_grad;
	}
private:
	// Staged computation cache.
	typename Root::Data in_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_SOFTPLUSACTIVATIONLAYER_H_ */
