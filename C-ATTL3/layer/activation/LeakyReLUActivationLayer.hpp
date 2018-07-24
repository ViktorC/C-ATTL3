/*
 * LeakyReLUActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_LEAKYRELUACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_LEAKYRELUACTIVATIONLAYER_H_

#include <cassert>
#include <utility>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template representing a leaky rectified linear unit activation function. Unlike
 * traditional ReLU layers leaky ReLU layers do not set negative elements of the input to
 * 0 but scale them by a small constant alpha. This function is not differentiable.
 *
 * \f[
 *   f(x) = \begin{cases}
 *     \alpha x & \text{for } x < 0\\
 *     x & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 *
 * \see https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
 */
template<typename Scalar, std::size_t Rank>
class LeakyReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param alpha The factor by which negative inputs are to be scaled.
	 */
	inline LeakyReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar alpha = 1e-1) :
			Base::ActivationLayer(dims),
			alpha(alpha) { }
	inline Root* clone() const {
		return new LeakyReLUActivationLayer(*this);
	}
	inline void empty_cache() {
		in_cache = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		in_cache = std::move(in);
		return in_cache.cwiseMax(in * alpha);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && in_cache.dimension(0) == out_grad.dimension(0));
		return in_cache.unaryExpr([alpha](Scalar e) { return (Scalar) (e >= 0 ? 1 : alpha); }) * out_grad;
	}
private:
	const Scalar alpha;
	typename Root::Data in_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_LEAKYRELUACTIVATIONLAYER_H_ */
