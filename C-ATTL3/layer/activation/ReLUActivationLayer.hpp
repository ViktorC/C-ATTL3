/*
 * ReLUActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_RELUACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_RELUACTIVATIONLAYER_H_

#include <cassert>
#include <utility>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template representing a rectified linear unit (ReLU) activation function. ReLU
 * layers set all negative elements of the input to 0. This function is not differentiable.

 * \f[
 *   f(x) = \begin{cases}
 *     0 & \text{for } x < 0\\
 *     x & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 */
template<typename Scalar, std::size_t Rank>
class ReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline ReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			Base::ActivationLayer(dims) { }
	inline Root* clone() const {
		return new ReLUActivationLayer(*this);
	}
	inline void empty_cache() {
		in_cache = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		in_cache = std::move(in);
		return in_cache.cwiseMax((Scalar) 0);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && in_cache.dimension(0) == out_grad.dimension(0));
		return in_cache.unaryExpr([](Scalar e) { return (Scalar) (e >= 0); }) * out_grad;
	}
private:
	typename Root::Data in_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_RELUACTIVATIONLAYER_H_ */
