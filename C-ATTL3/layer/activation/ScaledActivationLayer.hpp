/*
 * ScaledActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_SCALEDACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_SCALEDACTIVATIONLAYER_H_

#include <cassert>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template that represents a linearly scaled activation layer.
 *
 * \f$f(x) = c x\f$
 */
template<typename Scalar, std::size_t Rank>
class ScaledActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param scale The factor by which the input is to be scaled.
	 */
	inline ScaledActivationLayer(const typename Root::Dims& dims, Scalar scale) :
			Base::ActivationLayer(dims),
			scale(scale) { }
	inline Root* clone() const {
		return new ScaledActivationLayer(*this);
	}
	inline void empty_cache() { }
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return in * scale;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		if (Base::is_input_layer())
			return typename Root::Data();
		return out_grad * scale;
	}
private:
	const Scalar scale;
	std::size_t batch_size;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_SCALEDACTIVATIONLAYER_H_ */
