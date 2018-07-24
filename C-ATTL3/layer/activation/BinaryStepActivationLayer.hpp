/*
 * BinaryStepActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_BINARYSTEPACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_BINARYSTEPACTIVATIONLAYER_H_

#include <cassert>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template that represents a binary step activation function that outputs either
 * 1 or 0 based on the signum of its input. This function is not differentiable.
 *
 * \f[
 *   f(x) = \begin{cases}
 *     0 & \text{for } x < 0\\
 *     1 & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 */
template<typename Scalar, std::size_t Rank>
class BinaryStepActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline BinaryStepActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			Base::ActivationLayer(dims) { }
	inline Root* clone() const {
		return new BinaryStepActivationLayer(*this);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return in.unaryExpr([](Scalar e) { return (Scalar) (e >= 0); });
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		return out_grad.constant(0);
	}
private:
	std::size_t batch_size;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_BINARYSTEPACTIVATIONLAYER_H_ */
