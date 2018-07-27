/*
 * KernelLayer.hpp
 *
 *  Created on: 22 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_KERNELLAYER_H_
#define C_ATTL3_LAYER_KERNELLAYER_H_

#include <cassert>
#include <memory>

#include "core/Layer.hpp"

namespace cattle {

/**
 * An alias for a shared pointer to a Parameters instance.
 */
template<typename Scalar>
using ParamsSharedPtr = std::shared_ptr<Parameters<Scalar>>;

/**
 * An abstract class template that represents a kernel-based linear transformation function layer.
 */
template<typename Scalar, std::size_t Rank>
class KernelLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef KernelLayer<Scalar,Rank> Self;
public:
	virtual ~KernelLayer() = default;
	inline const Base& get_params_owner() const {
		return owner;
	}
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
		return output_dims;
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline std::vector<const Parameters<Scalar>*> get_params() const {
		return std::vector<const Parameters<Scalar>*>({ weights.get(), bias.get() });
	}
	inline std::vector<Parameters<Scalar>*> get_params() {
		return std::vector<Parameters<Scalar>*>({ weights.get(), bias.get() });
	}
protected:
	/**
	 * @param input_dims The input dimensions of the layer.
	 * @param output_dims The output dimensions of the layer.
	 * @param weights The weight parameters; cannot be null.
	 * @param bias The bias parameters; cannot be null.
	 */
	inline KernelLayer(const typename Base::Dims& input_dims, const typename Base::Dims& output_dims,
			ParamsSharedPtr<Scalar> weights, ParamsSharedPtr<Scalar> bias) :
				owner(*this),
				input_dims(input_dims),
				output_dims(output_dims),
				weights(weights),
				bias(bias),
				input_layer(false) {
		assert(weights && bias);
	}
	inline KernelLayer(const Self& layer, bool share_params = false) :
			owner(share_params ? layer.owner : *this),
			input_dims(layer.input_dims),
			output_dims(layer.output_dims),
			weights(share_params ? layer.weights : ParamsSharedPtr<Scalar>(layer.weights->clone())),
			bias(share_params ? layer.bias : ParamsSharedPtr<Scalar>(layer.bias->clone())),
			input_layer(layer.input_layer) { }
	const Self& owner;
	const typename Base::Dims input_dims, output_dims;
	ParamsSharedPtr<Scalar> weights, bias;
	bool input_layer;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_KERNELLAYER_H_ */
