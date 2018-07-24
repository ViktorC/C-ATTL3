/*
 * ActivationLayer.hpp
 *
 *  Created on: 22 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATIONLAYER_H_

#include "core/Layer.hpp"

namespace cattle {

/**
 * An abstract class template that represents an activation function layer.
 */
template<typename Scalar, std::size_t Rank>
class ActivationLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef ActivationLayer<Scalar,Rank> Self;
	typedef Dimensions<std::size_t,Rank> Dims;
public:
	virtual ~ActivationLayer() = default;
	inline Base* clone_with_shared_params() {
		return Base::clone();
	}
	inline const Base& get_params_owner() const {
		return *this;
	}
	inline const Dims& get_input_dims() const {
		return dims;
	}
	inline const Dims& get_output_dims() const {
		return dims;
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline std::vector<const Parameters<Scalar>*>& get_params() const {
		return std::vector<const Parameters<Scalar>*>(0);
	}
	inline std::vector<Parameters<Scalar>*>& get_params() {
		return std::vector<Parameters<Scalar>*>(0);
	}
protected:
	inline ActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			dims(dims),
			input_layer(false) { }
	const Dims dims;
	bool input_layer;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATIONLAYER_H_ */
