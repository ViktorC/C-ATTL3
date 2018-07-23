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
#include <utility>

#include "core/Layer.hpp"

namespace cattle {

template<typename Scalar, std::size_t Rank>
class KernelLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef KernelLayer<Scalar,Rank> Self;
	typedef Dimensions<std::size_t,Rank> Dims;
	typedef std::shared_ptr<Parameters<Scalar>*> ParamsSharedPtr;
public:
	inline KernelLayer(const Dims& input_dims, const Dims& output_dims, ParamsPtr filter,
			ParamsPtr bias) :
				owner(*this),
				input_dims(input_dims),
				output_dims(output_dims),
				filter(std::move(filter)),
				bias(std::move(bias)),
				input_layer(false) {
		assert(filter && bias);
	}
	inline KernelLayer(const Self& layer, bool share_params = false) :
			owner(share_params || layer.is_shared_params_clone() ? layer.owner : *this),
			input_dims(layer.input_dims),
			output_dims(layer.output_dims),
			filter(share_params || layer.is_shared_params_clone() ? layer.filter : ParamsSharedPtr(layer.filter.clone())),
			bias(share_params || layer.is_shared_params_clone() ? layer.bias : ParamsSharedPtr(layer.bias.clone())),
			input_layer(layer.input_layer) { }
	virtual ~KernelLayer() = default;
	inline const Base& get_params_owner() const {
		return owner;
	}
	inline const Dims& get_input_dims() const {
		return input_dims;
	}
	inline const Dims& get_output_dims() const {
		return output_dims;
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline std::vector<const Parameters<Scalar>*>& get_params() const {
		return std::vector<const Parameters<Scalar>*>({ filter.get(), bias.get()});
	}
	inline std::vector<Parameters<Scalar>*>& get_params() {
		return std::vector<Parameters<Scalar>*>({ filter.get(), bias.get()});
	}
protected:
	const Self& owner;
	const Dims input_dims, output_dims;
	ParamsSharedPtr filter, bias;
	bool input_layer;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_KERNELLAYER_H_ */
