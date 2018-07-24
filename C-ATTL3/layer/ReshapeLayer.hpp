/*
 * ReshapeLayer.hpp
 *
 *  Created on: 24 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_RESHAPELAYER_H_
#define C_ATTL3_LAYER_RESHAPELAYER_H_

#include <array>
#include <cassert>

#include "core/Layer.hpp"

namespace cattle {

/**
 * A class template representing a reshaping layer that outputs a reshaped copy of the input
 * tensor with the same volume. The data backing the tensor is not shifted in any way.
 */
template<typename Scalar, std::size_t Rank>
class ReshapeLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Base::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param input_dims The nominal input dimensions of the layer.
	 * @param output_dims The dimensions of the reshaped tensor. The output tensor must have
	 * the same volume as the input tensor.
	 */
	inline ReshapeLayer(const typename Base::Data& input_dims, const typename Base::Data& output_dims) :
				input_dims(input_dims),
				output_dims(output_dims),
				input_layer(false),
				input_conversion_dims(output_dims.template promote<>()),
				output_conversion_dims(input_dims.template promote<>()) {
		assert(input_dims.get_volume() == output_dims.get_volume());
	}
	inline Base* clone() const {
		return new ReshapeLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return clone();
	}
	inline const Base& get_params_owner() const {
		return *this;
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
	inline std::vector<const Parameters<Scalar>*>& get_params() const {
		return std::vector<const Parameters<Scalar>*>(0);
	}
	inline std::vector<Parameters<Scalar>*>& get_params() {
		return std::vector<Parameters<Scalar>*>(0);
	}
	inline void empty_cache() { }
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == input_dims);
		assert(in.dimension(0) > 0);
		input_conversion_dims[0] = in.dimension(0);
		return in.reshape(input_conversion_dims);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == output_dims);
		assert(out_grad.dimension(0) > 0 && input_conversion_dims[0] == out_grad.dimension(0));
		if (input_layer)
			return typename Base::Data();
		output_conversion_dims[0] = input_conversion_dims[0];
		return out_grad.reshape(output_conversion_dims);
	}
private:
	const typename Base::Dims input_dims, output_dims;
	RankwiseArray input_conversion_dims, output_conversion_dims;
	bool input_layer;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_RESHAPELAYER_H_ */
