/*
 * BroadcastLayer.hpp
 *
 *  Created on: 24 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_BROADCASTLAYER_H_
#define C_ATTL3_LAYER_BROADCASTLAYER_H_

#include <array>
#include <cassert>
#include <utility>

#include "core/Layer.hpp"

namespace cattle {

/**
 * A class template representing a broadcasting layer that repeats the contents of its input tensors
 * along its ranks.
 */
template<typename Scalar, std::size_t Rank>
class BroadcastLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Base::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param input_dims The nominal input dimensions of the layer.
	 * @param broadcast The number of times the input tensor's contents are
	 * repeated along each rank. All elements should be greater than 0.
	 */
	inline BroadcastLayer(const typename Base::Dims& input_dims, const typename Base::Dims& broadcast) :
				input_dims(input_dims),
				output_dims(input_dims * broadcast),
				input_layer(false),
				broadcast(broadcast.template promote<>()) {
		slice_offsets.fill(0);
		for (std::size_t i = 0; i < Rank; ++i)
			assert(broadcast(i) > 0);
	}
	inline Base* clone() const {
		return new BroadcastLayer(*this);
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
		rows = in.dimension(0);
		return in.broadcast(broadcast);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == output_dims);
		assert(out_grad.dimension(0) > 0 && rows == out_grad.dimension(0));
		if (input_layer)
			return typename Base::Data();
		typename Base::Data prev_out_grad = std::move(out_grad);
		slice_offsets.fill(0);
		slice_extents = output_dims.template promote<>();
		slice_extents[0] = rows;
		for (std::size_t i = 0; i < Rank; ++i) {
			if (broadcast[i + 1] <= 1)
				continue;
			slice_extents[i + 1] = input_dims(i);
			typename Base::Data work_tensor(slice_extents);
			work_tensor.setZero();
			for (std::size_t j = 0; j < broadcast[i + 1]; ++j) {
				work_tensor += prev_out_grad.slice(slice_offsets, slice_extents);
				slice_offsets[i + 1] += input_dims(i);
			}
			slice_offsets[i + 1] = 0;
			prev_out_grad = std::move(work_tensor);
		}
		return prev_out_grad;
	}
private:
	const typename Base::Dims input_dims, output_dims;
	RankwiseArray broadcast, slice_offsets, slice_extents;
	std::size_t rows;
	bool input_layer;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_BROADCASTLAYER_H_ */
