/*
 * DropoutLayer.hpp
 *
 *  Created on: 24 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_DROPOUTLAYER_H_
#define C_ATTL3_LAYER_DROPOUTLAYER_H_

#include <cassert>
#include <utility>

#include "core/Layer.hpp"
#include "core/NumericUtils.hpp"

namespace cattle {

/**
 * A class template representing a drop-out layer.
 *
 * \see https://arxiv.org/abs/1207.0580
 * \see http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
 */
template<typename Scalar, std::size_t Rank>
class DropoutLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param dropout_prob The probability of an element of the input tensor being set to 0.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline DropoutLayer(const typename Base::Dims& dims, Scalar dropout_prob,
			Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				dims(dims),
				dropout_prob(dropout_prob),
				epsilon(epsilon),
				input_layer(false) {
		assert(dropout_prob > 0 && dropout_prob <= 1 &&
				"dropout probability must be greater than 0 and no greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline Base* clone() const {
		return new DropoutLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return clone();
	}
	inline const Base& get_params_owner() const {
		return *this;
	}
	inline const typename Base::Dims& get_input_dims() const {
		return dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
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
	inline void empty_cache() {
		dropout_mask = typename Base::Data();
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		if (training) {
			// Inverted dropout.
			Scalar scaling_factor = (Scalar) 1 / (1 - dropout_prob + epsilon);
			dropout_mask = in.random().unaryExpr([this,scaling_factor](Scalar e) {
				return (Scalar) (e <= dropout_prob ? 0 : scaling_factor);
			});
			return in * dropout_mask;
		}
		return std::move(in);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == dims);
		assert(out_grad.dimension(0) > 0 && dropout_mask.dimension(0) == out_grad.dimension(0));
		if (input_layer)
			return typename Base::Data();
		// The derivative of the dropout function.
		return out_grad * dropout_mask;
	}
private:
	const typename Base::Dims dims;
	const Scalar dropout_prob, epsilon;
	bool input_layer;
	// Staged computation cache.
	typename Base::Data dropout_mask;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_DROPOUTLAYER_H_ */
