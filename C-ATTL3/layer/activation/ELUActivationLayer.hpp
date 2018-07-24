/*
 * ELUActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_ELUACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_ELUACTIVATIONLAYER_H_

#include <array>
#include <cassert>
#include <cmath>
#include <utility>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template representing an exponential linear unit (ELU) activation function. ELUs
 * apply an exponential (e based) function scaled by alpha to the negative elements of the input.
 * ELU layers are not differentiable.
 *
 * \f[
 *   f(x) = \begin{cases}
 *     \alpha (e^x - 1) & \text{for } x < 0\\
 *     x & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 *
 * \see https://arxiv.org/abs/1511.07289
 */
template<typename Scalar, std::size_t Rank>
class ELUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param alpha The factor by which negative inputs are to be scaled.
	 */
	inline ELUActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar alpha = 1e-1) :
			Base::ActivationLayer(dims),
			alpha(alpha),
			conversion_dims(dims.template promote<>()) { }
	inline Root* clone() const {
		return new ELUActivationLayer(*this);
	}
	inline void empty_cache() {
		in_cache = Matrix<Scalar>();
		out_cache = Matrix<Scalar>();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		in_cache = MatrixMap<Scalar>(in.data(), in.dimension(0), Base::dims.get_volume());
		out_cache = in_cache.unaryExpr([alpha](Scalar e) {
			return (Scalar) (e >= 0 ? i : (alpha * (exp(e) - 1)));
		});
		conversion_dims[0] = out.rows();
		return TensorMap<Scalar,Root::DATA_RANK>(out_cache.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && conversion_dims[0] == out_grad.dimension(0));
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), conversion_dims[0], Base::dims.get_volume());
		Matrix<Scalar> prev_out_grad(in_cache.rows(), in_cache.cols());
		for (int i = 0; i < in_cache.cols(); ++i) {
			for (int j = 0; j < in_cache.rows(); ++j)
				prev_out_grad(j,i) = (Scalar) ((in_cache(j,i) >= 0 ?
						1 : (out_cache(j,i) + alpha)) * out_grad_mat(j,i));
		}
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad.data(), conversion_dims);
	}
private:
	const Scalar alpha;
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in_cache;
	Matrix<Scalar> out_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_ELUACTIVATIONLAYER_H_ */
