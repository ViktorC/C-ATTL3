/*
 * ReLUActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_PRELUACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_PRELUACTIVATIONLAYER_H_

#include <array>
#include <cassert>
#include <utility>

#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template representing a parametric rectified linear unit (PReLU) activation function.
 * PReLU layers are Leaky ReLU activation functions with learnable alphas. PReLU activation
 * functions are not differentiable.
 *
 * \f[
 *   f(x) = \begin{cases}
 *     \alpha x & \text{for } x < 0\\
 *     x & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 *
 * \see https://arxiv.org/abs/1502.01852
 */
template<typename Scalar, std::size_t Rank>
class PReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param param_reg The regularization function to apply to the layer's parameters.
	 * @param init_alpha The initial factor by which negative inputs are to be scaled.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline PReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar init_alpha = 1e-1,
			ParamRegSharedPtr<Scalar> param_reg = Root::NO_PARAM_REG,
			Scalar init_alpha = 1e-1, Scalar max_norm_constraint = 0) :
				Base::ActivationLayer(dims, 1, dims.get_volume()),
				conversion_dims(dims.template promote<>()) { }
	inline PReLUActivationLayer(PReLUActivationLayer<Scalar,Rank>& layer, bool share_params) :
			Base::ActivationLayer(layer, share_params),
			param_reg(layer.param_reg),
			init_alpha(layer.init_alpha),
			max_norm_constraint(layer.max_norm_constraint),
			max_norm(layer.max_norm),
			conversion_dims(layer.conversion_dims) { }
	inline Root* clone() const {
		return new PReLUActivationLayer(*this);
	}
	inline Root* clone_with_shared_params() {
		return new PReLUActivationLayer(*this, true);
	}
	inline std::vector<const Parameters<Scalar>*>& get_params() const {
		return std::vector<const Parameters<Scalar>*>({ weights.get(), bias.get() });
	}
	inline std::vector<Parameters<Scalar>*>& get_params() {
		return std::vector<Parameters<Scalar>*>({ weights.get(), bias.get() });
	}
	inline void empty_cache() {
		in_cache = Matrix<Scalar>();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		in_cache = MatrixMap<Scalar>(in.data(), rows, Base::dims.get_volume());
		Matrix<Scalar> out = in_cache.cwiseMax(in_cache * Base::params_ref.row(0).asDiagonal());
		conversion_dims[0] = rows;
		return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && conversion_dims[0] == out_grad.dimension(0));
		Base::params_grad.row(0).setZero();
		MatrixMap<Scalar> out_grad_map(out_grad.data(), conversion_dims[0], Base::dims.get_volume());
		Matrix<Scalar> prev_out_grad = Matrix<Scalar>(in_cache.rows(), in_cache.cols());
		Matrix<Scalar> grad = Matrix<Scalar>::Zero()
		for (int i = 0; i < in_cache.cols(); ++i) {
			for (int j = 0; j < in_cache.rows(); ++j) {
				Scalar in_ji = in_cache(j,i);
				if (in_ji >= 0)
					prev_out_grad(j,i) = out_grad_map(j,i);
				else {
					Scalar out_ji = out_grad_map(j,i);
					prev_out_grad(j,i) = Base::params_ref(0,i) * out_ji;
					Base::params_grad(0,i) += in_ji * out_ji;
				}
			}
		}
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad.data(), conversion_dims);
	}
private:
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_PRELUACTIVATIONLAYER_H_ */
