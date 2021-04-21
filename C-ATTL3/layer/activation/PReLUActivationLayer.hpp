/*
 * PReLUActivationLayer.hpp
 *
 *  Created on: 24 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_PRELUACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_PRELUACTIVATIONLAYER_H_

#include <array>
#include <cassert>
#include <utility>

#include "layer/ActivationLayer.hpp"
#include "parameter_initialization/ConstantParameterInitialization.hpp"
#include "parameters/StandardParameters.hpp"

namespace cattle {

/**
 * A class template representing a parametric rectified linear unit (PReLU)
 * activation function. PReLU layers are Leaky ReLU activation functions with
 * learnable alphas. PReLU activation functions are not differentiable.
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
template <typename Scalar, std::size_t Rank>
class PReLUActivationLayer : public ActivationLayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef ActivationLayer<Scalar, Rank> Base;
  typedef std::array<std::size_t, Root::DATA_RANK> RankwiseArray;

 public:
  /**
   * @param dims The dimensionality of the input tensor.
   * @param init_alpha The initial factor by which negative inputs are to be
   * scaled.
   * @param alpha_reg An optional regularization function to apply to the
   * parameters.
   * @param alpha_clip The maximum allowed absolute parameter value. If it is 0
   * or less, no value clipping is performed.
   * @param alpha_max_l1_norm The maximum allowed L1 alpha value norm. If it is
   * 0 or less, no L1 max norm constraint is enforced.
   * @param alpha_max_l2_norm The maximum allowed L2 alpha value norm. If it is
   * 0 or less, no L2 max norm constraint is enforced.
   * @param alpha_grad_clip The maximum allowed absolute parameter gradient. If
   * it is 0 or less, no gradient clipping is performed.
   * @param alpha_grad_max_l1_norm The maximum allowed L1 alpha gradient norm.
   * If it is 0 or less, no L1 gradient max norm constraint is enforced.
   * @param alpha_grad_max_l2_norm The maximum allowed L2 alpha gradient norm.
   * If it is 0 or less, no L2 gradient max norm constraint is enforced.
   */
  inline PReLUActivationLayer(const typename Root::Dims& dims, Scalar init_alpha = 1e-1,
                              ParamRegSharedPtr<Scalar> alpha_reg = nullptr, Scalar alpha_clip = 0,
                              Scalar alpha_max_l1_norm = 0, Scalar alpha_max_l2_norm = 0, Scalar alpha_grad_clip = 0,
                              Scalar alpha_grad_max_l1_norm = 0, Scalar alpha_grad_max_l2_norm = 0)
      : Base::ActivationLayer(
            dims, std::make_shared<StandardParameters<Scalar>>(
                      1, dims.get_volume(), true, std::make_shared<ConstantParameterInitialization<Scalar>>(init_alpha),
                      alpha_reg, alpha_clip, alpha_max_l1_norm, alpha_max_l2_norm, alpha_grad_clip,
                      alpha_grad_max_l1_norm, alpha_grad_max_l2_norm)),
        conversion_dims(dims.template promote<>()) {}
  inline PReLUActivationLayer(const PReLUActivationLayer<Scalar, Rank>& layer, bool share_params = false)
      : Base::ActivationLayer(layer, share_params),
        conversion_dims(layer.conversion_dims),
        in_mat_cache(layer.in_mat_cache) {}
  inline Root* clone() const { return new PReLUActivationLayer(*this); }
  inline Root* clone_with_shared_params() { return new PReLUActivationLayer(*this, true); }
  inline void empty_cache() { in_mat_cache = Matrix<Scalar>(); }
  inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
    assert((Dimensions<std::size_t, Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
    assert(in.dimension(0) > 0);
    conversion_dims[0] = in.dimension(0);
    in_mat_cache = MatrixMap<Scalar>(in.data(), conversion_dims[0], in.size() / conversion_dims[0]);
    Matrix<Scalar> out_mat = in_mat_cache.cwiseMax(in_mat_cache * Base::params->get_values().asDiagonal());
    return TensorMap<Scalar, Root::DATA_RANK>(out_mat.data(), conversion_dims);
  }
  inline typename Root::Data pass_back(typename Root::Data out_grad) {
    assert((Dimensions<std::size_t, Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
    assert(out_grad.dimension(0) > 0 && conversion_dims[0] == out_grad.dimension(0));
    MatrixMap<Scalar> out_grad_mat(out_grad.data(), conversion_dims[0], out_grad.size() / conversion_dims[0]);
    Matrix<Scalar> prev_out_grad_mat = Matrix<Scalar>(in_mat_cache.rows(), in_mat_cache.cols());
    const Matrix<Scalar>& alphas = Base::params->get_values();
    Matrix<Scalar> alphas_grad = Matrix<Scalar>::Zero(1, out_grad_mat.cols());
    for (int i = 0; i < in_mat_cache.cols(); ++i) {
      for (int j = 0; j < in_mat_cache.rows(); ++j) {
        Scalar in_mat_ji = in_mat_cache(j, i);
        Scalar out_mat_ji = out_grad_mat(j, i);
        if (in_mat_ji >= 0)
          prev_out_grad_mat(j, i) = out_mat_ji;
        else {
          prev_out_grad_mat(j, i) = alphas(0, i) * out_mat_ji;
          alphas_grad(0, i) += in_mat_ji * out_mat_ji;
        }
      }
    }
    Base::params->accumulate_grad(alphas_grad);
    return TensorMap<Scalar, Root::DATA_RANK>(prev_out_grad_mat.data(), conversion_dims);
  }

 private:
  RankwiseArray conversion_dims;
  // Staged computation cache.
  Matrix<Scalar> in_mat_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_PRELUACTIVATIONLAYER_H_ */
