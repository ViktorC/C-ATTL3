/*
 * SoftmaxActivationLayer.hpp
 *
 *  Created on: 23 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATION_SOFTMAXACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATION_SOFTMAXACTIVATIONLAYER_H_

#include <array>
#include <cassert>
#include <utility>

#include "core/NumericUtils.hpp"
#include "layer/ActivationLayer.hpp"

namespace cattle {

/**
 * A class template for a softmax activation function layer. Unlike most other activation
 * functions, the softmax layer does not represent a simple coefficient-wise function but
 * a multivariate one. The per-sample sums of the elements of the output tensor of the layer
 * are always 1.
 *
 * \f$f(x_i) = \frac{e^{x_i}}{\epsilon + \sum\limits_{j = 1}^J e^{x_j}}\f$
 */
template<typename Scalar, std::size_t Rank>
class SoftmaxActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param epsilon A small constant to maintain numerical stability.
	 */
	inline SoftmaxActivationLayer(const typename Root::Dims& dims,
			Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				Base::ActivationLayer(dims),
				epsilon(epsilon),
				conversion_dims(dims.template promote<>()) { }
	inline Root* clone() const {
		return new SoftmaxActivationLayer(*this);
	}
	inline void empty_cache() {
		out_mat_cache = Matrix<Scalar>();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		conversion_dims[0] = rows;
		MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
		/* First subtract the value of the greatest coefficient from each element row-wise
		 * to avoid an overflow due to raising e to great powers. */
		Matrix<Scalar> act_in_mat = (in_mat.array().colwise() - in_mat.array().rowwise().maxCoeff()).exp();
		act_in_mat = act_in_mat.array().colwise() / (act_in_mat.array().rowwise().sum() + epsilon);
		out_mat_cache = std::move(act_in_mat);
		return TensorMap<Scalar,Root::DATA_RANK>(out_mat_cache.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && out_mat_cache.rows() == out_grad.dimension(0));
		if (Base::is_input_layer())
			return typename Root::Data();
		std::size_t rows = out_grad.dimension(0);
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), rows, out_grad.size() / rows);
		Matrix<Scalar> prev_out_grad_mat(rows, out_mat_cache.cols());
		for (int i = 0; i < rows; ++i) {
			RowVector<Scalar> row_i = out_mat_cache.row(i);
			// FIXME Do not evaluate the expressions into a temporary variable.
			Matrix<Scalar> jacobian = row_i.asDiagonal();
			jacobian -= row_i.transpose() * row_i;
			prev_out_grad_mat.row(i) = out_grad_mat.row(i) * jacobian;
		}
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad_mat.data(), conversion_dims);
	}
private:
	const Scalar epsilon;
	RankwiseArray conversion_dims;
	// Staged computation cache matrix.
	Matrix<Scalar> out_mat_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATION_SOFTMAXACTIVATIONLAYER_H_ */
