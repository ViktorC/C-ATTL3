/*
 * HingeLoss.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LOSS_HINGELOSS_H_
#define C_ATTL3_LOSS_HINGELOSS_H_

#include <algorithm>
#include <cassert>

#include "core/NumericUtils.hpp"
#include "UniversalLoss.hpp"

namespace cattle {

/**
 * A template class representing the hinge loss function. This class assumes the objectives for
 * each sample (and time step) to be a one-hot vector (tensor rank).
 *
 * \f$L_i = \sum_{j \neq y_i} \max(0, \hat{y_i}_j - \hat{y_i}_{y_i} + 1)\f$ or
 * \f$L_i = \sum_{j \neq y_i} \max(0, \hat{y_i}_j - \hat{y_i}_{y_i} + 1)^2\f$
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool Squared = false>
class HingeLoss : public UniversalLoss<Scalar,Rank,Sequential> {
	typedef Loss<Scalar,Rank,Sequential> Root;
	typedef UniversalLoss<Scalar,Rank,Sequential> Base;
protected:
	inline ColVector<Scalar> _function(typename Root::Data out, typename Root::Data obj) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		ColVector<Scalar> loss(rows);
		for (int i = 0; i < rows; ++i) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < cols; ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				if (NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar loss_i = 0;
			Scalar correct_class_score = out_mat(i,correct_class_ind);
			for (int j = 0; j < cols; ++j) {
				if (j == correct_class_ind)
					continue;
				Scalar loss_ij = std::max((Scalar) 0, (Scalar) (out_mat(i,j) - correct_class_score + 1));
				loss_i += Squared ? loss_ij * loss_ij : loss_ij;
			}
			loss(i) = loss_i;
		}
		return loss;
	}
	inline typename Root::Data _d_function(typename Root::Data out, typename Root::Data obj,
			const typename Base::RankwiseArray& grad_dims) const {
		std::size_t rows = out.dimension(0);
		std::size_t cols = out.size() / rows;
		MatrixMap<Scalar> out_mat(out.data(), rows, cols);
		MatrixMap<Scalar> obj_mat(obj.data(), rows, cols);
		Matrix<Scalar> out_grad(rows, cols);
		for (int i = 0; i < rows; ++i) {
			unsigned ones = 0;
			int correct_class_ind = -1;
			for (int j = 0; j < cols; ++j) {
				Scalar obj_ij = obj_mat(i,j);
				assert((NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 0) ||
						NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)));
				if (NumericUtils<Scalar>::almost_equal(obj_ij, (Scalar) 1)) {
					ones++;
					correct_class_ind = j;
				}
			}
			assert(ones == 1);
			Scalar total_out_grad = 0;
			Scalar correct_class_score = out_mat(i,correct_class_ind);
			for (int j = 0; j < cols; ++j) {
				if (j == correct_class_ind)
					continue;
				Scalar out_ij = out_mat(i,j);
				Scalar margin = out_ij - correct_class_score + 1;
				if (NumericUtils<Scalar>::decidedly_greater(margin, (Scalar) 0)) {
					Scalar out_grad_ij = Squared ? 2 * margin : 1;
					total_out_grad += out_grad_ij;
					out_grad(i,j) = out_grad_ij;
				} else
					out_grad(i,j) = 0;
			}
			out_grad(i,correct_class_ind) = -total_out_grad;
		}
		return TensorMap<Scalar,Root::DATA_RANK>(out_grad.data(), grad_dims);
	}
};

} /* namespace cattle */

#endif /* C_ATTL3_LOSS_HINGELOSS_H_ */
