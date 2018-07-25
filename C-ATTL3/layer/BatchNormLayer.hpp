/*
 * BatchNormLayer.hpp
 *
 *  Created on: 24 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_BATCHNORMLAYER_H_
#define C_ATTL3_LAYER_BATCHNORMLAYER_H_

#include <cassert>
#include <utility>

#include "core/Layer.hpp"
#include "core/NumericUtils.hpp"
#include "parameter_initialization/OneParameterInitialization.hpp"
#include "parameter_initialization/ZeroParameterInitialization.hpp"
#include "parameters/HostParameters.hpp"

namespace cattle {

/**
 * A class template for a per-channel batch normalization layer.
 *
 * \see https://arxiv.org/abs/1502.03167
 */
template<typename Scalar, std::size_t Rank, bool PerLastRank = (Rank == 3)>
class BatchNormLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef BatchNormLayer<Scalar,Rank,PerLastRank> Self;
	typedef std::array<std::size_t,Base::DATA_RANK> RankwiseArray;
	typedef std::shared_ptr<HostParameters<Scalar>> HostParamsSharedPtr;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param gamma_reg The regularization function to apply to the layer's gamma parameters.
	 * @param beta_reg The regularization function to apply to the layer's beta parameters.
	 * @param gamma_max_norm_constraint An optional max-norm constraint to enforce on the
	 * gamma parameters. If it is 0 or less, no constraint is applied.
	 * @param beta_max_norm_constraint An optional max-norm constraint to enforce on the
	 * beta parameters. If it is 0 or less, no constraint is applied.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(const Dimensions<std::size_t,Rank>& dims, Scalar norm_avg_decay = .1,
			Scalar epsilon = NumericUtils<Scalar>::EPSILON2, ParamRegSharedPtr<Scalar> gamma_reg = nullptr,
			Scalar gamma_clip = 0, Scalar gamma_max_l1_norm = 0, Scalar gamma_max_l2_norm = 0,
			Scalar gamma_grad_clip = 0, Scalar gamma_grad_max_l1_norm = 0, Scalar gamma_grad_max_l2_norm = 0,
			ParamRegSharedPtr<Scalar> beta_reg = nullptr, Scalar beta_clip = 0, Scalar beta_max_l1_norm = 0,
			Scalar beta_max_l2_norm = 0, Scalar beta_grad_clip = 0, Scalar beta_grad_max_l1_norm = 0,
			Scalar beta_grad_max_l2_norm = 0) :
				owner(*this),
				dims(dims),
				norm_avg_decay(norm_avg_decay),
				epsilon(epsilon),
				channels(dims(Rank - 1)),
				input_layer(false),
				offsets(),
				extents(dims.template promote<>()),
				memb_vec(channels) {
		assert(gamma_reg != nullptr);
		assert(beta_reg != nullptr);
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
		offsets.fill(0);
		extents[Rank] = 1;
		auto gamma_init = std::make_shared<OneParameterInitialization<Scalar>>();
		auto beta_init = std::make_shared<ZeroParameterInitialization<Scalar>>();
		for (std::size_t i; i < channels; ++i) {
			ChannelSpecificMembers& memb = memb_vec[i];
			memb.avg_means;
			memb.avg_inv_sds;
			memb.gammas;
			memb.betas;
		}
	}
	inline BatchNormLayer(const Self& layer, bool share_params = false) :
			dims(layer.dims),
			gamma_reg(layer.gamma_reg),
			beta_reg(layer.beta_reg),
			gamma_max_norm_constraint(layer.gamma_max_norm_constraint),
			beta_max_norm_constraint(layer.beta_max_norm_constraint),
			gamma_max_norm(layer.gamma_max_norm),
			beta_max_norm(layer.beta_max_norm),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			channels(layer.channels),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			offsets(layer.offsets),
			extents(layer.extents),
			avg_means(layer.avg_means),
			avg_inv_sds(layer.avg_inv_sds),
			avgs_init(layer.avgs_init),
			params(share_params ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_params ? layer.params_ref : params),
			owner(share_params ? layer.owner : *this),
			cache_vec(layer.cache_vec) { }
	inline Base* clone() const {
		return new BatchNormLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return new BatchNormLayer(*this, true);
	}
	inline const Base& get_params_owner() const {
		return owner;
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
		std::vector<const Parameters<Scalar>*> params_vec(channels * 4);
		populate_params_vector<std::vector<const Parameters<Scalar>*>>(params_vec);
		return params_vec;
	}
	inline std::vector<Parameters<Scalar>*>& get_params() {
		std::vector<Parameters<Scalar>*> params_vec(channels * 4);
		populate_params_vector<std::vector<const Parameters<Scalar>*>>(params_vec);
		return params_vec;
	}
	inline void empty_cache() {
		for (unsigned i = 0; i < memb_vec.size(); ++i)
			memb_vec[i].std_in_mat_cache = Matrix<Scalar>();
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		extents[0] = rows;
		typename Base::Data out;
		if (channels == 1) {
			MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
			Matrix<Scalar> out_mat = _pass_forward(in_mat, 0, training);
			out = TensorMap<Scalar,Base::DATA_RANK>(out_mat.data(), extents);
		} else {
			out = typename Base::Data(in.dimensions());
			for (std::size_t i = 0; i < channels; ++i) {
				offsets[Rank] = i;
				typename Base::Data in_slice = in.slice(offsets, extents);
				MatrixMap<Scalar> in_slice_mat(in_slice.data(), rows, in_slice.size() / rows);
				Matrix<Scalar> out_slice_mat = _pass_forward(in_slice_mat, i, training);
				out.slice(offsets, extents) = TensorMap<Scalar,Base::DATA_RANK>(out_slice_mat.data(), extents);
			}
		}
		return out;
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == dims);
		assert(out_grad.dimension(0) > 0 && extents[0] == out_grad.dimension(0));
		std::size_t rows = out_grad.dimension(0);
		typename Base::Data prev_out_grad;
		if (channels == 1) {
			MatrixMap<Scalar> out_grad_mat(out_grad.data(), rows, out_grad.size() / rows);
			if (input_layer) {
				_pass_back(out_grad_mat, 0);
				return typename Base::Data();
			} else {
				Matrix<Scalar> prev_out_grad_mat = _pass_back(out_grad_mat, 0);
				prev_out_grad = TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad_mat.data(), extents);
			}
		} else {
			prev_out_grad = input_layer ? typename Base::Data() : typename Base::Data(out_grad.dimensions());
			for (std::size_t i = 0; i < channels; ++i) {
				offsets[Rank] = i;
				typename Base::Data out_grad_slice = out_grad.slice(offsets, extents);
				MatrixMap<Scalar> out_grad_slice_mat(out_grad_slice.data(), rows, out_grad_slice.size() / rows);
				if (input_layer) {
					_pass_back(out_grad_slice_mat, i);
					continue;
				} else {
					Matrix<Scalar> prev_out_grad_slice_mat = _pass_back(out_grad_slice_mat, i);
					prev_out_grad.slice(offsets, extents) =
							TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad_slice_mat.data(), extents);
				}
			}
		}
		return prev_out_grad;
	}
private:
	inline Matrix<Scalar> _pass_forward(MatrixMap<Scalar>& in, std::size_t i, bool training) {
		Matrix<Scalar> out;
		ChannelSpecificMembers& memb = memb_vec[i];
		if (training) {
			Matrix<Scalar> mean(1, 1, in.mean());
			Matrix<Scalar> norm_in = in.array() - mean(0,0);
			memb.inv_in_sd_cache = 1 / sqrt(norm_in.array().square().mean() + epsilon);
			memb.std_in_mat_cache = norm_in * memb.inv_in_sd_cache;
			out = memb.std_in_mat_cache;
			if (memb.avgs_init) {
				memb.avg_means->update_values(norm_avg_decay * memb.avg_means->get_values() + norm_avg_decay * mean);
				memb.avg_inv_sds->update_values(norm_avg_decay * memb.avg_inv_sds->get_values().array() +
						norm_avg_decay * memb.inv_in_sd_cache);
			} else {
				memb.avg_means->update_values(mean);
				memb.avg_inv_sds->update_values(memb.inv_in_sd_cache);
				memb.avgs_init = true;
			}
		} else {
			assert(memb.avgs_init);
			out = (in.array() - memb.avg_means->get_values()) * memb.avg_inv_sds->get_values();
		}
		return (out * memb.gammas(0,0)).array() + memb.betas(0,0);
	}
	inline Matrix<Scalar> _pass_back(MatrixMap<Scalar>& out_grad, std::size_t i) {
		ChannelSpecificMembers& memb = memb_vec[i];
		memb.gammas->update_grad(out_grad.cwiseProduct(memb.std_in_mat_cache).sum());
		memb.betas->update_grad(Matrix<Scalar>(1, 1, out_grad.sum()));
		if (input_layer)
			return Matrix<Scalar>();
		std::size_t locations = out_grad.size();
		Matrix<Scalar> std_in_grad_mat = out_grad * memb.gammas->get_values()(0,0);
		return (((locations * std_in_grad_mat).array() - std_in_grad_mat.sum()).matrix() -
				memb.std_in_mat_cache * memb.std_in_mat_cache.cwiseProduct(std_in_grad_mat).sum()) *
				(((Scalar) 1 / locations) * memb.inv_in_sd_cache);
	}
	template<typename ParamsVec>
	inline void populate_params_vector(ParamsVec params_vec) {
		for (std::size_t i; i < channels; ++i) {
			ChannelSpecificMembers& memb = memb_vec[i];
			params_vec.push_back(memb.avg_means);
			params_vec.push_back(memb.avg_inv_sds);
			params_vec.push_back(memb.gammas);
			params_vec.push_back(memb.betas);
		}
	}
	const Self& owner;
	const typename Base::Dims dims;
	const Scalar norm_avg_decay, epsilon;
	const std::size_t channels;
	bool input_layer;
	RankwiseArray offsets, extents;
	struct ChannelSpecificMembers {
		// Dynamic batch normalization parameters.
		HostParamsSharedPtr avg_means, avg_inv_sds;
		// The optimizable parameters.
		HostParamsSharedPtr gammas, betas;
		bool avgs_init;
		// Staged computation cache.
		Scalar inv_in_sd_cache;
		Matrix<Scalar> std_in_mat_cache;
	};
	std::vector<ChannelSpecificMembers> memb_vec;
};

/**
 * A class template for a per-activation batch normalization layer.
 *
 * \see https://arxiv.org/abs/1502.03167
 */
template<typename Scalar, std::size_t Rank>
class BatchNormLayer<Scalar,Rank,false> : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef BatchNormLayer<Scalar,Rank,false> Self;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param gamma_reg The regularization function to apply to the layer's gamma parameters.
	 * @param beta_reg The regularization function to apply to the layer's beta parameters.
	 * @param gamma_max_norm_constraint An optional max-norm constraint to enforce on the
	 * gamma parameters. If it is 0 or less, no constraint is applied.
	 * @param beta_max_norm_constraint An optional max-norm constraint to enforce on the
	 * beta parameters. If it is 0 or less, no constraint is applied.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(const Dimensions<std::size_t,Rank>& dims, ParamRegSharedPtr<Scalar> gamma_reg = Base::NO_PARAM_REG,
			ParamRegSharedPtr<Scalar> beta_reg = Base::NO_PARAM_REG, Scalar gamma_max_norm_constraint = 0,
			Scalar beta_max_norm_constraint = 0, Scalar norm_avg_decay = .1, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				dims(dims),
				gamma_reg(gamma_reg),
				beta_reg(beta_reg),
				gamma_max_norm_constraint(gamma_max_norm_constraint),
				beta_max_norm_constraint(beta_max_norm_constraint),
				gamma_max_norm(NumericUtils<Scalar>::decidedly_greater(gamma_max_norm_constraint, (Scalar) 0)),
				beta_max_norm(NumericUtils<Scalar>::decidedly_greater(beta_max_norm_constraint, (Scalar) 0)),
				norm_avg_decay(norm_avg_decay),
				epsilon(epsilon),
				input_layer(false),
				frozen(false),
				avg_means(),
				avg_inv_sds(),
				avgs_init(false),
				params(),
				params_grad(),
				params_ref(params),
				owner(*this) {
		assert(gamma_reg != nullptr);
		assert(beta_reg != nullptr);
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline BatchNormLayer(const Self& layer) :
			dims(layer.dims),
			gamma_reg(layer.gamma_reg),
			beta_reg(layer.beta_reg),
			gamma_max_norm_constraint(layer.gamma_max_norm_constraint),
			beta_max_norm_constraint(layer.beta_max_norm_constraint),
			gamma_max_norm(layer.gamma_max_norm),
			beta_max_norm(layer.beta_max_norm),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			avg_means(layer.avg_means),
			avg_inv_sds(layer.avg_inv_sds),
			avgs_init(layer.avgs_init),
			params(layer.params),
			params_grad(layer.params_grad),
			params_ref(layer.is_shared_params_clone() ? layer.params_ref : params),
			owner(layer.is_shared_params_clone() ? layer.owner : *this),
			inv_in_sd(layer.inv_in_sd),
			std_in(layer.std_in) { }
	inline Base* clone() const {
		return new BatchNormLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return new BatchNormLayer(*this, true);
	}
	inline const Base& get_params_owner() const {
		return owner;
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return dims;
	}
	inline const Matrix<Scalar>& get_params() const {
		return params_ref;
	}
	inline const Matrix<Scalar>& get_params_grad() const {
		return params_grad;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() {
		params_ref = Matrix<Scalar>(dims.get_volume(), 2);
		// Gamma.
		params_ref.col(0).setOnes();
		// Beta.
		params_ref.col(1).setZero();
		params_grad = Matrix<Scalar>::Zero(params_ref.rows(), params_ref.cols());
		avg_means = Matrix<Scalar>(1, params_ref.rows());
		avg_inv_sds = Matrix<Scalar>(1, params_ref.rows());
		avgs_init = false;
	}
protected:
	inline BatchNormLayer(Self& layer, bool share_params) :
			dims(layer.dims),
			gamma_reg(layer.gamma_reg),
			beta_reg(layer.beta_reg),
			gamma_max_norm_constraint(layer.gamma_max_norm_constraint),
			beta_max_norm_constraint(layer.beta_max_norm_constraint),
			gamma_max_norm(layer.gamma_max_norm),
			beta_max_norm(layer.beta_max_norm),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			avg_means(layer.avg_means),
			avg_inv_sds(layer.avg_inv_sds),
			avgs_init(layer.avgs_init),
			params(share_params ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_params ? layer.params : params),
			owner(share_params ? layer.owner : *this),
			inv_in_sd(layer.inv_in_sd),
			std_in(layer.std_in) { }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void empty_cache() {
		inv_in_sd = RowVector<Scalar>(0);
		std_in = Matrix<Scalar>(0, 0);
	}
	inline Matrix<Scalar>& get_params() {
		return params_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() {
		params_grad.col(0) += gamma_reg->d_function(params_ref.col(0));
		params_grad.col(1) += beta_reg->d_function(params_ref.col(1));
	}
	inline Scalar get_regularization_penalty() const {
		return gamma_reg->function(params_ref.col(0)) + beta_reg->function(params_ref.col(1));
	}
	inline void enforce_constraints() {
		Scalar l2_norm;
		if (gamma_max_norm) {
			l2_norm = params_ref.col(0).squaredNorm();
			if (l2_norm > gamma_max_norm_constraint)
				params_ref.col(0) *= (gamma_max_norm_constraint / l2_norm);
		}
		if (beta_max_norm) {
			l2_norm = params_ref.col(1).squaredNorm();
			if (l2_norm > beta_max_norm_constraint)
				params_ref.col(1) *= (beta_max_norm_constraint / l2_norm);
		}
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
		if (training) {
			RowVector<Scalar> means = in_mat.colwise().mean();
			Matrix<Scalar> norm_in = in_mat.rowwise() - means;
			inv_in_sd = (norm_in.array().square().colwise().mean() + epsilon).sqrt().inverse();
			std_in = norm_in * inv_in_sd.asDiagonal();
			in_mat = std_in;
			// Maintain a moving average of means and variances for testing.
			if (avgs_init) {
				avg_means = (1.0 - norm_avg_decay) * avg_means + norm_avg_decay * means;
				avg_inv_sds = (1.0 - norm_avg_decay) * avg_inv_sds + norm_avg_decay * inv_in_sd;
			} else {
				avg_means = means;
				avg_inv_sds = inv_in_sd;
				avgs_init = true;
			}
		} else {
			// For testing, use the moving averages.
			assert(avgs_init);
			in_mat = (in_mat.rowwise() - avg_means) * avg_inv_sds.asDiagonal();
		}
		Matrix<Scalar> out = (in_mat * params_ref.col(0).asDiagonal()).rowwise() + params_ref.col(1).transpose();
		return TensorMap<Scalar,Base::DATA_RANK>(out.data(), in.dimensions());
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == dims);
		assert(out_grad.dimension(0) > 0 && std_in.rows() == out_grad.dimension(0));
		std::size_t rows = out_grad.dimension(0);
		/* Back-propagate the gradient through the batch normalization function and also calculate the
		 * gradients of the betas and gammas. */
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), rows, out_grad.size() / rows);
		params_grad.col(0) = out_grad_mat.cwiseProduct(std_in).colwise().sum().transpose();
		params_grad.col(1) = out_grad_mat.colwise().sum().transpose();
		if (input_layer)
			return typename Base::Data();
		Matrix<Scalar> std_in_grad = out_grad_mat * params_ref.col(0).asDiagonal();
		Matrix<Scalar> prev_out_grad = (((rows * std_in_grad).rowwise() - std_in_grad.colwise().sum()) -
				std_in * (std_in.cwiseProduct(std_in_grad).colwise().sum().asDiagonal())) *
				(((Scalar) 1 / rows) * inv_in_sd).asDiagonal();
		return TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad.data(), out_grad.dimensions());
	}
private:
	const Dimensions<std::size_t,Rank> dims;
	const ParamRegSharedPtr<Scalar> gamma_reg;
	const ParamRegSharedPtr<Scalar> beta_reg;
	const Scalar gamma_max_norm_constraint;
	const Scalar beta_max_norm_constraint;
	const bool gamma_max_norm;
	const bool beta_max_norm;
	const Scalar norm_avg_decay;
	const Scalar epsilon;
	bool input_layer;
	bool frozen;
	// Dynamic batch normalization parameters.
	RowVector<Scalar> avg_means;
	RowVector<Scalar> avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	Matrix<Scalar> params;
	Matrix<Scalar>& params_ref;
	Matrix<Scalar> params_grad;
	const Base& owner;
	// Staged computation caches.
	RowVector<Scalar> inv_in_sd;
	Matrix<Scalar> std_in;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_BATCHNORMLAYER_H_ */
