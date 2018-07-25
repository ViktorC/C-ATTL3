/*
 * BatchNormLayer.hpp
 *
 *  Created on: 24 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_BATCHNORMLAYER_H_
#define C_ATTL3_LAYER_BATCHNORMLAYER_H_

#include <array>
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
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 * @param gamma_reg An optional regularization function to apply to the gammas.
	 * @param gamma_clip The maximum allowed absolute gamma value. If it is 0 or less, no less, no
	 * L1 max norm constraint is enforced.
	 * @param gamma_max_l2_norm The maximum allowed L2 gamma value norm. If it is 0 or less, no L2
	 * max norm constraint is enforced.
	 * @param gamma_grad_clip The maximum allowed absolute gamma gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param gamma_grad_max_l1_norm The maximum allowed L1 gamma gradient norm. If it is 0 or less,
	 * no L1 gradient max norm constraint is enforced.
	 * @param gamma_grad_max_l2_norm The maximum allowed L2 gamma gradient norm. If it is 0 or less,
	 * no L2 gradient max norm constraint is enforced.
	 * @param beta_reg An optional regularization function to apply to the beta.
	 * @param beta_clip The maximum allowed absolute beta value. If it is 0 or less, no value clipping
	 * is performed.
	 * @param beta_max_l1_norm The maximum allowed L1 beta value norm. If it is 0 or less, no beta L1
	 * max norm constraint is enforced.
	 * @param beta_max_l2_norm The maximum allowed L2 beta value norm. If it is 0 or less, no beta L2
	 * max norm constraint is enforced.
	 * @param beta_grad_clip The maximum allowed absolute beta gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param beta_grad_max_l1_norm The maximum allowed L1 beta gradient norm. If it is 0 or less, no
	 * beta L1 gradient max norm constraint is enforced.
	 * @param beta_grad_max_l2_norm The maximum allowed L2 beta gradient norm. If it is 0 or less, no
	 * beta L2 gradient max norm constraint is enforced.
	 */
	inline BatchNormLayer(const typename Base::Dims& dims, Scalar norm_avg_decay = .1,
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
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
		offsets.fill(0);
		extents[Rank] = 1;
		auto gamma_init = std::make_shared<OneParameterInitialization<Scalar>>();
		auto beta_init = std::make_shared<ZeroParameterInitialization<Scalar>>();
		for (std::size_t i; i < channels; ++i) {
			ChannelSpecificMembers& memb = memb_vec[i];
			memb.avg_means = std::make_shared<HostParameters<Scalar>>(1, 1, false);
			memb.avg_inv_sds = std::make_shared<HostParameters<Scalar>>(1, 1, false);
			memb.gammas = std::make_shared<HostParameters<Scalar>>(1, 1, true, gamma_init, gamma_reg,
					gamma_clip, gamma_max_l1_norm, gamma_max_l2_norm, gamma_grad_clip, gamma_grad_max_l1_norm,
					gamma_grad_max_l2_norm);
			memb.betas = std::make_shared<HostParameters<Scalar>>(1, 1, true, beta_init, beta_reg, beta_clip,
					beta_max_l1_norm, beta_max_l2_norm, beta_grad_clip, beta_grad_max_l1_norm,
					beta_grad_max_l2_norm);
			memb.avgs_init = false;
		}
	}
	inline BatchNormLayer(const Self& layer, bool share_params = false) :
			owner(share_params || layer.is_shared_params_clone() ? layer.owner : *this),
			dims(layer.dims),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			channels(layer.channels),
			input_layer(layer.input_layer),
			offsets(layer.offsets),
			extents(layer.extents),
			memb_vec(channels) {
		for (std::size_t i; i < channels; ++i) {
			ChannelSpecificMembers& memb1 = memb_vec[i];
			ChannelSpecificMembers& memb2 = layer.memb_vec[i];
			if (share_params) {
				memb1.avg_means = memb2.avg_means;
				memb1.avg_inv_sds = memb2.avg_inv_sds;
				memb1.gammas = memb2.gammas;
				memb1.betas = memb2.betas;
			} else {
				memb1.avg_means = HostParamsSharedPtr(memb2.avg_means.clone());
				memb1.avg_inv_sds = HostParamsSharedPtr(memb2.avg_inv_sds.clone());
				memb1.gammas = HostParamsSharedPtr(memb2.gammas.clone());
				memb1.betas = HostParamsSharedPtr(memb2.betas.clone());
			}
			memb1.avgs_init = memb2.avgs_init;
			memb1.inv_in_sd_cache = memb2.inv_in_sd_cache;
			memb1.std_in_mat_cache = memb2.std_in_mat_cache;
		}
	}
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
		populate_params_vector<const Parameters<Scalar>*>(params_vec);
		return params_vec;
	}
	inline std::vector<Parameters<Scalar>*>& get_params() {
		std::vector<Parameters<Scalar>*> params_vec(channels * 4);
		populate_params_vector<Parameters<Scalar>*>(params_vec);
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
		Matrix<Scalar> out_mat;
		ChannelSpecificMembers& memb = memb_vec[i];
		if (training) {
			Matrix<Scalar> mean_mat(1, 1, in.mean());
			Matrix<Scalar> norm_in_mat = in.array() - mean_mat(0,0);
			memb.inv_in_sd_cache = 1 / sqrt(norm_in_mat.array().square().mean() + epsilon);
			memb.std_in_mat_cache = norm_in_mat * memb.inv_in_sd_cache;
			out_mat = memb.std_in_mat_cache;
			if (memb.avgs_init) {
				memb.avg_means->set_values((1 - norm_avg_decay) * memb.avg_means->get_values() +
						norm_avg_decay * mean_mat);
				memb.avg_inv_sds->set_values((1 - norm_avg_decay) * memb.avg_inv_sds->get_values().array() +
						norm_avg_decay * memb.inv_in_sd_cache);
			} else {
				memb.avg_means->set_values(mean_mat);
				memb.avg_inv_sds->set_values(memb.inv_in_sd_cache);
				memb.avgs_init = true;
			}
		} else {
			assert(memb.avgs_init);
			out_mat = (in.array() - memb.avg_means->get_values()(0,0)) * memb.avg_inv_sds->get_values()(0,0);
		}
		return (out_mat * memb.gammas(0,0)).array() + memb.betas(0,0);
	}
	inline Matrix<Scalar> _pass_back(MatrixMap<Scalar>& out_grad, std::size_t i) {
		ChannelSpecificMembers& memb = memb_vec[i];
		auto gammas_grad = out_grad.cwiseProduct(memb.std_in_mat_cache).sum();
		auto betas_grad = Matrix<Scalar>(1, 1, out_grad.sum());
		if (Base::is_shared_params_clone()) {
			memb.gammas->set_grad(memb.gammas->get_grad() + gammas_grad);
			memb.betas->set_grad(memb.betas->get_grad() + betas_grad);
		} else {
			memb.gammas->set_grad(gammas_grad);
			memb.betas->set_grad(betas_grad);
		}
		if (input_layer)
			return Matrix<Scalar>();
		std::size_t locations = out_grad.size();
		Matrix<Scalar> std_in_grad_mat = out_grad * memb.gammas->get_values()(0,0);
		return (((locations * std_in_grad_mat).array() - std_in_grad_mat.sum()).matrix() -
				memb.std_in_mat_cache * memb.std_in_mat_cache.cwiseProduct(std_in_grad_mat).sum()) *
				(((Scalar) 1 / locations) * memb.inv_in_sd_cache);
	}
	template<typename _ParamsPtr>
	inline void populate_params_vector(std::vector<_ParamsPtr> params_vec) {
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
		bool avgs_init;
		// The optimizable parameters.
		HostParamsSharedPtr gammas, betas;
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
	typedef std::shared_ptr<HostParameters<Scalar>> HostParamsSharedPtr;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 * @param gamma_reg An optional regularization function to apply to the gammas.
	 * @param gamma_clip The maximum allowed absolute gamma value. If it is 0 or less, no less, no
	 * L1 max norm constraint is enforced.
	 * @param gamma_max_l2_norm The maximum allowed L2 gamma value norm. If it is 0 or less, no L2
	 * max norm constraint is enforced.
	 * @param gamma_grad_clip The maximum allowed absolute gamma gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param gamma_grad_max_l1_norm The maximum allowed L1 gamma gradient norm. If it is 0 or less,
	 * no L1 gradient max norm constraint is enforced.
	 * @param gamma_grad_max_l2_norm The maximum allowed L2 gamma gradient norm. If it is 0 or less,
	 * no L2 gradient max norm constraint is enforced.
	 * @param beta_reg An optional regularization function to apply to the beta.
	 * @param beta_clip The maximum allowed absolute beta value. If it is 0 or less, no value clipping
	 * is performed.
	 * @param beta_max_l1_norm The maximum allowed L1 beta value norm. If it is 0 or less, no beta L1
	 * max norm constraint is enforced.
	 * @param beta_max_l2_norm The maximum allowed L2 beta value norm. If it is 0 or less, no beta L2
	 * max norm constraint is enforced.
	 * @param beta_grad_clip The maximum allowed absolute beta gradient. If it is 0 or less, no
	 * gradient clipping is performed.
	 * @param beta_grad_max_l1_norm The maximum allowed L1 beta gradient norm. If it is 0 or less, no
	 * beta L1 gradient max norm constraint is enforced.
	 * @param beta_grad_max_l2_norm The maximum allowed L2 beta gradient norm. If it is 0 or less, no
	 * beta L2 gradient max norm constraint is enforced.
	 */
	inline BatchNormLayer(const typename Base::Dims& dims, Scalar norm_avg_decay = .1,
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
				avg_means(std::make_shared<HostParameters<Scalar>>(1, dims.get_volume(), false)),
				avg_inv_sds(std::make_shared<HostParameters<Scalar>>(1, dims.get_volume(), false)),
				avgs_init(false),
				gammas(std::make_shared<HostParameters<Scalar>>(1, dims.get_volume(), true,
						std::make_shared<OneParameterInitialization<Scalar>>(), gamma_reg, gamma_clip,
						gamma_max_l1_norm, gamma_max_l2_norm, gamma_grad_clip, gamma_grad_max_l1_norm,
						gamma_grad_max_l2_norm)),
				betas(std::make_shared<HostParameters<Scalar>>(1, dims.get_volume(), true,
						std::make_shared<ZeroParameterInitialization<Scalar>>(), beta_reg, beta_clip,
						beta_max_l1_norm, beta_max_l2_norm, beta_grad_clip, beta_grad_max_l1_norm,
						beta_grad_max_l2_norm)),
				input_layer(false) {
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline BatchNormLayer(const Self& layer, bool share_params = false) :
			owner(share_params ? layer.owner : *this),
			dims(layer.dims),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			input_layer(layer.input_layer),
			avg_means(share_params ? layer.avg_means : HostParamsSharedPtr(layer.avg_means.clone())),
			avg_inv_sds(share_params ? layer.avg_inv_sds : HostParamsSharedPtr(layer.avg_inv_sds.clone())),
			avgs_init(layer.avgs_init),
			gammas(share_params ? layer.gammas : HostParamsSharedPtr(layer.gammas.clone())),
			betas(share_params ? layer.betas : HostParamsSharedPtr(layer.betas.clone())),
			inv_in_sd_cache(layer.inv_in_sd_cache),
			std_in_mat_cache(layer.std_in_mat_cache) { }
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
	inline std::vector<Parameters<Scalar>*>& get_params() {
		return std::vector<Parameters<Scalar>*>({ avg_means.get(), avg_inv_sds.get(),
				gammas.get(), betas.get() });
	}
	inline void empty_cache() {
		inv_in_sd_cache = RowVector<Scalar>();
		std_in_mat_cache = Matrix<Scalar>();
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
		if (training) {
			RowVector<Scalar> mean_vec = in_mat.colwise().mean();
			Matrix<Scalar> norm_in_mat = in_mat.rowwise() - mean_vec;
			inv_in_sd_cache = (norm_in_mat.array().square().colwise().mean() + epsilon).sqrt().inverse();
			std_in_mat_cache = norm_in_mat * inv_in_sd_cache.asDiagonal();
			in_mat = std_in_mat_cache;
			// Maintain a moving average of means and variances for testing.
			if (avgs_init) {
				avg_means->set_values((1 - norm_avg_decay) * avg_means->get_values() + norm_avg_decay * mean_vec);
				avg_inv_sds->set_values((1 - norm_avg_decay) * avg_inv_sds->get_values() +
						norm_avg_decay * inv_in_sd_cache);
			} else {
				avg_means->set_values(mean_vec);
				avg_inv_sds->set_values(inv_in_sd_cache);
				avgs_init = true;
			}
		} else {
			// For testing, use the moving averages.
			assert(avgs_init);
			in_mat = (in_mat.rowwise() - avg_means->get_values()) * avg_inv_sds.asDiagonal()->get_values();
		}
		Matrix<Scalar> out_mat = (in_mat * gammas->get_values().asDiagonal()).rowwise() + betas->get_values();
		return TensorMap<Scalar,Base::DATA_RANK>(out_mat.data(), in.dimensions());
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == dims);
		assert(out_grad.dimension(0) > 0 && std_in_mat_cache.rows() == out_grad.dimension(0));
		std::size_t rows = out_grad.dimension(0);
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), rows, out_grad.size() / rows);
		auto gammas_grad = out_grad_mat.cwiseProduct(std_in_mat_cache).colwise().sum();
		auto betas_grad = out_grad_mat.colwise().sum();
		if (Base::is_shared_params_clone()) {
			gammas->set_grad(gammas->get_grad() + gammas_grad);
			betas->set_grad(betas->get_grad() + betas_grad);
		} else {
			gammas->set_grad(gammas_grad);
			betas->set_grad(betas_grad);
		}
		if (input_layer)
			return typename Base::Data();
		Matrix<Scalar> std_in_grad_mat = out_grad_mat * gammas->get_values().asDiagonal();
		Matrix<Scalar> prev_out_grad_mat = (((rows * std_in_grad_mat).rowwise() -
				std_in_grad_mat.colwise().sum()) - std_in_mat_cache *
				(std_in_mat_cache.cwiseProduct(std_in_grad_mat).colwise().sum().asDiagonal())) *
				(((Scalar) 1 / rows) * inv_in_sd_cache).asDiagonal();
		return TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad_mat.data(), out_grad.dimensions());
	}
private:
	const Self& owner;
	const typename Base::Dims dims;
	const Scalar norm_avg_decay, epsilon;
	bool input_layer;
	// Dynamic batch normalization parameters.
	HostParamsSharedPtr avg_means, avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	HostParamsSharedPtr gammas, betas;
	// Staged computation caches.
	RowVector<Scalar> inv_in_sd_cache;
	Matrix<Scalar> std_in_mat_cache;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_BATCHNORMLAYER_H_ */
