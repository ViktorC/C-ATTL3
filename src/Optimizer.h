/*
 * Optimizer.h
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <algorithm>
#include <cassert>
#include <DataProvider.h>
#include <Loss.h>
#include <memory>
#include <NeuralNetwork.h>
#include <RegularizationPenalty.h>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <Utils.h>
#include <vector>
#include <WeightInitialization.h>

namespace cppnn {

template<typename Scalar>
using LossSharedPtr = std::shared_ptr<Loss<Scalar>>;

template<typename Scalar>
class Optimizer {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	Optimizer(LossSharedPtr<Scalar> loss) :
				loss(loss) {
		assert(loss != nullptr);
	};
	virtual ~Optimizer() = default;
	bool verify_gradients(NeuralNetwork<Scalar>& net, DataProvider<Scalar>& provider, Scalar step_size = 1e-5,
			Scalar abs_epsilon = Utils<Scalar>::EPSILON2, Scalar rel_epsilon = Utils<Scalar>::EPSILON3) const {
		assert(net.get_input_dims().equals(provider.get_obs_dims()));
		assert(net.get_output_dims().equals(provider.get_obj_dims()));
		assert(step_size > 0);
		assert(abs_epsilon >= 0 && rel_epsilon > 0);
		DataPair<Scalar> data_pair = provider.get_data(provider.instances());
		provider.reset();
		/* As the loss to minimize is the mean of the losses for all the training observations, the gradient to
		 * back-propagate is to be divided by the number of observations in the batch. */
		net.backpropagate(loss->d_function(net.propagate(data_pair.first, true), data_pair.second) /
				(Scalar) provider.instances());
		bool failure = false;
		std::vector<Layer<Scalar>*> layers = net.get_layers();
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar>& layer = *(layers[i]);
			if (layer.is_parametric()) {
				std::cout << "Layer " << std::setw(3) << std::to_string(i + 1) <<
						std::string(28, '-') << std::endl;
				Matrix<Scalar>& params = layer.get_params();
				const Matrix<Scalar>& param_grads = layer.get_param_grads();
				for (int j = 0; j < params.rows(); j++) {
					for (int k = 0; k < params.cols(); k++) {
						std::cout << "\tParam[" << i << "," << j << "," << k << "]:" << std::endl;
						Scalar ana_grad = param_grads(j,k);
						std::cout << "\t\tAnalytic gradient = " << ana_grad << std::endl;
						Scalar param = params(j,k);
						params(j,k) = param + step_size;
						/* Compute the numerical gradients in training mode to ensure that the means
						 * and standard deviations used for batch normalization are the same as those
						 * used during the analytic gradient computation. */
						Scalar loss_inc = loss->function(net.propagate(data_pair.first, true),
								data_pair.second).mean();
						params(j,k) = param - step_size;
						Scalar loss_dec = loss->function(net.propagate(data_pair.first, true),
								data_pair.second).mean();
						params(j,k) = param;
						Scalar num_grad = (loss_inc - loss_dec) / (2 * step_size);
						std::cout << "\t\tNumerical gradient = " << num_grad;
						if (!Utils<Scalar>::almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
							std::cout << " *****FAIL*****";
							failure = true;
						}
						std::cout << std::endl;
					}
				}
			}
		}
		// Empty the layer caches.
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i]->empty_cache();
		return !failure;
	};
	void optimize(NeuralNetwork<Scalar>& net, DataProvider<Scalar>& training_prov, DataProvider<Scalar>& test_prov,
			unsigned epochs, unsigned early_stop = 0) {
		assert(net.get_input_dims().equals(training_prov.get_obs_dims()));
		assert(net.get_output_dims().equals(training_prov.get_obj_dims()));
		assert(training_prov.get_obs_dims().equals(test_prov.get_obs_dims()));
		assert(training_prov.get_obj_dims().equals(test_prov.get_obj_dims()));
		assert(epochs > 0);
		// Fit the optimizer parameters to the network.
		fit(net);
		Scalar prev_test_loss = Utils<Scalar>::MAX;
		unsigned cons_loss_inc = 0;
		// Start the optimization iterations.
		for (unsigned i = 0; i <= epochs; i++) {
			std::cout << "Epoch " << std::setw(3) << i << std::string(28, '-') << std::endl;
			// Train.
			if (i != 0) {
				training_prov.reset();
				std::cout << "\ttraining loss: " << std::to_string(train(net, training_prov, i)) << std::endl;
			}
			// Validate.
			test_prov.reset();
			Scalar test_loss = test(net, test_prov, i);
			std::cout << "\ttest loss: " << std::to_string(test_loss);
			if (test_loss >= prev_test_loss) {
				cons_loss_inc++;
				std::cout << " *****INCREASED LOSS*****";
				if (early_stop > 0 && cons_loss_inc >= early_stop)
					break;
			} else
				cons_loss_inc = 0;
			std::cout << std::endl << std::endl;
			prev_test_loss = test_loss;
		}
	};
protected:
	virtual void fit(NeuralNetwork<Scalar>& net) = 0;
	virtual Scalar train(NeuralNetwork<Scalar>& net, DataProvider<Scalar>& training_prov, unsigned epoch) = 0;
	virtual Scalar test(NeuralNetwork<Scalar>& net, DataProvider<Scalar>& test_prov, unsigned epoch) = 0;
	inline static std::vector<Layer<Scalar>*> get_layers(NeuralNetwork<Scalar>& net) {
		return net.get_layers();
	};
	inline static Tensor4<Scalar> propagate(NeuralNetwork<Scalar>& net, Tensor4<Scalar> in) {
		return net.propagate(std::move(in), true);
	};
	inline static void backpropagate(NeuralNetwork<Scalar>& net, Tensor4<Scalar> out_grads) {
		net.backpropagate(std::move(out_grads));
	};
	inline static bool is_parametric(Layer<Scalar>& layer) {
		return layer.is_parametric();
	};
	inline static Matrix<Scalar>& get_params(Layer<Scalar>& layer) {
		return layer.get_params();
	};
	inline static const Matrix<Scalar>& get_param_grads(Layer<Scalar>& layer) {
		return layer.get_param_grads();
	};
	inline static const void enforce_constraints(Layer<Scalar>& layer) {
		layer.enforce_constraints();
	};
	LossSharedPtr<Scalar> loss;
};

template<typename Scalar>
using RegPenSharedPtr = std::shared_ptr<RegularizationPenalty<Scalar>>;

template<typename Scalar>
class SGDOptimizer : public Optimizer<Scalar> {
public:
	SGDOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size) :
				Optimizer<Scalar>::Optimizer(loss),
				reg(reg),
				batch_size(batch_size) {
		assert(reg != nullptr);
		assert(batch_size > 0);
	};
	virtual ~SGDOptimizer() = default;
protected:
	inline Scalar train(NeuralNetwork<Scalar>& net, DataProvider<Scalar>& training_prov, unsigned epoch) {
		// TODO Handle overflow.
		Scalar training_loss = 0;
		Scalar instances = (Scalar) training_prov.instances();
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		// Perform an entire training epoch.
		while (training_prov.has_more()) {
			DataPair<Scalar> data_pair = training_prov.get_data(batch_size);
			Tensor4<Scalar> out = Optimizer<Scalar>::propagate(net, std::move(data_pair.first));
			training_loss += Optimizer<Scalar>::loss->function(out, data_pair.second).sum();
			/* Again, the loss on a batch is the mean of the losses on the observations in the batch and not their
			 * sum (see the last line of the function), thus the gradients of the loss function w.r.t the output of
			 * the network have to be divided by the number of instances in the batch. */
			Optimizer<Scalar>::backpropagate(net, Optimizer<Scalar>::loss->d_function(out, data_pair.second) /
					instances);
			for (unsigned k = 0; k < layers.size(); k++) {
				Layer<Scalar>& layer = *(layers[k]);
				if (Optimizer<Scalar>::is_parametric(layer)) {
					update_params(layer, k, epoch - 1);
					Optimizer<Scalar>::enforce_constraints(layer);
				}
			}
		}
		return training_loss / instances;
	};
	inline Scalar test(NeuralNetwork<Scalar>& net, DataProvider<Scalar>& test_prov, unsigned epoch) {
		// TODO Handle overflow.
		Scalar obj_loss = 0;
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		// Perform an entire test epoch.
		while (test_prov.has_more()) {
			DataPair<Scalar> data_pair = test_prov.get_data(batch_size);
			obj_loss += Optimizer<Scalar>::loss->function(net.infer(std::move(data_pair.first)),
					data_pair.second).sum();
		}
		Scalar mean_obj_loss = obj_loss / test_prov.instances();
		Scalar reg_loss = 0;
		for (unsigned j = 0; j < layers.size(); j++)
			reg_loss += reg->function(Optimizer<Scalar>::get_params(*(layers[j])));
		std::cout << "\tobj loss: " << std::to_string(mean_obj_loss) << std::endl;
		std::cout << "\treg loss: " << std::to_string(reg_loss) << std::endl;
		return mean_obj_loss + reg_loss;
	};
	virtual void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) = 0;
	RegPenSharedPtr<Scalar> reg;
	unsigned batch_size;
};

template<typename Scalar>
class VanillaSGDOptimizer : public SGDOptimizer<Scalar> {
public:
	VanillaSGDOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size),
				learning_rate(learning_rate) {
		assert(learning_rate > 0);
	};
protected:
	inline void fit(NeuralNetwork<Scalar>& net) { };
	inline void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		params -= (learning_rate * (Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg->d_function(params)));
	};
	Scalar learning_rate;
};

template<typename Scalar>
class MomentumAcceleratedSGDOptimizer : public SGDOptimizer<Scalar> {
public:
	MomentumAcceleratedSGDOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg,
			unsigned batch_size = 1, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3,
			Scalar momentum = .9) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size),
				init_learning_rate(init_learning_rate),
				annealing_rate(annealing_rate),
				momentum(momentum) {
		assert(init_learning_rate > 0);
		assert(annealing_rate >= 0);
		assert(momentum > 0 && momentum < 1);
	};
	virtual ~MomentumAcceleratedSGDOptimizer() = default;
protected:
	inline void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		param_grads_vec = std::vector<Matrix<Scalar>>(layers.size());
		for (unsigned i = 0; i < param_grads_vec.size(); i++) {
			Layer<Scalar>& layer = *(layers[i]);
			Matrix<Scalar>& param_grads = param_grads_vec[i];
			param_grads = Matrix<Scalar>(Optimizer<Scalar>::get_param_grads(layer).rows(),
					Optimizer<Scalar>::get_param_grads(layer).cols());
			param_grads.setZero(param_grads.rows(), param_grads.cols());
		}
	};
	inline void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		Scalar learning_rate = calculate_learning_rate(epoch);
		Matrix<Scalar>& param_grads = param_grads_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		param_grads = momentum * param_grads - learning_rate * (Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg->d_function(params));
		params += param_grads;
	};
	Scalar calculate_learning_rate(unsigned epoch) {
		return init_learning_rate / (1.0 + annealing_rate * epoch);
	};
	Scalar init_learning_rate;
	Scalar annealing_rate;
	Scalar momentum;
	std::vector<Matrix<Scalar>> param_grads_vec;
};

template<typename Scalar>
class NesterovMomentumAcceleratedSGDOptimizer : public MomentumAcceleratedSGDOptimizer<Scalar> {
public:
	NesterovMomentumAcceleratedSGDOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg,
			unsigned batch_size = 1, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3,
			Scalar momentum = .9) :
				MomentumAcceleratedSGDOptimizer<Scalar>::MomentumAcceleratedSGDOptimizer(loss, reg, batch_size,
						init_learning_rate, annealing_rate, momentum) { };
protected:
	inline void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		Scalar learning_rate = MomentumAcceleratedSGDOptimizer<Scalar>::calculate_learning_rate(epoch);
		Matrix<Scalar>& param_grads = MomentumAcceleratedSGDOptimizer<Scalar>::param_grads_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads_bak = param_grads;
		param_grads = MomentumAcceleratedSGDOptimizer<Scalar>::momentum * param_grads -
				learning_rate * (Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg->d_function(params));
		params += -MomentumAcceleratedSGDOptimizer<Scalar>::momentum * param_grads_bak +
				(1 + MomentumAcceleratedSGDOptimizer<Scalar>::momentum) * param_grads;
	};
};

template<typename Scalar>
class AdagradOptimizer : public SGDOptimizer<Scalar> {
public:
	AdagradOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-2, Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size),
				learning_rate(learning_rate),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(epsilon > 0);
	};
	virtual ~AdagradOptimizer() = default;
protected:
	inline void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		param_grad_sqrs_vec = std::vector<Matrix<Scalar>>(layers.size());
		for (unsigned i = 0; i < param_grad_sqrs_vec.size(); i++) {
			Layer<Scalar>& layer = *(layers[i]);
			Matrix<Scalar>& param_grad_sqrs = param_grad_sqrs_vec[i];
			param_grad_sqrs = Matrix<Scalar>(Optimizer<Scalar>::get_param_grads(layer).rows(),
					Optimizer<Scalar>::get_param_grads(layer).cols());
			param_grad_sqrs.setZero(param_grad_sqrs.rows(), param_grad_sqrs.cols());
		}
	};
	inline virtual void update_acc_weight_grad_sqrs(Matrix<Scalar>& acc_weight_grad_sqrs,
			const Matrix<Scalar>& param_grads) {
		acc_weight_grad_sqrs += param_grads.cwiseProduct(param_grads);
	};
	inline virtual void update_acc_batch_norm_grad_sqrs(RowVector<Scalar>& acc_batch_norm_grad_sqrs,
			const RowVector<Scalar>& batch_norm_grads) {
		acc_batch_norm_grad_sqrs += batch_norm_grads.cwiseProduct(batch_norm_grads);
	};
	inline void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		Matrix<Scalar>& param_grad_sqrs = param_grad_sqrs_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg->d_function(params);
		update_acc_weight_grad_sqrs(param_grad_sqrs, param_grads);
		params -= (learning_rate * param_grads.array() / (param_grad_sqrs.array().sqrt() + epsilon)).matrix();
	};
	Scalar learning_rate;
	Scalar epsilon;
	std::vector<Matrix<Scalar>> param_grad_sqrs_vec;
};

template<typename Scalar>
class RMSPropOptimizer : public AdagradOptimizer<Scalar> {
public:
	RMSPropOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l2_decay = 1e-1, Scalar epsilon = Utils<Scalar>::EPSILON) :
				AdagradOptimizer<Scalar>::AdagradOptimizer(loss, reg, batch_size, learning_rate, epsilon),
				l2_decay(l2_decay) {
		assert(l2_decay >= 0 && l2_decay <= 1);
	};
protected:
	inline void update_acc_weight_grad_sqrs(Matrix<Scalar>& acc_weight_grad_sqrs,
			const Matrix<Scalar>& param_grads) {
		acc_weight_grad_sqrs = (1 - l2_decay) * acc_weight_grad_sqrs + l2_decay * param_grads.cwiseProduct(param_grads);
	};
	inline void update_acc_batch_norm_grad_sqrs(RowVector<Scalar>& acc_batch_norm_grad_sqrs,
			const RowVector<Scalar>& batch_norm_grads) {
		acc_batch_norm_grad_sqrs = (1 - l2_decay) * acc_batch_norm_grad_sqrs +
				l2_decay * batch_norm_grads.cwiseProduct(batch_norm_grads);
	};
	Scalar l2_decay;
};

template<typename Scalar>
class AdadeltaOptimizer : public SGDOptimizer<Scalar> {
public:
	AdadeltaOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar decay = 5e-2, Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size),
				decay(decay),
				epsilon(epsilon) {
		assert(decay >= 0 && decay <= 1);
		assert(epsilon > 0);
	};
protected:
	inline void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		pgus_vec = std::vector<ParamGradAndUpdateSqrs>(layers.size());
		for (unsigned i = 0; i < pgus_vec.size(); i++) {
			Layer<Scalar>& layer = *(layers[i]);
			ParamGradAndUpdateSqrs& pgus = pgus_vec[i];
			pgus.param_grad = Matrix<Scalar>(Optimizer<Scalar>::get_param_grads(layer).rows(),
					Optimizer<Scalar>::get_param_grads(layer).cols());
			pgus.param_grad.setZero(pgus.param_grad.rows(), pgus.param_grad.cols());
			pgus.param_update = Matrix<Scalar>(pgus.param_grad.rows(), pgus.param_grad.cols());
			pgus.param_update.setZero(pgus.param_update.rows(), pgus.param_update.cols());
		}
	};
	inline void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		ParamGradAndUpdateSqrs& pgus = pgus_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg->d_function(params);
		pgus.param_grad = (1 - decay) * pgus.param_grad + decay * param_grads.cwiseProduct(param_grads);
		Matrix<Scalar> weight_updates = -param_grads.array() * (pgus.param_update.array() + epsilon).sqrt() /
				(pgus.param_grad.array() + epsilon).sqrt();
		params += weight_updates;
		pgus.param_update = (1 - decay) * pgus.param_update + decay * weight_updates.cwiseProduct(weight_updates);
	};
	Scalar decay;
	Scalar epsilon;
	struct ParamGradAndUpdateSqrs {
		Matrix<Scalar> param_grad;
		Matrix<Scalar> param_update;
	};
	std::vector<ParamGradAndUpdateSqrs> pgus_vec;
};

template<typename Scalar>
class AdamOptimizer : public SGDOptimizer<Scalar> {
public:
	AdamOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size),
				learning_rate(learning_rate),
				l1_decay(l1_decay),
				l2_decay(l2_decay),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(l1_decay >= 0 && l1_decay <= 1);
		assert(l2_decay >= 0 && l2_decay <= 1);
		assert(epsilon > 0);
	};
	virtual ~AdamOptimizer() = default;
protected:
	inline void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		pgn_vec = std::vector<ParamGradNorms>(layers.size());
		for (unsigned i = 0; i < pgn_vec.size(); i++) {
			Layer<Scalar>& layer = *(layers[i]);
			ParamGradNorms& vel = pgn_vec[i];
			vel.param_grad_l1 = Matrix<Scalar>(Optimizer<Scalar>::get_param_grads(layer).rows(),
					Optimizer<Scalar>::get_param_grads(layer).cols());
			vel.param_grad_l1.setZero(vel.param_grad_l1.rows(), vel.param_grad_l1.cols());
			vel.param_grad_l2 = Matrix<Scalar>(vel.param_grad_l1.rows(), vel.param_grad_l1.cols());
			vel.param_grad_l2.setZero(vel.param_grad_l2.rows(), vel.param_grad_l2.cols());
		}
	};
	inline void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		ParamGradNorms& grad_norms = pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - l1_decay, epoch + 1) + epsilon);
		Scalar l2_corr = 1.0 / (1.0 - pow(1.0 - l2_decay, epoch + 1) + epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg->d_function(params);
		grad_norms.param_grad_l1 = (1 - l1_decay) * grad_norms.param_grad_l1 + l1_decay * param_grads;
		grad_norms.param_grad_l2 = (1 - l2_decay) * grad_norms.param_grad_l2 +
				l2_decay * param_grads.cwiseProduct(param_grads);
		params -= (learning_rate * (grad_norms.param_grad_l1 * l1_corr).array() /
				((grad_norms.param_grad_l2 * l2_corr).array().sqrt() + epsilon)).matrix();
	};
	Scalar learning_rate;
	Scalar l1_decay;
	Scalar l2_decay;
	Scalar epsilon;
	struct ParamGradNorms {
		Matrix<Scalar> param_grad_l1;
		Matrix<Scalar> param_grad_l2;
	};
	std::vector<ParamGradNorms> pgn_vec;
};

template<typename Scalar>
class AdaMaxOptimizer : public AdamOptimizer<Scalar> {
public:
	AdaMaxOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				AdamOptimizer<Scalar>::AdamOptimizer(loss, reg, batch_size, learning_rate,
						l1_decay, l2_decay, epsilon) { };
protected:
	inline void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		typename AdamOptimizer<Scalar>::ParamGradNorms& grad_norms = AdamOptimizer<Scalar>::pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg->d_function(params);
		grad_norms.param_grad_l1 = (1 - AdamOptimizer<Scalar>::l1_decay) * grad_norms.param_grad_l1 +
				AdamOptimizer<Scalar>::l1_decay * param_grads;
		grad_norms.param_grad_l2 = ((1 - AdamOptimizer<Scalar>::l2_decay) * grad_norms.param_grad_l2)
				.cwiseMax(param_grads.cwiseAbs());
		params -= (AdamOptimizer<Scalar>::learning_rate * (grad_norms.param_grad_l1 * l1_corr).array() /
				(grad_norms.param_grad_l2.array() + AdamOptimizer<Scalar>::epsilon)).matrix();
	};
};

template<typename Scalar>
class NadamOptimizer : public AdamOptimizer<Scalar> {
public:
	NadamOptimizer(LossSharedPtr<Scalar> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				AdamOptimizer<Scalar>::AdamOptimizer(loss, reg, batch_size, learning_rate,
						l1_decay, l2_decay, epsilon) { };
protected:
	inline void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		typename AdamOptimizer<Scalar>::ParamGradNorms& grad_norms = AdamOptimizer<Scalar>::pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Scalar l1_next_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 2) +
				AdamOptimizer<Scalar>::epsilon);
		Scalar l2_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l2_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg->d_function(params);
		grad_norms.param_grad_l1 = (1 - AdamOptimizer<Scalar>::l1_decay) * grad_norms.param_grad_l1 +
				AdamOptimizer<Scalar>::l1_decay * param_grads;
		grad_norms.param_grad_l2 = (1 - AdamOptimizer<Scalar>::l2_decay) * grad_norms.param_grad_l2 +
				AdamOptimizer<Scalar>::l2_decay * param_grads.cwiseProduct(param_grads);
		params -= (AdamOptimizer<Scalar>::learning_rate * (AdamOptimizer<Scalar>::l1_decay * l1_corr * param_grads +
				(1.0 - AdamOptimizer<Scalar>::l1_decay) * l1_next_corr * grad_norms.param_grad_l1).array() /
				((grad_norms.param_grad_l2 * l2_corr).array().sqrt() + AdamOptimizer<Scalar>::epsilon)).matrix();
	};
};

// TODO: Conjugate Gradient, L-BFGS, (LMA), Particle Swarm, GA, PBIL

} /* namespace cppnn */

#endif /* OPTIMIZER_H_ */
