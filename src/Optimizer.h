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
#include <cstddef>
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

namespace cattle {

template<typename Scalar, size_t Rank, bool Sequential>
using LossSharedPtr = std::shared_ptr<Loss<Scalar,Rank,Sequential>>;

template<typename Scalar, size_t Rank, bool Sequential>
class Optimizer {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal optimizer rank");
protected:
	typedef NeuralNetwork<Scalar,Rank,Sequential> Net;
	typedef DataProvider<Scalar,Rank,Sequential> Provider;
	typedef Tensor<Scalar,Rank + Sequential + 1> Data;
public:
	Optimizer(LossSharedPtr<Scalar,Rank,Sequential> loss) :
				loss(loss) {
		assert(loss != nullptr);
	};
	virtual ~Optimizer() = default;
	inline bool verify_gradients(Net& net, Provider& provider, Scalar step_size = 1e-5, Scalar abs_epsilon = Utils<Scalar>::EPSILON2,
			Scalar rel_epsilon = Utils<Scalar>::EPSILON3) const {
		assert(net.get_input_dims() == provider.get_obs_dims());
		assert(net.get_output_dims() == provider.get_obj_dims());
		assert(step_size > 0);
		assert(abs_epsilon >= 0 && rel_epsilon > 0);
		DataPair<Scalar,Rank,Sequential> data_pair = provider.get_data(provider.instances());
		provider.reset();
		/* As the loss to minimize is the mean of the losses for all the training observations, the gradient to
		 * back-propagate is to be divided by the number of observations in the batch. */
		net.backpropagate(loss->d_function(net.propagate(data_pair.first, true), data_pair.second) /
				(Scalar) provider.instances());
		bool failure = false;
		std::vector<Layer<Scalar,Rank>*> layers = net.get_layers();
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
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
	inline void optimize(Net& net, Provider& training_prov, Provider& test_prov, unsigned epochs, unsigned early_stop = 0) {
		assert(net.get_input_dims() == training_prov.get_obs_dims());
		assert(net.get_output_dims() == training_prov.get_obj_dims());
		assert(training_prov.get_obs_dims() == test_prov.get_obs_dims());
		assert(training_prov.get_obj_dims() == test_prov.get_obj_dims());
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
	virtual void fit(Net& net) = 0;
	virtual Scalar train(Net& net, Provider& training_prov,
			unsigned epoch) = 0;
	virtual Scalar test(Net& net, Provider& test_prov,
			unsigned epoch) = 0;
	inline static std::vector<Layer<Scalar,Rank>*> get_layers(Net& net) {
		return net.get_layers();
	};
	inline static Data propagate(Net& net, Data in) {
		return net.propagate(std::move(in), true);
	};
	inline static void backpropagate(Net& net, Data out_grads) {
		net.backpropagate(std::move(out_grads));
	};
	inline static bool is_parametric(Layer<Scalar,Rank>& layer) {
		return layer.is_parametric();
	};
	inline static Matrix<Scalar>& get_params(Layer<Scalar,Rank>& layer) {
		return layer.get_params();
	};
	inline static const Matrix<Scalar>& get_param_grads(Layer<Scalar,Rank>& layer) {
		return layer.get_param_grads();
	};
	inline static const void enforce_constraints(Layer<Scalar,Rank>& layer) {
		layer.enforce_constraints();
	};
	const LossSharedPtr<Scalar,Rank,Sequential> loss;
};

template<typename Scalar>
using RegPenSharedPtr = std::shared_ptr<RegularizationPenalty<Scalar>>;

template<typename Scalar, size_t Rank, bool Sequential>
class SGDOptimizer : public Optimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Base;
public:
	SGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size) :
			Base::Optimizer(loss),
				reg(reg),
				batch_size(batch_size) {
		assert(reg != nullptr);
		assert(batch_size > 0);
	};
	virtual ~SGDOptimizer() = default;
protected:
	inline Scalar train(typename Base::Net& net, typename Base::Provider& training_prov, unsigned epoch) {
		// TODO Handle overflow.
		Scalar training_loss = 0;
		Scalar instances = (Scalar) training_prov.instances();
		std::vector<Layer<Scalar,Rank>*> layers = Base::get_layers(net);
		// Perform an entire training epoch.
		while (training_prov.has_more()) {
			DataPair<Scalar,Rank> data_pair = training_prov.get_data(batch_size);
			typename Base::Data out = Base::propagate(net, std::move(data_pair.first));
			training_loss += Base::loss->function(out, data_pair.second).sum();
			/* Again, the loss on a batch is the mean of the losses on the observations in the batch and not their
			 * sum (see the last line of the function), thus the gradients of the loss function w.r.t the output of
			 * the network have to be divided by the number of instances in the batch. */
			Base::backpropagate(net, Base::loss->d_function(out, data_pair.second) / instances);
			for (unsigned k = 0; k < layers.size(); k++) {
				Layer<Scalar,Rank>& layer = *(layers[k]);
				if (Base::is_parametric(layer)) {
					update_params(layer, k, epoch - 1);
					Base::enforce_constraints(layer);
				}
			}
		}
		return training_loss / instances;
	};
	inline Scalar test(typename Base::Net& net, typename Base::Net& test_prov, unsigned epoch) {
		// TODO Handle overflow.
		Scalar obj_loss = 0;
		std::vector<Layer<Scalar,Rank>*> layers = Base::get_layers(net);
		// Perform an entire test epoch.
		while (test_prov.has_more()) {
			DataPair<Scalar,Rank> data_pair = test_prov.get_data(batch_size);
			obj_loss += Base::loss->function(net.infer(std::move(data_pair.first)), data_pair.second).sum();
		}
		Scalar mean_obj_loss = obj_loss / test_prov.instances();
		Scalar reg_loss = 0;
		for (unsigned j = 0; j < layers.size(); j++)
			reg_loss += reg->function(Base::get_params(*(layers[j])));
		std::cout << "\tobj loss: " << std::to_string(mean_obj_loss) << std::endl;
		std::cout << "\treg loss: " << std::to_string(reg_loss) << std::endl;
		return mean_obj_loss + reg_loss;
	};
	virtual void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) = 0;
	const RegPenSharedPtr<Scalar> reg;
	const unsigned batch_size;
};

template<typename Scalar, size_t Rank, bool Sequential>
class VanillaSGDOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
public:
	VanillaSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, reg, batch_size),
				learning_rate(learning_rate) {
		assert(learning_rate > 0);
	};
protected:
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) { };
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		params -= (learning_rate * (Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer) +
				SGDOptimizer<Scalar,Rank,Sequential>::reg->d_function(params)));
	};
	const Scalar learning_rate;
};

template<typename Scalar, size_t Rank, bool Sequential>
class MomentumAcceleratedSGDOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
public:
	MomentumAcceleratedSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg,
			unsigned batch_size = 1, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3,
			Scalar momentum = .9) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, reg, batch_size),
				init_learning_rate(init_learning_rate),
				annealing_rate(annealing_rate),
				momentum(momentum) {
		assert(init_learning_rate > 0);
		assert(annealing_rate >= 0);
		assert(momentum > 0 && momentum < 1);
	};
	virtual ~MomentumAcceleratedSGDOptimizer() = default;
protected:
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Optimizer<Scalar,Rank,Sequential>::get_layers(net);
		param_grads_vec = std::vector<Matrix<Scalar>>(layers.size());
		for (unsigned i = 0; i < param_grads_vec.size(); i++) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			Matrix<Scalar>& param_grads = param_grads_vec[i];
			param_grads = Matrix<Scalar>(Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer).rows(),
					Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer).cols());
			param_grads.setZero(param_grads.rows(), param_grads.cols());
		}
	};
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		Scalar learning_rate = calculate_learning_rate(epoch);
		Matrix<Scalar>& param_grads = param_grads_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		param_grads = momentum * param_grads - learning_rate * (Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer) +
				SGDOptimizer<Scalar,Rank,Sequential>::reg->d_function(params));
		params += param_grads;
	};
	Scalar calculate_learning_rate(unsigned epoch) {
		return init_learning_rate / (1.0 + annealing_rate * epoch);
	};
	const Scalar init_learning_rate;
	const Scalar annealing_rate;
	const Scalar momentum;
	std::vector<Matrix<Scalar>> param_grads_vec;
};

template<typename Scalar, size_t Rank, bool Sequential>
class NesterovMomentumAcceleratedSGDOptimizer : public MomentumAcceleratedSGDOptimizer<Scalar,Rank,Sequential> {
	typedef MomentumAcceleratedSGDOptimizer<Scalar,Rank,Sequential> Base;
public:
	NesterovMomentumAcceleratedSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg,
			unsigned batch_size = 1, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3,
			Scalar momentum = .9) :
				Base::MomentumAcceleratedSGDOptimizer(loss, reg, batch_size, init_learning_rate,
						annealing_rate, momentum) { };
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		Scalar learning_rate = Base::calculate_learning_rate(epoch);
		Matrix<Scalar>& param_grads = Base::param_grads_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		Matrix<Scalar> param_grads_bak = param_grads;
		param_grads = Base::momentum * param_grads - learning_rate * (Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer) +
				SGDOptimizer<Scalar,Rank,Sequential>::reg->d_function(params));
		params += -Base::momentum * param_grads_bak + (1 + Base::momentum) * param_grads;
	};
};

template<typename Scalar, size_t Rank, bool Sequential>
class AdagradOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
public:
	AdagradOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-2, Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, reg, batch_size),
				learning_rate(learning_rate),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(epsilon > 0);
	};
	virtual ~AdagradOptimizer() = default;
protected:
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Optimizer<Scalar,Rank,Sequential>::get_layers(net);
		param_grad_sqrs_vec = std::vector<Matrix<Scalar>>(layers.size());
		for (unsigned i = 0; i < param_grad_sqrs_vec.size(); i++) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			Matrix<Scalar>& param_grad_sqrs = param_grad_sqrs_vec[i];
			param_grad_sqrs = Matrix<Scalar>(Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer).rows(),
					Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer).cols());
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
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		Matrix<Scalar>& param_grad_sqrs = param_grad_sqrs_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer) +
				SGDOptimizer<Scalar,Rank,Sequential>::reg->d_function(params);
		update_acc_weight_grad_sqrs(param_grad_sqrs, param_grads);
		params -= (learning_rate * param_grads.array() / (param_grad_sqrs.array().sqrt() + epsilon)).matrix();
	};
	const Scalar learning_rate;
	const Scalar epsilon;
	std::vector<Matrix<Scalar>> param_grad_sqrs_vec;
};

template<typename Scalar, size_t Rank, bool Sequential>
class RMSPropOptimizer : public AdagradOptimizer<Scalar,Rank,Sequential> {
public:
	RMSPropOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l2_decay = 1e-1, Scalar epsilon = Utils<Scalar>::EPSILON) :
				AdagradOptimizer<Scalar,Rank,Sequential>::AdagradOptimizer(loss, reg, batch_size, learning_rate, epsilon),
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
	const Scalar l2_decay;
};

template<typename Scalar, size_t Rank, bool Sequential>
class AdadeltaOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
public:
	AdadeltaOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar decay = 5e-2, Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, reg, batch_size),
				decay(decay),
				epsilon(epsilon) {
		assert(decay >= 0 && decay <= 1);
		assert(epsilon > 0);
	};
protected:
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Optimizer<Scalar,Rank,Sequential>::get_layers(net);
		pgus_vec = std::vector<ParamGradAndUpdateSqrs>(layers.size());
		for (unsigned i = 0; i < pgus_vec.size(); i++) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			ParamGradAndUpdateSqrs& pgus = pgus_vec[i];
			pgus.param_grad = Matrix<Scalar>(Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer).rows(),
					Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer).cols());
			pgus.param_grad.setZero(pgus.param_grad.rows(), pgus.param_grad.cols());
			pgus.param_update = Matrix<Scalar>(pgus.param_grad.rows(), pgus.param_grad.cols());
			pgus.param_update.setZero(pgus.param_update.rows(), pgus.param_update.cols());
		}
	};
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		ParamGradAndUpdateSqrs& pgus = pgus_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer) +
				SGDOptimizer<Scalar,Rank,Sequential>::reg->d_function(params);
		pgus.param_grad = (1 - decay) * pgus.param_grad + decay * param_grads.cwiseProduct(param_grads);
		Matrix<Scalar> weight_updates = -param_grads.array() * (pgus.param_update.array() + epsilon).sqrt() /
				(pgus.param_grad.array() + epsilon).sqrt();
		params += weight_updates;
		pgus.param_update = (1 - decay) * pgus.param_update + decay * weight_updates.cwiseProduct(weight_updates);
	};
	const Scalar decay;
	const Scalar epsilon;
	struct ParamGradAndUpdateSqrs {
		Matrix<Scalar> param_grad;
		Matrix<Scalar> param_update;
	};
	std::vector<ParamGradAndUpdateSqrs> pgus_vec;
};

template<typename Scalar, size_t Rank, bool Sequential>
class AdamOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
public:
	AdamOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, reg, batch_size),
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
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Optimizer<Scalar,Rank,Sequential>::get_layers(net);
		pgn_vec = std::vector<ParamGradNorms>(layers.size());
		for (unsigned i = 0; i < pgn_vec.size(); i++) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			ParamGradNorms& vel = pgn_vec[i];
			vel.param_grad_l1 = Matrix<Scalar>(Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer).rows(),
					Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer).cols());
			vel.param_grad_l1.setZero(vel.param_grad_l1.rows(), vel.param_grad_l1.cols());
			vel.param_grad_l2 = Matrix<Scalar>(vel.param_grad_l1.rows(), vel.param_grad_l1.cols());
			vel.param_grad_l2.setZero(vel.param_grad_l2.rows(), vel.param_grad_l2.cols());
		}
	};
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		ParamGradNorms& grad_norms = pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - l1_decay, epoch + 1) + epsilon);
		Scalar l2_corr = 1.0 / (1.0 - pow(1.0 - l2_decay, epoch + 1) + epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer) +
				SGDOptimizer<Scalar,Rank,Sequential>::reg->d_function(params);
		grad_norms.param_grad_l1 = (1 - l1_decay) * grad_norms.param_grad_l1 + l1_decay * param_grads;
		grad_norms.param_grad_l2 = (1 - l2_decay) * grad_norms.param_grad_l2 +
				l2_decay * param_grads.cwiseProduct(param_grads);
		params -= (learning_rate * (grad_norms.param_grad_l1 * l1_corr).array() /
				((grad_norms.param_grad_l2 * l2_corr).array().sqrt() + epsilon)).matrix();
	};
	const Scalar learning_rate;
	const Scalar l1_decay;
	const Scalar l2_decay;
	const Scalar epsilon;
	struct ParamGradNorms {
		Matrix<Scalar> param_grad_l1;
		Matrix<Scalar> param_grad_l2;
	};
	std::vector<ParamGradNorms> pgn_vec;
};

template<typename Scalar, size_t Rank, bool Sequential>
class AdaMaxOptimizer : public AdamOptimizer<Scalar,Rank,Sequential> {
	typedef AdamOptimizer<Scalar,Rank,Sequential> Base;
public:
	AdaMaxOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				Base::AdamOptimizer(loss, reg, batch_size, learning_rate,
						l1_decay, l2_decay, epsilon) { };
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		typename Base::ParamGradNorms& grad_norms = Base::pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - Base::l1_decay, epoch + 1) + Base::epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer) +
				SGDOptimizer<Scalar,Rank,Sequential>::reg->d_function(params);
		grad_norms.param_grad_l1 = (1 - Base::l1_decay) * grad_norms.param_grad_l1 + Base::l1_decay * param_grads;
		grad_norms.param_grad_l2 = ((1 - Base::l2_decay) * grad_norms.param_grad_l2).cwiseMax(param_grads.cwiseAbs());
		params -= (Base::learning_rate * (grad_norms.param_grad_l1 * l1_corr).array() /
				(grad_norms.param_grad_l2.array() + Base::epsilon)).matrix();
	};
};

template<typename Scalar, size_t Rank, bool Sequential>
class NadamOptimizer : public AdamOptimizer<Scalar,Rank,Sequential> {
	typedef AdamOptimizer<Scalar,Rank,Sequential> Base;
public:
	NadamOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, RegPenSharedPtr<Scalar> reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				Base::AdamOptimizer(loss, reg, batch_size, learning_rate, l1_decay, l2_decay, epsilon) { };
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		typename Base::ParamGradNorms& grad_norms = Base::pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - Base::l1_decay, epoch + 1) + Base::epsilon);
		Scalar l1_next_corr = 1.0 / (1.0 - pow(1.0 - Base::l1_decay, epoch + 2) + Base::epsilon);
		Scalar l2_corr = 1.0 / (1.0 - pow(1.0 - Base::l2_decay, epoch + 1) + Base::epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar,Rank,Sequential>::get_param_grads(layer) +
				SGDOptimizer<Scalar,Rank,Sequential>::reg->d_function(params);
		grad_norms.param_grad_l1 = (1 - Base::l1_decay) * grad_norms.param_grad_l1 + Base::l1_decay * param_grads;
		grad_norms.param_grad_l2 = (1 - Base::l2_decay) * grad_norms.param_grad_l2 + Base::l2_decay *
				param_grads.cwiseProduct(param_grads);
		params -= (Base::learning_rate * (Base::l1_decay * l1_corr * param_grads +
				(1.0 - Base::l1_decay) * l1_next_corr * grad_norms.param_grad_l1).array() /
				((grad_norms.param_grad_l2 * l2_corr).array().sqrt() + Base::epsilon)).matrix();
	};
};

// TODO: Hessian-free w/ Conjugate Gradient, L-BFGS, Supervised Descent Method, Particle Swarm, GA, PBIL

} /* namespace cattle */

#endif /* OPTIMIZER_H_ */
