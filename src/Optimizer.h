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
#include <limits>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <RegularizationPenalty.h>
#include <random>
#include <sstream>
#include <string>
#include <Utils.h>
#include <vector>
#include <Vector.h>
#include <WeightInitialization.h>

namespace cppnn {

template<typename Scalar>
class Optimizer {
public:
	Optimizer(const Loss<Scalar>& loss) :
				loss(loss) { };
	virtual ~Optimizer() = default;
	virtual void train(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			unsigned epochs, unsigned cons_loss_inc_for_early_stop) = 0;
	bool verify_gradients(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x,
			const Matrix<Scalar>& y, Scalar step_size = 1e-5, Scalar abs_epsilon = Utils<Scalar>::EPSILON2,
			Scalar rel_epsilon = Utils<Scalar>::EPSILON3) const {
		assert((unsigned) x.cols() == net.get_input_size() && (unsigned) y.cols() == net.get_output_size());
		assert(x.rows() == y.rows());
		assert(step_size > 0);
		assert(abs_epsilon >= 0 && rel_epsilon > 0);
		net.backpropagate(loss.d_function(net.propagate(x, true), y));
		bool failure = false;
		for (unsigned i = 0; i < net.get_layers().size(); i++) {
			Layer<Scalar>& layer = *(net.get_layers()[i]);
			Matrix<Scalar>& weights = layer.get_weights();
			const Matrix<Scalar>& weight_grads = layer.get_weight_grads();
			for (int j = 0; j < weights.rows(); j++) {
				for (int k = 0; k < weights.cols(); k++) {
					std::cout << "Weight[" << i << "," << j << "," << k << "]:" << std::endl;
					Scalar ana_grad = weight_grads(j,k);
					std::cout << "\tAnalytic gradient = " << ana_grad << std::endl;
					Scalar weight = weights(j,k);
					weights(j,k) = weight + step_size;
					Scalar loss_inc = loss.function(net.propagate(x, true), y).mean();
					weights(j,k) = weight - step_size;
					Scalar loss_dec = loss.function(net.propagate(x, true), y).mean();
					weights(j,k) = weight;
					Scalar num_grad = (loss_inc - loss_dec) / (2 * step_size);
					std::cout << "\tNumerical gradient = " << num_grad;
					if (!Utils<Scalar>::almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
						std::cout << " *****FAIL*****";
						failure = true;
					}
					std::cout << std::endl;
				}
			}
			if (layer.get_batch_norm()) {
				RowVector<Scalar>& betas = layer.get_betas();
				const RowVector<Scalar>& beta_grads = layer.get_beta_grads();
				for (int j = 0; j < betas.cols(); j++) {
					std::cout << "Beta[" << i << "," << j << "]:" << std::endl;
					Scalar ana_grad = beta_grads(j);
					std::cout << "\tAnalytic gradient = " << ana_grad << std::endl;
					Scalar beta = betas(j);
					betas(j) = beta + step_size;
					Scalar loss_inc = loss.function(net.propagate(x, true), y).mean();
					betas(j) = beta - step_size;
					Scalar loss_dec = loss.function(net.propagate(x, true), y).mean();
					betas(j) = beta;
					Scalar num_grad = (loss_inc - loss_dec) / (2 * step_size);
					std::cout << "\tNumerical gradient = " << num_grad;
					if (!Utils<Scalar>::almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
						std::cout << " *****FAIL*****";
						failure = true;
					}
					std::cout << std::endl;
				}
				RowVector<Scalar>& gammas = layer.get_gammas();
				const RowVector<Scalar>& gamma_grads = layer.get_gamma_grads();
				for (int j = 0; j < gammas.cols(); j++) {
					std::cout << "Gamma[" << i << "," << j << "]:" << std::endl;
					Scalar ana_grad = gamma_grads(j);
					std::cout << "\tAnalytic gradient = " << ana_grad << std::endl;
					Scalar gamma = gammas(j);
					gammas(j) = gamma + step_size;
					Scalar loss_inc = loss.function(net.propagate(x, true), y).mean();
					gammas(j) = gamma - step_size;
					Scalar loss_dec = loss.function(net.propagate(x, true), y).mean();
					gammas(j) = gamma;
					Scalar num_grad = (loss_inc - loss_dec) / (2 * step_size);
					std::cout << "\tNumerical gradient = " << num_grad;
					if (!Utils<Scalar>::almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
						std::cout << " *****FAIL*****";
						failure = true;
					}
					std::cout << std::endl;
				}
			}
		}
		empty_layer_caches(net);
		return !failure;
	};
protected:
	void empty_layer_caches(NeuralNetwork<Scalar>& net) const {
		for (unsigned i = 0; i < net.get_layers().size(); i++) {
			net.get_layers()[i]->empty_cache();
		}
	};
	ColVector<Scalar> compute_training_loss(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x,
			const Matrix<Scalar>& y) const {
		return loss.function(net.propagate(x, true), y);
	};
	ColVector<Scalar> compute_training_loss_and_backprop(NeuralNetwork<Scalar>& net,
			const Matrix<Scalar>& x, const Matrix<Scalar>& y) const {
		Matrix<Scalar> out = net.propagate(x, true);
		ColVector<Scalar> loss_vec = loss.function(out, y);
		net.backpropagate(loss.d_function(out, y));
		return loss_vec;
	};
	static std::vector<Layer<Scalar>*>& get_layers(NeuralNetwork<Scalar>& net) {
		return net.get_layers();
	};
	static Matrix<Scalar>& get_weights(Layer<Scalar>* layer_ptr) {
		return layer_ptr->get_weights();
	};
	static const Matrix<Scalar>& get_weight_grads(Layer<Scalar>* layer_ptr) {
		return layer_ptr->get_weight_grads();
	};
	static RowVector<Scalar>& get_betas(Layer<Scalar>* layer_ptr) {
		return layer_ptr->get_betas();
	};
	static const RowVector<Scalar>& get_beta_grads(Layer<Scalar>* layer_ptr) {
		return layer_ptr->get_beta_grads();
	};
	static RowVector<Scalar>& get_gammas(Layer<Scalar>* layer_ptr) {
		return layer_ptr->get_gammas();
	};
	static const RowVector<Scalar>& get_gamma_grads(Layer<Scalar>* layer_ptr) {
		return layer_ptr->get_gamma_grads();
	};
	static const void enforce_constraints(Layer<Scalar>* layer_ptr) {
		layer_ptr->enforce_constraints();
	};
	const Loss<Scalar>& loss;
};

template<typename Scalar>
class SGDOptimizer : public Optimizer<Scalar> {
public:
	SGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
			unsigned batch_size, Scalar k) :
				Optimizer<Scalar>::Optimizer(loss),
				reg(reg),
				batch_size(batch_size),
				k(k) {
		assert(batch_size > 0);
		assert(k > 0 && k < 1);
	};
	virtual ~SGDOptimizer() = default;
	void train(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x, const Matrix<Scalar>& y,
				unsigned epochs = 1000, unsigned early_stop = 0) {
		assert((unsigned) x.cols() == net.get_input_size() && (unsigned) y.cols() == net.get_output_size());
		assert(x.rows() == y.rows());
		assert(x.rows() > 1);
		assert(epochs > 0);
		// Fit the optimizer parameters to the network.
		fit(net);
		// Divide the data into training and test partitions.
		unsigned training_row_num = std::min((unsigned) (x.rows() - 1),
				std::max((unsigned) 1, (unsigned) (x.rows() * k)));
		unsigned test_row_num = x.rows() - training_row_num;
		std::vector<unsigned> training_rows(x.rows());
		for (unsigned i = 0; i < (unsigned) x.rows(); i++) training_rows[i] = i;
		std::random_shuffle(training_rows.begin(), training_rows.end());
		std::vector<unsigned> test_rows(test_row_num);
		for (unsigned i = 0; i < test_row_num; i++) {
			test_rows[i] = training_rows[i];
			training_rows.erase(training_rows.begin() + i);
		}
		Scalar prev_test_loss = std::numeric_limits<Scalar>::max();
		unsigned cons_loss_inc = 0;
		NeuralNetwork<Scalar>& test_net(net);
		for (unsigned i = 0; i <= epochs; i++) {
			std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(test_net);
			std::cout << "Epoch " << std::setw(4) << i << "----------------------------" << std::endl;
			// Train.
			if (i != 0) {
				Scalar training_loss = 0;
				unsigned batch_ind = 0;
				Matrix<Scalar> batch_x(batch_size, x.cols());
				Matrix<Scalar> batch_y(batch_size, y.cols());
				for (unsigned j = 0; j < training_row_num; j ++) {
					unsigned row = training_rows[j];
					batch_x.row(batch_ind) = x.row(row);
					batch_y.row(batch_ind) = y.row(row);
					batch_ind++;
					if (batch_ind == batch_size || j == training_row_num - 1) {
						training_loss += (batch_ind == batch_size ?
								Optimizer<Scalar>::compute_training_loss_and_backprop(test_net, batch_x, batch_y) :
								Optimizer<Scalar>::compute_training_loss_and_backprop(test_net,
										batch_x.topRows(batch_ind), batch_y.topRows(batch_ind))).sum();
						for (unsigned k = 0; k < layers.size(); k++) {
							Layer<Scalar>* layer_ptr = layers[k];
							update_params(layer_ptr, k, i - 1);
							Optimizer<Scalar>::enforce_constraints(layer_ptr);
						}
						batch_ind = 0;
					}
				}
				Scalar mean_training_loss = training_loss / training_row_num;
				std::cout << "\ttraining loss: " << std::to_string(mean_training_loss) << std::endl;
				Optimizer<Scalar>::empty_layer_caches(test_net);
			}
			// Validate.
			Scalar obj_loss = 0;
			unsigned batch_ind = 0;
			Matrix<Scalar> test_batch_x(batch_size, x.cols());
			Matrix<Scalar> test_batch_y(batch_size, y.cols());
			for (unsigned j = 0; j < test_row_num; j ++) {
				unsigned row = test_rows[j];
				test_batch_x.row(batch_ind) = x.row(row);
				test_batch_y.row(batch_ind) = y.row(row);
				batch_ind++;
				if (batch_ind == batch_size || j == test_row_num - 1) {
					obj_loss += (batch_ind == batch_size ?
							Optimizer<Scalar>::loss.function(net.infer(test_batch_x), test_batch_y) :
							Optimizer<Scalar>::loss.function(net.infer(test_batch_x.topRows(batch_ind)),
									test_batch_y.topRows(batch_ind))).sum();
					batch_ind = 0;
				}
			}
			Scalar mean_obj_loss = obj_loss / test_row_num;
			Scalar reg_loss = 0;
			for (unsigned j = 0; j < layers.size(); j++)
				reg_loss += reg.function(Optimizer<Scalar>::get_weights(layers[j]));
			Scalar test_loss = mean_obj_loss + reg_loss;
			std::cout << "\tobj loss: " << std::to_string(mean_obj_loss) << std::endl;
			std::cout << "\treg loss: " << std::to_string(reg_loss) << std::endl;
			std::cout << "\tvalidation loss: " << std::to_string(test_loss);
			Optimizer<Scalar>::empty_layer_caches(test_net);
			if (test_loss >= prev_test_loss) {
				cons_loss_inc++;
				std::cout << " *****INCREASED LOSS*****";
				if (early_stop > 0 && cons_loss_inc >= early_stop)
					break;
			} else {
				cons_loss_inc = 0;
				net = test_net;
			}
			std::cout << std::endl << std::endl;
			prev_test_loss = test_loss;
		}
	};
protected:
	virtual void fit(NeuralNetwork<Scalar>& net) = 0;
	virtual void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) = 0;
	const RegularizationPenalty<Scalar>& reg;
	unsigned batch_size;
	Scalar k;
};

template<typename Scalar>
class VanillaSGDOptimizer : public SGDOptimizer<Scalar> {
public:
	VanillaSGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar k = 0.8, Scalar learning_rate = 1e-3) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size, k),
				learning_rate(learning_rate) {
		assert(learning_rate > 0);
	};
protected:
	void fit(NeuralNetwork<Scalar>& net) { };
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		Matrix<Scalar>& weights = Optimizer<Scalar>::get_weights(layer_ptr);
		weights -= (learning_rate * (Optimizer<Scalar>::get_weight_grads(layer_ptr) +
				SGDOptimizer<Scalar>::reg.d_function(weights)));
		if (layer_ptr->get_batch_norm()) {
			Optimizer<Scalar>::get_betas(layer_ptr) -= (learning_rate * Optimizer<Scalar>::get_beta_grads(layer_ptr));
			Optimizer<Scalar>::get_gammas(layer_ptr) -= (learning_rate * Optimizer<Scalar>::get_gamma_grads(layer_ptr));
		}
	};
	Scalar learning_rate;
};

template<typename Scalar>
class MomentumAcceleratedSGDOptimizer : public SGDOptimizer<Scalar> {
public:
	MomentumAcceleratedSGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
			unsigned batch_size = 1, Scalar k = 0.8, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3,
			Scalar momentum = .9) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size, k),
				init_learning_rate(init_learning_rate),
				annealing_rate(annealing_rate),
				momentum(momentum) {
		assert(init_learning_rate > 0);
		assert(annealing_rate >= 0);
		assert(momentum > 0 && momentum < 1);
	};
	virtual ~MomentumAcceleratedSGDOptimizer() = default;
protected:
	void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		grads_vec = std::vector<Gradients>(layers.size());
		for (unsigned i = 0; i < grads_vec.size(); i++) {
			Layer<Scalar>* layer_ptr = layers[i];
			Gradients& vel = grads_vec[i];
			vel.weight = Matrix<Scalar>(Optimizer<Scalar>::get_weight_grads(layer_ptr).rows(),
					Optimizer<Scalar>::get_weight_grads(layer_ptr).cols());
			vel.weight.setZero(vel.weight.rows(), vel.weight.cols());
			if (layer_ptr->get_batch_norm()) {
				vel.beta = RowVector<Scalar>(Optimizer<Scalar>::get_beta_grads(layer_ptr).cols());
				vel.beta.setZero(vel.beta.cols());
				vel.gamma = RowVector<Scalar>(Optimizer<Scalar>::get_gamma_grads(layer_ptr).cols());
				vel.gamma.setZero(vel.gamma.cols());
			}
		}
	};
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		Scalar learning_rate = calculate_learning_rate(epoch);
		Gradients& grads = grads_vec[i];
		Matrix<Scalar>& weights = Optimizer<Scalar>::get_weights(layer_ptr);
		grads.weight = momentum * grads.weight - learning_rate * (Optimizer<Scalar>::get_weight_grads(layer_ptr) +
				SGDOptimizer<Scalar>::reg.d_function(weights));
		weights += grads.weight;
		if (layer_ptr->get_batch_norm()) {
			grads.beta = momentum * grads.beta - learning_rate * Optimizer<Scalar>::get_beta_grads(layer_ptr);
			Optimizer<Scalar>::get_betas(layer_ptr) += grads.beta;
			grads.gamma = momentum * grads.gamma - learning_rate * Optimizer<Scalar>::get_gamma_grads(layer_ptr);
			Optimizer<Scalar>::get_gammas(layer_ptr) += grads.gamma;
		}
	};
	Scalar calculate_learning_rate(unsigned epoch) {
		return init_learning_rate / (1.0 + annealing_rate * epoch);
	};
	Scalar init_learning_rate;
	Scalar annealing_rate;
	Scalar momentum;
	struct Gradients {
		Matrix<Scalar> weight;
		RowVector<Scalar> beta;
		RowVector<Scalar> gamma;
	};
	std::vector<Gradients> grads_vec;
};

template<typename Scalar>
class NesterovMomentumAcceleratedSGDOptimizer : public MomentumAcceleratedSGDOptimizer<Scalar> {
public:
	NesterovMomentumAcceleratedSGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
			unsigned batch_size = 1, Scalar k = 0.8, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3,
			Scalar momentum = .9) :
				MomentumAcceleratedSGDOptimizer<Scalar>::MomentumAcceleratedSGDOptimizer(loss, reg, batch_size,
						k, init_learning_rate, annealing_rate, momentum) { };
protected:
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		Scalar learning_rate = MomentumAcceleratedSGDOptimizer<Scalar>::calculate_learning_rate(epoch);
		typename MomentumAcceleratedSGDOptimizer<Scalar>::Gradients& grads =
				MomentumAcceleratedSGDOptimizer<Scalar>::grads_vec[i];
		Matrix<Scalar>& weights = Optimizer<Scalar>::get_weights(layer_ptr);
		Matrix<Scalar> acc_weight_grads = grads.weight;
		grads.weight = MomentumAcceleratedSGDOptimizer<Scalar>::momentum * grads.weight -
				learning_rate * (Optimizer<Scalar>::get_weight_grads(layer_ptr) +
				SGDOptimizer<Scalar>::reg.d_function(weights));
		weights += -MomentumAcceleratedSGDOptimizer<Scalar>::momentum * acc_weight_grads +
				(1 + MomentumAcceleratedSGDOptimizer<Scalar>::momentum) * grads.weight;
		if (layer_ptr->get_batch_norm()) {
			RowVector<Scalar> acc_beta_grads = grads.beta;
			grads.beta = MomentumAcceleratedSGDOptimizer<Scalar>::momentum * grads.beta -
					learning_rate * Optimizer<Scalar>::get_beta_grads(layer_ptr);
			Optimizer<Scalar>::get_betas(layer_ptr) += -MomentumAcceleratedSGDOptimizer<Scalar>::momentum * acc_beta_grads +
					(1 + MomentumAcceleratedSGDOptimizer<Scalar>::momentum) * grads.beta;
			RowVector<Scalar> acc_gamma_grads = grads.gamma;
			grads.gamma = MomentumAcceleratedSGDOptimizer<Scalar>::momentum * grads.gamma -
					learning_rate * Optimizer<Scalar>::get_gamma_grads(layer_ptr);
			Optimizer<Scalar>::get_gammas(layer_ptr) += -MomentumAcceleratedSGDOptimizer<Scalar>::momentum * acc_gamma_grads +
					(1 + MomentumAcceleratedSGDOptimizer<Scalar>::momentum) * grads.gamma;
		}
	};
};

template<typename Scalar>
class AdagradOptimizer : public SGDOptimizer<Scalar> {
public:
	AdagradOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar k = 0.8, Scalar learning_rate = 1e-2, Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size, k),
				learning_rate(learning_rate),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(epsilon > 0);
	};
	virtual ~AdagradOptimizer() = default;
protected:
	void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		grads_vec = std::vector<GradientSquares>(layers.size());
		for (unsigned i = 0; i < grads_vec.size(); i++) {
			Layer<Scalar>* layer_ptr = layers[i];
			GradientSquares& grad_sqrs = grads_vec[i];
			grad_sqrs.weight = Matrix<Scalar>(Optimizer<Scalar>::get_weight_grads(layer_ptr).rows(),
					Optimizer<Scalar>::get_weight_grads(layer_ptr).cols());
			grad_sqrs.weight.setZero(grad_sqrs.weight.rows(), grad_sqrs.weight.cols());
			if (layer_ptr->get_batch_norm()) {
				grad_sqrs.beta = RowVector<Scalar>(Optimizer<Scalar>::get_beta_grads(layer_ptr).cols());
				grad_sqrs.beta.setZero(grad_sqrs.beta.cols());
				grad_sqrs.gamma = RowVector<Scalar>(Optimizer<Scalar>::get_gamma_grads(layer_ptr).cols());
				grad_sqrs.gamma.setZero(grad_sqrs.gamma.cols());
			}
		}
	};
	virtual void update_acc_weight_grad_sqrs(Matrix<Scalar>& acc_weight_grad_sqrs,
			const Matrix<Scalar>& weight_grads) {
		acc_weight_grad_sqrs += weight_grads.cwiseProduct(weight_grads);
	};
	virtual void update_acc_batch_norm_grad_sqrs(RowVector<Scalar>& acc_batch_norm_grad_sqrs,
			const RowVector<Scalar>& batch_norm_grads) {
		acc_batch_norm_grad_sqrs += batch_norm_grads.cwiseProduct(batch_norm_grads);
	};
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		GradientSquares& grad_sqrs = grads_vec[i];
		Matrix<Scalar>& weights = Optimizer<Scalar>::get_weights(layer_ptr);
		Matrix<Scalar> weight_grads = Optimizer<Scalar>::get_weight_grads(layer_ptr) +
				SGDOptimizer<Scalar>::reg.d_function(weights);
		update_acc_weight_grad_sqrs(grad_sqrs.weight, weight_grads);
		weights -= (learning_rate * weight_grads.array() / (grad_sqrs.weight.array().sqrt() + epsilon)).matrix();
		if (layer_ptr->get_batch_norm()) {
			const RowVector<Scalar>& beta_grads = Optimizer<Scalar>::get_beta_grads(layer_ptr);
			update_acc_batch_norm_grad_sqrs(grad_sqrs.beta, beta_grads);
			Optimizer<Scalar>::get_betas(layer_ptr) -= (learning_rate * beta_grads.array() /
					(grad_sqrs.beta.array().sqrt() + epsilon)).matrix();
			const RowVector<Scalar>& gamma_grads = Optimizer<Scalar>::get_gamma_grads(layer_ptr);
			update_acc_batch_norm_grad_sqrs(grad_sqrs.gamma, gamma_grads);
			Optimizer<Scalar>::get_gammas(layer_ptr) -= (learning_rate * gamma_grads.array() /
					(grad_sqrs.gamma.array().sqrt() + epsilon)).matrix();
		}
	};
	Scalar learning_rate;
	Scalar epsilon;
	struct GradientSquares {
		Matrix<Scalar> weight;
		RowVector<Scalar> beta;
		RowVector<Scalar> gamma;
	};
	std::vector<GradientSquares> grads_vec;
};

template<typename Scalar>
class RMSPropOptimizer : public AdagradOptimizer<Scalar> {
public:
	RMSPropOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
			unsigned batch_size = 1, Scalar k = 0.8, Scalar learning_rate = 1e-3, Scalar l2_decay = 1e-1,
			Scalar epsilon = Utils<Scalar>::EPSILON) :
				AdagradOptimizer<Scalar>::AdagradOptimizer(loss, reg, batch_size, k, learning_rate, epsilon),
				l2_decay(l2_decay) {
		assert(l2_decay >= 0 && l2_decay <= 1);
	};
protected:
	void update_acc_weight_grad_sqrs(Matrix<Scalar>& acc_weight_grad_sqrs,
			const Matrix<Scalar>& weight_grads) {
		acc_weight_grad_sqrs = (1 - l2_decay) * acc_weight_grad_sqrs + l2_decay * weight_grads.cwiseProduct(weight_grads);
	};
	void update_acc_batch_norm_grad_sqrs(RowVector<Scalar>& acc_batch_norm_grad_sqrs,
			const RowVector<Scalar>& batch_norm_grads) {
		acc_batch_norm_grad_sqrs = (1 - l2_decay) * acc_batch_norm_grad_sqrs +
				l2_decay * batch_norm_grads.cwiseProduct(batch_norm_grads);
	};
	Scalar l2_decay;
};

template<typename Scalar>
class AdadeltaOptimizer : public SGDOptimizer<Scalar> {
public:
	AdadeltaOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar k = 0.8, Scalar decay = 5e-2, Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size, k),
				decay(decay),
				epsilon(epsilon) {
		assert(decay >= 0 && decay <= 1);
		assert(epsilon > 0);
	};
protected:
	void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		gus_vec = std::vector<GradientAndUpdateSquares>(layers.size());
		for (unsigned i = 0; i < gus_vec.size(); i++) {
			Layer<Scalar>* layer_ptr = layers[i];
			GradientAndUpdateSquares& gus = gus_vec[i];
			gus.weight_grad = Matrix<Scalar>(Optimizer<Scalar>::get_weight_grads(layer_ptr).rows(),
					Optimizer<Scalar>::get_weight_grads(layer_ptr).cols());
			gus.weight_grad.setZero(gus.weight_grad.rows(), gus.weight_grad.cols());
			gus.weight_update = Matrix<Scalar>(gus.weight_grad.rows(), gus.weight_grad.cols());
			gus.weight_update.setZero(gus.weight_update.rows(), gus.weight_update.cols());
			if (layer_ptr->get_batch_norm()) {
				gus.beta_grad = RowVector<Scalar>(Optimizer<Scalar>::get_beta_grads(layer_ptr).cols());
				gus.beta_grad.setZero(gus.beta_grad.cols());
				gus.beta_update = RowVector<Scalar>(gus.beta_grad.cols());
				gus.beta_update.setZero(gus.beta_update.cols());
				gus.gamma_grad = RowVector<Scalar>(Optimizer<Scalar>::get_gamma_grads(layer_ptr).cols());
				gus.gamma_grad.setZero(gus.gamma_grad.cols());
				gus.gamma_update = RowVector<Scalar>(gus.gamma_grad.cols());
				gus.gamma_update.setZero(gus.gamma_update.cols());
			}
		}
	};
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		GradientAndUpdateSquares& gus = gus_vec[i];
		Matrix<Scalar>& weights = Optimizer<Scalar>::get_weights(layer_ptr);
		Matrix<Scalar> weight_grads = Optimizer<Scalar>::get_weight_grads(layer_ptr) +
				SGDOptimizer<Scalar>::reg.d_function(weights);
		gus.weight_grad = (1 - decay) * gus.weight_grad + decay * weight_grads.cwiseProduct(weight_grads);
		Matrix<Scalar> weight_updates = -weight_grads.array() * (gus.weight_update.array() + epsilon).sqrt() /
				(gus.weight_grad.array() + epsilon).sqrt();
		weights += weight_updates;
		gus.weight_update = (1 - decay) * gus.weight_update + decay * weight_updates.cwiseProduct(weight_updates);
		if (layer_ptr->get_batch_norm()) {
			const RowVector<Scalar>& beta_grads = Optimizer<Scalar>::get_beta_grads(layer_ptr);
			gus.beta_grad = (1 - decay) * gus.beta_grad + decay * beta_grads.cwiseProduct(beta_grads);
			Matrix<Scalar> beta_updates = -beta_grads.array() * (gus.beta_update.array() + epsilon).sqrt() /
					(gus.beta_grad.array() + epsilon).sqrt();
			Optimizer<Scalar>::get_betas(layer_ptr) += beta_updates;
			gus.beta_update = (1 - decay) * gus.beta_update + decay * beta_updates.cwiseProduct(beta_updates);
			const RowVector<Scalar>& gamma_grads = Optimizer<Scalar>::get_gamma_grads(layer_ptr);
			gus.gamma_grad = (1 - decay) * gus.gamma_grad + decay * gamma_grads.cwiseProduct(gamma_grads);
			Matrix<Scalar> gamma_updates = -beta_grads.array() * (gus.gamma_update.array() + epsilon).sqrt() /
					(gus.gamma_grad.array() + epsilon).sqrt();
			Optimizer<Scalar>::get_gammas(layer_ptr) += gamma_updates;
			gus.gamma_update = (1 - decay) * gus.gamma_update + decay * gamma_updates.cwiseProduct(gamma_updates);
		}
	};
	Scalar decay;
	Scalar epsilon;
	struct GradientAndUpdateSquares {
		Matrix<Scalar> weight_grad;
		Matrix<Scalar> weight_update;
		RowVector<Scalar> beta_grad;
		RowVector<Scalar> beta_update;
		RowVector<Scalar> gamma_grad;
		RowVector<Scalar> gamma_update;
	};
	std::vector<GradientAndUpdateSquares> gus_vec;
};

template<typename Scalar>
class AdamOptimizer : public SGDOptimizer<Scalar> {
public:
	AdamOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar k = 0.8, Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size, k),
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
	void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		grad_norms_vec = std::vector<GradientNorms>(layers.size());
		for (unsigned i = 0; i < grad_norms_vec.size(); i++) {
			Layer<Scalar>* layer_ptr = layers[i];
			GradientNorms& vel = grad_norms_vec[i];
			vel.weight_l1 = Matrix<Scalar>(Optimizer<Scalar>::get_weight_grads(layer_ptr).rows(),
					Optimizer<Scalar>::get_weight_grads(layer_ptr).cols());
			vel.weight_l1.setZero(vel.weight_l1.rows(), vel.weight_l1.cols());
			vel.weight_l2 = Matrix<Scalar>(vel.weight_l1.rows(), vel.weight_l1.cols());
			vel.weight_l2.setZero(vel.weight_l2.rows(), vel.weight_l2.cols());
			if (layer_ptr->get_batch_norm()) {
				vel.beta_l1 = RowVector<Scalar>(Optimizer<Scalar>::get_beta_grads(layer_ptr).cols());
				vel.beta_l1.setZero(vel.beta_l1.cols());
				vel.beta_l2 = RowVector<Scalar>(vel.beta_l1.cols());
				vel.beta_l2.setZero(vel.beta_l2.cols());
				vel.gamma_l1 = RowVector<Scalar>(Optimizer<Scalar>::get_gamma_grads(layer_ptr).cols());
				vel.gamma_l1.setZero(vel.gamma_l1.cols());
				vel.gamma_l2 = RowVector<Scalar>(vel.gamma_l1.cols());
				vel.gamma_l2.setZero(vel.gamma_l2.cols());
			}
		}
	};
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		GradientNorms& grad_norms = grad_norms_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - l1_decay, epoch + 1) + epsilon);
		Scalar l2_corr = 1.0 / (1.0 - pow(1.0 - l2_decay, epoch + 1) + epsilon);
		Matrix<Scalar>& weights = Optimizer<Scalar>::get_weights(layer_ptr);
		Matrix<Scalar> weight_grads = Optimizer<Scalar>::get_weight_grads(layer_ptr) +
				SGDOptimizer<Scalar>::reg.d_function(weights);
		grad_norms.weight_l1 = (1 - l1_decay) * grad_norms.weight_l1 + l1_decay * weight_grads;
		grad_norms.weight_l2 = (1 - l2_decay) * grad_norms.weight_l2 +
				l2_decay * weight_grads.cwiseProduct(weight_grads);
		weights -= (learning_rate * (grad_norms.weight_l1 * l1_corr).array() /
				((grad_norms.weight_l2 * l2_corr).array().sqrt() + epsilon)).matrix();
		if (layer_ptr->get_batch_norm()) {
			const RowVector<Scalar>& beta_grads = Optimizer<Scalar>::get_beta_grads(layer_ptr);
			grad_norms.beta_l1 = (1 - l1_decay) * grad_norms.beta_l1 + l1_decay * beta_grads;
			grad_norms.beta_l2 = (1 - l2_decay) * grad_norms.beta_l2 + l2_decay * beta_grads.cwiseProduct(beta_grads);
			Optimizer<Scalar>::get_betas(layer_ptr) -= (learning_rate * (grad_norms.beta_l1 * l1_corr).array() /
					((grad_norms.beta_l2 * l2_corr).array().sqrt() + epsilon)).matrix();
			const RowVector<Scalar>& gamma_grads = Optimizer<Scalar>::get_gamma_grads(layer_ptr);
			grad_norms.gamma_l1 = (1 - l1_decay) * grad_norms.gamma_l1 + l1_decay * gamma_grads;
			grad_norms.gamma_l2 = (1 - l2_decay) * grad_norms.gamma_l2 + l2_decay * gamma_grads.cwiseProduct(gamma_grads);
			Optimizer<Scalar>::get_betas(layer_ptr) -= (learning_rate * (grad_norms.gamma_l1 * l1_corr).array() /
					((grad_norms.gamma_l2 * l2_corr).array().sqrt() + epsilon)).matrix();
		}
	};
	Scalar learning_rate;
	Scalar l1_decay;
	Scalar l2_decay;
	Scalar epsilon;
	struct GradientNorms {
		Matrix<Scalar> weight_l1;
		Matrix<Scalar> weight_l2;
		RowVector<Scalar> beta_l1;
		RowVector<Scalar> beta_l2;
		RowVector<Scalar> gamma_l1;
		RowVector<Scalar> gamma_l2;
	};
	std::vector<GradientNorms> grad_norms_vec;
};

template<typename Scalar>
class AdaMaxOptimizer : public AdamOptimizer<Scalar> {
public:
	AdaMaxOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar k = 0.8, Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				AdamOptimizer<Scalar>::AdamOptimizer(loss, reg, batch_size, k, learning_rate, l1_decay,
						l2_decay, epsilon) { };
protected:
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		typename AdamOptimizer<Scalar>::GradientNorms& grad_norms = AdamOptimizer<Scalar>::grad_norms_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Matrix<Scalar>& weights = Optimizer<Scalar>::get_weights(layer_ptr);
		Matrix<Scalar> weight_grads = Optimizer<Scalar>::get_weight_grads(layer_ptr) +
				SGDOptimizer<Scalar>::reg.d_function(weights);
		grad_norms.weight_l1 = (1 - AdamOptimizer<Scalar>::l1_decay) * grad_norms.weight_l1 +
				AdamOptimizer<Scalar>::l1_decay * weight_grads;
		grad_norms.weight_l2 = ((1 - AdamOptimizer<Scalar>::l2_decay) * grad_norms.weight_l2)
				.cwiseMax(weight_grads.cwiseAbs());
		weights -= (AdamOptimizer<Scalar>::learning_rate * (grad_norms.weight_l1 * l1_corr).array() /
				(grad_norms.weight_l2.array() + AdamOptimizer<Scalar>::epsilon)).matrix();
		if (layer_ptr->get_batch_norm()) {
			const RowVector<Scalar>& beta_grads = Optimizer<Scalar>::get_beta_grads(layer_ptr);
			grad_norms.beta_l1 = (1 - AdamOptimizer<Scalar>::l1_decay) * grad_norms.beta_l1 +
					AdamOptimizer<Scalar>::l1_decay * beta_grads;
			grad_norms.beta_l2 = ((1 - AdamOptimizer<Scalar>::l2_decay) * grad_norms.beta_l2)
					.cwiseMax(beta_grads.cwiseAbs());
			Optimizer<Scalar>::get_betas(layer_ptr) -= (AdamOptimizer<Scalar>::learning_rate *
					(grad_norms.beta_l1 * l1_corr).array() /
					(grad_norms.beta_l2.array() + AdamOptimizer<Scalar>::epsilon)).matrix();
			const RowVector<Scalar>& gamma_grads = Optimizer<Scalar>::get_gamma_grads(layer_ptr);
			grad_norms.gamma_l1 = (1 - AdamOptimizer<Scalar>::l1_decay) * grad_norms.gamma_l1 +
					AdamOptimizer<Scalar>::l1_decay * gamma_grads;
			grad_norms.gamma_l2 = ((1 - AdamOptimizer<Scalar>::l2_decay) * grad_norms.gamma_l2)
					.cwiseMax(gamma_grads.cwiseAbs());
			Optimizer<Scalar>::get_gammas(layer_ptr) -= (AdamOptimizer<Scalar>::learning_rate *
					(grad_norms.gamma_l1 * l1_corr).array() /
					(grad_norms.gamma_l2.array() + AdamOptimizer<Scalar>::epsilon)).matrix();
		}
	};
};

template<typename Scalar>
class NadamOptimizer : public AdamOptimizer<Scalar> {
public:
	NadamOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar k = 0.8, Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				AdamOptimizer<Scalar>::AdamOptimizer(loss, reg, batch_size, k, learning_rate, l1_decay,
						l2_decay, epsilon) { };
protected:
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		typename AdamOptimizer<Scalar>::GradientNorms& grad_norms = AdamOptimizer<Scalar>::grad_norms_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Scalar l1_next_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 2) +
				AdamOptimizer<Scalar>::epsilon);
		Scalar l2_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l2_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Matrix<Scalar>& weights = Optimizer<Scalar>::get_weights(layer_ptr);
		Matrix<Scalar> weight_grads = Optimizer<Scalar>::get_weight_grads(layer_ptr) +
				SGDOptimizer<Scalar>::reg.d_function(weights);
		grad_norms.weight_l1 = (1 - AdamOptimizer<Scalar>::l1_decay) * grad_norms.weight_l1 +
				AdamOptimizer<Scalar>::l1_decay * weight_grads;
		grad_norms.weight_l2 = (1 - AdamOptimizer<Scalar>::l2_decay) * grad_norms.weight_l2 +
				AdamOptimizer<Scalar>::l2_decay * weight_grads.cwiseProduct(weight_grads);
		weights -= (AdamOptimizer<Scalar>::learning_rate * (AdamOptimizer<Scalar>::l1_decay * l1_corr * weight_grads +
				(1.0 - AdamOptimizer<Scalar>::l1_decay) * l1_next_corr * grad_norms.weight_l1).array() /
				((grad_norms.weight_l2 * l2_corr).array().sqrt() + AdamOptimizer<Scalar>::epsilon)).matrix();
		if (layer_ptr->get_batch_norm()) {
			const RowVector<Scalar>& beta_grads = Optimizer<Scalar>::get_beta_grads(layer_ptr);
			grad_norms.beta_l1 = (1 - AdamOptimizer<Scalar>::l1_decay) * grad_norms.beta_l1 +
					AdamOptimizer<Scalar>::l1_decay * beta_grads;
			grad_norms.beta_l2 = (1 - AdamOptimizer<Scalar>::l2_decay) * grad_norms.beta_l2 +
					AdamOptimizer<Scalar>::l2_decay * beta_grads.cwiseProduct(beta_grads);
			Optimizer<Scalar>::get_betas(layer_ptr) -= (AdamOptimizer<Scalar>::learning_rate *
					(AdamOptimizer<Scalar>::l1_decay * l1_corr * beta_grads +
					(1.0 - AdamOptimizer<Scalar>::l1_decay) * l1_next_corr * grad_norms.beta_l1).array() /
					(grad_norms.beta_l2.array() + AdamOptimizer<Scalar>::epsilon)).matrix();
			const RowVector<Scalar>& gamma_grads = Optimizer<Scalar>::get_gamma_grads(layer_ptr);
			grad_norms.gamma_l1 = (1 - AdamOptimizer<Scalar>::l1_decay) * grad_norms.gamma_l1 +
					AdamOptimizer<Scalar>::l1_decay * gamma_grads;
			grad_norms.gamma_l2 = (1 - AdamOptimizer<Scalar>::l2_decay) * grad_norms.gamma_l2 +
					AdamOptimizer<Scalar>::l2_decay * gamma_grads.cwiseProduct(gamma_grads);
			Optimizer<Scalar>::get_gammas(layer_ptr) -= (AdamOptimizer<Scalar>::learning_rate *
					(AdamOptimizer<Scalar>::l1_decay * l1_corr * gamma_grads +
					(1.0 - AdamOptimizer<Scalar>::l1_decay) * l1_next_corr * grad_norms.gamma_l1).array() /
					(grad_norms.gamma_l2.array() + AdamOptimizer<Scalar>::epsilon)).matrix();
		}
	};
};

// TODO: Conjugate Gradient, L-BFGS, LMA, Particle Swarm, GA, PBIL

} /* namespace cppnn */

#endif /* OPTIMIZER_H_ */
