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
			unsigned epochs) = 0;
	bool verify_gradients(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x,
			const Matrix<Scalar>& y, Scalar step_size = 1e-5, Scalar abs_epsilon = 1e-10,
			Scalar rel_epsilon = 1e-4) const {
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
					if (!almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
						std::cout << " *****FAIL*****";
						failure = true;
					}
					std::cout << std::endl;
				}
			}
			if (layer.get_batch_norm()) {
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
					if (!almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
						std::cout << " *****FAIL*****";
						failure = true;
					}
					std::cout << std::endl;
				}
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
					if (!almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
						std::cout << " *****FAIL*****";
						failure = true;
					}
					std::cout << std::endl;
				}
			}
		}
		return !failure;
	};
protected:
	void empty_layer_caches(NeuralNetwork<Scalar>& net) const {
		for (unsigned i = 0; i < net.get_layers().size(); i++) {
			net.get_layers()[i]->empty_cache();
		}
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
	static void max_norm(Layer<Scalar>* layer_ptr, Scalar constraint) {
		Matrix<Scalar>& weights = layer_ptr->get_weights();
		Scalar l2_norm = weights.squaredNorm();
		if (l2_norm > constraint)
			weights *= (constraint / l2_norm);
	};
	const Loss<Scalar>& loss;
};

template<typename Scalar>
class SGDOptimizer : public Optimizer<Scalar> {
public:
	SGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, Scalar k,
			unsigned batch_size, Scalar max_norm_constraint) :
				Optimizer<Scalar>::Optimizer(loss),
				reg(reg),
				k(k),
				batch_size(batch_size),
				max_norm_constraint(max_norm_constraint),
				max_norm(decidedly_greater(max_norm_constraint, .0)) {
		assert(batch_size > 0);
		assert(k > 0 && k < 1);
	};
	virtual ~SGDOptimizer() = default;
	void train(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x, const Matrix<Scalar>& y,
				unsigned epochs = 1000) {
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
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		Scalar prev_total_loss = std::numeric_limits<Scalar>::max();
		for (unsigned i = 0; i <= epochs; i++) {
			std::cout << "Epoch " << std::setw(2) << i << "----------------------------" << std::endl;
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
								Optimizer<Scalar>::compute_training_loss_and_backprop(net, batch_x, batch_y) :
								Optimizer<Scalar>::compute_training_loss_and_backprop(net,
								batch_x.topRows(batch_ind), batch_y.topRows(batch_ind))).sum();
						for (unsigned k = 0; k < layers.size(); k++) {
							Layer<Scalar>* layer_ptr = layers[k];
							update_params(layer_ptr, k, i - 1);
							// Max norm constraint.
							if (max_norm)
								Optimizer<Scalar>::max_norm(layer_ptr, max_norm_constraint);
						}
						batch_ind = 0;
					}
				}
				Scalar mean_training_loss = training_loss / training_row_num;
				std::cout << "\ttraining loss: " << std::to_string(mean_training_loss) << std::endl;
				Optimizer<Scalar>::empty_layer_caches(net);
			}
			// Validate.
			Scalar test_loss = 0;
			unsigned batch_ind = 0;
			Matrix<Scalar> test_batch_x(batch_size, x.cols());
			Matrix<Scalar> test_batch_y(batch_size, y.cols());
			for (unsigned j = 0; j < test_row_num; j ++) {
				unsigned row = test_rows[j];
				test_batch_x.row(batch_ind) = x.row(row);
				test_batch_y.row(batch_ind) = y.row(row);
				batch_ind++;
				if (batch_ind == batch_size || j == test_row_num - 1) {
					test_loss += (batch_ind == batch_size ?
							Optimizer<Scalar>::loss.function(net.infer(test_batch_x), test_batch_y) :
							Optimizer<Scalar>::loss.function(net.infer(test_batch_x.topRows(batch_ind)),
							test_batch_y.topRows(batch_ind))).sum();
					batch_ind = 0;
				}
			}
			Scalar mean_test_loss = test_loss / test_row_num;
			Scalar reg_loss = 0;
			std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
			for (unsigned j = 0; j < layers.size(); j++) {
				reg_loss += reg.function(Optimizer<Scalar>::get_weights(layers[j]));
			}
			Scalar total_test_loss = mean_test_loss + reg_loss;
			std::cout << "\ttest loss: " << std::to_string(mean_test_loss) << std::endl;
			std::cout << "\treg loss: " << std::to_string(reg_loss) << std::endl;
			std::cout << "\ttotal test loss: " << std::to_string(total_test_loss) << std::endl << std::endl;
			Optimizer<Scalar>::empty_layer_caches(net);
			// If the test loss increases, terminate early to prevent overfitting.
			if (total_test_loss >= prev_total_loss)
				std::cout << "*****INCREASED LOSS*****" << std::endl << std::endl;
			prev_total_loss = total_test_loss;
		}
	};
protected:
	virtual void fit(NeuralNetwork<Scalar>& net) = 0;
	virtual void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) = 0;
	const RegularizationPenalty<Scalar>& reg;
	Scalar k;
	unsigned batch_size;
	Scalar max_norm_constraint;
	bool max_norm;
};

template<typename Scalar>
class VanillaSGDOptimizer : public SGDOptimizer<Scalar> {
public:
	VanillaSGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, Scalar learning_rate = 1e-3,
			Scalar k = 0.8, unsigned batch_size = 1, Scalar max_norm_constraint = 0) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, k, batch_size, max_norm_constraint),
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
			Scalar init_learning_rate = 1e-3, Scalar decay = 1e-3, Scalar momentum = .9, Scalar k = 0.8,
			unsigned batch_size = 1, Scalar max_norm_constraint = 0) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, k, batch_size, max_norm_constraint),
				init_learning_rate(init_learning_rate),
				decay(decay),
				momentum(momentum) {
		assert(init_learning_rate > 0);
		assert(decay >= 0);
		assert(momentum > 0 && momentum < 1);
	};
	virtual ~MomentumAcceleratedSGDOptimizer() = default;
protected:
	void fit(NeuralNetwork<Scalar>& net) {
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		grads_vec = std::vector<AcceleratedGradients>(layers.size());
		for (unsigned i = 0; i < grads_vec.size(); i++) {
			Layer<Scalar>* layer_ptr = layers[i];
			AcceleratedGradients& vel = grads_vec[i];
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
		AcceleratedGradients& grads = grads_vec[i];
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
		return init_learning_rate / (1.0 + decay * epoch);
	};
	Scalar init_learning_rate;
	Scalar decay;
	Scalar momentum;
	struct AcceleratedGradients {
		Matrix<Scalar> weight;
		RowVector<Scalar> beta;
		RowVector<Scalar> gamma;
	};
	std::vector<AcceleratedGradients> grads_vec;
};

template<typename Scalar>
class NesterovMomentumAcceleratedSGDOptimizer : public MomentumAcceleratedSGDOptimizer<Scalar> {
public:
	NesterovMomentumAcceleratedSGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
			Scalar init_learning_rate = 1e-3, Scalar decay = 1e-3, Scalar momentum = .9, Scalar k = 0.8,
			unsigned batch_size = 1, Scalar max_norm_constraint = 0) :
				MomentumAcceleratedSGDOptimizer<Scalar>::MomentumDrivenSGDOptimizer(loss, reg, init_learning_rate,
				decay, momentum, k, batch_size, max_norm_constraint) { };
protected:
	void update_params(Layer<Scalar>* layer_ptr, unsigned i, unsigned epoch) {
		Scalar learning_rate = MomentumAcceleratedSGDOptimizer<Scalar>::calculate_learning_rate(epoch);
		typename MomentumAcceleratedSGDOptimizer<Scalar>::AcceleratedGradients& grads =
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

// TODO: RMSProp, Adagrad, Nadam, Conjugate Gradient, L-BFGS, LMA, EA, and Particle Swarm

} /* namespace cppnn */

#endif /* OPTIMIZER_H_ */
