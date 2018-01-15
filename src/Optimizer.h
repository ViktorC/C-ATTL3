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
#include <Tensor.h>
#include <Utils.h>
#include <vector>
#include <Vector.h>
#include <WeightInitialization.h>

#define X_DIMS x.dimension(1), x.dimension(2), x.dimension(3)
#define Y_DIMS y.dimension(1), y.dimension(2), y.dimension(3)

namespace cppnn {

template<typename Scalar>
class Optimizer {
public:
	Optimizer(const Loss<Scalar>& loss) :
				loss(loss) { };
	virtual ~Optimizer() = default;
	bool verify_gradients(NeuralNetwork<Scalar>& net, const Tensor4D<Scalar>& x,
			const Tensor4D<Scalar>& y, Scalar step_size = 1e-5, Scalar abs_epsilon = Utils<Scalar>::EPSILON2,
			Scalar rel_epsilon = Utils<Scalar>::EPSILON3) const {
		int rows = x.dimension(0);
		assert(rows > 0);
		assert((unsigned) x.dimension(1) == net.get_input_dims().get_dim1() &&
				(unsigned) x.dimension(2) == net.get_input_dims().get_dim2() &&
				(unsigned) x.dimension(3) == net.get_input_dims().get_dim3());
		assert(rows == y.dimension(0));
		assert(step_size > 0);
		assert(abs_epsilon >= 0 && rel_epsilon > 0);
		net.backpropagate(loss.d_function(net.propagate(x, true), y) / rows);
		bool failure = false;
		for (unsigned i = 0; i < net.get_layers().size(); i++) {
			Layer<Scalar>& layer = *(net.get_layers()[i]);
			if (layer.is_parametric()) {
				std::cout << "Layer " << std::setw(3) << std::to_string(i + 1) <<
						"----------------------------" << std::endl;
				Matrix<Scalar>& params = layer.get_params();
				const Matrix<Scalar>& param_grads = layer.get_param_grads();
				for (int j = 0; j < params.rows(); j++) {
					for (int k = 0; k < params.cols(); k++) {
						std::cout << "\tParam[" << i << "," << j << "," << k << "]:" << std::endl;
						Scalar ana_grad = param_grads(j,k);
						std::cout << "\t\tAnalytic gradient = " << ana_grad << std::endl;
						Scalar param = params(j,k);
						params(j,k) = param + step_size;
						Scalar loss_inc = loss.function(net.propagate(x, true), y).mean();
						params(j,k) = param - step_size;
						Scalar loss_dec = loss.function(net.propagate(x, true), y).mean();
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
		empty_layer_caches(net);
		return !failure;
	};
	void optimize(NeuralNetwork<Scalar>& net, const Tensor4D<Scalar>& x, const Tensor4D<Scalar>& y,
			unsigned epochs, Scalar k = .8, unsigned early_stop = 0) {
		int rows = x.dimension(0);
		assert(rows > 1);
		assert((unsigned) x.dimension(1) == net.get_input_dims().get_dim1() &&
				(unsigned) x.dimension(2) == net.get_input_dims().get_dim2() &&
				(unsigned) x.dimension(3) == net.get_input_dims().get_dim3());
		assert(rows == y.dimension(0));
		assert(epochs > 0);
		assert(k > 0 && k < 1);
		// Fit the optimizer parameters to the network.
		fit(net);
		// Divide the data into training and test partitions.
		unsigned training_row_num = std::min((unsigned) (rows - 1),
				std::max((unsigned) 1, (unsigned) (rows * k)));
		unsigned test_row_num = rows - training_row_num;
		std::vector<unsigned> training_rows(rows);
		for (int i = 0; i < rows; i++) training_rows[i] = (unsigned) i;
		std::random_shuffle(training_rows.begin(), training_rows.end());
		std::vector<unsigned> test_rows(test_row_num);
		for (unsigned i = 0; i < test_row_num; i++) {
			test_rows[i] = training_rows[i];
			training_rows.erase(training_rows.begin() + i);
		}
		Scalar prev_valid_loss = std::numeric_limits<Scalar>::max();
		unsigned cons_loss_inc = 0;
		NeuralNetwork<Scalar>& test_net(net);
		for (unsigned i = 0; i <= epochs; i++) {
			std::cout << "Epoch " << std::setw(4) << i << "----------------------------" << std::endl;
			// Train.
			if (i != 0) {
				Scalar training_loss = train(test_net, x, y, training_rows, i);
				std::cout << "\ttraining loss: " << std::to_string(training_loss) << std::endl;
				empty_layer_caches(test_net);
			}
			// Validate.
			Scalar valid_loss = validate(test_net, x, y, test_rows, i);
			std::cout << "\tvalidation loss: " << std::to_string(valid_loss);
			empty_layer_caches(test_net);
			if (valid_loss >= prev_valid_loss) {
				cons_loss_inc++;
				std::cout << " *****INCREASED LOSS*****";
				if (early_stop > 0 && cons_loss_inc >= early_stop)
					break;
			} else {
				cons_loss_inc = 0;
				net = test_net;
			}
			std::cout << std::endl << std::endl;
			prev_valid_loss = valid_loss;
		}
	};
protected:
	virtual void fit(NeuralNetwork<Scalar>& net) = 0;
	virtual Scalar train(NeuralNetwork<Scalar>& net, const Tensor4D<Scalar>& x, const Tensor4D<Scalar>& y,
			const std::vector<unsigned>& rows, unsigned epoch) = 0;
	virtual Scalar validate(NeuralNetwork<Scalar>& net, const Tensor4D<Scalar>& x, const Tensor4D<Scalar>& y,
			const std::vector<unsigned>& rows, unsigned epoch) = 0;
	void empty_layer_caches(NeuralNetwork<Scalar>& net) const {
		for (unsigned i = 0; i < net.get_layers().size(); i++) {
			net.get_layers()[i]->empty_cache();
		}
	};
	static std::vector<Layer<Scalar>*>& get_layers(NeuralNetwork<Scalar>& net) {
		return net.get_layers();
	};
	static Tensor4D<Scalar> propagate(NeuralNetwork<Scalar>& net, const Tensor4D<Scalar>& x) {
		return net.propagate(x, true);
	};
	static void backpropagate(NeuralNetwork<Scalar>& net, const Tensor4D<Scalar>& grads) {
		net.backpropagate(grads);
	};
	static bool is_parametric(Layer<Scalar>& layer) {
		return layer.is_parametric();
	};
	static Matrix<Scalar>& get_params(Layer<Scalar>& layer) {
		return layer.get_params();
	};
	static const Matrix<Scalar>& get_param_grads(Layer<Scalar>& layer) {
		return layer.get_param_grads();
	};
	static const void enforce_constraints(Layer<Scalar>& layer) {
		layer.enforce_constraints();
	};
	const Loss<Scalar>& loss;
};

template<typename Scalar>
class SGDOptimizer : public Optimizer<Scalar> {
public:
	SGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
			unsigned batch_size) :
				Optimizer<Scalar>::Optimizer(loss),
				reg(reg),
				batch_size(batch_size) {
		assert(batch_size > 0);
	};
	virtual ~SGDOptimizer() = default;
protected:
	Scalar train(NeuralNetwork<Scalar>& net, const Tensor4D<Scalar>& x, const Tensor4D<Scalar>& y,
			const std::vector<unsigned>& rows, unsigned epoch) {
		Scalar training_loss = 0;
		unsigned batch_ind = 0;
		int training_row_num = rows.size();
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		Tensor4D<Scalar> batch_x(batch_size, X_DIMS);
		Tensor4D<Scalar> batch_y(batch_size, Y_DIMS);
		Eigen::array<int, 4> offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> x_extents = { 1, X_DIMS };
		Eigen::array<int, 4> y_extents = { 1, Y_DIMS };
		Eigen::array<int, 4> batch_x_extents = { 0, X_DIMS };
		Eigen::array<int, 4> batch_y_extents = { 0, Y_DIMS };
		for (int j = 0; j < training_row_num; j ++) {
			unsigned row = rows[j];
			offsets[0] = (int) row;
			batch_x.slice(offsets, x_extents) = x.slice(offsets, x_extents);
			batch_y.slice(offsets, y_extents) = y.slice(offsets, y_extents);
			batch_ind++;
			if (batch_ind == batch_size || j == training_row_num - 1) {
				offsets[0] = 0;
				batch_x_extents[0] = (int) batch_ind;
				batch_y_extents[0] = (int) batch_ind;
				Tensor4D<Scalar> out = Optimizer<Scalar>::propagate(net, batch_x.slice(offsets, batch_x_extents));
				training_loss += Optimizer<Scalar>::loss.function(out, batch_y.slice(offsets, batch_y_extents)).sum();
				/* As the loss to minimize is the mean of the losses for all the training observations
				 * (see the last line of the function), the gradient to back-propagate is to be divided by
				 * the number of observations in the batch. */
				Optimizer<Scalar>::backpropagate(net, Optimizer<Scalar>::loss.d_function(out,
						batch_y.slice(offsets, batch_y_extents)) / batch_ind);
				for (unsigned k = 0; k < layers.size(); k++) {
					Layer<Scalar>& layer = *(layers[k]);
					if (Optimizer<Scalar>::is_parametric(layer)) {
						update_params(layer, k, epoch - 1);
						Optimizer<Scalar>::enforce_constraints(layer);
					}
				}
				batch_ind = 0;
			}
		}
		return training_loss / training_row_num;
	};
	Scalar validate(NeuralNetwork<Scalar>& net, const Tensor4D<Scalar>& x, const Tensor4D<Scalar>& y,
			const std::vector<unsigned>& rows, unsigned epoch) {
		Scalar obj_loss = 0;
		unsigned batch_ind = 0;
		int test_row_num = rows.size();
		Tensor4D<Scalar> test_batch_x(batch_size, X_DIMS);
		Tensor4D<Scalar> test_batch_y(batch_size, Y_DIMS);
		Eigen::array<int, 4> offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> x_extents = { 1, X_DIMS };
		Eigen::array<int, 4> y_extents = { 1, Y_DIMS };
		Eigen::array<int, 4> batch_x_extents = { 0, X_DIMS };
		Eigen::array<int, 4> batch_y_extents = { 0, Y_DIMS };
		std::vector<Layer<Scalar>*> layers = Optimizer<Scalar>::get_layers(net);
		for (int j = 0; j < test_row_num; j ++) {
			unsigned row = rows[j];
			offsets[0] = (int) row;
			test_batch_x.slice(offsets, x_extents) = x.slice(offsets, x_extents);
			test_batch_y.slice(offsets, y_extents) = y.slice(offsets, y_extents);
			batch_ind++;
			if (batch_ind == batch_size || j == test_row_num - 1) {
				offsets[0] = 0;
				batch_x_extents[0] = (int) batch_ind;
				batch_y_extents[0] = (int) batch_ind;
				obj_loss += Optimizer<Scalar>::loss.function(net.infer(test_batch_x.slice(offsets, batch_x_extents)),
						test_batch_y.slice(offsets, batch_y_extents)).sum();
				batch_ind = 0;
			}
		}
		Scalar mean_obj_loss = obj_loss / test_row_num;
		Scalar reg_loss = 0;
		for (unsigned j = 0; j < layers.size(); j++)
			reg_loss += reg.function(Optimizer<Scalar>::get_params(*(layers[j])));
		std::cout << "\tobj loss: " << std::to_string(mean_obj_loss) << std::endl;
		std::cout << "\treg loss: " << std::to_string(reg_loss) << std::endl;
		return mean_obj_loss + reg_loss;
	};
	virtual void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) = 0;
	const RegularizationPenalty<Scalar>& reg;
	unsigned batch_size;
};

template<typename Scalar>
class VanillaSGDOptimizer : public SGDOptimizer<Scalar> {
public:
	VanillaSGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size),
				learning_rate(learning_rate) {
		assert(learning_rate > 0);
	};
protected:
	void fit(NeuralNetwork<Scalar>& net) { };
	void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		params -= (learning_rate * (Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg.d_function(params)));
	};
	Scalar learning_rate;
};

template<typename Scalar>
class MomentumAcceleratedSGDOptimizer : public SGDOptimizer<Scalar> {
public:
	MomentumAcceleratedSGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
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
	void fit(NeuralNetwork<Scalar>& net) {
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
	void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		Scalar learning_rate = calculate_learning_rate(epoch);
		Matrix<Scalar>& param_grads = param_grads_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		param_grads = momentum * param_grads - learning_rate * (Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg.d_function(params));
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
	NesterovMomentumAcceleratedSGDOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
			unsigned batch_size = 1, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3,
			Scalar momentum = .9) :
				MomentumAcceleratedSGDOptimizer<Scalar>::MomentumAcceleratedSGDOptimizer(loss, reg, batch_size,
						init_learning_rate, annealing_rate, momentum) { };
protected:
	void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		Scalar learning_rate = MomentumAcceleratedSGDOptimizer<Scalar>::calculate_learning_rate(epoch);
		Matrix<Scalar>& param_grads = MomentumAcceleratedSGDOptimizer<Scalar>::param_grads_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads_bak = param_grads;
		param_grads = MomentumAcceleratedSGDOptimizer<Scalar>::momentum * param_grads -
				learning_rate * (Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg.d_function(params));
		params += -MomentumAcceleratedSGDOptimizer<Scalar>::momentum * param_grads_bak +
				(1 + MomentumAcceleratedSGDOptimizer<Scalar>::momentum) * param_grads;
	};
};

template<typename Scalar>
class AdagradOptimizer : public SGDOptimizer<Scalar> {
public:
	AdagradOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-2, Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size),
				learning_rate(learning_rate),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(epsilon > 0);
	};
	virtual ~AdagradOptimizer() = default;
protected:
	void fit(NeuralNetwork<Scalar>& net) {
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
	virtual void update_acc_weight_grad_sqrs(Matrix<Scalar>& acc_weight_grad_sqrs,
			const Matrix<Scalar>& param_grads) {
		acc_weight_grad_sqrs += param_grads.cwiseProduct(param_grads);
	};
	virtual void update_acc_batch_norm_grad_sqrs(RowVector<Scalar>& acc_batch_norm_grad_sqrs,
			const RowVector<Scalar>& batch_norm_grads) {
		acc_batch_norm_grad_sqrs += batch_norm_grads.cwiseProduct(batch_norm_grads);
	};
	void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		Matrix<Scalar>& param_grad_sqrs = param_grad_sqrs_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg.d_function(params);
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
	RMSPropOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg,
			unsigned batch_size = 1, Scalar learning_rate = 1e-3, Scalar l2_decay = 1e-1,
			Scalar epsilon = Utils<Scalar>::EPSILON) :
				AdagradOptimizer<Scalar>::AdagradOptimizer(loss, reg, batch_size, learning_rate, epsilon),
				l2_decay(l2_decay) {
		assert(l2_decay >= 0 && l2_decay <= 1);
	};
protected:
	void update_acc_weight_grad_sqrs(Matrix<Scalar>& acc_weight_grad_sqrs,
			const Matrix<Scalar>& param_grads) {
		acc_weight_grad_sqrs = (1 - l2_decay) * acc_weight_grad_sqrs + l2_decay * param_grads.cwiseProduct(param_grads);
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
			Scalar decay = 5e-2, Scalar epsilon = Utils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar>::SGDOptimizer(loss, reg, batch_size),
				decay(decay),
				epsilon(epsilon) {
		assert(decay >= 0 && decay <= 1);
		assert(epsilon > 0);
	};
protected:
	void fit(NeuralNetwork<Scalar>& net) {
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
	void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		ParamGradAndUpdateSqrs& pgus = pgus_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg.d_function(params);
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
	AdamOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
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
	void fit(NeuralNetwork<Scalar>& net) {
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
	void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		ParamGradNorms& grad_norms = pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - l1_decay, epoch + 1) + epsilon);
		Scalar l2_corr = 1.0 / (1.0 - pow(1.0 - l2_decay, epoch + 1) + epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg.d_function(params);
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
	AdaMaxOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				AdamOptimizer<Scalar>::AdamOptimizer(loss, reg, batch_size, learning_rate,
						l1_decay, l2_decay, epsilon) { };
protected:
	void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		typename AdamOptimizer<Scalar>::ParamGradNorms& grad_norms = AdamOptimizer<Scalar>::pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg.d_function(params);
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
	NadamOptimizer(const Loss<Scalar>& loss, const RegularizationPenalty<Scalar>& reg, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3, Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3,
			Scalar epsilon = Utils<Scalar>::EPSILON2) :
				AdamOptimizer<Scalar>::AdamOptimizer(loss, reg, batch_size, learning_rate,
						l1_decay, l2_decay, epsilon) { };
protected:
	void update_params(Layer<Scalar>& layer, unsigned i, unsigned epoch) {
		typename AdamOptimizer<Scalar>::ParamGradNorms& grad_norms = AdamOptimizer<Scalar>::pgn_vec[i];
		Scalar l1_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Scalar l1_next_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l1_decay, epoch + 2) +
				AdamOptimizer<Scalar>::epsilon);
		Scalar l2_corr = 1.0 / (1.0 - pow(1.0 - AdamOptimizer<Scalar>::l2_decay, epoch + 1) +
				AdamOptimizer<Scalar>::epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar>::get_params(layer);
		Matrix<Scalar> param_grads = Optimizer<Scalar>::get_param_grads(layer) +
				SGDOptimizer<Scalar>::reg.d_function(params);
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
