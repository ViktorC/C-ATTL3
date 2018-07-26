/*
 * Optimizer.hpp
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_OPTIMIZER_H_
#define CATTL3_OPTIMIZER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "DataProvider.hpp"
#include "Loss.hpp"
#include "EigenProxy.hpp"
#include "NeuralNetwork.hpp"
#include "NumericUtils.hpp"
#include "ParameterInitialization.hpp"
#include "ParameterRegularization.hpp"

namespace cattle {

/**
 * An abstract class template for neural network optimizer algorithm implementations.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class Optimizer {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal optimizer rank");
protected:
	typedef NeuralNetwork<Scalar,Rank,Sequential> Net;
	typedef DataProvider<Scalar,Rank,Sequential> Provider;
	typedef Tensor<Scalar,Rank + Sequential + 1> Data;
public:
	inline Optimizer(LossSharedPtr<Scalar,Rank,Sequential> loss) :
				loss(loss) {
		assert(loss != nullptr);
	}
	virtual ~Optimizer() = default;
	/**
	 * It performs a gradient check to verify the correctness of the neural network and layer implementations.
	 * It is recommended to use double precision floating points; however, non-differentiable layers such as
	 * rectified linear units may still fail.
	 *
	 * @param net A reference to the network on which the gradient check is to be performed.
	 * @param provider A reference to the data provider to use for the gradient check.
	 * @param verbose Whether the analytic and numerical derivatives of the variables should be printed to the
	 * standard out stream.
	 * @param step_size The step size for numerical differentiation.
	 * @param abs_epsilon The maximum acceptable absolute difference between the numerical and analytic
	 * gradients.
	 * @param rel_epsilon The maximum acceptable relative (to the greater out of the two) difference between
	 * the numerical and analytic gradients.
	 * @return Whether the gradient check has been passed or failed.
	 */
	inline bool verify_gradients(Net& net, Provider& provider, bool verbose = true,
			Scalar step_size = NumericUtils<Scalar>::EPSILON2 / 2,
			Scalar abs_epsilon = NumericUtils<Scalar>::EPSILON2,
			Scalar rel_epsilon = NumericUtils<Scalar>::EPSILON2) const {
		assert(net.get_input_dims() == provider.get_obs_dims());
		assert(net.get_output_dims() == provider.get_obj_dims());
		assert(step_size > 0);
		assert(abs_epsilon >= 0 && rel_epsilon > 0);
		bool failure = false;
		DataPair<Scalar,Rank,Sequential> data_pair = provider.get_data(std::numeric_limits<std::size_t>::max());
		std::size_t instances = data_pair.first.dimension(0);
		provider.reset();
		/* As the loss to minimize is the mean of the losses for all the training observations, the gradient to
		 * back-propagate is to be divided by the number of observations in the batch. */
		net.backpropagate(loss->d_function(net.propagate(data_pair.first, true),
				data_pair.second) / (Scalar) instances);
		std::vector<Layer<Scalar,Rank>*> layers = net.get_layers();
		for (std::size_t i = 0; i < layers.size(); ++i) {
			std::vector<Parameters<Scalar>*> params_vec = layers[i]->get_params();
			for (auto params_ptr : params_vec) {
				if (!params_ptr || !params_ptr->are_optimizable())
					continue;
				Parameters<Scalar>& params = *params_ptr;
				if (verbose) {
					std::cout << "Layer " << std::setw(3) << std::to_string(i + 1) <<
							std::string(28, '-') << std::endl;
				}
				/* Add the derivative of the regularization function w.r.t. to the parameters of the layer to the
				 * parameters' gradient. */
				params.regularize();
				Matrix<Scalar> params_values = params.get_values();
				const Matrix<Scalar>& params_grad = params.get_grad();
				for (int j = 0; j < params_values.rows(); ++j) {
					for (int k = 0; k < params_values.cols(); ++k) {
						if (verbose)
							std::cout << "\tParam[" << i << "," << j << "," << k << "]:" << std::endl;
						Scalar ana_grad = params_grad(j,k);
						if (verbose)
							std::cout << "\t\tAnalytic gradient = " << ana_grad << std::endl;
						Scalar param = params_values(j,k);
						params_values(j,k) = param + step_size;
						params.set_values(params_values);
						/* Compute the numerical gradients in training mode to ensure that the means and standard
						 * deviations used for batch normalization are the same as those used during the analytic
						 * gradient computation. */
						Scalar loss_inc = loss->function(net.propagate(data_pair.first, true),
								data_pair.second).mean();
						/* Calculate the new regularization penalty as its derivative w.r.t. the layer's
						 * parameters is included in the gradient. */
						Scalar reg_pen_inc = params.get_regularization_penalty();
						params_values(j,k) = param - step_size;
						params.set_values(params_values);
						Scalar loss_dec = loss->function(net.propagate(data_pair.first, true),
								data_pair.second).mean();
						Scalar reg_pen_dec = params.get_regularization_penalty();
						params_values(j,k) = param;
						params.set_values(params_values);
						// Include the regularization penalty as well.
						Scalar num_grad = (loss_inc + reg_pen_inc - (loss_dec + reg_pen_dec)) / (2 * step_size);
						if (verbose)
							std::cout << "\t\tNumerical gradient = " << num_grad;
						if (!NumericUtils<Scalar>::almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
							if (verbose)
								std::cout << " *****FAIL*****";
							failure = true;
						}
						if (verbose)
							std::cout << std::endl;
					}
				}
				params.reset_grad();
			}
		}
		// Empty the network caches.
		net.empty_caches();
		return !failure;
	}
	/**
	 * It optimizes the specified neural network using the given data providers according to the
	 * optimizers loss function. It also fits the optimizer to the network before the otpimization
	 * begins.
	 *
	 * @param net A reference to the network whose parameters are to be optimized.
	 * @param training_prov A reference to the provider of the training data.
	 * @param test_prov A reference to the provider of the test data.
	 * @param epochs The number of epochs for which the optimization should proceed.
	 * @param early_stop An std::size_t integer denoting the number of consecutive loss increases
	 * after which the optimization process is to be terminated prematurely. If it is 0, the
	 * process is never terminated prematurely.
	 * @param target_loss The target test loss value. If the test loss reaches this value or
	 * drops below it, the optimization process is terminated.
	 * @param verbose Whether the training, test, and regularization losses for each epoch should
	 * be printed to the standard out stream.
	 * @return The test loss of the last epoch.
	 */
	inline Scalar optimize(Net& net, Provider& training_prov, Provider& test_prov, std::size_t epochs,
			std::size_t early_stop = 0, Scalar target_loss = NumericUtils<Scalar>::MIN, bool verbose = true) {
		assert(net.get_input_dims() == training_prov.get_obs_dims());
		assert(net.get_output_dims() == training_prov.get_obj_dims());
		assert(training_prov.get_obs_dims() == test_prov.get_obs_dims());
		assert(training_prov.get_obj_dims() == test_prov.get_obj_dims());
		// Fit the optimizer parameters to the network.
		fit(net);
		Scalar prev_test_loss = NumericUtils<Scalar>::MAX;
		std::size_t cons_loss_inc = 0;
		if (verbose)
			std::cout << "<Optimization>" << std::endl;
		// Start the optimization iterations.
		for (std::size_t i = 0; i <= epochs; ++i) {
			if (verbose)
				std::cout << "Epoch " << std::setw(3) << i << std::string(28, '-') << std::endl;
			// Train.
			if (i != 0) {
				training_prov.reset();
				if (verbose) {
					std::cout << "\ttraining loss: " <<
							std::to_string(_train(net, training_prov, i, verbose)) << std::endl;
				}
			}
			// Validate.
			test_prov.reset();
			Scalar test_loss = _test(net, test_prov, i, verbose);
			if (verbose)
				std::cout << "\ttest loss: " << std::to_string(test_loss);
			if (test_loss >= prev_test_loss) {
				cons_loss_inc++;
				if (verbose)
					std::cout << " *****INCREASED LOSS*****";
				if (early_stop > 0 && cons_loss_inc >= early_stop)
					break;
			} else
				cons_loss_inc = 0;
			if (verbose)
				std::cout << std::endl << std::endl;
			prev_test_loss = test_loss;
			if (prev_test_loss <= target_loss)
				break;
		}
		// Reset the providers.
		training_prov.reset();
		test_prov.reset();
		// Empty the network caches.
		net.empty_caches();
		return prev_test_loss;
	}
	/**
	 * It trains the specified neural network using the given training data provider according to
	 * the optimizers loss function for the specified number of epochs. It does not fit the
	 * optimizer to the network, thus the #fit(Net&) method might need to be invoked beforehand.
	 *
	 * @param net A reference to the network whose parameters are to be optimized.
	 * @param prov A reference to the provider of the training data.
	 * @param epochs The number of epochs for which the training should proceed.
	 * @param early_stop An std::size_t integer denoting the number of consecutive loss increases
	 * after which the optimization process is to be terminated prematurely. If it is 0, the
	 * process is never terminated prematurely.
	 * @param target_loss The target test loss value. If the test loss reaches this value or
	 * drops below it, the optimization process is terminated.
	 * @param verbose Whether the training losses for of the epochs should be printed to the
	 * standard out stream.
	 * @return The training loss of the last epoch.
	 */
	inline Scalar train(Net& net, Provider& prov, std::size_t epochs, std::size_t early_stop = 0,
			Scalar target_loss = NumericUtils<Scalar>::MIN, bool verbose = false) {
		assert(net.get_input_dims() == prov.get_obs_dims());
		assert(net.get_output_dims() == prov.get_obj_dims());
		Scalar train_loss;
		Scalar prev_train_loss = NumericUtils<Scalar>::MAX;
		std::size_t cons_loss_inc = 0;
		if (verbose)
			std::cout << "<Training>" << std::endl;
		for (std::size_t i = 1; i <= epochs; ++i) {
			if (verbose)
				std::cout << "Epoch " << std::setw(3) << i << std::string(28, '-') << std::endl;
			prov.reset();
			train_loss = _train(net, prov, i, verbose);
			if (verbose)
				std::cout << "\ttraining loss: " << std::to_string(train_loss);
			if (train_loss >= prev_train_loss) {
				cons_loss_inc++;
				if (verbose)
					std::cout << " *****INCREASED LOSS*****";
				if (early_stop > 0 && cons_loss_inc >= early_stop)
					break;
			} else
				cons_loss_inc = 0;
			if (verbose)
				std::cout << std::endl << std::endl;
			prev_train_loss = train_loss;
			if (prev_train_loss <= target_loss)
				break;
		}
		prov.reset();
		net.empty_caches();
		return prev_train_loss;
	}
	/**
	 * It tests the specified neural network using the given test data provider according to the
	 * optimizers loss function. It does not fit the optimizer to the network, thus the
	 * #fit(Net&) method might need to be invoked beforehand.
	 *
	 * @param net A reference to the network whose parameters are to be optimized.
	 * @param prov A reference to the provider of the training data.
	 * @param verbose Whether the training, test, and regularization losses for each epoch should
	 * be printed to the standard out stream.
	 * @return The test loss of the last epoch.
	 */
	inline Scalar test(Net& net, Provider& prov, bool verbose = false) {
		assert(net.get_input_dims() == prov.get_obs_dims());
		assert(net.get_output_dims() == prov.get_obj_dims());
		if (verbose)
			std::cout << "<Testing>" << std::endl;
		prov.reset();
		Scalar test_loss = _test(net, prov, 0, verbose);
		if (verbose)
			std::cout << "\ttest loss: " << std::to_string(test_loss);
		prov.reset();
		net.empty_caches();
		return test_loss;
	}
	/**
	 * It fits the optimizer to the neural network. It allows optimizers with individual
	 * learning rates for each parameter to set up their necessary internal data structures.
	 *
	 * @param net A reference to the neural network that is to be optimized.
	 */
	virtual void fit(Net& net) = 0;
protected:
	/**
	 * It trains the specified neural network for a single epoch on data provided by the
	 * specified data provider.
	 *
	 * @param net A reference to the neural network to optimize.
	 * @param training_prov A reference to the training data provider.
	 * @param epoch The index of the current epoch. starting from 1.
	 * @param verbose Whether the optimization is performed in verbose mode; i.e. whether
	 * information should be printed to the standard out stream.
	 * @return The training loss of the epoch.
	 */
	virtual Scalar _train(Net& net, Provider& training_prov, std::size_t epoch, bool verbose) = 0;
	/**
	 * It tests the specified neural network for a single epoch on the test data
	 * provided by the specified data provider.
	 *
	 * @param net A reference to the neural network to test.
	 * @param test_prov A reference to the test data provider.
	 * @param epoch The index of the epoch starting from 0.
	 * @param verbose Whether the optimization is performed in verbose mode; i.e. whether
	 * information should be printed to the standard out stream.
	 * @return The test loss of the epoch.
	 */
	virtual Scalar _test(Net& net, Provider& test_prov, std::size_t epoch, bool verbose) = 0;
private:
	const LossSharedPtr<Scalar,Rank,Sequential> loss;
};

/**
 * A class template for a vanilla SGD optimizer.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class VanillaSGDOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 */
	inline VanillaSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar learning_rate = 1e-3) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				learning_rate(learning_rate) {
		assert(learning_rate > 0);
	}
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) { }
protected:
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		Matrix<Scalar>& params = Root::get_params(layer);
		params -= Root::get_params_grad(layer) * learning_rate;
	}
	const Scalar learning_rate;
};

/**
 * A class template for a momentum accelerated SGD optimizer.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class MomentumAcceleratedSGDOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param init_learning_rate The initial learning rate (a.k.a. step size) to use. It
	 * is expected to be greater than 0.
	 * @param annealing_rate The rate at which the learning rate is to be annealed. It is
	 * expected to be greater than or equal to 0. The greater it is, the faster the learning
	 * rate decreases.
	 * @param momentum The momentum rate to use. The greater the momentum, the lesser the
	 * effect of newer gradients. It is expected to be greater than 0 and less than 1.
	 */
	inline MomentumAcceleratedSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3, Scalar momentum = .9) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				init_learning_rate(init_learning_rate),
				annealing_rate(annealing_rate),
				momentum(momentum) {
		assert(init_learning_rate > 0);
		assert(annealing_rate >= 0);
		assert(momentum > 0 && momentum < 1);
	}
	virtual ~MomentumAcceleratedSGDOptimizer() = default;
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Root::get_layers(net);
		params_grad_vec = std::vector<Matrix<Scalar>>(layers.size());
		for (std::size_t i = 0; i < params_grad_vec.size(); ++i) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			const Matrix<Scalar>& params_grad = Root::get_params_grad(layer);
			Matrix<Scalar> acc_params_grad = Matrix<Scalar>::Zero(params_grad.rows(), params_grad.cols());
			params_grad_vec[i] = acc_params_grad;
		}
	}
protected:
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		Scalar learning_rate = calculate_learning_rate(epoch);
		Matrix<Scalar>& params_grad = params_grad_vec[i];
		Matrix<Scalar>& params = Root::get_params(layer);
		params += params_grad * momentum - Root::get_params_grad(layer) * learning_rate;
	}
	/**
	 * It calculates the annealed learning rate as a function of the epoch index.
	 *
	 * @param epoch The epoch index.
	 * @return The learning rate to use.
	 */
	Scalar calculate_learning_rate(std::size_t epoch) {
		return init_learning_rate / (1 + annealing_rate * epoch);
	}
	const Scalar init_learning_rate;
	const Scalar annealing_rate;
	const Scalar momentum;
	std::vector<Matrix<Scalar>> params_grad_vec;
};

/**
 * A class template for Nesterov momentum accelerated SGD optimizers.
 *
 * \see https://arxiv.org/abs/1212.0901
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NesterovMomentumAcceleratedSGDOptimizer : public MomentumAcceleratedSGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
	typedef MomentumAcceleratedSGDOptimizer<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param init_learning_rate The initial learning rate (a.k.a. step size) to use. It is
	 * expected to be greater than 0.
	 * @param annealing_rate The rate at which the learning rate is to be annealed. It is
	 * expected to be greater than or equal to 0. The greater it is, the faster the learning
	 * rate decreases.
	 * @param momentum The momentum rate to use. The greater the momentum, the lesser the
	 * effect of newer gradients. It is expected to be greater than 0 and less than 1.
	 */
	inline NesterovMomentumAcceleratedSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss,
			std::size_t batch_size = 1, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3, Scalar momentum = .9) :
				Base::MomentumAcceleratedSGDOptimizer(loss, batch_size, init_learning_rate, annealing_rate, momentum) { };
protected:
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		Scalar learning_rate = Base::calculate_learning_rate(epoch);
		Matrix<Scalar>& acc_params_grad = Base::params_grad_vec[i];
		Matrix<Scalar>& params = Root::get_params(layer);
		Matrix<Scalar> params_grad_bak = acc_params_grad;
		acc_params_grad = acc_params_grad * Base::momentum - Root::get_params_grad(layer) * learning_rate;
		params += params_grad_bak * -Base::momentum + acc_params_grad * (1 + Base::momentum);
	}
};

/**
 * A class template for the AdaGrad optimization algorithm.
 *
 * \see http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdaGradOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AdaGradOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar learning_rate = 1e-2, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				learning_rate(learning_rate),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(epsilon > 0);
	}
	virtual ~AdaGradOptimizer() = default;
protected:
public:
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers =Root::get_layers(net);
		params_grad_sqrs_vec = std::vector<Matrix<Scalar>>(layers.size());
		for (std::size_t i = 0; i < params_grad_sqrs_vec.size(); ++i) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			const Matrix<Scalar>& params_grad = Root::get_params_grad(layer);
			Matrix<Scalar> params_grad_sqrs = Matrix<Scalar>::Zero(params_grad.rows(), params_grad.cols());
			params_grad_sqrs_vec[i] = params_grad_sqrs;
		}
	}
	/**
	 * It updates the accumulated squared parameter gradients.
	 *
	 * @param acc_params_grad_sqrs The accumulated squared parameter gradients.
	 * @param params_grad The new parameter gradients.
	 */
	inline virtual void _update_acc_params_grad_sqrs(Matrix<Scalar>& acc_params_grad_sqrs,
			const Matrix<Scalar>& params_grad) {
		// Accumulate the squares of the gradients.
		acc_params_grad_sqrs += params_grad.cwiseProduct(params_grad);
	}
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		Matrix<Scalar>& params_grad_sqrs = params_grad_sqrs_vec[i];
		Matrix<Scalar>& params = Root::get_params(layer);
		const Matrix<Scalar>& params_grad = Root::get_params_grad(layer);
		_update_acc_params_grad_sqrs(params_grad_sqrs, params_grad);
		params -= (params_grad.array() * learning_rate / (params_grad_sqrs.array().sqrt() + epsilon)).matrix();
	}
	const Scalar learning_rate;
	const Scalar epsilon;
	std::vector<Matrix<Scalar>> params_grad_sqrs_vec;
};

/**
 * A class template for the RMSProp optimizer.
 *
 * \see https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class RMSPropOptimizer : public AdaGradOptimizer<Scalar,Rank,Sequential> {
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param l2_decay The decay rate of the accumulated squared parameter gradients.
	 * It is expected to be in the range [0,1]. The greater it is, the faster the accumulated
	 * gradients decay.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline RMSPropOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l2_decay = 1e-1, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				AdaGradOptimizer<Scalar,Rank,Sequential>::AdaGradOptimizer(loss, batch_size, learning_rate, epsilon),
				l2_decay(l2_decay) {
		assert(l2_decay >= 0 && l2_decay <= 1);
	}
protected:
	inline void _update_acc_params_grad_sqrs(Matrix<Scalar>& acc_params_grad_sqrs,
			const Matrix<Scalar>& params_grad) {
		acc_params_grad_sqrs = acc_params_grad_sqrs * (1 - l2_decay) + params_grad.cwiseProduct(params_grad) * l2_decay;
	}
	const Scalar l2_decay;
};

/**
 * A class template for the ADADELTA optimization algorithm.
 *
 * \see https://arxiv.org/abs/1212.5701
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdaDeltaOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param decay The decay rate of the accelerated accumulated parameter gradients.
	 * It is expected to be in the range [0,1]. The greater it is, the faster the accumulated
	 * gradients decay.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AdaDeltaOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1,
			Scalar decay = 5e-2, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				decay(decay),
				epsilon(epsilon) {
		assert(decay >= 0 && decay <= 1);
		assert(epsilon > 0);
	}
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Root::get_layers(net);
		pgus_vec = std::vector<ParamGradAndUpdateSqrs>(layers.size());
		for (std::size_t i = 0; i < pgus_vec.size(); ++i) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			const Matrix<Scalar>& param_grad = Root::get_params_grad(layer);
			ParamGradAndUpdateSqrs pgus;
			pgus.params_grad = Matrix<Scalar>::Zero(param_grad.rows(), param_grad.cols());
			pgus.params_update = Matrix<Scalar>::Zero(pgus.params_grad.rows(), pgus.params_grad.cols());
			pgus_vec[i] = pgus;
		}
	}
protected:
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		ParamGradAndUpdateSqrs& pgus = pgus_vec[i];
		Matrix<Scalar>& params = Root::get_params(layer);
		const Matrix<Scalar>& params_grad = Root::get_params_grad(layer);
		pgus.params_grad = pgus.params_grad * (1 - decay) + params_grad.cwiseProduct(params_grad) * decay;
		Matrix<Scalar> weight_updates = -params_grad.array() * (pgus.params_update.array() + epsilon).sqrt() /
				(pgus.params_grad.array() + epsilon).sqrt();
		params += weight_updates;
		pgus.params_update = pgus.params_update * (1 - decay) + weight_updates.cwiseProduct(weight_updates) * decay;
	}
	const Scalar decay;
	const Scalar epsilon;
	/**
	 * A struct containing the moving averages of the squared gradients and squared updates of a layer.
	 */
	struct ParamGradAndUpdateSqrs {
		Matrix<Scalar> params_grad;
		Matrix<Scalar> params_update;
	};
	std::vector<ParamGradAndUpdateSqrs> pgus_vec;
};

/**
 * A class template for the Adam optimization algorithm.
 *
 * \see https://arxiv.org/abs/1412.6980
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdamOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param l1_decay The decay rate of the accumulated parameter gradients. It is expected
	 * to be in the range [0,1]. The greater it is, the faster the accumulated gradients
	 * decay.
	 * @param l2_decay The decay rate of the accumulated squared parameter gradients. It is
	 * expected to be in the range [0,1]. The greater it is, the faster the accumulated
	 * squared gradients decay.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AdamOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				learning_rate(learning_rate),
				l1_decay(l1_decay),
				l2_decay(l2_decay),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(l1_decay >= 0 && l1_decay <= 1);
		assert(l2_decay >= 0 && l2_decay <= 1);
		assert(epsilon > 0);
	}
	virtual ~AdamOptimizer() = default;
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Root::get_layers(net);
		pgn_vec = std::vector<ParamGradNorms>(layers.size());
		for (std::size_t i = 0; i < pgn_vec.size(); ++i) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			const Matrix<Scalar>& param_grad = Root::get_params_grad(layer);
			ParamGradNorms vel;
			vel.params_grad_l1 = Matrix<Scalar>::Zero(param_grad.rows(), param_grad.cols());
			vel.params_grad_l2 = Matrix<Scalar>::Zero(vel.params_grad_l1.rows(), vel.params_grad_l1.cols());
			pgn_vec[i] = vel;
		}
	}
protected:
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		ParamGradNorms& grad_norms = pgn_vec[i];
		Scalar l1_corr = (Scalar) 1 / (1 - pow(1 - l1_decay, epoch + 1) + epsilon);
		Scalar l2_corr = (Scalar) 1 / (1 - pow(1 - l2_decay, epoch + 1) + epsilon);
		Matrix<Scalar>& params = Root::get_params(layer);
		const Matrix<Scalar>& params_grad = Root::get_params_grad(layer);
		grad_norms.params_grad_l1 = grad_norms.params_grad_l1 * (1 - l1_decay) + params_grad * l1_decay;
		grad_norms.params_grad_l2 = grad_norms.params_grad_l2 * (1 - l2_decay) +
				params_grad.cwiseProduct(params_grad) * l2_decay;
		params -= ((grad_norms.params_grad_l1 * (learning_rate * l1_corr)).array() /
				((grad_norms.params_grad_l2 * l2_corr).array() + epsilon).sqrt()).matrix();
	}
	const Scalar learning_rate;
	const Scalar l1_decay;
	const Scalar l2_decay;
	const Scalar epsilon;
	/**
	 * A struct containing the moving averages of the first and second norms of the parameter gradients
	 * of a layer.
	 */
	struct ParamGradNorms {
		Matrix<Scalar> params_grad_l1;
		Matrix<Scalar> params_grad_l2;
	};
	std::vector<ParamGradNorms> pgn_vec;
};

/**
 * A class template for the AdaMax optimization algorithm.
 *
 * \see https://arxiv.org/abs/1412.6980
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdaMaxOptimizer : public AdamOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
	typedef AdamOptimizer<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param l1_decay The decay rate of the accumulated parameter gradients. It is expected
	 * to be in the range [0,1]. The greater it is, the faster the accumulated gradients
	 * decay.
	 * @param l2_decay The decay rate of the accumulated squared parameter gradients. It is
	 * expected to be in the range [0,1]. The greater it is, the faster the accumulated
	 * squared gradients decay.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AdaMaxOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				Base::AdamOptimizer(loss, batch_size, learning_rate, l1_decay, l2_decay, epsilon) { }
protected:
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		typename Base::ParamGradNorms& grad_norms = Base::pgn_vec[i];
		Scalar l1_corr = (Scalar) 1 / (1 - pow(1 - Base::l1_decay, epoch + 1) + Base::epsilon);
		Matrix<Scalar>& params = Root::get_params(layer);
		const Matrix<Scalar>& params_grad = Root::get_params_grad(layer);
		grad_norms.params_grad_l1 = grad_norms.params_grad_l1 * (1 - Base::l1_decay) + params_grad * Base::l1_decay;
		grad_norms.params_grad_l2 = (grad_norms.params_grad_l2 * (1 - Base::l2_decay)).cwiseMax(params_grad.cwiseAbs());
		params -= ((grad_norms.params_grad_l1 * (Base::learning_rate * l1_corr)).array() /
				(grad_norms.params_grad_l2.array() + Base::epsilon)).matrix();
	}
};

/**
 * A class template for the Nesterov accelerated Adam (Nadam) optimization algorithm.
 *
 * \see http://cs229.stanford.edu/proj2015/054_report.pdf
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NadamOptimizer : public AdamOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
	typedef AdamOptimizer<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param l1_decay The decay rate of the accumulated parameter gradients. It is expected
	 * to be in the range [0,1]. The greater it is, the faster the accumulated gradients
	 * decay.
	 * @param l2_decay The decay rate of the accumulated squared parameter gradients. It is
	 * expected to be in the range [0,1]. The greater it is, the faster the accumulated
	 * squared gradients decay.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline NadamOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				Base::AdamOptimizer(loss, batch_size, learning_rate, l1_decay, l2_decay, epsilon) { }
protected:
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		typename Base::ParamGradNorms& grad_norms = Base::pgn_vec[i];
		Scalar l1_corr = (Scalar) 1 / (1 - pow(1 - Base::l1_decay, epoch + 1) + Base::epsilon);
		Scalar l1_next_corr = (Scalar) 1 / (1 - pow(1 - Base::l1_decay, epoch + 2) + Base::epsilon);
		Scalar l2_corr = (Scalar) 1 / (1 - pow(1 - Base::l2_decay, epoch + 1) + Base::epsilon);
		Matrix<Scalar>& params = Root::get_params(layer);
		const Matrix<Scalar>& params_grad = Root::get_params_grad(layer);
		grad_norms.params_grad_l1 = grad_norms.params_grad_l1 * (1 - Base::l1_decay) + params_grad * Base::l1_decay;
		grad_norms.params_grad_l2 = grad_norms.params_grad_l2 * (1 - Base::l2_decay) +
				params_grad.cwiseProduct(params_grad) * Base::l2_decay;
		params -= ((params_grad * (Base::l1_decay * l1_corr) + grad_norms.params_grad_l1 *
				((1.0 - Base::l1_decay) * l1_next_corr)).array() * Base::learning_rate /
				((grad_norms.params_grad_l2 * l2_corr).array() + Base::epsilon).sqrt()).matrix();
	}
};

/**
 * A class template for the AMSGrad optimization algorithm.
 *
 * \see https://openreview.net/pdf?id=ryQu7f-RZ
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AMSGradOptimizer : public AdamOptimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Root;
	typedef AdamOptimizer<Scalar,Rank,Sequential> Base;
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param l1_decay The decay rate of the accumulated parameter gradients. It is expected
	 * to be in the range [0,1]. The greater it is, the faster the accumulated gradients
	 * decay.
	 * @param l2_decay The decay rate of the accumulated squared parameter gradients. It is
	 * expected to be in the range [0,1]. The greater it is, the faster the accumulated
	 * squared gradients decay.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AMSGradOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				Base::AdamOptimizer(loss, batch_size, learning_rate, l1_decay, l2_decay, epsilon) { }
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		Base::fit(net);
		params_grad_l2_max = std::vector<Matrix<Scalar>>(Base::pgn_vec.size());
		for (std::size_t i = 0; i < params_grad_l2_max.size(); ++i)
			params_grad_l2_max[i] = Base::pgn_vec[i].params_grad_l2;
	}
protected:
	inline void _update_params(Layer<Scalar,Rank>& layer, std::size_t i, std::size_t epoch) {
		typename Base::ParamGradNorms& grad_norms = Base::pgn_vec[i];
		Matrix<Scalar>& params = Root::get_params(layer);
		const Matrix<Scalar>& params_grad = Root::get_params_grad(layer);
		grad_norms.params_grad_l1 = grad_norms.params_grad_l1 * (1 - Base::l1_decay) + params_grad * Base::l1_decay;
		grad_norms.params_grad_l2 = grad_norms.params_grad_l2 * (1 - Base::l2_decay) +
				params_grad.cwiseProduct(params_grad) * Base::l2_decay;
		params_grad_l2_max[i] = grad_norms.params_grad_l2.cwiseMax(params_grad_l2_max[i]);
		params -= (grad_norms.params_grad_l1.array() * Base::learning_rate /
				(params_grad_l2_max[i].array() + Base::epsilon).sqrt()).matrix();
	}
private:
	std::vector<Matrix<Scalar>> params_grad_l2_max;
};

} /* namespace cattle */

#endif /* CATTL3_OPTIMIZER_H_ */
