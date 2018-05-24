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
#include "NeuralNetwork.hpp"
#include "ParameterRegularization.hpp"
#include "utils/EigenProxy.hpp"
#include "utils/NumericUtils.hpp"
#include "WeightInitialization.hpp"

namespace cattle {

// TODO Hessian-free w/ Conjugate Gradient
// TODO L-BFGS
// TODO Particle Swarm
// TODO GA
// TODO PBIL

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
			Scalar step_size = internal::NumericUtils<Scalar>::EPSILON2 / 2,
			Scalar abs_epsilon = internal::NumericUtils<Scalar>::EPSILON2,
			Scalar rel_epsilon = internal::NumericUtils<Scalar>::EPSILON2) const {
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
		net.backpropagate(loss->d_function(net.propagate(data_pair.first, true), data_pair.second) / (Scalar) instances);
		std::vector<Layer<Scalar,Rank>*> layers = net.get_layers();
		for (std::size_t i = 0; i < layers.size(); ++i) {
			Layer<Scalar,Rank>& layer = *layers[i];
			if (layer.is_parametric()) {
				if (verbose)
					std::cout << "Layer " << std::setw(3) << std::to_string(i + 1) << std::string(28, '-') << std::endl;
				/* Add the derivative of the regularization function w.r.t. to the parameters of the layer to the
				 * parameters' gradient. */
				layer.regularize();
				Matrix<Scalar>& params = layer.get_params();
				const Matrix<Scalar>& params_grad = layer.get_params_grad();
				for (int j = 0; j < params.rows(); ++j) {
					for (int k = 0; k < params.cols(); ++k) {
						if (verbose)
							std::cout << "\tParam[" << i << "," << j << "," << k << "]:" << std::endl;
						Scalar ana_grad = params_grad(j,k);
						if (verbose)
							std::cout << "\t\tAnalytic gradient = " << ana_grad << std::endl;
						Scalar param = params(j,k);
						params(j,k) = param + step_size;
						/* Compute the numerical gradients in training mode to ensure that the means and standard
						 * deviations used for batch normalization are the same as those used during the analytic
						 * gradient computation. */
						Scalar loss_inc = loss->function(net.propagate(data_pair.first, true), data_pair.second).mean();
						/* Calculate the new regularization penalty as its derivative w.r.t. the layer's parameters
						 * is included in the gradient. */
						Scalar reg_pen_inc = layer.get_regularization_penalty();
						params(j,k) = param - step_size;
						Scalar loss_dec = loss->function(net.propagate(data_pair.first, true), data_pair.second).mean();
						Scalar reg_pen_dec = layer.get_regularization_penalty();
						params(j,k) = param;
						// Include the regularization penalty as well.
						Scalar num_grad = (loss_inc + reg_pen_inc - (loss_dec + reg_pen_dec)) / (2 * step_size);
						if (verbose)
							std::cout << "\t\tNumerical gradient = " << num_grad;
						if (!internal::NumericUtils<Scalar>::almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
							if (verbose)
								std::cout << " *****FAIL*****";
							failure = true;
						}
						if (verbose)
							std::cout << std::endl;
					}
				}
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
	 * @param early_stop An unsigned integer denoting the number of consecutive loss increases
	 * after which the optimization process is to be terminated prematurely. If it is 0, the
	 * process is never terminated prematurely.
	 * @param verbose Whether the training, test, and regularization losses for each epoch should
	 * be printed to the standard out stream.
	 * @return The test loss of the last epoch.
	 */
	inline Scalar optimize(Net& net, Provider& training_prov, Provider& test_prov, unsigned epochs, unsigned early_stop = 0,
			bool verbose = true) {
		assert(net.get_input_dims() == training_prov.get_obs_dims());
		assert(net.get_output_dims() == training_prov.get_obj_dims());
		assert(training_prov.get_obs_dims() == test_prov.get_obs_dims());
		assert(training_prov.get_obj_dims() == test_prov.get_obj_dims());
		// Fit the optimizer parameters to the network.
		fit(net);
		Scalar prev_test_loss = internal::NumericUtils<Scalar>::MAX;
		unsigned cons_loss_inc = 0;
		if (verbose)
			std::cout << "<Optimization>" << std::endl;
		// Start the optimization iterations.
		for (unsigned i = 0; i <= epochs; ++i) {
			if (verbose)
				std::cout << "Epoch " << std::setw(3) << i << std::string(28, '-') << std::endl;
			// Train.
			if (i != 0) {
				training_prov.reset();
				if (verbose)
					std::cout << "\ttraining loss: " << std::to_string(_train(net, training_prov, i, verbose)) << std::endl;
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
					return prev_test_loss;
			} else
				cons_loss_inc = 0;
			if (verbose)
				std::cout << std::endl << std::endl;
			prev_test_loss = test_loss;
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
	 * @param verbose Whether the training losses for of the epochs should be printed to the
	 * standard out stream.
	 * @return The training loss of the last epoch.
	 */
	inline Scalar train(Net& net, Provider& prov, unsigned epochs, bool verbose = true) {
		assert(net.get_input_dims() == prov.get_obs_dims());
		assert(net.get_output_dims() == prov.get_obj_dims());
		Scalar train_loss;
		if (verbose)
			std::cout << "<Training>" << std::endl;
		for (unsigned i = 1; i <= epochs; ++i) {
			if (verbose)
				std::cout << "Epoch " << std::setw(3) << i << std::string(28, '-') << std::endl;
			prov.reset();
			train_loss = _train(net, prov, i, verbose);
			if (verbose)
				std::cout << "\ttraining loss: " << std::to_string(train_loss) << std::endl;
		}
		prov.reset();
		net.empty_caches();
		return train_loss;
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
	inline Scalar test(Net& net, Provider& prov, bool verbose = true) {
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
	virtual Scalar _train(Net& net, Provider& training_prov, unsigned epoch, bool verbose) = 0;
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
	virtual Scalar _test(Net& net, Provider& test_prov, unsigned epoch, bool verbose) = 0;
	/**
	 * A method to expose protected methods of the NeuralNetwork class to subclasses of
	 * Optimizer that are not friend classes of NeuralNetwork.
	 *
	 * \see NeuralNetwork#get_layers()
	 *
	 * It returns a vector of pointers to the layers of the specified network.
	 *
	 * @param net A reference to the network whose layers are to be fetched.
	 * @return A vector of pointers to the layers of the network.
	 */
	inline static std::vector<Layer<Scalar,Rank>*> get_layers(Net& net) {
		return net.get_layers();
	}
	/**
	 * A method to expose protected methods of the NeuralNetwork class to subclasses of
	 * Optimizer that are not friend classes of NeuralNetwork.
	 *
	 * \see NeuralNetwork#propagate(Tensor<Scalar,Rank + Sequential + 1>,bool)
	 *
	 * It propagates the input tensor through the specified network in training mode.
	 *
	 * @param net A reference to the network through which the input is to be propagated.
	 * @param in The input tensor.
	 * @return The output of the network in response to the input.
	 */
	inline static Data propagate(Net& net, Data in) {
		return net.propagate(std::move(in), true);
	}
	/**
	 * A method to expose protected methods of the NeuralNetwork class to subclasses of
	 * Optimizer that are not friend classes of NeuralNetwork.
	 *
	 * \see NeuralNetwork#backpropagate(Tensor<Scalar,Rank + Sequential + 1>)
	 *
	 * It back-propagates the gradients through the specified network.
	 *
	 * @param net A reference to the network throught which the gradients are to be
	 * back-propagated.
	 * @param out_grads The gradient tensor.
	 */
	inline static void backpropagate(Net& net, Data out_grads) {
		net.backpropagate(std::move(out_grads));
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * Optimizer that are not friend classes of Layer.
	 *
	 * \see Layer#is_parametric()
	 *
	 * It returns whether the layer has learnable parameters.
	 *
	 * @param layer The layer whose parametric status is to be determined.
	 * @return Whether the layer has learnable parameters.
	 */
	inline static bool is_parametric(Layer<Scalar,Rank>& layer) {
		return layer.is_parametric();
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * Optimizer that are not friend classes of Layer.
	 *
	 * \see Layer#get_params()
	 *
	 * It returns a non-constant reference to the parameter matrix of the specified
	 * layer.
	 *
	 * @param layer The layer whose parameters are to be fetched.
	 * @return A non-constant reference to the parameter matrix of the layer.
	 */
	inline static Matrix<Scalar>& get_params(Layer<Scalar,Rank>& layer) {
		return layer.get_params();
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * Optimizer that are not friend classes of Layer.
	 *
	 * \see Layer#get_params_grad()
	 *
	 * It returns a non-constant reference to the gradient of the specified
	 * layer's parameters.
	 *
	 * @param layer The layer whose parameters' gradient is to be fetched.
	 * @return A non-constant reference to the gradient of the layer's
	 * parameters.
	 */
	inline static const Matrix<Scalar>& get_params_grad(Layer<Scalar,Rank>& layer) {
		return layer.get_params_grad();
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * Optimizer that are not friend classes of Layer.
	 *
	 * \see Layer#regularize()
	 *
	 * It regularizes the layer's parameters by adding the derivative of the regularization
	 * function of the layer w.r.t. the parameters to the parameters' gradient.
	 *
	 * @param layer The layer whose parameters are to be regularized.
	 */
	inline static void regularize(Layer<Scalar,Rank>& layer) {
		return layer.regularize();
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * Optimizer that are not friend classes of Layer.
	 *
	 * \see Layer#get_regularization_penalty()
	 *
	 * It calculates the regularization penalty of the layer's parameters.
	 *
	 * @param layer The layer whose parameters are to be constrained.
	 * @return A scalar representing the penalty on the magnitude of the layer's parameters.
	 */
	inline static Scalar get_regularization_penalty(Layer<Scalar,Rank>& layer) {
		return layer.get_regularization_penalty();
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * Optimizer that are not friend classes of Layer.
	 *
	 * \see Layer#enforce_constraints()
	 *
	 * It enforces constraints on the parameters of the specified layer if applicable.
	 *
	 * @param layer The layer whose parameters are to be constrained.
	 */
	inline static void enforce_constraints(Layer<Scalar,Rank>& layer) {
		layer.enforce_constraints();
	}
	const LossSharedPtr<Scalar,Rank,Sequential> loss;
};

/**
 * An abstract class template for stochastic gradient descent (SGD) optimizers.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class SGDOptimizer : public Optimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Base;
public:
	inline SGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size) :
			Base::Optimizer(loss),
				batch_size(batch_size) {
		assert(batch_size > 0);
	}
	virtual ~SGDOptimizer() = default;
protected:
	inline Scalar _train(typename Base::Net& net, typename Base::Provider& training_prov, unsigned epoch, bool verbose) {
		Scalar training_loss = 0;
		Scalar instances = 0;
		std::vector<Layer<Scalar,Rank>*> layers = Base::get_layers(net);
		// Perform an entire training epoch.
		while (training_prov.has_more()) {
			DataPair<Scalar,Rank,Sequential> data_pair = training_prov.get_data(batch_size);
			instances += data_pair.first.dimension(0);
			typename Base::Data out = Base::propagate(net, std::move(data_pair.first));
			training_loss += Base::loss->function(out, data_pair.second).sum();
			/* Divide the gradient by the batch size to decouple the learning rate and the batch
			 * size hyper-parameters. Use the nominal batch size as the denominator even if the
			 * actual batch size is different (in case the number of samples in the data set is
			 * not divisible by the batch size and the last batch of the epoch contains fewer
			 * instances than the others) to make sure that the magnitude of the gradient is
			 * proportional to the batch size (just like its 'accuracy' is). */
			Base::backpropagate(net, Base::loss->d_function(std::move(out), std::move(data_pair.second)) / (Scalar) batch_size);
			for (unsigned k = 0; k < layers.size(); ++k) {
				Layer<Scalar,Rank>& layer = *(layers[k]);
				if (Base::is_parametric(layer) && !layer.is_frozen()) {
					Base::regularize(layer);
					update_params(layer, k, epoch - 1);
					Base::enforce_constraints(layer);
				}
			}
		}
		return training_loss / instances;
	}
	inline Scalar _test(typename Base::Net& net, typename Base::Provider& test_prov, unsigned epoch, bool verbose) {
		Scalar obj_loss = 0;
		Scalar instances = 0;
		std::vector<Layer<Scalar,Rank>*> layers = Base::get_layers(net);
		// Perform an entire test epoch.
		while (test_prov.has_more()) {
			DataPair<Scalar,Rank,Sequential> data_pair = test_prov.get_data(batch_size);
			instances += data_pair.first.dimension(0);
			obj_loss += Base::loss->function(net.infer(std::move(data_pair.first)), std::move(data_pair.second)).sum();
		}
		Scalar mean_obj_loss = obj_loss / instances;
		Scalar reg_loss = 0;
		for (unsigned j = 0; j < layers.size(); ++j) {
			Layer<Scalar,Rank>& layer = *(layers[j]);
			if (Base::is_parametric(layer) && !layer.is_frozen())
				reg_loss += Base::get_regularization_penalty(layer);
		}
		if (verbose) {
			std::cout << "\tobj loss: " << std::to_string(mean_obj_loss) << std::endl;
			std::cout << "\treg loss: " << std::to_string(reg_loss) << std::endl;
		}
		return mean_obj_loss + reg_loss;
	}
	/**
	 * It updates the parameters of the specified layer based on their gradients after
	 * back-propagation.
	 *
	 * @param layer A reference to the layer whose parameters are to be updated.
	 * @param i The index of the layer.
	 * @param epoch The index of the epoch.
	 */
	virtual void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) = 0;
	const unsigned batch_size;
};

/**
 * A class template for a vanilla SGD optimizer.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class VanillaSGDOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 */
	inline VanillaSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size = 1,
			Scalar learning_rate = 1e-3) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				learning_rate(learning_rate) {
		assert(learning_rate > 0);
	}
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) { }
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		params -= learning_rate * Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
	}
	const Scalar learning_rate;
};

/**
 * A class template for a momentum accelerated SGD optimizer.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class MomentumAcceleratedSGDOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
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
	inline MomentumAcceleratedSGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size = 1,
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
		std::vector<Layer<Scalar,Rank>*> layers = Optimizer<Scalar,Rank,Sequential>::get_layers(net);
		params_grad_vec = std::vector<Matrix<Scalar>>(layers.size());
		for (unsigned i = 0; i < params_grad_vec.size(); ++i) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			const Matrix<Scalar>& params_grad = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
			Matrix<Scalar> acc_params_grad = Matrix<Scalar>::Zero(params_grad.rows(), params_grad.cols());
			params_grad_vec[i] = acc_params_grad;
		}
	}
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		Scalar learning_rate = calculate_learning_rate(epoch);
		Matrix<Scalar>& params_grad = params_grad_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		params += momentum * params_grad - learning_rate * Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
	}
	/**
	 * It calculates the annealed learning rate as a function of the epoch index.
	 *
	 * @param epoch The epoch index.
	 * @return The learning rate to use.
	 */
	Scalar calculate_learning_rate(unsigned epoch) {
		return init_learning_rate / (1 + annealing_rate * epoch);
	}
	const Scalar init_learning_rate;
	const Scalar annealing_rate;
	const Scalar momentum;
	std::vector<Matrix<Scalar>> params_grad_vec;
};

/**
 * A class template for Nesterov momentum accelerated SGD optimizers.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NesterovMomentumAcceleratedSGDOptimizer : public MomentumAcceleratedSGDOptimizer<Scalar,Rank,Sequential> {
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
			unsigned batch_size = 1, Scalar init_learning_rate = 1e-3, Scalar annealing_rate = 1e-3, Scalar momentum = .9) :
				Base::MomentumAcceleratedSGDOptimizer(loss, batch_size, init_learning_rate, annealing_rate, momentum) { };
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		Scalar learning_rate = Base::calculate_learning_rate(epoch);
		Matrix<Scalar>& acc_params_grad = Base::params_grad_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		Matrix<Scalar> params_grad_bak = acc_params_grad;
		acc_params_grad = Base::momentum * acc_params_grad - learning_rate * Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
		params += -Base::momentum * params_grad_bak + (1 + Base::momentum) * acc_params_grad;
	}
};

/**
 * A class template for the Adagrad optimization algorithm.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdagradOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
public:
	/**
	 * @param loss A shared pointer to the loss function to use.
	 * @param batch_size The batch size to use for training and testing. It is expected to
	 * be greater than 0.
	 * @param learning_rate The learning rate (a.k.a. step size) to use. It is expected to
	 * be greater than 0.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline AdagradOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size = 1,
			Scalar learning_rate = 1e-2, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				learning_rate(learning_rate),
				epsilon(epsilon) {
		assert(learning_rate > 0);
		assert(epsilon > 0);
	}
	virtual ~AdagradOptimizer() = default;
protected:
public:
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Optimizer<Scalar,Rank,Sequential>::get_layers(net);
		params_grad_sqrs_vec = std::vector<Matrix<Scalar>>(layers.size());
		for (unsigned i = 0; i < params_grad_sqrs_vec.size(); ++i) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			const Matrix<Scalar>& params_grad = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
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
	inline virtual void update_acc_params_grad_sqrs(Matrix<Scalar>& acc_params_grad_sqrs,
			const Matrix<Scalar>& params_grad) {
		// Accumulate the squares of the gradients.
		acc_params_grad_sqrs += params_grad.cwiseProduct(params_grad);
	}
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		Matrix<Scalar>& params_grad_sqrs = params_grad_sqrs_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		const Matrix<Scalar>& params_grad = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
		update_acc_params_grad_sqrs(params_grad_sqrs, params_grad);
		params -= (learning_rate * params_grad.array() / (params_grad_sqrs.array().sqrt() + epsilon)).matrix();
	}
	const Scalar learning_rate;
	const Scalar epsilon;
	std::vector<Matrix<Scalar>> params_grad_sqrs_vec;
};

/**
 * A class template for the RMSProp optimizer.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class RMSPropOptimizer : public AdagradOptimizer<Scalar,Rank,Sequential> {
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
	inline RMSPropOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l2_decay = 1e-1, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				AdagradOptimizer<Scalar,Rank,Sequential>::AdagradOptimizer(loss, batch_size, learning_rate, epsilon),
				l2_decay(l2_decay) {
		assert(l2_decay >= 0 && l2_decay <= 1);
	}
protected:
	inline void update_acc_params_grad_sqrs(Matrix<Scalar>& acc_params_grad_sqrs,
			const Matrix<Scalar>& params_grad) {
		acc_params_grad_sqrs = (1 - l2_decay) * acc_params_grad_sqrs + l2_decay * params_grad.cwiseProduct(params_grad);
	}
	const Scalar l2_decay;
};

/**
 * A class template for the Adadelta optimization algorithm.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdadeltaOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
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
	inline AdadeltaOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size = 1,
			Scalar decay = 5e-2, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				SGDOptimizer<Scalar,Rank,Sequential>::SGDOptimizer(loss, batch_size),
				decay(decay),
				epsilon(epsilon) {
		assert(decay >= 0 && decay <= 1);
		assert(epsilon > 0);
	}
	inline void fit(NeuralNetwork<Scalar,Rank,Sequential>& net) {
		std::vector<Layer<Scalar,Rank>*> layers = Optimizer<Scalar,Rank,Sequential>::get_layers(net);
		pgus_vec = std::vector<ParamGradAndUpdateSqrs>(layers.size());
		for (unsigned i = 0; i < pgus_vec.size(); ++i) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			const Matrix<Scalar>& param_grads = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
			ParamGradAndUpdateSqrs pgus;
			pgus.params_grad = Matrix<Scalar>::Zero(param_grads.rows(), param_grads.cols());
			pgus.params_update = Matrix<Scalar>::Zero(pgus.params_grad.rows(), pgus.params_grad.cols());
			pgus_vec[i] = pgus;
		}
	}
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		ParamGradAndUpdateSqrs& pgus = pgus_vec[i];
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		const Matrix<Scalar>& params_grad = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
		pgus.params_grad = (1 - decay) * pgus.params_grad + decay * params_grad.cwiseProduct(params_grad);
		Matrix<Scalar> weight_updates = -params_grad.array() * (pgus.params_update.array() + epsilon).sqrt() /
				(pgus.params_grad.array() + epsilon).sqrt();
		params += weight_updates;
		pgus.params_update = (1 - decay) * pgus.params_update + decay * weight_updates.cwiseProduct(weight_updates);
	}
	const Scalar decay;
	const Scalar epsilon;
	/**
	 * A struct containing the accumulated squared gradients and squared gradients updates of a layer.
	 */
	struct ParamGradAndUpdateSqrs {
		Matrix<Scalar> params_grad;
		Matrix<Scalar> params_update;
	};
	std::vector<ParamGradAndUpdateSqrs> pgus_vec;
};

/**
 * A class template for the Adam optimization algorithm.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdamOptimizer : public SGDOptimizer<Scalar,Rank,Sequential> {
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
	inline AdamOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
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
		std::vector<Layer<Scalar,Rank>*> layers = Optimizer<Scalar,Rank,Sequential>::get_layers(net);
		pgn_vec = std::vector<ParamGradNorms>(layers.size());
		for (unsigned i = 0; i < pgn_vec.size(); ++i) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			const Matrix<Scalar>& param_grads = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
			ParamGradNorms vel;
			vel.params_grad_l1 = Matrix<Scalar>::Zero(param_grads.rows(), param_grads.cols());
			vel.params_grad_l2 = Matrix<Scalar>::Zero(vel.params_grad_l1.rows(), vel.params_grad_l1.cols());
			pgn_vec[i] = vel;
		}
	}
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		ParamGradNorms& grad_norms = pgn_vec[i];
		Scalar l1_corr = (Scalar) 1 / (1 - pow(1 - l1_decay, epoch + 1) + epsilon);
		Scalar l2_corr = (Scalar) 1 / (1 - pow(1 - l2_decay, epoch + 1) + epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		const Matrix<Scalar>& params_grad = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
		grad_norms.params_grad_l1 = (1 - l1_decay) * grad_norms.params_grad_l1 + l1_decay * params_grad;
		grad_norms.params_grad_l2 = (1 - l2_decay) * grad_norms.params_grad_l2 +
				l2_decay * params_grad.cwiseProduct(params_grad);
		params -= (learning_rate * (grad_norms.params_grad_l1 * l1_corr).array() /
				((grad_norms.params_grad_l2 * l2_corr).array() + epsilon).sqrt()).matrix();
	}
	const Scalar learning_rate;
	const Scalar l1_decay;
	const Scalar l2_decay;
	const Scalar epsilon;
	/**
	 * A struct containing the accumulated first and second norms of the parameter gradients
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
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class AdaMaxOptimizer : public AdamOptimizer<Scalar,Rank,Sequential> {
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
	inline AdaMaxOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				Base::AdamOptimizer(loss, batch_size, learning_rate, l1_decay, l2_decay, epsilon) { }
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		typename Base::ParamGradNorms& grad_norms = Base::pgn_vec[i];
		Scalar l1_corr = (Scalar) 1 / (1 - pow(1 - Base::l1_decay, epoch + 1) + Base::epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		const Matrix<Scalar>& params_grad = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
		grad_norms.params_grad_l1 = (1 - Base::l1_decay) * grad_norms.params_grad_l1 + Base::l1_decay * params_grad;
		grad_norms.params_grad_l2 = ((1 - Base::l2_decay) * grad_norms.params_grad_l2).cwiseMax(params_grad.cwiseAbs());
		params -= (Base::learning_rate * (grad_norms.params_grad_l1 * l1_corr).array() /
				(grad_norms.params_grad_l2.array() + Base::epsilon)).matrix();
	}
};

/**
 * A class template for the Nesterov accelerated Adam (Nadam) optimization algorithm.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NadamOptimizer : public AdamOptimizer<Scalar,Rank,Sequential> {
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
	inline NadamOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, unsigned batch_size = 1, Scalar learning_rate = 1e-3,
			Scalar l1_decay = 1e-1, Scalar l2_decay = 1e-3, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				Base::AdamOptimizer(loss, batch_size, learning_rate, l1_decay, l2_decay, epsilon) { }
protected:
	inline void update_params(Layer<Scalar,Rank>& layer, unsigned i, unsigned epoch) {
		typename Base::ParamGradNorms& grad_norms = Base::pgn_vec[i];
		Scalar l1_corr = (Scalar) 1 / (1 - pow(1 - Base::l1_decay, epoch + 1) + Base::epsilon);
		Scalar l1_next_corr = (Scalar) 1 / (1 - pow(1 - Base::l1_decay, epoch + 2) + Base::epsilon);
		Scalar l2_corr = (Scalar) 1 / (1 - pow(1 - Base::l2_decay, epoch + 1) + Base::epsilon);
		Matrix<Scalar>& params = Optimizer<Scalar,Rank,Sequential>::get_params(layer);
		const Matrix<Scalar>& params_grad = Optimizer<Scalar,Rank,Sequential>::get_params_grad(layer);
		grad_norms.params_grad_l1 = (1 - Base::l1_decay) * grad_norms.params_grad_l1 + Base::l1_decay * params_grad;
		grad_norms.params_grad_l2 = (1 - Base::l2_decay) * grad_norms.params_grad_l2 + Base::l2_decay *
				params_grad.cwiseProduct(params_grad);
		params -= (Base::learning_rate * (Base::l1_decay * l1_corr * params_grad +
				(1.0 - Base::l1_decay) * l1_next_corr * grad_norms.params_grad_l1).array() /
				((grad_norms.params_grad_l2 * l2_corr).array() + Base::epsilon).sqrt()).matrix();
	}
};

} /* namespace cattle */

#endif /* CATTL3_OPTIMIZER_H_ */
