/*
 * Optimizer.hpp
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_OPTIMIZER_H_
#define CATTL3_OPTIMIZER_H_

#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <type_traits>

#include "DataProvider.hpp"
#include "Loss.hpp"
#include "NeuralNetwork.hpp"
#include "NumericUtils.hpp"

namespace cattle {

/**
 * An alias for a unique pointer to a loss function of arbitrary rank, scalar type and
 * sequentiality.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using LossSharedPtr = std::shared_ptr<Loss<Scalar,Rank,Sequential>>;

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
		std::vector<Parameters<Scalar>*> params_vec = get_unique_optimizable_params(net);
		for (std::size_t i = 0; i < params_vec.size(); ++i) {
			Parameters<Scalar>& params = *(params_vec[i]);
			if (verbose) {
				std::cout << "Parameter Set " << std::setw(3) << std::to_string(i) <<
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
	/**
	 * Returns a vector of unique pointers to the optimizable parameters of a network.
	 *
	 * @param net The network whose optimizable parameters are to be retrieved.
	 * @return A vector of pointers to the unique, optimizable parameters of the network.
	 */
	inline static std::vector<Parameters<Scalar>*> get_unique_optimizable_params(Net& net) {
		std::vector<Parameters<Scalar>*> params_vec;
		std::set<Parameters<Scalar>*> params_set;
		for (auto layer_ptr : net.get_layers()) {
			if (!layer_ptr)
				continue;
			for (auto params_ptr : layer_ptr->get_params()) {
				if (params_ptr && params_ptr->are_optimizable() &&
						params_set.find(params_ptr) == params_set.end()) {
					params_set.insert(params_ptr);
					params_vec.push_back(params_ptr);
				}
			}
		}
		return params_vec;
	}
	const LossSharedPtr<Scalar,Rank,Sequential> loss;
};

} /* namespace cattle */

#endif /* CATTL3_OPTIMIZER_H_ */
