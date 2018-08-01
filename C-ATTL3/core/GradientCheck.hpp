/*
 * GradientCheck.hpp
 *
 *  Created on: 31 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GRADIENTCHECK_H_
#define C_ATTL3_CORE_GRADIENTCHECK_H_

#include <cassert>
#include <type_traits>

#include "DataProvider.hpp"
#include "NeuralNetwork.hpp"
#include "Loss.hpp"

namespace cattle {

/**
 * A utility class for performing gradient checks on neural network and loss function
 * implementations.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class GradientCheck {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal optimizer rank");
	typedef DataProvider<Scalar,Rank,Sequential> Provider;
	typedef NeuralNetwork<Scalar,Rank,Sequential> Net;
	typedef Loss<Scalar,Rank,Sequential> Loss;
public:
	/**
	 * It performs a gradient check to verify the correctness of the neural network and layer implementations.
	 * It is recommended to use double precision floating points.
	 *
	 * @param provider A reference to the data provider to use for the gradient check.
	 * @param net A reference to the network on which the gradient check is to be performed.
	 * @param loss The loss function to use for the gradient check.
	 * @param verbose Whether the analytic and numerical derivatives of the variables should be printed to the
	 * standard out stream.
	 * @param step_size The step size for numerical differentiation.
	 * @param abs_epsilon The maximum acceptable absolute difference between the numerical and analytic
	 * gradients.
	 * @param rel_epsilon The maximum acceptable relative (to the greater out of the two) difference between
	 * the numerical and analytic gradients.
	 * @return Whether the gradient check has been passed or failed.
	 */
	inline static bool verify_gradients(Provider& provider, Net& net, const Loss& loss, bool verbose = true,
			Scalar step_size = NumericUtils<Scalar>::EPSILON2 / 2,
			Scalar abs_epsilon = NumericUtils<Scalar>::EPSILON2,
			Scalar rel_epsilon = NumericUtils<Scalar>::EPSILON2) {
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
		net.backpropagate(loss.d_function(net.propagate(data_pair.first, true),
				data_pair.second) / (Scalar) instances);
		std::vector<Parameters<Scalar>*> params_vec = net.get_unique_optimizable_params();
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
						std::cout << "\tparam[" << i << "," << j << "," << k << "]:" << std::endl;
					Scalar ana_grad = params_grad(j,k);
					if (verbose)
						std::cout << "\t\tanalytic gradient = " << ana_grad << std::endl;
					Scalar param = params_values(j,k);
					params_values(j,k) = param + step_size;
					params.set_values(params_values);
					/* Compute the numerical gradients in training mode to ensure that the means and standard
					 * deviations used for batch normalization are the same as those used during the analytic
					 * gradient computation. */
					Scalar loss_inc = loss.function(net.propagate(data_pair.first, true),
							data_pair.second).mean();
					/* Calculate the new regularization penalty as its derivative w.r.t. the layer's
					 * parameters is included in the gradient. */
					Scalar reg_pen_inc = params.get_regularization_penalty();
					params_values(j,k) = param - step_size;
					params.set_values(params_values);
					Scalar loss_dec = loss.function(net.propagate(data_pair.first, true),
							data_pair.second).mean();
					Scalar reg_pen_dec = params.get_regularization_penalty();
					params_values(j,k) = param;
					params.set_values(params_values);
					// Include the regularization penalty as well.
					Scalar num_grad = (loss_inc + reg_pen_inc - (loss_dec + reg_pen_dec)) / (2 * step_size);
					if (verbose)
						std::cout << "\t\tnumerical gradient = " << num_grad;
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
};

} /* namespace cattle */

#endif /* C_ATTL3_CORE_GRADIENTCHECK_H_ */
