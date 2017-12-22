/*
 * Optimizer.h
 *
 *  Created on: 6 Dec 2017
 *      Author: Viktor Csomor
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <Loss.h>
#include <Matrix.h>
#include <NeuralNetwork.h>
#include <NumericUtils.h>
#include <random>
#include <Regularization.h>
#include <sstream>
#include <string>
#include <vector>

namespace cppnn {

template<typename Scalar>
class Optimizer {
public:
	Optimizer(const Regularization<Scalar>& reg, const Loss<Scalar>& loss) :
			reg(reg),
			loss(loss) { };
	virtual ~Optimizer() = default;
	virtual bool validate_gradients(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x,
			const Matrix<Scalar>& y, Scalar step_size = 1e-8, Scalar abs_epsilon = 1e-10,
			Scalar rel_epsilon = 1e-4) const {
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
					Scalar loss_inc = loss.function(net.propagate(x, true), y).sum();
					weights(j,k) = weight - step_size;
					Scalar loss_dec = loss.function(net.propagate(x, true), y).sum();
					weights(j,k) = weight;
					Scalar num_grad = (loss_inc - loss_dec) / (2 * step_size);
					std::cout << "\tNumerical gradient = " << num_grad;
					if (!almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
						std::cout << " <<<<<FAIL>>>>>";
						failure = true;
					}
					std::cout << std::endl;
				}
			}
			RowVector<Scalar>& gammas = layer.get_gammas();
			const RowVector<Scalar>& gamma_grads = layer.get_gamma_grads();
			for (int j = 0; j < gammas.cols(); j++) {
				std::cout << "Gamma[" << i << "," << j << "]:" << std::endl;
				Scalar ana_grad = gamma_grads(j);
				std::cout << "\tAnalytic gradient = " << ana_grad << std::endl;
				Scalar gamma = gammas(j);
				gammas(j) = gamma + step_size;
				Scalar loss_inc = loss.function(net.propagate(x, true), y).sum();
				gammas(j) = gamma - step_size;
				Scalar loss_dec = loss.function(net.propagate(x, true), y).sum();
				gammas(j) = gamma;
				Scalar num_grad = (loss_inc - loss_dec) / (2 * step_size);
				std::cout << "\tNumerical gradient = " << num_grad;
				if (!almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
					std::cout << " <<<<<FAIL>>>>>";
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
				Scalar loss_inc = loss.function(net.propagate(x, true), y).sum();
				betas(j) = beta - step_size;
				Scalar loss_dec = loss.function(net.propagate(x, true), y).sum();
				betas(j) = beta;
				Scalar num_grad = (loss_inc - loss_dec) / (2 * step_size);
				std::cout << "\tNumerical gradient = " << num_grad;
				if (!almost_equal(ana_grad, num_grad, abs_epsilon, rel_epsilon)) {
					std::cout << " <<<<<FAIL>>>>>";
					failure = true;
				}
				std::cout << std::endl;
			}
		}
		return !failure;
	};
	virtual void train(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x, const Matrix<Scalar>& y,
			unsigned epochs = 1000, unsigned batch_size = 1, bool cross_validate = false,
			unsigned k = 5) = 0;
protected:
	const Regularization<Scalar>& reg;
	const Loss<Scalar>& loss;
};

template<typename Scalar>
class SGDOptimizer : public Optimizer<Scalar> {
public:
	SGDOptimizer(const Regularization<Scalar>& reg, const Loss<Scalar>& loss) :
			Optimizer<Scalar>::Optimizer(reg, loss) {

	};
	void train(NeuralNetwork<Scalar>& net, const Matrix<Scalar>& x, const Matrix<Scalar>& y,
				unsigned epochs = 1000, unsigned batch_size = 1, bool cross_validate = false,
				unsigned k = 5) {

	};
};

template<typename Scalar>
class NadamOptimizer : public Optimizer<Scalar> {
public:
	NadamOptimizer(const Regularization<Scalar>& reg, const Loss<Scalar>& loss) :
			Optimizer<Scalar>::Optimizer(reg, loss) {

	};
};

} /* namespace cppnn */

#endif /* OPTIMIZER_H_ */
