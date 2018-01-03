/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <Activation.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include <Matrix.h>
#include <Utils.h>
#include <utility>
#include <Vector.h>
#include <WeightInitialization.h>

#include <iostream>

namespace cppnn {

// Forward declarations to NeuralNetwork and Optimizer so they can be friended.
template<typename Scalar>
class NeuralNetwork;
template<typename Scalar>
class Optimizer;

template<typename Scalar>
class Layer {
	friend class NeuralNetwork<Scalar>;
	friend class Optimizer<Scalar>;
public:
	virtual ~Layer() = default;
	// Clone pattern.
	virtual Layer<Scalar>* clone();
	virtual unsigned get_prev_size() const = 0;
	virtual unsigned get_size() const = 0;
	virtual bool get_batch_norm() const = 0;
protected:
	/* Only expose methods that allow for the modification of the
	 * layer's state to friends and sub-classes. */
	virtual void init() = 0;
	virtual void empty_cache() = 0;
	virtual Matrix<Scalar>& get_weights() = 0;
	virtual const Matrix<Scalar>& get_weight_grads() const = 0;
	virtual RowVector<Scalar>& get_betas() = 0;
	virtual const RowVector<Scalar>& get_beta_grads() const = 0;
	virtual RowVector<Scalar>& get_gammas() = 0;
	virtual const RowVector<Scalar>& get_gamma_grads() const = 0;
	virtual void enforce_constraints() = 0;
	virtual Matrix<Scalar> pass_forward(Matrix<Scalar> prev_out, bool training) = 0;
	virtual Matrix<Scalar> pass_back(Matrix<Scalar> out_grads) = 0;
};

/**
 * A class template for a fully connected layer of a neural network.
 *
 * The layer representation has its weights before its neurons. This less
 * intuitive, reverse implementation allows for a more convenient
 * definition of neural network architectures as the input layer is not
 * normally activated, while the output layer often is.
 */
template<typename Scalar>
class FCLayer : public Layer<Scalar> {
public:
	FCLayer(unsigned prev_size, unsigned size, const WeightInitialization<Scalar>& weight_init,
			const Activation<Scalar>& act, Scalar dropout_prob = 0, bool batch_norm = false,
			Scalar norm_avg_decay = 5e-2, Scalar max_norm_constraint = 0, Scalar epsilon = Utils<Scalar>::EPSILON) :
				prev_size(prev_size),
				size(size),
				weight_init(weight_init),
				act(act),
				dropout_prob(dropout_prob),
				dropout(Utils<Scalar>::decidedly_greater(dropout_prob, (Scalar) .0, epsilon, epsilon)),
				batch_norm(batch_norm),
				norm_avg_decay(norm_avg_decay),
				max_norm_constraint(max_norm_constraint),
				max_norm(Utils<Scalar>::decidedly_greater(max_norm_constraint, (Scalar) .0, epsilon, epsilon)),
				epsilon(epsilon),
				weights(prev_size + 1, size),
				weight_grads(prev_size + 1, size),
				betas(batch_norm ? prev_size : 0),
				beta_grads(batch_norm ? prev_size : 0),
				gammas(batch_norm ? prev_size : 0),
				gamma_grads(batch_norm ? prev_size : 0),
				avg_means(batch_norm ? prev_size : 0),
				avg_vars(batch_norm ? prev_size : 0),
				moving_means_init(false) {
		assert(prev_size > 0 && "prev size must be greater than 0");
		assert(size > 0 && "size must be greater than 0");
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(dropout_prob <= 1 && "dropout prob must not be greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	Layer<Scalar>* clone() {
		return new FCLayer(*this);
	};
	unsigned get_prev_size() const {
		return prev_size;
	};
	unsigned get_size() const {
		return size;
	};
	bool get_batch_norm() const {
		return batch_norm;
	};
protected:
	void init() {
		weight_init.apply(weights);
		if (batch_norm) {
			betas.setZero(betas.cols());
			gammas.setOnes(gammas.cols());
		}
		avg_means.setZero(avg_means.cols());
		avg_vars.setZero(avg_vars.cols());
		moving_means_init = false;
	};
	void empty_cache() {
		std_prev_out_factor = RowVector<Scalar>(0);
		centered_prev_out = Matrix<Scalar>(0, 0);
		dropout_mask = Matrix<Scalar>(0, 0);
		biased_prev_out = Matrix<Scalar>(0, 0);
		in = Matrix<Scalar>(0, 0);
		out = Matrix<Scalar>(0, 0);
	};
	Matrix<Scalar>& get_weights() {
		return weights;
	};
	const Matrix<Scalar>& get_weight_grads() const {
		return weight_grads;
	};
	// Batch normalization parameters.
	RowVector<Scalar>& get_betas(){
		return betas;
	};
	const RowVector<Scalar>& get_beta_grads() const {
		return beta_grads;
	};
	RowVector<Scalar>& get_gammas() {
		return gammas;
	};
	const RowVector<Scalar>& get_gamma_grads() const {
		return gamma_grads;
	};
	void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = weights.squaredNorm();
			if (l2_norm > max_norm_constraint)
				weights *= (max_norm_constraint / l2_norm);
		}
	};
	Matrix<Scalar> pass_forward(Matrix<Scalar> prev_out, bool training) {
		assert((unsigned) prev_out.cols() == prev_size &&
				"illegal input matrix size for feed forward");
		// Batch normalization.
		if (batch_norm) {
			if (training) {
//				std::cout << "prev_out:" << std::endl << prev_out << std::endl << std::endl;
				RowVector<Scalar> means = prev_out.colwise().mean();
				centered_prev_out = prev_out.rowwise() - means;
//				std::cout << "centered_prev_out:" << std::endl << centered_prev_out << std::endl << std::endl;
				RowVector<Scalar> vars = centered_prev_out.array().square().matrix().colwise().mean();
				std_prev_out_factor = (vars.array() + epsilon).cwiseSqrt().cwiseInverse();
				prev_out = centered_prev_out * std_prev_out_factor.asDiagonal();
//				std::cout << "standardized_prev_out:" << std::endl << prev_out << std::endl << std::endl;
				/* Maintain a moving average of means and variances
				 * for testing. */
				if (moving_means_init) {
					avg_means = (1.0 - norm_avg_decay) * avg_means + norm_avg_decay * means;
					avg_vars = (1.0 - norm_avg_decay) * avg_vars + norm_avg_decay * vars;
				} else {
					avg_means = means;
					avg_vars = vars;
					moving_means_init = true;
				}
			} else if (moving_means_init) { // For testing, use the moving averages.
				prev_out = (prev_out.rowwise() - avg_means) *
						(avg_vars.array() + epsilon).cwiseSqrt().cwiseInverse().matrix().asDiagonal();
			}
			prev_out = (prev_out * gammas.asDiagonal()).rowwise() + betas;
//			std::cout << "beta_gamma_adjusted_prev_out:" << std::endl << prev_out << std::endl << std::endl;
		}
		// Dropout.
		if (training && dropout) {
			Matrix<Scalar> dropout_mask(prev_out.rows(), prev_out.cols());
			dropout_mask.setRandom(prev_out.rows(), prev_out.cols());
			Scalar scaling_factor = 1 / (1 - dropout_prob + epsilon);
			dropout_mask = ((dropout_mask.array() + 1) / 2).unaryExpr([this,scaling_factor](Scalar i) {
				return (Scalar) (i <= dropout_prob ? .0 : scaling_factor);
			});
			prev_out = prev_out.cwiseProduct(dropout_mask);
			this->dropout_mask = std::move(dropout_mask);
		}
		// Add a 1-column to the input for the bias trick.
		biased_prev_out = Matrix<Scalar>(prev_out.rows(), prev_size + 1);
		biased_prev_out.leftCols(prev_size) = std::move(prev_out);
		biased_prev_out.col(prev_size).setOnes();
		/* Compute the neuron inputs by multiplying the output of the
		 * previous layer by the weights. */
		in = biased_prev_out * weights;
		// Activate the neurons.
		out = act.function(in);
		return out;
	};
	Matrix<Scalar> pass_back(Matrix<Scalar> out_grads) {
		assert((unsigned) out_grads.cols() == size &&
				out_grads.rows() == out.rows() &&
				"illegal input matrix size for feed back");
		/* Compute the gradients of the outputs with respect to the
		 * weighted inputs. */
		Matrix<Scalar> in_grads = act.d_function(in, out, out_grads);
		weight_grads = biased_prev_out.transpose() * in_grads;
		/* Remove the bias column from the transposed weight matrix
		 * and compute the out-gradients of the previous layer. */
		Matrix<Scalar> prev_out_grads = in_grads * weights.transpose().leftCols(prev_size);
		// Derivate the dropout 'function' as well and apply the chain rule.
		if (dropout)
			prev_out_grads = prev_out_grads.cwiseProduct(dropout_mask);
		if (batch_norm) {
			/* Back-propagate the gradient through the batch
			 * normalization 'function' and also calculate the
			 * gradients on the betas and gammas. */
			beta_grads = prev_out_grads.colwise().sum();
			gamma_grads = (prev_out_grads.cwiseProduct(centered_prev_out * std_prev_out_factor.asDiagonal()))
					.colwise().sum();
			Matrix<Scalar> normalized_out_grads = prev_out_grads * gammas.asDiagonal();
			Matrix<Scalar> centered_out_grads = normalized_out_grads * std_prev_out_factor.asDiagonal() +
					((Matrix<Scalar>::Ones(prev_out_grads.rows(), prev_out_grads.cols()) / prev_out_grads.rows()) *
					((normalized_out_grads.cwiseProduct(centered_prev_out).colwise().sum()).array() *
					(std_prev_out_factor.array().pow(3) * -.5)).matrix().asDiagonal())
					.cwiseProduct(2 * centered_prev_out);
			prev_out_grads = centered_out_grads +
					(Matrix<Scalar>::Ones(prev_out_grads.rows(), prev_out_grads.cols()) / prev_out_grads.rows()) *
					(-1 * centered_out_grads.colwise().sum()).asDiagonal();
		}
		return prev_out_grads;
	};
private:
	unsigned prev_size;
	unsigned size;
	const WeightInitialization<Scalar>& weight_init;
	const Activation<Scalar>& act;
	Scalar dropout_prob;
	bool dropout;
	bool batch_norm;
	Scalar norm_avg_decay;
	Scalar max_norm_constraint;
	bool max_norm;
	Scalar epsilon;
	/* Eigen matrices are backed by arrays allocated the heap, so these
	 * members do not burden the stack. */
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	// Dynamic batch normalization parameters.
	RowVector<Scalar> betas;
	RowVector<Scalar> beta_grads;
	RowVector<Scalar> gammas;
	RowVector<Scalar> gamma_grads;
	RowVector<Scalar> avg_means;
	RowVector<Scalar> avg_vars;
	bool moving_means_init;
	// Staged computation caches
	RowVector<Scalar> std_prev_out_factor;
	Matrix<Scalar> centered_prev_out;
	Matrix<Scalar> dropout_mask;
	Matrix<Scalar> biased_prev_out;
	Matrix<Scalar> in;
	Matrix<Scalar> out;
};

} /* namespace cppnn */

#endif /* LAYER_H_ */
