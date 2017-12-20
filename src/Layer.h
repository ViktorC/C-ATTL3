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
#include <Initialization.h>
#include <Matrix.h>
#include <NumericalUtils.h>
#include <string>
#include <utility>
#include <Vector.h>

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
	virtual unsigned get_prev_nodes() const = 0;
	virtual unsigned get_nodes() const = 0;
	virtual bool get_batch_norm() const = 0;
	virtual Scalar get_dropout() const = 0;
	virtual Scalar get_norm_stats_momentum() const = 0;
	virtual const Activation<Scalar>& get_act() const = 0;
	virtual const Initialization<Scalar>& get_init() const = 0;
protected:
	/* Only expose methods that allow for the modification of the
	 * layer's state to friends and sub-classes. */
	virtual void reset() = 0;
	virtual void set_batch_norm(bool on) = 0;
	virtual void set_dropout(Scalar dropout) = 0;
	virtual void set_norm_stats_momentum(Scalar norm_stats_momentum) = 0;
	virtual Matrix<Scalar>& get_weights() = 0;
	virtual Matrix<Scalar>& get_weight_grads() = 0;
	virtual RowVector<Scalar>& get_betas() = 0;
	virtual RowVector<Scalar>& get_beta_grads() = 0;
	virtual RowVector<Scalar>& get_gammas() = 0;
	virtual RowVector<Scalar>& get_gamma_grads() = 0;
	virtual RowVector<Scalar>& get_moving_means() = 0;
	virtual RowVector<Scalar>& get_moving_vars() = 0;
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
	FCLayer(unsigned prev_nodes, unsigned nodes, const Activation<Scalar>& act,
			const Initialization<Scalar>& init, bool batch_norm = true, Scalar dropout = .5,
			Scalar norm_stats_momentum = .9, Scalar epsilon = EPSILON) :
				prev_nodes(prev_nodes),
				nodes(nodes),
				act(act),
				init(init),
				batch_norm(batch_norm),
				dropout(dropout),
				norm_stats_momentum(norm_stats_momentum),
				epsilon(epsilon),
				weights(prev_nodes + 1, nodes),
				weight_grads(prev_nodes + 1, nodes),
				betas(batch_norm ? prev_nodes : 0),
				beta_grads(batch_norm ? prev_nodes : 0),
				gammas(batch_norm ? prev_nodes : 0),
				gamma_grads(batch_norm ? prev_nodes : 0),
				moving_means(batch_norm ? prev_nodes : 0),
				moving_vars(batch_norm ? prev_nodes : 0),
				moving_means_init(false) {
		assert(prev_nodes > 0 && "prev_nodes must be greater than 0");
		assert(nodes > 0 && "nodes must be greater than 0");
		assert(norm_stats_momentum >= 0 && norm_stats_momentum <= 1 &&
				"norm_stats_momentum must not be less than 0 or greater than 1");
		assert(dropout >= 0 && dropout <= 1 && "dropout must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
		reset();
	};
	// Clone pattern.
	Layer<Scalar>* clone() {
		return new FCLayer(*this);
	};
	unsigned get_prev_nodes() const {
		return prev_nodes;
	};
	unsigned get_nodes() const {
		return nodes;
	};
	bool get_batch_norm() const {
		return batch_norm;
	};
	Scalar get_dropout() const {
		return dropout;
	};
	Scalar get_norm_stats_momentum() const {
		return norm_stats_momentum;
	};
	const Activation<Scalar>& get_act() const {
		return act;
	};
	const Initialization<Scalar>& get_init() const {
		return init;
	};
protected:
	void reset() {
		init.init(weights);
		if (batch_norm) {
			betas.setZero(betas.cols());
			gammas.setOnes(gammas.cols());
		}
		moving_means.setZero(moving_means.cols());
		moving_vars.setZero(moving_vars.cols());
		moving_means_init = false;
		// Empty the caches.
		std_prev_out_factor = RowVector<Scalar>(0);
		centered_prev_out = Matrix<Scalar>(0, 0);
		biased_prev_out = Matrix<Scalar>(0, 0);
		in = Matrix<Scalar>(0, 0);
		out = Matrix<Scalar>(0, 0);
	};
	void set_batch_norm(bool on) {
		batch_norm = on;
	};
	void set_dropout(Scalar dropout) {
		this->dropout = dropout;
	};
	void set_norm_stats_momentum(Scalar norm_stats_momentum) {
		this->norm_stats_momentum = norm_stats_momentum;
	};
	Matrix<Scalar>& get_weights() {
		return weights;
	};
	Matrix<Scalar>& get_weight_grads() {
		return weight_grads;
	};
	// Batch normalization parameters.
	RowVector<Scalar>& get_betas(){
		return betas;
	};
	RowVector<Scalar>& get_beta_grads() {
		return beta_grads;
	};
	RowVector<Scalar>& get_gammas() {
		return gammas;
	};
	RowVector<Scalar>& get_gamma_grads() {
		return gamma_grads;
	};
	RowVector<Scalar>& get_moving_means() {
		return moving_means;
	};
	RowVector<Scalar>& get_moving_vars() {
		return moving_vars;
	};
	Matrix<Scalar> pass_forward(Matrix<Scalar> prev_out,
			bool training) {
		assert((unsigned) prev_out.cols() == prev_nodes &&
				"illegal input matrix size for feed forward");
		// Batch normalization.
		if (batch_norm) {
			if (training) {
				RowVector<Scalar> means = prev_out.colwise().mean();
				centered_prev_out = prev_out.rowwise() - means;
				RowVector<Scalar> vars = centered_prev_out.array().square().matrix().colwise().mean();
				std_prev_out_factor = (vars.array() + epsilon).cwiseSqrt().cwiseInverse();
				prev_out = centered_prev_out * std_prev_out_factor.asDiagonal();
				/* Maintain a moving average of means and variances
				 * for testing. */
				if (moving_means_init) {
					moving_means = moving_means * norm_stats_momentum + means * (1 - norm_stats_momentum);
					moving_vars = moving_vars * norm_stats_momentum + vars * (1 - norm_stats_momentum);
				} else {
					moving_means = means;
					moving_vars = vars;
					moving_means_init = true;
				}
			} else {
				// For testing, use the moving averages.
				prev_out = (prev_out.rowwise() - moving_means) *
						(moving_vars.array() + epsilon).cwiseSqrt().cwiseInverse().matrix().asDiagonal();
			}
			prev_out = (prev_out * gammas.asDiagonal()).rowwise() + betas;
		}
		// Dropout.
		if (greater(dropout, 0)) {
			Matrix<Scalar> dropout_mask;
			dropout_mask.setRandom(prev_out.rows(), prev_out.cols());
			dropout_mask = (dropout_mask.array() + 1) / 2;
			dropout_mask = dropout_mask.unaryExpr([this](Scalar i) {
				return (Scalar) (greater(i, dropout) ? 1 : 0);
			});
			dropout_mask /= std::max(1 - dropout, EPSILON);
			prev_out = prev_out.cwiseProduct(dropout_mask);
		}
		// Add a 1-column to the input for the bias trick.
		biased_prev_out = Matrix<Scalar>(prev_out.rows(), prev_nodes + 1);
		biased_prev_out.leftCols(prev_nodes) = std::move(prev_out);
		biased_prev_out.col(prev_nodes).setOnes();
		/* Compute the neuron inputs by multiplying the output of the
		 * previous layer by the weights. */
		in = biased_prev_out * weights;
		// Activate the neurons.
		out = act.function(in);
		return out;
	};
	Matrix<Scalar> pass_back(Matrix<Scalar> out_grads) {
		assert((unsigned) out_grads.cols() == nodes &&
				out_grads.rows() == out.rows() &&
				"illegal input matrix size for feed back");
		/* Compute the gradients of the outputs with respect to the
		 * weighted inputs. */
		Matrix<Scalar> in_grads = act.d_function(in, out).cwiseProduct(out_grads);
		weight_grads = biased_prev_out.transpose() * in_grads;
		/* Remove the bias column from the transposed weight matrix
		 * and compute the out-gradients of the previous layer. */
		Matrix<Scalar> prev_out_grads = in_grads * weights.transpose().leftCols(prev_nodes);
		if (batch_norm) {
			/* Back-propagate the gradient through the batch
			 * normalization function and also calculate the
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
	static constexpr Scalar EPSILON = 1e-5;
	unsigned prev_nodes;
	unsigned nodes;
	const Activation<Scalar>& act;
	const Initialization<Scalar>& init;
	bool batch_norm;
	Scalar dropout;
	Scalar norm_stats_momentum;
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
	RowVector<Scalar> moving_means;
	RowVector<Scalar> moving_vars;
	bool moving_means_init;
	// Staged computation caches
	RowVector<Scalar> std_prev_out_factor;
	Matrix<Scalar> centered_prev_out;
	Matrix<Scalar> biased_prev_out;
	Matrix<Scalar> in;
	Matrix<Scalar> out;
};

} /* namespace cppnn */

#endif /* LAYER_H_ */
