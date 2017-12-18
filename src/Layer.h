/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <Activation.h>
#include <cassert>
#include <cmath>
#include <Initialization.h>
#include <Matrix.h>
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
	virtual Scalar get_dropout() const = 0;
	virtual bool get_batch_norm() const = 0;
	virtual Scalar get_norm_stats_momentum() const = 0;
	virtual const Activation<Scalar>& get_act() const = 0;
	virtual const Initialization<Scalar>& get_init() const = 0;
protected:
	/* Only expose methods that allow for the modification of the
	 * layer's state to friends and sub-classes. */
	virtual Matrix<Scalar>& get_weights() = 0;
	virtual const Matrix<Scalar>& get_weight_grads() const = 0;
	// Batch normalization parameters.
	virtual RowVector<Scalar>& get_betas() = 0;
	virtual const RowVector<Scalar>& get_beta_grads() const = 0;
	virtual RowVector<Scalar>& get_gammas() = 0;
	virtual const RowVector<Scalar>& get_gamma_grads() const = 0;
	virtual const RowVector<Scalar>& get_moving_means() const = 0;
	virtual const RowVector<Scalar>& get_moving_vars() const = 0;
	virtual void init_params() = 0;
	virtual Matrix<Scalar> feed_forward(Matrix<Scalar> prev_out,
			bool training) = 0;
	virtual Matrix<Scalar> feed_back(Matrix<Scalar> out_grads) = 0;
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
			const Initialization<Scalar>& init, Scalar dropout = 0,
			bool batch_norm = false, Scalar norm_stats_momentum = 0.9,
			Scalar epsilon = EPSILON) :
				prev_nodes(prev_nodes),
				nodes(nodes),
				act(act),
				init(init),
				dropout(dropout),
				batch_norm(batch_norm),
				norm_stats_momentum(norm_stats_momentum),
				epsilon(epsilon),
				weights(prev_nodes + 1, nodes),
				weight_grads(prev_nodes + 1, nodes),
				betas(batch_norm ? nodes : 0),
				beta_grads(batch_norm ? nodes : 0),
				gammas(batch_norm ? nodes : 0),
				gamma_grads(batch_norm ? nodes : 0),
				moving_means(batch_norm ? nodes : 0),
				moving_vars(batch_norm ? nodes : 0),
				moving_means_init(false) {
		assert(prev_nodes > 0 && "prev_nodes must be greater than 0");
		assert(nodes > 0 && "nodes must be greater than 0");
		assert(norm_stats_momentum >= 0 && norm_stats_momentum <= 1 &&
				"norm_stats_momentum must not be less than 0 or greater "
				"than 1");
		assert(dropout >= 0 && dropout <= 1 && "dropout must not be less "
				"than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
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
	Scalar get_dropout() const {
		return dropout;
	};
	bool get_batch_norm() const {
		return batch_norm;
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
	static constexpr Scalar EPSILON = 1e-5;
	unsigned prev_nodes;
	unsigned nodes;
	const Activation<Scalar>& act;
	const Initialization<Scalar>& init;
	Scalar dropout;
	bool batch_norm;
	Scalar norm_stats_momentum;
	Scalar epsilon;
	/* Eigen matrices are backed by arrays allocated the heap, so these
	 * members do not burden the stack. */
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	RowVector<Scalar> betas;
	RowVector<Scalar> beta_grads;
	RowVector<Scalar> gammas;
	RowVector<Scalar> gamma_grads;
	RowVector<Scalar> moving_means;
	RowVector<Scalar> moving_vars;
	bool moving_means_init;
	Matrix<Scalar> biased_prev_out;
	Matrix<Scalar> in;
	Matrix<Scalar> out;
	RowVector<Scalar> means;
	RowVector<Scalar> vars;
	Matrix<Scalar> normalized_out;
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
	const RowVector<Scalar>& get_moving_means() const {
		return moving_means;
	};
	const RowVector<Scalar>& get_moving_vars() const {
		return moving_vars;
	};
	void init_params() {
		init.init(weights);
		if (batch_norm) {
			betas.setZero(nodes);
			gammas.setOnes(nodes);
		}
		moving_means.setZero(moving_means.cols());
		moving_vars.setZero(moving_vars.cols());
		moving_means_init = false;
	};
	Matrix<Scalar> feed_forward(Matrix<Scalar> prev_out,
			bool training) {
		assert((unsigned) prev_out.cols() == prev_nodes &&
				"illegal input matrix size for feed forward");
		// Add a 1-column to the input for the bias trick.
		biased_prev_out = Matrix<Scalar>(prev_out.rows(), prev_nodes + 1);
		biased_prev_out.leftCols(prev_nodes) = std::move(prev_out);
		biased_prev_out.col(prev_nodes).setOnes();
		/* Compute the neuron inputs by multiplying the output of the
		 * previous layer by the weights. */
		in = biased_prev_out * weights;
		// Activate the neurons.
		out = act.function(in);
		if (batch_norm) { // Batch normalization.
			means = out.colwise().mean();
			vars = (out.rowwise() - means).array().square()
					.matrix().colwise().mean();
			if (training) {
				normalized_out = (out.rowwise() - means);
				for (int i = 0; i < normalized_out.cols(); i++) {
					normalized_out.col(i) /= sqrt(vars(i) + epsilon);
				}
				/* Maintain a moving average of means and variances
				 * for testing. */
				if (moving_means_init) {
					moving_means = moving_means * norm_stats_momentum +
							means * (1 - norm_stats_momentum);
					moving_vars = moving_vars * norm_stats_momentum +
							vars * (1 - norm_stats_momentum);
				} else {
					moving_means = means;
					moving_vars = vars;
					moving_means_init = true;
				}
			} else {
				// For testing, use the moving averages.
				normalized_out = (out.rowwise() - moving_means);
				for (int i = 0; i < normalized_out.cols(); i++) {
					normalized_out.col(i) /= sqrt(moving_vars(i) +
							epsilon);
				}
			}
			return (normalized_out * gammas.asDiagonal()).rowwise() + betas;
		}
		return out;
	};
	Matrix<Scalar> feed_back(Matrix<Scalar> out_grads) {
		assert((unsigned) out_grads.cols() == nodes &&
				out_grads.rows() == out.rows() &&
				"illegal input matrix size for feed back");
		if (batch_norm) {
			/* Back-propagate the gradient through the batch
			 * normalization function and also calculate the
			 * gradients on the betas and gammas. */
			beta_grads = out_grads.colwise().sum();
			gamma_grads = out_grads.cwiseProduct(normalized_out)
					.colwise().sum();
			Matrix<Scalar> normalized_out_grads = out_grads * gammas
					.asDiagonal();
			Matrix<Scalar> centered_out = out.rowwise() - means;
			RowVector<Scalar> std_factor = (vars.array() + epsilon)
					.cwiseSqrt().cwiseInverse();
			Matrix<Scalar> centered_out_grads1 = normalized_out_grads *
					std_factor.asDiagonal();
			RowVector<Scalar> std_factor_grads = normalized_out_grads
					.cwiseProduct(centered_out).colwise().sum();
			RowVector<Scalar> var_grads = std_factor_grads.array() *
					(std_factor.array().pow(3) * -.5);
			Matrix<Scalar> centered_out_grads2 = ((Matrix<Scalar>::Ones(
					centered_out.rows(), centered_out.cols()) /
					centered_out.rows()) * var_grads.asDiagonal())
					.cwiseProduct(2 * centered_out);
			Matrix<Scalar> centered_out_grads = centered_out_grads1 +
					centered_out_grads2;
			Matrix<Scalar> out_grads1 = centered_out_grads;
			RowVector<Scalar> mean_grads = -1 * centered_out_grads.colwise()
					.sum();
			Matrix<Scalar> out_grads2 = (Matrix<Scalar>::Ones(out.rows(),
					out.cols()) / out.rows()) * mean_grads.asDiagonal();
			out_grads = out_grads1 + out_grads2;
		}
		/* Compute the gradients of the outputs with respect to the
		 * weighted inputs. */
		Matrix<Scalar> in_grads = act.d_function(in, out)
				.cwiseProduct(out_grads);
		weight_grads = biased_prev_out.transpose() * in_grads;
		/* Remove the bias column from the transposed weight matrix
		 * and compute the out-gradients of the previous layer. */
		return in_grads * weights.transpose().block(0, 0, nodes,
				prev_nodes);
	};
};

} /* namespace cppnn */

#endif /* LAYER_H_ */
