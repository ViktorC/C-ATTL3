/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include <Matrix.h>
#include <Utils.h>
#include <utility>
#include <Vector.h>
#include <WeightInitialization.h>

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
	virtual Layer<Scalar>* clone() = 0;
	virtual unsigned get_input_size() const = 0;
	virtual unsigned get_output_size() const = 0;
protected:
	bool is_parametric() {
		return get_params().rows() > 0 && get_params().cols() > 0;
	};
	/* Only expose methods that allow for the modification of the
	 * layer's state to friends and sub-classes. */
	virtual void init() = 0;
	virtual void empty_cache() = 0;
	virtual Matrix<Scalar>& get_params() = 0;
	virtual const Matrix<Scalar>& get_param_grads() const = 0;
	virtual void enforce_constraints() = 0;
	virtual Matrix<Scalar> pass_forward(Matrix<Scalar> in, bool training) = 0;
	virtual Matrix<Scalar> pass_back(Matrix<Scalar> out_grads) = 0;
};

template<typename Scalar>
class DenseLayer : public Layer<Scalar> {
public:
	DenseLayer(unsigned input_size, unsigned output_size, const WeightInitialization<Scalar>& weight_init,
			Scalar max_norm_constraint = 0) :
				input_size(input_size),
				output_size(output_size),
				weight_init(weight_init),
				max_norm_constraint(max_norm_constraint),
				max_norm(Utils<Scalar>::decidedly_greater(max_norm_constraint, .0)),
				weights(input_size + 1, output_size),
				weight_grads(input_size + 1, output_size) {
		assert(input_size > 0 && "prev size must be greater than 0");
		assert(output_size > 0 && "size must be greater than 0");
	};
	Layer<Scalar>* clone() {
		return new DenseLayer(*this);
	};
	unsigned get_input_size() const {
		return input_size;
	};
	unsigned get_output_size() const {
		return output_size;
	};
protected:
	void init() {
		weight_init.apply(weights);
		weight_grads.setZero(weight_grads.rows(), weight_grads.cols());
	};
	void empty_cache() {
		biased_prev_out = Matrix<Scalar>(0, 0);
	};
	Matrix<Scalar>& get_params() {
		return weights;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return weight_grads;
	};
	void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = weights.squaredNorm();
			if (l2_norm > max_norm_constraint)
				weights *= (max_norm_constraint / l2_norm);
		}
	};
	Matrix<Scalar> pass_forward(Matrix<Scalar> in, bool training) {
		assert((unsigned) in.cols() == input_size &&
				"illegal input matrix size for feed forward");
		// Add a 1-column to the input for the bias trick.
		biased_prev_out = Matrix<Scalar>(in.rows(), input_size + 1);
		biased_prev_out.leftCols(input_size) = std::move(in);
		biased_prev_out.col(input_size).setOnes();
		return biased_prev_out * weights;
	};
	Matrix<Scalar> pass_back(Matrix<Scalar> out_grads) {
		assert((unsigned) out_grads.cols() == output_size &&
				out_grads.rows() == biased_prev_out.rows() &&
				"illegal input matrix size for feed back");
		/* Compute the gradients of the outputs with respect to the
		 * weighted inputs. */
		weight_grads = biased_prev_out.transpose() * out_grads;
		/* Remove the bias column from the transposed weight matrix
		 * and compute the out-gradients of the previous layer. */
		return out_grads * weights.topRows(input_size).transpose();
	};
private:
	unsigned input_size;
	unsigned output_size;
	const WeightInitialization<Scalar>& weight_init;
	Scalar max_norm_constraint;
	bool max_norm;
	/* Eigen matrices are backed by arrays allocated on the heap, so these
	 * members do not burden the stack. */
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	// Staged computation caches
	Matrix<Scalar> biased_prev_out;
};

template<typename Scalar>
class ActivationLayer : public Layer<Scalar> {
public:
	ActivationLayer(unsigned size) :
			size(size),
			params(0, 0),
			param_grads(0, 0) {
		assert(size > 0 && "size must be greater than 0");
	};
	virtual ~ActivationLayer() = default;
	unsigned get_input_size() const {
		return size;
	};
	unsigned get_output_size() const {
		return size;
	};
protected:
	void init() { };
	void empty_cache() {
		in = Matrix<Scalar>(0, 0);
		out = Matrix<Scalar>(0, 0);
	};
	Matrix<Scalar>& get_params() {
		return params;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return param_grads;
	};
	void enforce_constraints() { };
	Matrix<Scalar> pass_forward(Matrix<Scalar> in, bool training) {
		assert((unsigned) in.cols() == size &&
				"illegal input matrix size for feed forward");
		this->in = std::move(in);
		out = activate(this->in);
		return out;
	};
	Matrix<Scalar> pass_back(Matrix<Scalar> out_grads) {
		assert((unsigned) out_grads.cols() == size &&
				out_grads.rows() == out.rows() &&
				"illegal input matrix size for feed back");
		return d_activate(in, out, out_grads);
	};
	virtual Matrix<Scalar> activate(const Matrix<Scalar>& in) const = 0;
	virtual Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const = 0;
	unsigned size;
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation caches
	Matrix<Scalar> in;
	Matrix<Scalar> out;
};

template<typename Scalar>
class IdentityActivationLayer : public ActivationLayer<Scalar> {
public:
	IdentityActivationLayer(unsigned size) :
			ActivationLayer<Scalar>::ActivationLayer(size) { };
	Layer<Scalar>* clone() {
		return new IdentityActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return in;
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		return out_grads;
	};
};

template<typename Scalar>
class ScalingActivationLayer : public ActivationLayer<Scalar> {
public:
	ScalingActivationLayer(unsigned size, Scalar scale) :
			ActivationLayer<Scalar>::ActivationLayer(size),
			scale(scale) { };
	Layer<Scalar>* clone() {
		return new ScalingActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return in * scale;
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		return out_grads * scale;
	};
private:
	Scalar scale;
};

template<typename Scalar>
class BinaryStepActivationLayer : public ActivationLayer<Scalar> {
public:
	BinaryStepActivationLayer(unsigned size) :
			ActivationLayer<Scalar>::ActivationLayer(size) { };
	Layer<Scalar>* clone() {
		return new BinaryStepActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return in.unaryExpr([](Scalar i) { return i >= .0 ? 1.0 : .0; });
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		return Matrix<Scalar>::Zero(in.rows(), in.cols());
	};
};

template<typename Scalar>
class SigmoidActivationLayer : public ActivationLayer<Scalar> {
public:
	SigmoidActivationLayer(unsigned size) :
			ActivationLayer<Scalar>::ActivationLayer(size) { };
	Layer<Scalar>* clone() {
		return new SigmoidActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return ((-in).array().exp() + 1).inverse();
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		return (out.array() *  (-out.array() + 1)) * out_grads.array();
	};
};

template<typename Scalar>
class TanhActivationLayer : public ActivationLayer<Scalar> {
public:
	TanhActivationLayer(unsigned size) :
			ActivationLayer<Scalar>::ActivationLayer(size) { };
	Layer<Scalar>* clone() {
		return new TanhActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return in.array().tanh();
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		return (-out.array() * out.array() + 1) * out_grads.array();
	};
};

template<typename Scalar>
class SoftmaxActivationLayer : public ActivationLayer<Scalar> {
public:
	SoftmaxActivationLayer(unsigned size, Scalar epsilon = Utils<Scalar>::EPSILON2) :
			ActivationLayer<Scalar>::ActivationLayer(size),
			epsilon(epsilon) { };
	Layer<Scalar>* clone() {
		return new SoftmaxActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		/* First subtract the value of the greatest coefficient from each element row-wise
		 * to avoid an overflow due to raising e to great powers. */
		Matrix<Scalar> out = (in.array().colwise() - in.array().rowwise().maxCoeff()).exp();
		return out.array().colwise() / (out.array().rowwise().sum() + epsilon);
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		Matrix<Scalar> d_in(in.rows(), in.cols());
		for (int i = 0; i < d_in.rows(); i++) {
			Matrix<Scalar> jacobian = out.row(i).asDiagonal();
			jacobian -= out.row(i).transpose() * out.row(i);
			out.row(i) = out_grads.row(i) * jacobian;
		}
		return d_in;
	};
private:
	Scalar epsilon;
};

template<typename Scalar>
class ReLUActivationLayer : public ActivationLayer<Scalar> {
public:
	ReLUActivationLayer(unsigned size) :
			ActivationLayer<Scalar>::ActivationLayer(size) { };
	Layer<Scalar>* clone() {
		return new ReLUActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return in.cwiseMax(.0);
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		return in.unaryExpr([](Scalar i) { return i >= .0 ? 1.0 : .0; })
				.cwiseProduct(out_grads);
	};
};

template<typename Scalar>
class LeakyReLUActivationLayer : public ActivationLayer<Scalar> {
public:
	LeakyReLUActivationLayer(unsigned size, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar>::ActivationLayer(size),
			alpha(alpha) { };
	Layer<Scalar>* clone() {
		return new LeakyReLUActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return in.cwiseMax(in * alpha);
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		return in.unaryExpr([this](Scalar i) { return i >= .0 ? 1.0 : alpha; })
				.cwiseProduct(out_grads);
	};
private:
	Scalar alpha;
};

template<typename Scalar>
class ELUActivationLayer : public ActivationLayer<Scalar> {
public:
	ELUActivationLayer(unsigned size, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar>::ActivationLayer(size),
			alpha(alpha) { };
	Layer<Scalar>* clone() {
		return new ELUActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return in.unaryExpr([this](Scalar i) { return i > .0 ? i : (alpha * (exp(i) - 1)); });
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		Matrix<Scalar> d_in(in.rows(), in.cols());
		for (int i = 0; i < in.rows(); i++) {
			for (int j = 0; j < in.cols(); j++)
				d_in(i,j) = (in(i,j) > .0 ? 1.0 : (out(i,j) + alpha)) * out_grads(i,j);
		}
		return d_in;
	};
private:
	Scalar alpha;
};

template<typename Scalar>
class PReLUActivationLayer : public ActivationLayer<Scalar> {
public:
	PReLUActivationLayer(unsigned size, Scalar init_alpha = 1e-1) :
			ActivationLayer<Scalar>::ActivationLayer(size),
			init_alpha(init_alpha) {
		ActivationLayer<Scalar>::params.resize(1, size);
		ActivationLayer<Scalar>::param_grads.resize(1, size);
	};
	Layer<Scalar>* clone() {
		return new PReLUActivationLayer(*this);
	};
protected:
	void init() {
		ActivationLayer<Scalar>::params.setConstant(init_alpha);
		ActivationLayer<Scalar>::param_grads.setZero(1, ActivationLayer<Scalar>::size);
	};
	Matrix<Scalar> activate(const Matrix<Scalar>& in) const {
		return in.cwiseMax(in * ActivationLayer<Scalar>::params.row(0).asDiagonal());
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) const {
		Matrix<Scalar> d_in = Matrix<Scalar>(in.rows(), in.cols());
		for (int i = 0; i < in.rows(); i++) {
			ActivationLayer<Scalar>::param_grads(0,j) = 0;
			for (int j = 0; j < in.cols(); j++) {
				Scalar in_ij = in(i,j);
				if (in_ij >= 0)
					d_in(i,j) = out_grads(i,j);
				else {
					Scalar out_ij = out_grads(i,j);
					d_in(i,j) = ActivationLayer<Scalar>::params(0,j) * out_ij;
					ActivationLayer<Scalar>::param_grads(0,j) += in_ij * out_ij;
				}
			}
		}
		return d_in;
	};
private:
	Scalar init_alpha;
};

template<typename Scalar>
class BatchNormLayer : public Layer<Scalar> {
public:
	BatchNormLayer(unsigned size, Scalar norm_avg_decay = .1, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			size(size),
			norm_avg_decay(norm_avg_decay),
			epsilon(epsilon),
			avg_means(size),
			avg_inv_sds(size),
			avgs_init(false),
			params(2, size),
			param_grads(2, size) {
		assert(size > 0 && "size must be greater than 0");
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	Layer<Scalar>* clone() {
		return new BatchNormLayer(*this);
	};
	unsigned get_input_size() const {
		return size;
	};
	unsigned get_output_size() const {
		return size;
	};
protected:
	void init() {
		params.row(0).setOnes();
		params.row(1).setZero();
		param_grads.setZero(params.rows(), params.cols());
		avg_means.setZero(avg_means.cols());
		avg_inv_sds.setZero(avg_inv_sds.cols());
		avgs_init = false;
	};
	void empty_cache() {
		inv_in_sd = RowVector<Scalar>(0);
		std_in = Matrix<Scalar>(0, 0);
	};
	Matrix<Scalar>& get_params() {
		return params;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return param_grads;
	};
	void enforce_constraints() { };
	Matrix<Scalar> pass_forward(Matrix<Scalar> in, bool training) {
		assert((unsigned) in.cols() == size &&
				"illegal input matrix size for feed forward");
		if (training) {
			RowVector<Scalar> means = in.colwise().mean();
			Matrix<Scalar> norm_in = in.rowwise() - means;
			inv_in_sd = (norm_in.array().square().colwise().mean() + epsilon).sqrt().inverse();
			std_in = norm_in * inv_in_sd.asDiagonal();
			// Maintain a moving average of means and variances for testing.
			if (avgs_init) {
				avg_means = (1.0 - norm_avg_decay) * avg_means + norm_avg_decay * means;
				avg_inv_sds = (1.0 - norm_avg_decay) * avg_inv_sds + norm_avg_decay * inv_in_sd;
			} else {
				avg_means = means;
				avg_inv_sds = inv_in_sd;
				avgs_init = true;
			}
		} else // For testing, use the moving averages.
			std_in = (in.rowwise() - avg_means) * avg_inv_sds.asDiagonal();
		return (std_in * params.row(0).asDiagonal()).rowwise() + params.row(1);
	};
	Matrix<Scalar> pass_back(Matrix<Scalar> out_grads) {
		assert((unsigned) out_grads.cols() == size &&
				out_grads.rows() == std_in.rows() &&
				"illegal input matrix size for feed back");
		/* Back-propagate the gradient through the batch
		 * normalization 'function' and also calculate the
		 * gradients on the betas and gammas. */
		param_grads.row(0) = out_grads.cwiseProduct(std_in).colwise().sum();
		param_grads.row(1) = out_grads.colwise().sum();
		Matrix<Scalar> std_in_grads = out_grads * params.row(0).asDiagonal();
		int rows = std_in.rows();
		return (((rows * std_in_grads).rowwise() - std_in_grads.colwise().sum()) -
				std_in * (std_in.cwiseProduct(std_in_grads).colwise().sum().asDiagonal())) *
				((1.0 / rows) * inv_in_sd).asDiagonal();
	};
private:
	unsigned size;
	Scalar norm_avg_decay;
	Scalar epsilon;
	// Dynamic batch normalization parameters.
	RowVector<Scalar> avg_means;
	RowVector<Scalar> avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation caches
	RowVector<Scalar> inv_in_sd;
	Matrix<Scalar> std_in;
};

template<typename Scalar>
class DropoutLayer : public Layer<Scalar> {
public:
	DropoutLayer(unsigned size, Scalar dropout_prob, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			size(size),
			dropout_prob(dropout_prob),
			epsilon(epsilon),
			dropout(Utils<Scalar>::decidedly_greater(dropout_prob, .0)),
			params(0, 0),
			param_grads(0, 0) {
		assert(size > 0 && "size must be greater than 0");
		assert(dropout_prob <= 1 && "dropout prob must not be greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	Layer<Scalar>* clone() {
		return new DropoutLayer(*this);
	};
	unsigned get_input_size() const {
		return size;
	};
	unsigned get_output_size() const {
		return size;
	};
protected:
	void init() { };
	void empty_cache() {
		dropout_mask = Matrix<Scalar>(0, 0);
	};
	Matrix<Scalar>& get_params() {
		return params;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return param_grads;
	};
	void enforce_constraints() { };
	Matrix<Scalar> pass_forward(Matrix<Scalar> in, bool training) {
		assert((unsigned) in.cols() == size &&
				"illegal input matrix size for feed forward");
		if (training && dropout) {
			Matrix<Scalar> dropout_mask(in.rows(), in.cols());
			dropout_mask.setRandom(in.rows(), in.cols());
			Scalar scaling_factor = 1 / (1 - dropout_prob + epsilon);
			dropout_mask = ((dropout_mask.array() + 1) / 2).unaryExpr([this,scaling_factor](Scalar i) {
				return (Scalar) (i <= dropout_prob ? .0 : scaling_factor);
			});
			in = in.cwiseProduct(dropout_mask);
			this->dropout_mask = std::move(dropout_mask);
		}
		return in;
	};
	Matrix<Scalar> pass_back(Matrix<Scalar> out_grads) {
		assert((unsigned) out_grads.cols() == size &&
				out_grads.rows() == dropout_mask.rows() &&
				"illegal input matrix size for feed back");
		// The derivative of the dropout 'function'.
		return out_grads.cwiseProduct(dropout_mask);
	};
private:
	unsigned size;
	Scalar dropout_prob;
	Scalar epsilon;
	bool dropout;
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation caches
	Matrix<Scalar> dropout_mask;
};

} /* namespace cppnn */

#endif /* LAYER_H_ */
