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
#include <Dimensions.h>
#include <Eigen/Dense>
#include <limits>
#include <Matrix.h>
#include <Tensor.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Utils.h>
#include <utility>
#include <Vector.h>
#include <WeightInitialization.h>

namespace cppnn {

// Forward declarations to NeuralNetwork and Optimizer so they can be friended.
template<typename Scalar> class NeuralNetwork;
template<typename Scalar> class Optimizer;

template<typename Scalar>
class Layer {
	friend class NeuralNetwork<Scalar>;
	friend class Optimizer<Scalar>;
public:
	virtual ~Layer() = default;
	// Clone pattern.
	virtual Layer<Scalar>* clone() = 0;
	virtual Dimensions get_input_dims() const = 0;
	virtual Dimensions get_output_dims() const = 0;
	bool is_parametric() {
		return get_params().rows() > 0 && get_params().cols() > 0;
	};
protected:
	/* Only expose methods that allow for the modification of the
	 * layer's state to friends and sub-classes. */
	void set_input_layer() {
		input_layer = true;
	};
	bool is_input_layer() {
		return input_layer;
	};
	virtual void init() = 0;
	virtual void empty_cache() = 0;
	virtual Matrix<Scalar>& get_params() = 0;
	virtual const Matrix<Scalar>& get_param_grads() const = 0;
	virtual void enforce_constraints() = 0;
	virtual Tensor4D<Scalar> pass_forward(Tensor4D<Scalar> in, bool training) = 0;
	virtual Tensor4D<Scalar> pass_back(Tensor4D<Scalar> out_grads) = 0;
	static constexpr Tensor4D<Scalar> EMPTY_TENSOR = Tensor4D<Scalar>(0, 0, 0, 0);
private:
	bool input_layer = false;
};

template<typename Scalar>
class DenseLayer : public Layer<Scalar> {
public:
	DenseLayer(Dimensions input_dims, unsigned output_size, const WeightInitialization<Scalar>& weight_init,
			Scalar max_norm_constraint = 0) :
				input_dims(input_dims),
				output_dims(output_size, 1, 1),
				weight_init(weight_init),
				max_norm_constraint(max_norm_constraint),
				max_norm(Utils<Scalar>::decidedly_greater(max_norm_constraint, .0)),
				weights(input_dims.get_points() + 1, output_size),
				weight_grads(input_dims.get_points() + 1, output_size) { };
	Layer<Scalar>* clone() {
		return new DenseLayer(*this);
	};
	Dimensions get_input_dims() const {
		return input_dims;
	};
	Dimensions get_output_dims() const {
		return output_dims;
	};
protected:
	void init() {
		weight_init.apply(weights);
		weight_grads.setZero(weight_grads.rows(), weight_grads.cols());
	};
	void empty_cache() {
		biased_in = Matrix<Scalar>(0, 0);
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
	Tensor4D<Scalar> pass_forward(Tensor4D<Scalar> in, bool training) {
		assert(in.dimension(1) == input_dims.get_dim1() && in.dimension(2) == input_dims.get_dim2() &&
				in.dimension(3) == input_dims.get_dim3());
		assert(in.dimension(0) > 0);
		unsigned input_size = input_dims.get_points();
		// Add a 1-column to the input for the bias trick.
		biased_in = Matrix<Scalar>(in.dimension(0), input_size + 1);
		biased_in.leftCols(input_size) = Utils<Scalar>::tensor4d_to_mat(in);
		biased_in.col(input_size).setOnes();
		return Utils<Scalar>::mat_to_tensor4d((biased_in * weights).eval(), output_dims);
	};
	Tensor4D<Scalar> pass_back(Tensor4D<Scalar> out_grads) {
		assert(out_grads.dimension(1) == output_dims.get_dim1() && out_grads.dimension(2) == output_dims.get_dim2() &&
				out_grads.dimension(3) == output_dims.get_dim3());
		assert(out_grads.dimension(0) > 0 && biased_in.rows() == out_grads.dimension(0));
		Matrix<Scalar> out_grads_mat = Utils<Scalar>::tensor4d_to_mat(out_grads);
		// Compute the gradients of the outputs with respect to the weights.
		weight_grads = biased_in.transpose() * out_grads_mat;
		if (is_input_layer())
			return Layer<Scalar>::EMPTY_TENSOR;
		/* Remove the bias row from the weight matrix, transpose it, and compute gradients w.r.t. the
		 * previous layer's output. */
		return Utils<Scalar>::mat_to_tensor4d((out_grads_mat * weights.topRows(input_dims.get_points()).transpose())
				.eval(), input_dims);
	};
private:
	Dimensions input_dims;
	Dimensions output_dims;
	const WeightInitialization<Scalar>& weight_init;
	Scalar max_norm_constraint;
	bool max_norm;
	/* Eigen matrices are backed by arrays allocated on the heap, so these
	 * members do not burden the stack. */
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	// Staged computation caches
	Matrix<Scalar> biased_in;
};

template<typename Scalar>
class ActivationLayer : public Layer<Scalar> {
public:
	ActivationLayer(Dimensions dims) :
			dims(dims),
			params(0, 0),
			param_grads(0, 0) { };
	virtual ~ActivationLayer() = default;
	Dimensions get_input_dims() const {
		return dims;
	};
	Dimensions get_output_dims() const {
		return dims;
	};
protected:
	virtual Matrix<Scalar> activate(const Matrix<Scalar>& in) = 0;
	virtual Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) = 0;
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
	Tensor4D<Scalar> pass_forward(Tensor4D<Scalar> in, bool training) {
		assert(in.dimension(1) == dims.get_dim1() && in.dimension(2) == dims.get_dim2() &&
				in.dimension(3) == dims.get_dim3());
		assert(in.dimension(0) > 0);
		this->in = Utils<Scalar>::tensor4d_to_mat(in);
		out = activate(this->in);
		return Utils<Scalar>::mat_to_tensor4d(out, dims);
	};
	Tensor4D<Scalar> pass_back(Tensor4D<Scalar> out_grads) {
		assert(out_grads.dimension(1) == dims.get_dim1() && out_grads.dimension(2) == dims.get_dim2() &&
				out_grads.dimension(3) == dims.get_dim3());
		assert(out_grads.dimension(0) > 0 && out.rows() == out_grads.dimension(0));
		if (is_input_layer())
			return Layer<Scalar>::EMPTY_TENSOR;
		return Utils<Scalar>::mat_to_tensor4d(d_activate(in, out, Utils<Scalar>::tensor4d_to_mat(out_grads)), dims);
	};
	Dimensions dims;
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation caches
	Matrix<Scalar> in;
	Matrix<Scalar> out;
};

template<typename Scalar>
class IdentityActivationLayer : public ActivationLayer<Scalar> {
public:
	IdentityActivationLayer(Dimensions dims) :
			ActivationLayer<Scalar>::ActivationLayer(dims) { };
	Layer<Scalar>* clone() {
		return new IdentityActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in;
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return out_grads;
	};
};

template<typename Scalar>
class ScalingActivationLayer : public ActivationLayer<Scalar> {
public:
	ScalingActivationLayer(Dimensions dims, Scalar scale) :
			ActivationLayer<Scalar>::ActivationLayer(dims),
			scale(scale) { };
	Layer<Scalar>* clone() {
		return new ScalingActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in * scale;
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return out_grads * scale;
	};
private:
	Scalar scale;
};

template<typename Scalar>
class BinaryStepActivationLayer : public ActivationLayer<Scalar> {
public:
	BinaryStepActivationLayer(Dimensions dims) :
			ActivationLayer<Scalar>::ActivationLayer(dims) { };
	Layer<Scalar>* clone() {
		return new BinaryStepActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.unaryExpr([](Scalar i) { return i >= .0 ? 1.0 : .0; });
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return Matrix<Scalar>::Zero(in.rows(), in.cols());
	};
};

template<typename Scalar>
class SigmoidActivationLayer : public ActivationLayer<Scalar> {
public:
	SigmoidActivationLayer(Dimensions dims) :
			ActivationLayer<Scalar>::ActivationLayer(dims) { };
	Layer<Scalar>* clone() {
		return new SigmoidActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return ((-in).array().exp() + 1).inverse();
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return (out.array() *  (-out.array() + 1)) * out_grads.array();
	};
};

template<typename Scalar>
class TanhActivationLayer : public ActivationLayer<Scalar> {
public:
	TanhActivationLayer(Dimensions dims) :
			ActivationLayer<Scalar>::ActivationLayer(dims) { };
	Layer<Scalar>* clone() {
		return new TanhActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.array().tanh();
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return (-out.array() * out.array() + 1) * out_grads.array();
	};
};

template<typename Scalar>
class SoftmaxActivationLayer : public ActivationLayer<Scalar> {
public:
	SoftmaxActivationLayer(Dimensions dims, Scalar epsilon = Utils<Scalar>::EPSILON2) :
			ActivationLayer<Scalar>::ActivationLayer(dims),
			epsilon(epsilon) { };
	Layer<Scalar>* clone() {
		return new SoftmaxActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		/* First subtract the value of the greatest coefficient from each element row-wise
		 * to avoid an overflow due to raising e to great powers. */
		Matrix<Scalar> out = (in.array().colwise() - in.array().rowwise().maxCoeff()).exp();
		return out.array().colwise() / (out.array().rowwise().sum() + epsilon);
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
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
	ReLUActivationLayer(Dimensions dims) :
			ActivationLayer<Scalar>::ActivationLayer(dims) { };
	Layer<Scalar>* clone() {
		return new ReLUActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(.0);
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return in.unaryExpr([](Scalar i) { return i >= .0 ? 1.0 : .0; })
				.cwiseProduct(out_grads);
	};
};

template<typename Scalar>
class LeakyReLUActivationLayer : public ActivationLayer<Scalar> {
public:
	LeakyReLUActivationLayer(Dimensions dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar>::ActivationLayer(dims),
			alpha(alpha) { };
	Layer<Scalar>* clone() {
		return new LeakyReLUActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(in * alpha);
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return in.unaryExpr([this](Scalar i) { return i >= .0 ? 1.0 : alpha; })
				.cwiseProduct(out_grads);
	};
private:
	Scalar alpha;
};

template<typename Scalar>
class ELUActivationLayer : public ActivationLayer<Scalar> {
public:
	ELUActivationLayer(Dimensions dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar>::ActivationLayer(dims),
			alpha(alpha) { };
	Layer<Scalar>* clone() {
		return new ELUActivationLayer(*this);
	};
protected:
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.unaryExpr([this](Scalar i) { return i > .0 ? i : (alpha * (exp(i) - 1)); });
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		Matrix<Scalar> d_in(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); i++) {
			for (int j = 0; j < in.rows(); j++)
				d_in(j,i) = (in(j,i) > .0 ? 1.0 : (out(j,i) + alpha)) * out_grads(j,i);
		}
		return d_in;
	};
private:
	Scalar alpha;
};

template<typename Scalar>
class PReLUActivationLayer : public ActivationLayer<Scalar> {
public:
	PReLUActivationLayer(Dimensions dims, Scalar init_alpha = 1e-1) :
			ActivationLayer<Scalar>::ActivationLayer(dims),
			init_alpha(init_alpha) {
		ActivationLayer<Scalar>::params.resize(1, dims.get_points());
		ActivationLayer<Scalar>::param_grads.resize(1, dims.get_points());
	};
	Layer<Scalar>* clone() {
		return new PReLUActivationLayer(*this);
	};
protected:
	void init() {
		ActivationLayer<Scalar>::params.setConstant(init_alpha);
		ActivationLayer<Scalar>::param_grads.setZero(1, ActivationLayer<Scalar>::dims.get_points());
	};
	Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(in * ActivationLayer<Scalar>::params.row(0).asDiagonal());
	};
	Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		ActivationLayer<Scalar>::param_grads.row(0).setZero();
		Matrix<Scalar> d_in = Matrix<Scalar>(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); i++) {
			for (int j = 0; j < in.rows(); j++) {
				Scalar in_ji = in(j,i);
				if (in_ji >= 0)
					d_in(j,i) = out_grads(j,i);
				else {
					Scalar out_ji = out_grads(j,i);
					d_in(j,i) = ActivationLayer<Scalar>::params(0,i) * out_ji;
					ActivationLayer<Scalar>::param_grads(0,i) += in_ji * out_ji;
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
	BatchNormLayer(Dimensions dims, Scalar norm_avg_decay = .1, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			dims(dims),
			norm_avg_decay(norm_avg_decay),
			epsilon(epsilon),
			avg_means(dims.get_dim3(), dims.get_dim1() * dims.get_dim2()),
			avg_inv_sds(dims.get_dim3(), dims.get_dim1() * dims.get_dim2()),
			avgs_init(false),
			params(2 * dims.get_dim3(), dims.get_dim1() * dims.get_dim2()),
			param_grads(2 * dims.get_dim3(), dims.get_dim1() * dims.get_dim2()),
			cache_vec(dims.get_dim3()) {
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	Layer<Scalar>* clone() {
		return new BatchNormLayer(*this);
	};
	Dimensions get_input_dims() const {
		return dims;
	};
	Dimensions get_output_dims() const {
		return dims;
	};
protected:
	void init() {
		for (int i = 0; i < dims.get_dim3(); i += 2) {
			// Gamma
			params.row(i).setOnes();
			// Beta
			params.row(i + 1).setZero();
		}
		param_grads.setZero(params.rows(), params.cols());
		avg_means.setZero(avg_means.rows(), avg_means.cols());
		avg_inv_sds.setZero(avg_means.rows(), avg_inv_sds.cols());
		avgs_init = false;
	};
	void empty_cache() {
		for (unsigned i = 0; i < cache_vec.size(); i++) {
			Cache& cache = cache_vec[i];
			cache.inv_in_sd = RowVector<Scalar>(0);
			cache.std_in = Matrix<Scalar>(0, 0);
		}
	};
	Matrix<Scalar>& get_params() {
		return params;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return param_grads;
	};
	void enforce_constraints() { };
	Tensor4D<Scalar> pass_forward(Tensor4D<Scalar> in, bool training) {
		assert(in.dimension(1) == dims.get_dim1() && in.dimension(2) == dims.get_dim2() &&
				in.dimension(3) == dims.get_dim3());
		assert(in.dimension(0) > 0);
		int rows = in.dimension(0);
		Tensor4D<Scalar> out(rows, dims.get_dim1(), dims.get_dim2(), dims.get_dim3());
		Dimensions slice_dims(dims.get_dim1(), dims.get_dim2(), 1);
		Eigen::array<int, 4> offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> extents = { rows, slice_dims.get_dim1(), slice_dims.get_dim2(), slice_dims.get_dim3() };
		for (int i = 0; i < dims.get_dim3(); i++) {
			offsets[3] = i;
			Tensor4D<Scalar> in_slice_i = in.slice(offsets, extents);
			Matrix<Scalar> in_ch_i = Utils<Scalar>::tensor4d_to_mat(std::move(in_slice_i));
			if (training) {
				Cache& cache = cache_vec[i];
				RowVector<Scalar> means = in_ch_i.colwise().mean();
				Matrix<Scalar> norm_in = in_ch_i.rowwise() - means;
				cache.inv_in_sd = (norm_in.array().square().colwise().mean() + epsilon).sqrt().inverse();
				cache.std_in = norm_in * cache.inv_in_sd.asDiagonal();
				in_ch_i = cache.std_in;
				// Maintain a moving average of means and variances for testing.
				if (avgs_init) {
					avg_means.row(i) = (1.0 - norm_avg_decay) * avg_means.row(i) + norm_avg_decay * means;
					avg_inv_sds.row(i) = (1.0 - norm_avg_decay) * avg_inv_sds.row(i) + norm_avg_decay *
							cache.inv_in_sd;
				} else {
					avg_means.row(i) = means;
					avg_inv_sds.row(i) = cache.inv_in_sd;
					avgs_init = true;
				}
			} else // For testing, use the moving averages.
				in_ch_i = (in_ch_i.rowwise() - avg_means.row(i)) * avg_inv_sds.row(i).asDiagonal();
			out.slice(offsets, extents) = Utils<Scalar>::mat_to_tensor4d(((in_ch_i * params.row(2 * i).asDiagonal())
					.rowwise() + params.row(2 * i + 1)).eval(), slice_dims);
		}
		return out;
	};
	Tensor4D<Scalar> pass_back(Tensor4D<Scalar> out_grads) {
		assert(out_grads.dimension(1) == dims.get_dim1() && out_grads.dimension(2) == dims.get_dim2() &&
				out_grads.dimension(3) == dims.get_dim3());
		assert(out_grads.dimension(0) > 0 && cache_vec[0].std_in.rows() == out_grads.dimension(0));
		int rows = out_grads.dimension(0);
		Tensor4D<Scalar> prev_out_grads(rows, dims.get_dim1(), dims.get_dim2(), dims.get_dim3());
		Dimensions slice_dims(dims.get_dim1(), dims.get_dim2(), 1);
		Eigen::array<int, 4> offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> extents = { rows, slice_dims.get_dim1(), slice_dims.get_dim2(), slice_dims.get_dim3() };
		/* Back-propagate the gradient through the batch
		 * normalization 'function' and also calculate the
		 * gradients on the betas and gammas. */
		for (int i = 0; i < dims.get_dim3(); i++) {
			offsets[3] = i;
			Tensor4D<Scalar> out_grads_slice_i = out_grads.slice(offsets, extents);
			Matrix<Scalar> out_grads_ch_i = Utils<Scalar>::tensor4d_to_mat(std::move(out_grads_slice_i));
			Cache& cache = cache_vec[i];
			param_grads.row(2 * i) = out_grads_ch_i.cwiseProduct(cache.std_in).colwise().sum();
			param_grads.row(2 * i + 1) = out_grads_ch_i.colwise().sum();
			if (is_input_layer())
				continue;
			Matrix<Scalar> std_in_grads = out_grads_ch_i * params.row(2 * i).asDiagonal();
			prev_out_grads.slice(offsets, extents) =
					Utils<Scalar>::mat_to_tensor4d(((((rows * std_in_grads).rowwise() - std_in_grads.colwise().sum()) -
					cache.std_in * (cache.std_in.cwiseProduct(std_in_grads).colwise().sum().asDiagonal())) *
					((1.0 / rows) * cache.inv_in_sd).asDiagonal()).eval(), slice_dims);
		}
		return prev_out_grads;
	};
private:
	Dimensions dims;
	Scalar norm_avg_decay;
	Scalar epsilon;
	// Dynamic batch normalization parameters.
	Matrix<Scalar> avg_means;
	Matrix<Scalar> avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation cache_vec
	struct Cache {
		RowVector<Scalar> inv_in_sd;
		Matrix<Scalar> std_in;
	};
	std::vector<Cache> cache_vec;
};

template<typename Scalar>
class DropoutLayer : public Layer<Scalar> {
public:
	DropoutLayer(Dimensions dims, Scalar dropout_prob, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			dims(dims),
			dropout_prob(dropout_prob),
			epsilon(epsilon),
			dropout(Utils<Scalar>::decidedly_greater(dropout_prob, .0)),
			params(0, 0),
			param_grads(0, 0) {
		assert(dropout_prob <= 1 && "dropout prob must not be greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	Layer<Scalar>* clone() {
		return new DropoutLayer(*this);
	};
	Dimensions get_input_dims() const {
		return dims;
	};
	Dimensions get_output_dims() const {
		return dims;
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
	Tensor4D<Scalar> pass_forward(Tensor4D<Scalar> in, bool training) {
		assert(in.dimension(1) == dims.get_dim1() && in.dimension(2) == dims.get_dim2() &&
				in.dimension(3) == dims.get_dim3());
		assert(in.dimension(0) > 0);
		if (training && dropout) {
			Matrix<Scalar> in_mat = Utils<Scalar>::tensor4d_to_mat(in);
			Matrix<Scalar> dropout_mask(in_mat.rows(), in_mat.cols());
			dropout_mask.setRandom(in_mat.rows(), in_mat.cols());
			// Inverted dropout.
			Scalar scaling_factor = 1 / (1 - dropout_prob + epsilon);
			dropout_mask = ((dropout_mask.array() + 1) / 2).unaryExpr([this,scaling_factor](Scalar i) {
				return (Scalar) (i <= dropout_prob ? .0 : scaling_factor);
			});
			in = Utils<Scalar>::mat_to_tensor4d(in_mat.cwiseProduct(dropout_mask).eval(), dims);
			this->dropout_mask = std::move(dropout_mask);
		}
		return in;
	};
	Tensor4D<Scalar> pass_back(Tensor4D<Scalar> out_grads) {
		assert(out_grads.dimension(1) == dims.get_dim1() && out_grads.dimension(2) == dims.get_dim2() &&
				out_grads.dimension(3) == dims.get_dim3());
		assert(out_grads.dimension(0) > 0 && dropout_mask.rows() == out_grads.dimension(0));
		if (is_input_layer())
			return Layer<Scalar>::EMPTY_TENSOR;
		// The derivative of the dropout 'function'.
		return Utils<Scalar>::mat_to_tensor4d(Utils<Scalar>::tensor4d_to_mat(out_grads).cwiseProduct(dropout_mask)
				.eval(), dims);
	};
private:
	Dimensions dims;
	Scalar dropout_prob;
	Scalar epsilon;
	bool dropout;
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation cache_vec
	Matrix<Scalar> dropout_mask;
};

template<typename Scalar>
class ConvLayer : public Layer<Scalar> {
	// TODO Dilation.
public:
	ConvLayer(Dimensions input_dims, const WeightInitialization<Scalar>& weight_init, unsigned filters,
			unsigned receptor_size = 3, unsigned stride = 1, unsigned padding = 1, Scalar max_norm_constraint = 0) :
				input_dims(input_dims),
				output_dims(calculate_output_dim(input_dims.get_dim1(), receptor_size, padding, stride),
						calculate_output_dim(input_dims.get_dim2(), receptor_size, padding, stride), filters),
				weight_init(weight_init),
				filters(filters),
				receptor_size(receptor_size),
				stride(stride),
				padding(padding),
				max_norm_constraint(max_norm_constraint),
				max_norm(Utils<Scalar>::decidedly_greater(max_norm_constraint, .0)),
				weights(receptor_size * receptor_size * input_dims.get_dim3() + 1, filters),
				weight_grads(weights.rows(), filters) {
		assert(input_dims.get_dim1() + 2 * padding >= receptor_size &&
				input_dims.get_dim2() + 2 * padding >= receptor_size);
		assert(filters > 0);
		assert(receptor_size > 0);
		assert(stride > 0);
	};
	Layer<Scalar>* clone() {
		return new ConvLayer(*this);
	};
	Dimensions get_input_dims() const {
		return input_dims;
	};
	Dimensions get_output_dims() const {
		return output_dims;
	};
protected:
	void init() {
		/* For every filter, there is a column in the weight matrix with the same number of
		 * elements as the size of the receptive field (F * F * D) + 1 for the bias row. */
		weight_init.apply(weights);
		weight_grads.setZero(weight_grads.rows(), weight_grads.cols());
	};
	void empty_cache() {
		biased_in_vec = std::vector<Matrix<Scalar>>(0);
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
	Tensor4D<Scalar> pass_forward(Tensor4D<Scalar> in, bool training) {
		assert(in.dimension(1) == input_dims.get_dim1() && in.dimension(2) == input_dims.get_dim2() &&
				in.dimension(3) == input_dims.get_dim3());
		assert(in.dimension(0) > 0);
		// Spatial padding.
		Eigen::array<std::pair<int, int>, 4> paddings;
		paddings[0] = std::make_pair(0, 0);
		paddings[1] = std::make_pair(padding, padding);
		paddings[2] = std::make_pair(padding, padding);
		paddings[3] = std::make_pair(0, 0);
		Tensor4D<Scalar> padded_in = in.pad(paddings);
		// Prepare the base offsets and extents for slicing.
		int rows = padded_in.dimension(0);
		int depth = input_dims.get_dim3();
		int patches = output_dims.get_dim1() * output_dims.get_dim2();
		int receptor_vol = receptor_size * receptor_size * depth;
		int height_rem = padded_in.dimension(1) - receptor_size;
		int width_rem = padded_in.dimension(2) - receptor_size;
		Eigen::array<int, 4> row_offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> row_extents = { 1, output_dims.get_dim1(), output_dims.get_dim2(), output_dims.get_dim3() };
		Eigen::array<int, 4> patch_offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> patch_extents = { 1, (int) receptor_size, (int) receptor_size, depth };
		Tensor4D<Scalar> out(rows, output_dims.get_dim1(), output_dims.get_dim2(), output_dims.get_dim3());
		biased_in_vec = std::vector<Matrix<Scalar>>(rows);
		/* 'Tensor-row' by 'tensor-row', stretch the receptor locations into row vectors, form a matrix of
		 * them, and multiply them by the weight matrix. */
		for (int i = 0; i < rows; i++) {
			row_offsets[0] = i;
			patch_offsets[0] = i;
			int patch_ind = 0;
			Matrix<Scalar> in_mat_i(patches, receptor_vol);
			for (int j = 0; j <= height_rem; j += stride) {
				patch_offsets[1] = j;
				for (int k = 0; k <= width_rem; k += stride) {
					patch_offsets[2] = k;
					Tensor4D<Scalar> patch = padded_in.slice(patch_offsets, patch_extents);
					in_mat_i.row(patch_ind++) = Utils<Scalar>::tensor4d_to_mat(patch);
				}
			}
			assert(patch_ind == patches);
			// Set the additional column's elements to 1.
			Matrix<Scalar> biased_in = Matrix<Scalar>(patches, receptor_vol + 1);
			biased_in.leftCols(receptor_vol) = std::move(in_mat_i);
			biased_in.col(receptor_vol).setOnes();
			biased_in_vec[i] = std::move(biased_in);
			/* Flatten the matrix product into a row vector, reshape it into a 'single-row' sub-tensor, and
			 * assign it to the output tensor's corresponding 'row'. */
			Matrix<Scalar> out_i = biased_in_vec[i] * weights;
			out.slice(row_offsets, row_extents) = Utils<Scalar>::mat_to_tensor4d(Eigen::Map<Matrix<Scalar>>(out_i.data(),
					1, out_i.rows() * out_i.cols()).matrix(), output_dims);
		}
		return out;
	};
	Tensor4D<Scalar> pass_back(Tensor4D<Scalar> out_grads) {
		assert(out_grads.dimension(1) == output_dims.get_dim1() && out_grads.dimension(2) == output_dims.get_dim2() &&
				out_grads.dimension(3) == output_dims.get_dim3());
		assert(out_grads.dimension(0) > 0 && biased_in_vec.size() == (unsigned) out_grads.dimension(0));
		int rows = out_grads.dimension(0);
		int padded_height = input_dims.get_dim1() + 2 * padding;
		int padded_width = input_dims.get_dim2() + 2 * padding;
		int height_rem = padded_height - receptor_size;
		int width_rem = padded_width - receptor_size;
		int depth = input_dims.get_dim3();
		Dimensions patch_dims((int) receptor_size, (int) receptor_size, depth);
		Eigen::array<int, 4> out_grads_row_offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> out_grads_row_extents = { 1, output_dims.get_dim1(), output_dims.get_dim2(),
				output_dims.get_dim3() };
		Eigen::array<int, 4> patch_offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> patch_extents = { 1, patch_dims.get_dim1(), patch_dims.get_dim2(), depth };
		Tensor4D<Scalar> prev_out_grads(rows, padded_height, padded_width, input_dims.get_dim3());
		prev_out_grads.setZero();
		weight_grads.setZero(weight_grads.rows(), weight_grads.cols());
		for (int i = 0; i < rows; i++) {
			out_grads_row_offsets[0] = i;
			Tensor4D<Scalar> slice_i = out_grads.slice(out_grads_row_offsets, out_grads_row_extents);
			Matrix<Scalar> out_grads_mat_i = Eigen::Map<Matrix<Scalar>>(slice_i.data(), output_dims.get_dim1() *
					output_dims.get_dim2(), filters);
			// Accumulate the gradients of the outputs w.r.t. the weights for each observation a.k.a 'tensor-row'.
			weight_grads += biased_in_vec[i].transpose() * out_grads_mat_i;
			if (is_input_layer())
				continue;
			/* Remove the bias row from the weight matrix, transpose it, and compute the gradients w.r.t. the
			 * previous layer's output. */
			Matrix<Scalar> prev_out_grads_mat_i = out_grads_mat_i * weights.topRows(weights.rows() - 1).transpose();
			/* Given the gradients w.r.t. the stretched out receptor patches, perform a 'backwards' convolution
			 * to get the gradients w.r.t. the individual input nodes. */
			int patch_ind = 0;
			patch_offsets[0] = i;
			for (int j = 0; j <= height_rem; j += stride) {
				patch_offsets[1] = j;
				for (int k = 0; k <= width_rem; k += stride) {
					patch_offsets[2] = k;
					// Accumulate the gradients where the receptor patch tensors overlap.
					prev_out_grads.slice(patch_offsets, patch_extents) +=
							Utils<Scalar>::mat_to_tensor4d(prev_out_grads_mat_i.row(patch_ind++), patch_dims);
				}
			}
			assert(patch_ind == prev_out_grads_mat_i.rows());
		}
		// Cut off the padding.
		Eigen::array<int, 4> no_padding_offsets = { 0, (int) padding, (int) padding, 0 };
		Eigen::array<int, 4> no_padding_extents = { rows, input_dims.get_dim1(), input_dims.get_dim2(), depth };
		return prev_out_grads.slice(no_padding_offsets, no_padding_extents);
	};
	static int calculate_output_dim(int input_dim, int receptor_size, int padding, int stride) {
		return (input_dim - receptor_size + 2 * padding) / stride + 1;
	};
private:
	Dimensions input_dims;
	Dimensions output_dims;
	const WeightInitialization<Scalar>& weight_init;
	unsigned filters;
	unsigned receptor_size;
	unsigned stride;
	unsigned padding;
	Scalar max_norm_constraint;
	bool max_norm;
	// The learnable parameters.
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	// Staged computation caches
	std::vector<Matrix<Scalar>> biased_in_vec;
};

template<typename Scalar>
class PoolingLayer : public Layer<Scalar> {
public:
	PoolingLayer(Dimensions input_dims, unsigned receptor_size, unsigned stride) :
				input_dims(input_dims),
				output_dims(calculate_output_dim(input_dims.get_dim1(), receptor_size, stride),
						calculate_output_dim(input_dims.get_dim2(), receptor_size, stride), input_dims.get_dim3()),
				receptor_size(receptor_size),
				stride(stride),
				params(0, 0),
				param_grads(0, 0) {
		assert(input_dims.get_dim1() >= (int) receptor_size && input_dims.get_dim2() >= (int) receptor_size);
		assert(receptor_size > 0);
		assert(stride > 0);
	};
	virtual ~PoolingLayer() = default;
	Dimensions get_input_dims() const {
		return input_dims;
	};
	Dimensions get_output_dims() const {
		return output_dims;
	};
protected:
	virtual void init_cache() = 0;
	virtual Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) = 0;
	virtual RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) = 0;
	void init() { };
	Matrix<Scalar>& get_params() {
		return params;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return param_grads;
	};
	void enforce_constraints() { };
	Tensor4D<Scalar> pass_forward(Tensor4D<Scalar> in, bool training) {
		assert(in.dimension(1) == input_dims.get_dim1() && in.dimension(2) == input_dims.get_dim2() &&
				in.dimension(3) == input_dims.get_dim3());
		assert(in.dimension(0) > 0);
		rows = in.dimension(0);
		int depth = input_dims.get_dim3();
		int height_rem = input_dims.get_dim1() - receptor_size;
		int width_rem = input_dims.get_dim2() - receptor_size;
		Eigen::array<int, 4> patch_offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> patch_extents = { 1, (int) receptor_size, (int) receptor_size, 1 };
		Tensor4D<Scalar> out(rows, output_dims.get_dim1(), output_dims.get_dim2(), depth);
		init_cache();
		int patch_ind = 0;
		for (int i = 0; i < rows; i++) {
			patch_offsets[0] = i;
			int out_j = 0;
			for (int j = 0; j <= height_rem; j += stride, out_j++) {
				patch_offsets[1] = j;
				int out_k = 0;
				for (int k = 0; k <= width_rem; k += stride, out_k++) {
					patch_offsets[2] = k;
					for (int l = 0; l < depth; l++) {
						patch_offsets[3] = l;
						// Reduce the patches to scalars.
						Tensor4D<Scalar> patch = in.slice(patch_offsets, patch_extents);
						out(i,out_j,out_k,l) = reduce(Utils<Scalar>::tensor4d_to_mat(patch), patch_ind++);
					}
				}
			}
		}
		return out;
	};
	Tensor4D<Scalar> pass_back(Tensor4D<Scalar> out_grads) {
		assert(out_grads.dimension(1) == output_dims.get_dim1() && out_grads.dimension(2) == output_dims.get_dim2() &&
				out_grads.dimension(3) == output_dims.get_dim3());
		assert(out_grads.dimension(0) > 0 && rows == out_grads.dimension(0));
		if (is_input_layer())
			return Layer<Scalar>::EMPTY_TENSOR;
		int rows = out_grads.dimension(0);
		int depth = input_dims.get_dim3();
		int height_rem = input_dims.get_dim1() - receptor_size;
		int width_rem = input_dims.get_dim2() - receptor_size;
		Dimensions patch_dims((int) receptor_size, (int) receptor_size, 1);
		Eigen::array<int, 4> patch_offsets = { 0, 0, 0, 0 };
		Eigen::array<int, 4> patch_extents = { 1, patch_dims.get_dim1(), patch_dims.get_dim2(), patch_dims.get_dim3() };
		Tensor4D<Scalar> prev_out_grads(rows, input_dims.get_dim1(), input_dims.get_dim2(), depth);
		prev_out_grads.setZero();
		int patch_ind = 0;
		for (int i = 0; i < rows; i++) {
			patch_offsets[0] = i;
			int out_grads_j = 0;
			for (int j = 0; j <= height_rem; j += stride, out_grads_j++) {
				patch_offsets[1] = j;
				int out_grads_k = 0;
				for (int k = 0; k <= width_rem; k += stride, out_grads_k++) {
					patch_offsets[2] = k;
					for (int l = 0; l < depth; l++) {
						patch_offsets[3] = l;
						/* Expand the scalars back into patches and accumulate the gradients where the patches
						 * overlap. */
						prev_out_grads.slice(patch_offsets, patch_extents) +=
								Utils<Scalar>::mat_to_tensor4d(d_reduce(out_grads(i,out_grads_j,out_grads_k,l),
										patch_ind++), patch_dims);
					}
				}
			}
		}
		return prev_out_grads;
	};
	static int calculate_output_dim(int input_dim, int receptor_size, int stride) {
		return (input_dim - receptor_size) / stride + 1;
	};
	Dimensions input_dims;
	Dimensions output_dims;
	unsigned receptor_size;
	unsigned stride;
	// No actual parameters.
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Keep track of the input rows.
	int rows;
};

template<typename Scalar>
class MeanPoolingLayer : public PoolingLayer<Scalar> {
public:
	MeanPoolingLayer(Dimensions input_dims, unsigned receptor_size = 2, unsigned stride = 2) :
			PoolingLayer<Scalar>::PoolingLayer(input_dims, receptor_size, stride),
			receptor_area(PoolingLayer<Scalar>::receptor_size * PoolingLayer<Scalar>::receptor_size) { };
	Layer<Scalar>* clone() {
		return new MeanPoolingLayer(*this);
	};
protected:
	void init_cache() { };
	Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) {
		return patch.mean();
	};
	RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) {
		RowVector<Scalar> d_patch(receptor_area);
		d_patch.setConstant(grad / (Scalar) receptor_area);
		return d_patch;
	};
	void empty_cache() { };
private:
	int receptor_area;
};

template<typename Scalar>
class MaxPoolingLayer : public PoolingLayer<Scalar> {
public:
	MaxPoolingLayer(Dimensions input_dims, unsigned receptor_size = 2, unsigned stride = 2) :
			PoolingLayer<Scalar>::PoolingLayer(input_dims, receptor_size, stride),
			receptor_area(PoolingLayer<Scalar>::receptor_size * PoolingLayer<Scalar>::receptor_size) { };
	Layer<Scalar>* clone() {
		return new MaxPoolingLayer(*this);
	};
protected:
	void init_cache() {
		max_inds = std::vector<unsigned>(PoolingLayer<Scalar>::rows *
				PoolingLayer<Scalar>::output_dims.get_points());
	};
	Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) {
		int max_ind = 0;
		Scalar max = Utils<Scalar>::MIN;
		for (int i = 0; i < patch.cols(); i++) {
			Scalar val_i = patch(i);
			if (val_i > max) {
				max = val_i;
				max_ind = i;
			}
		}
		max_inds[patch_ind] = max_ind;
		return max;
	};
	RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) {
		RowVector<Scalar> d_patch(receptor_area);
		d_patch.setZero();
		d_patch(max_inds[patch_ind]) = grad;
		return d_patch;
	};
	void empty_cache() {
		max_inds = std::vector<unsigned>(0);
	};
private:
	int receptor_area;
	// Cache
	std::vector<unsigned> max_inds;
};

} /* namespace cppnn */

#endif /* LAYER_H_ */
