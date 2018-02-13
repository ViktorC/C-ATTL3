/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <Dimensions.h>
#include <memory>
#include <type_traits>
#include <Utils.h>
#include <utility>
#include <WeightInitialization.h>

namespace cattle {

// Forward declarations to NeuralNetwork and Optimizer so they can be friended.
template<typename Scalar, size_t Rank> class NeuralNetwork;
template<typename Scalar, size_t Rank> class Optimizer;

template<typename Scalar, size_t Rank>
class Layer {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal rank");
	friend class NeuralNetwork<Scalar,Rank>;
	friend class Optimizer<Scalar,Rank>;
	typedef Tensor<Scalar,Rank + 1> DataBatch;
public:
	virtual ~Layer() = default;
	// Clone pattern.
	virtual Layer<Scalar,Rank>* clone() const = 0;
	virtual const Dimensions<int,Rank>& get_input_dims() const = 0;
	virtual const Dimensions<int,Rank>& get_output_dims() const = 0;
	bool is_parametric() {
		return get_params().rows() > 0 && get_params().cols() > 0;
	};
protected:
	/* Only expose methods that allow for the modification of the
	 * layer's state to friends and sub-classes. */
	bool is_input_layer() {
		return input_layer;
	};
	virtual void init() = 0;
	virtual void empty_cache() = 0;
	virtual Matrix<Scalar>& get_params() = 0;
	virtual const Matrix<Scalar>& get_param_grads() const = 0;
	virtual void enforce_constraints() = 0;
	// Rank is increased by one to allow for batch training.
	virtual DataBatch pass_forward(DataBatch in, bool training) = 0;
	virtual DataBatch pass_back(DataBatch out_grads) = 0;
private:
	bool input_layer = false;
};

template<typename Scalar>
using WeightInitSharedPtr = std::shared_ptr<WeightInitialization<Scalar>>;

template<typename Scalar, size_t Rank>
class FCLayer : public Layer<Scalar,Rank> {
	typedef Tensor<Scalar,Rank + 1> DataBatch;
public:
	FCLayer(Dimensions<int,Rank> input_dims, size_t output_size, WeightInitSharedPtr<Scalar> weight_init,
			Scalar max_norm_constraint = 0) :
				input_dims(input_dims),
				output_dims(),
				weight_init(weight_init),
				max_norm_constraint(max_norm_constraint),
				max_norm(Utils<Scalar>::decidedly_greater(max_norm_constraint, .0)),
				weights(input_dims.get_volume() + 1, output_size),
				weight_grads(input_dims.get_volume() + 1, output_size) {
		assert(weight_init != nullptr);
		output_dims(0) = output_size;
	};
	Layer<Scalar,Rank>* clone() const {
		return new FCLayer(*this);
	};
	const Dimensions<int,Rank>& get_input_dims() const {
		return input_dims;
	};
	const Dimensions<int,Rank>& get_output_dims() const {
		return output_dims;
	};
protected:
	void init() {
		weight_init->apply(weights);
		weight_grads.setZero(weight_grads.rows(), weight_grads.cols());
	};
	inline void empty_cache() {
		biased_in = Matrix<Scalar>(0, 0);
	};
	Matrix<Scalar>& get_params() {
		return weights;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return weight_grads;
	};
	inline void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = weights.squaredNorm();
			if (l2_norm > max_norm_constraint)
				weights *= (max_norm_constraint / l2_norm);
		}
	};
	inline DataBatch pass_forward(DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(in).demote() == input_dims);
		assert(in.dimension(0) > 0);
		unsigned input_size = input_dims.get_volume();
		// Add a 1-column to the input for the bias trick.
		biased_in = Matrix<Scalar>(in.dimension(0), input_size + 1);
		biased_in.leftCols(input_size) = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(in));
		biased_in.col(input_size).setOnes();
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>((biased_in * weights).eval(), output_dims);
	};
	inline DataBatch pass_back(DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).demote() == output_dims);
		assert(out_grads.dimension(0) > 0 && biased_in.rows() == out_grads.dimension(0));
		Matrix<Scalar> out_grads_mat = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads));
		// Compute the gradients of the outputs with respect to the weights.
		weight_grads = biased_in.transpose() * out_grads_mat;
		if (Layer<Scalar,Rank>::is_input_layer())
			return Utils<Scalar>::template get_null_tensor<Rank + 1>();
		/* Remove the bias row from the weight matrix, transpose it, and compute gradients w.r.t. the
		 * previous layer's output. */
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>((out_grads_mat * weights.topRows(input_dims.get_volume())
				.transpose()).eval(), input_dims);
	};
private:
	Dimensions<int,Rank> input_dims;
	Dimensions<int,Rank> output_dims;
	WeightInitSharedPtr<Scalar> weight_init;
	Scalar max_norm_constraint;
	bool max_norm;
	/* Eigen matrices are backed by arrays allocated on the heap, so these
	 * members do not burden the stack. */
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	// Staged computation caches
	Matrix<Scalar> biased_in;
};

template<typename Scalar, size_t Rank>
class ActivationLayer : public Layer<Scalar,Rank> {
	typedef Tensor<Scalar,Rank + 1> DataBatch;
public:
	ActivationLayer(Dimensions<int,Rank> dims) :
			dims(dims),
			params(0, 0),
			param_grads(0, 0) { };
	virtual ~ActivationLayer() = default;
	const Dimensions<int,Rank>& get_input_dims() const {
		return dims;
	};
	const Dimensions<int,Rank>& get_output_dims() const {
		return dims;
	};
protected:
	virtual Matrix<Scalar> activate(const Matrix<Scalar>& in) = 0;
	virtual Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) = 0;
	void init() { };
	inline void empty_cache() {
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
	inline DataBatch pass_forward(DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(in).demote() == dims);
		assert(in.dimension(0) > 0);
		this->in = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(in));
		out = activate(this->in);
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(out, dims);
	};
	inline DataBatch pass_back(DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).demote() == dims);
		assert(out_grads.dimension(0) > 0 && out.rows() == out_grads.dimension(0));
		if (Layer<Scalar,Rank>::is_input_layer())
			return Utils<Scalar>::template get_null_tensor<Rank + 1>();
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(d_activate(in, out,
				Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads))), dims);
	};
	Dimensions<int,Rank> dims;
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation caches
	Matrix<Scalar> in;
	Matrix<Scalar> out;
};

template<typename Scalar, size_t Rank>
class IdentityActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	IdentityActivationLayer(Dimensions<int,Rank> dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { };
	Layer<Scalar,Rank>* clone() const {
		return new IdentityActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in;
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return out_grads;
	};
};

template<typename Scalar, size_t Rank>
class ScalingActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	ScalingActivationLayer(Dimensions<int,Rank> dims, Scalar scale) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			scale(scale) { };
	Layer<Scalar,Rank>* clone() const {
		return new ScalingActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in * scale;
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return out_grads * scale;
	};
private:
	Scalar scale;
};

template<typename Scalar, size_t Rank>
class BinaryStepActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	BinaryStepActivationLayer(Dimensions<int,Rank> dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { };
	Layer<Scalar,Rank>* clone() const {
		return new BinaryStepActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : .0); });
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return Matrix<Scalar>::Zero(in.rows(), in.cols());
	};
};

template<typename Scalar, size_t Rank>
class SigmoidActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	SigmoidActivationLayer(Dimensions<int,Rank> dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { };
	Layer<Scalar,Rank>* clone() const {
		return new SigmoidActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return ((-in).array().exp() + 1).inverse();
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return (out.array() *  (-out.array() + 1)) * out_grads.array();
	};
};

template<typename Scalar, size_t Rank>
class TanhActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	TanhActivationLayer(Dimensions<int,Rank> dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { };
	Layer<Scalar,Rank>* clone() const {
		return new TanhActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.array().tanh();
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return (-out.array() * out.array() + 1) * out_grads.array();
	};
};

template<typename Scalar, size_t Rank>
class SoftmaxActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	SoftmaxActivationLayer(Dimensions<int,Rank> dims, Scalar epsilon = Utils<Scalar>::EPSILON2) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			epsilon(epsilon) { };
	Layer<Scalar,Rank>* clone() const {
		return new SoftmaxActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		/* First subtract the value of the greatest coefficient from each element row-wise
		 * to avoid an overflow due to raising e to great powers. */
		Matrix<Scalar> out = (in.array().colwise() - in.array().rowwise().maxCoeff()).exp();
		return out.array().colwise() / (out.array().rowwise().sum() + epsilon);
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		Matrix<Scalar> d_in(in.rows(), in.cols());
		for (int i = 0; i < d_in.rows(); i++) {
			RowVector<Scalar> row_i = out.row(i);
			auto jacobian = (row_i.asDiagonal() - (row_i.transpose() * row_i));
			d_in.row(i) = out_grads.row(i) * jacobian;
		}
		return d_in;
	};
private:
	Scalar epsilon;
};

template<typename Scalar, size_t Rank>
class ReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	ReLUActivationLayer(Dimensions<int,Rank> dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { };
	Layer<Scalar,Rank>* clone() const {
		return new ReLUActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(.0);
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : .0); })
				.cwiseProduct(out_grads);
	};
};

template<typename Scalar, size_t Rank>
class LeakyReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	LeakyReLUActivationLayer(Dimensions<int,Rank> dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha) { };
	Layer<Scalar,Rank>* clone() const {
		return new LeakyReLUActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(in * alpha);
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : alpha); })
				.cwiseProduct(out_grads);
	};
private:
	Scalar alpha;
};

template<typename Scalar, size_t Rank>
class ELUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	ELUActivationLayer(Dimensions<int,Rank> dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha) { };
	Layer<Scalar,Rank>* clone() const {
		return new ELUActivationLayer(*this);
	};
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i > .0 ? i : (alpha * (exp(i) - 1))); });
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
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

template<typename Scalar, size_t Rank>
class PReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	PReLUActivationLayer(Dimensions<int,Rank> dims, Scalar init_alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			init_alpha(init_alpha) {
		ActivationLayer<Scalar,Rank>::params.resize(1, dims.get_volume());
		ActivationLayer<Scalar,Rank>::param_grads.resize(1, dims.get_volume());
	};
	Layer<Scalar,Rank>* clone() const {
		return new PReLUActivationLayer(*this);
	};
protected:
	void init() {
		ActivationLayer<Scalar,Rank>::params.setConstant(init_alpha);
		ActivationLayer<Scalar,Rank>::param_grads.setZero(1, ActivationLayer<Scalar,Rank>::dims.get_volume());
	};
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(in * ActivationLayer<Scalar,Rank>::params.row(0).asDiagonal());
	};
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		ActivationLayer<Scalar,Rank>::param_grads.row(0).setZero();
		Matrix<Scalar> d_in = Matrix<Scalar>(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); i++) {
			for (int j = 0; j < in.rows(); j++) {
				Scalar in_ji = in(j,i);
				if (in_ji >= 0)
					d_in(j,i) = out_grads(j,i);
				else {
					Scalar out_ji = out_grads(j,i);
					d_in(j,i) = ActivationLayer<Scalar,Rank>::params(0,i) * out_ji;
					ActivationLayer<Scalar,Rank>::param_grads(0,i) += in_ji * out_ji;
				}
			}
		}
		return d_in;
	};
private:
	Scalar init_alpha;
};

template<typename Scalar, size_t Rank>
class BatchNormLayerBase : public Layer<Scalar,Rank> {
public:
	BatchNormLayerBase(Dimensions<int,Rank> dims, int depth, Scalar norm_avg_decay, Scalar epsilon) :
			dims(dims),
			depth(depth),
			norm_avg_decay(norm_avg_decay),
			epsilon(epsilon),
			avg_means(depth, dims.get_volume() / depth),
			avg_inv_sds(depth, dims.get_volume() / depth),
			avgs_init(false),
			params(2 * depth, dims.get_volume() / depth),
			param_grads(2 * depth, dims.get_volume() / depth),
			cache_vec(depth) {
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	virtual ~BatchNormLayerBase() = default;
	const Dimensions<int,Rank>& get_input_dims() const {
		return dims;
	};
	const Dimensions<int,Rank>& get_output_dims() const {
		return dims;
	};
protected:
	typedef Tensor<Scalar,Rank + 1> DataBatch;
	void init() {
		for (int i = 0; i < params.rows(); i += 2) {
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
	inline void empty_cache() {
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
	inline DataBatch _pass_forward(DataBatch in, const Dimensions<int,Rank>& output_dims, bool training, int i) {
		Matrix<Scalar> in_ch_i = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(in));
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
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(((in_ch_i * params.row(2 * i).asDiagonal())
				.rowwise() + params.row(2 * i + 1)).eval(), output_dims);
	};
	inline DataBatch _pass_back(DataBatch out_grads, const Dimensions<int,Rank>& output_dims, int i) {
		int rows = out_grads.dimension(0);
		Cache& cache = cache_vec[i];
		Matrix<Scalar> std_in_grads_i;
		{ // Manage memory by scope restriction.
			Matrix<Scalar> out_grads_ch_map_i = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads));
			param_grads.row(2 * i) = out_grads_ch_map_i.cwiseProduct(cache.std_in).colwise().sum();
			param_grads.row(2 * i + 1) = out_grads_ch_map_i.colwise().sum();
			if (Layer<Scalar,Rank>::is_input_layer())
				return Utils<Scalar>::template get_null_tensor<Rank + 1>();
			std_in_grads_i = out_grads_ch_map_i * params.row(2 * i).asDiagonal();
		}
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(((((rows * std_in_grads_i).rowwise() -
				std_in_grads_i.colwise().sum()) - cache.std_in *
				(cache.std_in.cwiseProduct(std_in_grads_i).colwise().sum().asDiagonal())) *
				((1.0 / rows) * cache.inv_in_sd).asDiagonal()).eval(), output_dims);
	};
	Dimensions<int,Rank> dims;
	int depth;
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

template<typename Scalar, size_t Rank>
class BatchNormLayer : public BatchNormLayerBase<Scalar,Rank> {
	typedef BatchNormLayerBase<Scalar,Rank> Base;
public:
	BatchNormLayer(Dimensions<int,Rank> dims, Scalar norm_avg_decay = .1, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			Base::template BatchNormLayerBase(dims, 1, norm_avg_decay, epsilon) { };
	Layer<Scalar,Rank>* clone() const {
		return new BatchNormLayer(*this);
	};
protected:
	inline typename Base::DataBatch pass_forward(typename Base::DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(in).demote() == Base::dims);
		assert(in.dimension(0) > 0);
		return Base::_pass_forward(std::move(in), Base::dims, training, 0);
	};
	inline typename Base::DataBatch pass_back(typename Base::DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).demote() == Base::dims);
		assert(out_grads.dimension(0) > 0 && Base::cache_vec[0].std_in.rows() == out_grads.dimension(0));
		return _pass_back(std::move(out_grads), Base::dims, 0);
	};
};

// Partial specialization for multi-channel tensors.
template<typename Scalar>
class BatchNormLayer<Scalar,3> : public BatchNormLayerBase<Scalar,3> {
	typedef BatchNormLayerBase<Scalar,3> Base;
public:
	BatchNormLayer(Dimensions<int,3> dims, Scalar norm_avg_decay = .1, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			Base::template BatchNormLayerBase(dims, dims(2), norm_avg_decay, epsilon) { };
	Layer<Scalar,3>* clone() const {
		return new BatchNormLayer(*this);
	};
protected:
	inline typename Base::DataBatch pass_forward(typename Base::DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<4>(in).demote() == Base::dims);
		assert(in.dimension(0) > 0);
		if (Base::depth == 1)
			return Base::_pass_forward(std::move(in), Base::dims, training, 0);
		else { // Multi-channel image data; depth-wise normalization.
			int rows = in.dimension(0);
			typename Base::DataBatch out(rows, Base::dims(0), Base::dims(1), Base::dims(2));
			Dimensions<int,3> slice_dims({ Base::dims(0), Base::dims(1), 1 });
			std::array<int,4> offsets({ 0, 0, 0, 0 });
			std::array<int,4> extents({ rows, slice_dims(0), slice_dims(1), slice_dims(2) });
			for (int i = 0; i < Base::depth; i++) {
				offsets[3] = i;
				typename Base::DataBatch in_slice_i = in.slice(offsets, extents);
				out.slice(offsets, extents) = Base::_pass_forward(std::move(in_slice_i), slice_dims, training, i);
			}
			return out;
		}
	};
	inline typename Base::DataBatch pass_back(typename Base::DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<4>(out_grads).demote() == Base::dims);
		assert(out_grads.dimension(0) > 0 && Base::cache_vec[0].std_in.rows() == out_grads.dimension(0));
		if (Base::depth == 1)
			return Base::_pass_back(std::move(out_grads), Base::dims, 0);
		else {
			int rows = out_grads.dimension(0);
			typename Base::DataBatch prev_out_grads;
			if (!Layer<Scalar,3>::is_input_layer())
				prev_out_grads = typename Base::DataBatch(rows, Base::dims(0), Base::dims(1), Base::dims(2));
			Dimensions<int,3> slice_dims({ Base::dims(0), Base::dims(1), 1 });
			std::array<int,4> offsets({ 0, 0, 0, 0 });
			std::array<int,4> extents({ rows, slice_dims(0), slice_dims(1), slice_dims(2) });
			/* Back-propagate the gradient through the batch normalization 'function' and also calculate the
			 * gradients on the betas and gammas. */
			for (int i = 0; i < Base::depth; i++) {
				offsets[3] = i;
				typename Base::DataBatch out_grads_slice_i = out_grads.slice(offsets, extents);
				if (Layer<Scalar,3>::is_input_layer())
					Base::_pass_back(std::move(out_grads_slice_i), slice_dims, i);
				else
					prev_out_grads.slice(offsets, extents) = Base::_pass_back(std::move(out_grads_slice_i), slice_dims, i);
			}
			return prev_out_grads;
		}
	};
};

template<typename Scalar, size_t Rank>
class DropoutLayer : public Layer<Scalar,Rank> {
	typedef Tensor<Scalar,Rank + 1> DataBatch;
public:
	DropoutLayer(Dimensions<int,Rank> dims, Scalar dropout_prob, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			dims(dims),
			dropout_prob(dropout_prob),
			epsilon(epsilon),
			dropout(Utils<Scalar>::decidedly_greater(dropout_prob, .0)),
			params(0, 0),
			param_grads(0, 0) {
		assert(dropout_prob <= 1 && "dropout prob must not be greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	Layer<Scalar,Rank>* clone() const {
		return new DropoutLayer(*this);
	};
	const Dimensions<int,Rank>& get_input_dims() const {
		return dims;
	};
	const Dimensions<int,Rank>& get_output_dims() const {
		return dims;
	};
protected:
	void init() { };
	inline void empty_cache() {
		dropout_mask = Matrix<Scalar>(0, 0);
	};
	Matrix<Scalar>& get_params() {
		return params;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return param_grads;
	};
	inline void enforce_constraints() { };
	inline DataBatch pass_forward(DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(in).demote() == dims);
		assert(in.dimension(0) > 0);
		if (training && dropout) {
			Matrix<Scalar> in_mat = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(in));
			dropout_mask = Matrix<Scalar>(in_mat.rows(), in_mat.cols());
			dropout_mask.setRandom(in_mat.rows(), in_mat.cols());
			// Inverted dropout.
			Scalar scaling_factor = 1 / (1 - dropout_prob + epsilon);
			dropout_mask = ((dropout_mask.array() + 1) / 2).unaryExpr([this,scaling_factor](Scalar i) {
				return (Scalar) (i <= dropout_prob ? .0 : scaling_factor);
			});
			return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(in_mat.cwiseProduct(dropout_mask).eval(), dims);
		}
		return in;
	};
	inline DataBatch pass_back(DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).demote() == dims);
		assert(out_grads.dimension(0) > 0 && dropout_mask.rows() == out_grads.dimension(0));
		if (Layer<Scalar,Rank>::is_input_layer())
			return Utils<Scalar>::template get_null_tensor<Rank + 1>();
		// The derivative of the dropout 'function'.
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads))
				.cwiseProduct(dropout_mask).eval(), dims);
	};
private:
	Dimensions<int,Rank> dims;
	Scalar dropout_prob;
	Scalar epsilon;
	bool dropout;
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation cache_vec
	Matrix<Scalar> dropout_mask;
};
#include <iostream>
template<typename Scalar, size_t Rank>
class ConvLayerBase : public Layer<Scalar,Rank> {
	static_assert(Rank == 2 || Rank == 3, "illegal convolutaional layer rank");
public:
	ConvLayerBase(Dimensions<int,Rank> input_dims, int input_depth, size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			size_t receptor_size, size_t padding, size_t stride, size_t dilation, Scalar max_norm_constraint) :
				input_dims(input_dims),
				output_dims({ calculate_output_dim(input_dims(0), receptor_size, padding, dilation, stride),
						calculate_output_dim(input_dims(1), receptor_size, padding, dilation, stride), (int) filters }),
				input_depth(input_depth),
				filters(filters),
				weight_init(weight_init),
				receptor_size(receptor_size),
				padding(padding),
				stride(stride),
				dilation(dilation),
				max_norm_constraint(max_norm_constraint),
				max_norm(Utils<Scalar>::decidedly_greater(max_norm_constraint, .0)),
				padded_height(input_dims(0) + 2 * padding),
				padded_width(input_dims(1) + 2 * padding),
				dil_receptor_size(receptor_size + (receptor_size - 1) * dilation),
				weights(receptor_size * receptor_size * input_depth + 1, filters),
				weight_grads(weights.rows(), (int) filters) {
		assert(filters > 0);
		assert(weight_init != nullptr);
		assert(receptor_size > 0);
		assert(stride > 0);
		assert(input_dims(1) + 2 * padding >= receptor_size + (receptor_size - 1) * dilation &&
				input_dims(2) + 2 * padding >= receptor_size + (receptor_size - 1) * dilation);
	};
	virtual ~ConvLayerBase() = default;
	const Dimensions<int,Rank>& get_input_dims() const {
		return input_dims;
	};
	const Dimensions<int,Rank>& get_output_dims() const {
		return output_dims;
	};
protected:
	typedef Tensor<Scalar,Rank + 1> DataBatch;
	typedef std::array<int,Rank + 1> RankwiseArray;
	void init() {
		/* For every filter, there is a column in the weight matrix with the same number of
		 * elements as the size of the receptive field (F * F * D) + 1 for the bias row. */
		weight_init->apply(weights);
		weight_grads.setZero(weight_grads.rows(), weight_grads.cols());
	};
	inline void empty_cache() {
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
	inline void _pass_forward(const DataBatch& padded_in, DataBatch& out, const RankwiseArray& row_extents,
			const RankwiseArray& patch_extents, const RankwiseArray& dil_strides, bool training) {
		int rows = padded_in.dimension(0);
		int patches = output_dims(0) * output_dims(1);
		int receptor_vol = receptor_size * receptor_size * input_depth;
		int height_rem = padded_height - dil_receptor_size;
		int width_rem = padded_width - dil_receptor_size;
		RankwiseArray row_offsets;
		RankwiseArray patch_offsets;
		row_offsets.fill(0);
		patch_offsets.fill(0);
		/* 'Tensor-row' by 'tensor-row', stretch the receptor locations into row vectors, form a matrix out of
		 * them, and multiply it by the weight matrix. */
		for (int i = 0; i < rows; i++) {
			row_offsets[0] = i;
			patch_offsets[0] = i;
			int patch_ind = 0;
			{
				Matrix<Scalar> in_mat_i(patches, receptor_vol);
				for (int j = 0; j <= height_rem; j += stride) {
					patch_offsets[1] = j;
					for (int k = 0; k <= width_rem; k += stride) {
						patch_offsets[2] = k;
						// If the patch is dilated, skip the internal padding when stretching it into a row.
						DataBatch patch;
						if (dilation > 0)
							patch = padded_in.slice(patch_offsets, patch_extents).stride(dil_strides);
						else
							patch = padded_in.slice(patch_offsets, patch_extents);
						in_mat_i.row(patch_ind++) = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(patch));
					}
				}
				assert(patch_ind == patches);
				// Set the additional column's elements to 1.
				biased_in_vec[i] = Matrix<Scalar>(patches, receptor_vol + 1);
				Matrix<Scalar>& biased_in = biased_in_vec[i];
				biased_in.col(receptor_vol).setOnes();
				biased_in.leftCols(receptor_vol) = in_mat_i;
			}
			/* Flatten the matrix product into a row vector, reshape it into a 'single-row' sub-tensor, and
			 * assign it to the output tensor's corresponding 'row'. */
			Matrix<Scalar> out_i = biased_in_vec[i] * weights;
			out.slice(row_offsets, row_extents) = Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(MatrixMap<Scalar>(out_i.data(),
					1, out_i.rows() * out_i.cols()).matrix(), output_dims);
		}
	};
	inline void _pass_back(const DataBatch& out_grads, DataBatch& prev_out_grads, const Dimensions<int,Rank> comp_patch_dims,
			const RankwiseArray& out_grads_row_extents, const RankwiseArray& patch_extents, const RankwiseArray& strides) {
		int rows = out_grads.dimension(0);
		int height_rem = padded_height - dil_receptor_size;
		int width_rem = padded_width - dil_receptor_size;
		RankwiseArray out_grads_row_offsets;
		RankwiseArray patch_offsets;
		out_grads_row_offsets.fill(0);
		patch_offsets.fill(0);
		weight_grads.setZero(weight_grads.rows(), weight_grads.cols());
		for (int i = 0; i < rows; i++) {
			out_grads_row_offsets[0] = i;
			Matrix<Scalar> prev_out_grads_mat_i;
			{
				DataBatch slice_i = out_grads.slice(out_grads_row_offsets, out_grads_row_extents);
				MatrixMap<Scalar> out_grads_mat_map_i = MatrixMap<Scalar>(slice_i.data(), output_dims(0) *
						output_dims(1), filters);
				// Accumulate the gradients of the outputs w.r.t. the weights for each observation a.k.a 'tensor-row'.
				weight_grads += biased_in_vec[i].transpose() * out_grads_mat_map_i;
				if (Layer<Scalar,Rank>::is_input_layer())
					continue;
				/* Remove the bias row from the weight matrix, transpose it, and compute the gradients w.r.t. the
				 * previous layer's output. */
				prev_out_grads_mat_i = out_grads_mat_map_i * weights.topRows(weights.rows() - 1).transpose();
			}
			/* Given the gradients w.r.t. the stretched out receptor patches, perform a 'backwards' convolution
			 * to get the gradients w.r.t. the individual input nodes. */
			int patch_ind = 0;
			patch_offsets[0] = i;
			for (int j = 0; j <= height_rem; j += stride) {
				patch_offsets[1] = j;
				for (int k = 0; k <= width_rem; k += stride) {
					patch_offsets[2] = k;
					// Accumulate the gradients where the receptor-patch-tensors overlap.
					if (dilation > 0) {
						prev_out_grads.slice(patch_offsets, patch_extents).stride(strides) +=
								Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(prev_out_grads_mat_i.row(patch_ind++), comp_patch_dims);
					} else {
						prev_out_grads.slice(patch_offsets, patch_extents) += Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(
								prev_out_grads_mat_i.row(patch_ind++), comp_patch_dims);
					}
				}
			}
			assert(patch_ind == prev_out_grads_mat_i.rows());
		}
	};
	static int calculate_output_dim(int input_dim, int receptor_size, int padding, int dilation, int stride) {
		return (input_dim - receptor_size - (receptor_size - 1) * dilation + 2 * padding) / stride + 1;
	};
	Dimensions<int,Rank> input_dims;
	Dimensions<int,Rank> output_dims;
	int input_depth;
	size_t filters;
	WeightInitSharedPtr<Scalar> weight_init;
	size_t receptor_size;
	size_t padding;
	size_t stride;
	size_t dilation;
	Scalar max_norm_constraint;
	bool max_norm;
	int padded_height;
	int padded_width;
	int dil_receptor_size;
	// The learnable parameters.
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
	// Staged computation caches
	std::vector<Matrix<Scalar>> biased_in_vec;
};

// Convolutional layer for 2D input data.
template<typename Scalar, size_t Rank>
class ConvLayer : public ConvLayerBase<Scalar,Rank> {
	typedef ConvLayerBase<Scalar,Rank> Base;
public:
	ConvLayer(Dimensions<int,Rank> input_dims, size_t filters, WeightInitSharedPtr<Scalar> weight_init, size_t receptor_size = 3,
			size_t padding = 1, size_t stride = 1, size_t dilation = 0, Scalar max_norm_constraint = 0) :
				Base::ConvLayerBase(input_dims, 1, filters, weight_init, receptor_size, padding, stride, dilation,
						max_norm_constraint) { };
	Layer<Scalar,Rank>* clone() const {
		return new ConvLayer(*this);
	};
protected:
	inline typename Base::DataBatch pass_forward(typename Base::DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<3>(in).demote() == Base::input_dims);
		assert(in.dimension(0) > 0);
		// Spatial padding.
		std::array<std::pair<int,int>,3> paddings;
		paddings[0] = std::make_pair(0, 0);
		paddings[1] = std::make_pair(Base::padding, Base::padding);
		paddings[2] = std::make_pair(Base::padding, Base::padding);
		typename Base::DataBatch padded_in = in.pad(paddings);
		// Free the memory occupied by the now-expendable input tensor.
		in = Utils<Scalar>::template get_null_tensor<3>();
		// Prepare the base offsets and extents for slicing and dilation.
		int dil_receptor_size = Base::receptor_size + (Base::receptor_size - 1) * Base::dilation;
		typename Base::RankwiseArray row_extents({ 1, Base::output_dims(0), Base::output_dims(1) });
		typename Base::RankwiseArray patch_extents({ 1, dil_receptor_size, dil_receptor_size });
		typename Base::RankwiseArray dil_strides({ 1, (int) Base::dilation + 1, (int) Base::dilation + 1 });
		typename Base::DataBatch out({ in.dimension(0), Base::output_dims(0), Base::output_dims(1) });
		Base::biased_in_vec = std::vector<Matrix<Scalar>>(padded_in.dimension(0));
		Base::_pass_forward(padded_in, out, row_extents, patch_extents, dil_strides, training);
		return out;
	};
	inline typename Base::DataBatch pass_back(typename Base::DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<3>(out_grads).demote() == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && Base::biased_in_vec.size() == (unsigned) out_grads.dimension(0));
		int rows = out_grads.dimension(0);
		Dimensions<int,2> comp_patch_dims({ (int) Base::receptor_size, (int) Base::receptor_size });
		typename Base::RankwiseArray out_grads_row_extents({ 1, Base::output_dims(0), Base::output_dims(1) });
		typename Base::RankwiseArray patch_extents({ 1, Base::dil_receptor_size, Base::dil_receptor_size });
		typename Base::RankwiseArray strides({ 1, (int) Base::dilation + 1, (int) Base::dilation + 1 });
		typename Base::DataBatch prev_out_grads({ rows, Base::padded_height, Base::padded_width });
		prev_out_grads.setZero();
		Base::weight_grads.setZero(Base::weight_grads.rows(), Base::weight_grads.cols());
		Base::_pass_back(out_grads, prev_out_grads, comp_patch_dims, out_grads_row_extents, patch_extents, strides);
		// Cut off the padding.
		typename Base::RankwiseArray no_padding_offsets({ 0, (int) Base::padding, (int) Base::padding });
		typename Base::RankwiseArray no_padding_extents({ rows, Base::input_dims(0), Base::input_dims(1) });
		return prev_out_grads.slice(no_padding_offsets, no_padding_extents);
	};
};

// Partial specialization for 3D convolution.
template<typename Scalar>
class ConvLayer<Scalar,3> : public ConvLayerBase<Scalar,3> {
	typedef ConvLayerBase<Scalar,3> Base;
public:
	ConvLayer(Dimensions<int,3> input_dims, size_t filters, WeightInitSharedPtr<Scalar> weight_init, size_t receptor_size = 3,
			size_t padding = 1, size_t stride = 1, size_t dilation = 0, Scalar max_norm_constraint = 0) :
				Base::ConvLayerBase(input_dims, input_dims(2), filters, weight_init, receptor_size, padding, stride,
						dilation, max_norm_constraint) { };
	Layer<Scalar,3>* clone() const {
		return new ConvLayer(*this);
	};
protected:
	inline typename Base::DataBatch pass_forward(typename Base::DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<4>(in).demote() == Base::input_dims);
		assert(in.dimension(0) > 0);
		// Spatial padding.
		std::array<std::pair<int,int>,4> paddings;
		paddings[0] = std::make_pair(0, 0);
		paddings[1] = std::make_pair(Base::padding, Base::padding);
		paddings[2] = std::make_pair(Base::padding, Base::padding);
		paddings[3] = std::make_pair(0, 0);
		typename Base::DataBatch padded_in = in.pad(paddings);
		// Free the memory occupied by the now-expendable input tensor.
		in = Utils<Scalar>::template get_null_tensor<4>();
		// Prepare the base offsets and extents for slicing and dilation.
		int dil_receptor_size = Base::receptor_size + (Base::receptor_size - 1) * Base::dilation;
		typename Base::RankwiseArray row_extents({ 1, Base::output_dims(0), Base::output_dims(1), Base::output_dims(2) });
		typename Base::RankwiseArray patch_extents({ 1, dil_receptor_size, dil_receptor_size, padded_in.dimension(3) });
		typename Base::RankwiseArray dil_strides({ 1, (int) Base::dilation + 1, (int) Base::dilation + 1, 1 });
		typename Base::DataBatch out(padded_in.dimension(0), Base::output_dims(0), Base::output_dims(1), Base::output_dims(2));
		Base::biased_in_vec = std::vector<Matrix<Scalar>>(padded_in.dimension(0));
		Base::_pass_forward(padded_in, out, row_extents, patch_extents, dil_strides, training);
		return out;
	};
	inline typename Base::DataBatch pass_back(typename Base::DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<4>(out_grads).demote() == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && Base::biased_in_vec.size() == (unsigned) out_grads.dimension(0));
		int rows = out_grads.dimension(0);
		int depth = Base::input_dims(2);
		Dimensions<int,3> comp_patch_dims({ (int) Base::receptor_size, (int) Base::receptor_size, depth });
		typename Base::RankwiseArray out_grads_row_extents({ 1, Base::output_dims(0), Base::output_dims(1), Base::output_dims(2) });
		typename Base::RankwiseArray patch_extents({ 1, Base::dil_receptor_size, Base::dil_receptor_size, depth });
		typename Base::RankwiseArray strides({ 1, (int) Base::dilation + 1, (int) Base::dilation + 1, 1 });
		typename Base::DataBatch prev_out_grads(rows, Base::padded_height, Base::padded_width, depth);
		prev_out_grads.setZero();
		Base::weight_grads.setZero(Base::weight_grads.rows(), Base::weight_grads.cols());
		Base::_pass_back(out_grads, prev_out_grads, comp_patch_dims, out_grads_row_extents, patch_extents, strides);
		// Cut off the padding.
		typename Base::RankwiseArray no_padding_offsets({ 0, (int) Base::padding, (int) Base::padding, 0 });
		typename Base::RankwiseArray no_padding_extents({ rows, Base::input_dims(0), Base::input_dims(1), Base::input_dims(2) });
		return prev_out_grads.slice(no_padding_offsets, no_padding_extents);
	};
};

template<typename Scalar, size_t Rank>
class PoolingLayerBase : public Layer<Scalar,Rank> {
	static_assert(Rank == 2 || Rank == 3, "illegal pooling layer rank");
public:
	PoolingLayerBase(Dimensions<int,Rank> input_dims, size_t receptor_size, size_t stride) :
			input_dims(input_dims),
			output_dims({ calculate_output_dim(input_dims(0), receptor_size, stride),
					calculate_output_dim(input_dims(1), receptor_size, stride), input_dims(2) }),
			receptor_size(receptor_size),
			stride(stride),
			receptor_area(receptor_size * receptor_size),
			height_rem(input_dims(0) - receptor_size),
			width_rem(input_dims(1) - receptor_size),
			params(0, 0),
			param_grads(0, 0) {
		assert(input_dims(0) >= (int) receptor_size && input_dims(1) >= (int) receptor_size);
		assert(receptor_size > 0);
		assert(stride > 0);
	};
	virtual ~PoolingLayerBase() = default;
	const Dimensions<int,Rank>& get_input_dims() const {
		return input_dims;
	};
	const Dimensions<int,Rank>& get_output_dims() const {
		return output_dims;
	};
protected:
	typedef Tensor<Scalar,Rank + 1> DataBatch;
	typedef std::array<int,Rank + 1> RankwiseArray;
	void init() { };
	Matrix<Scalar>& get_params() {
		return params;
	};
	const Matrix<Scalar>& get_param_grads() const {
		return param_grads;
	};
	void enforce_constraints() { };
	static int calculate_output_dim(int input_dim, int receptor_size, int stride) {
		return (input_dim - receptor_size) / stride + 1;
	};
	Dimensions<int,Rank> input_dims;
	Dimensions<int,Rank> output_dims;
	size_t receptor_size;
	size_t stride;
	int receptor_area;
	int height_rem;
	int width_rem;
	// No actual parameters.
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Keep track of the input rows.
	int rows;
};

// 2D pooling layer.
template<typename Scalar, size_t Rank>
class PoolingLayer : public PoolingLayerBase<Scalar,Rank> {
	typedef PoolingLayerBase<Scalar,Rank> Base;
public:
	PoolingLayer(Dimensions<int,Rank> input_dims, size_t receptor_size, size_t stride) :
			Base::PoolingLayerBase(input_dims, receptor_size, stride) { };
	virtual ~PoolingLayer() = default;
protected:
	virtual void init_cache() = 0;
	virtual Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) = 0;
	virtual RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) = 0;
	inline typename Base::DataBatch pass_forward(typename Base::DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<3>(in).demote() == Base::input_dims);
		assert(in.dimension(0) > 0);
		Base::rows = in.dimension(0);
		typename Base::RankwiseArray patch_offsets({ 0, 0, 0 });
		typename Base::RankwiseArray patch_extents({ 1, (int) Base::receptor_size, (int) Base::receptor_size });
		typename Base::DataBatch out(Base::rows, Base::output_dims(0), Base::output_dims(1));
		init_cache();
		int patch_ind = 0;
		for (int i = 0; i < Base::rows; i++) {
			patch_offsets[0] = i;
			int out_j = 0;
			for (int j = 0; j <= Base::height_rem; j += Base::stride, out_j++) {
				patch_offsets[1] = j;
				int out_k = 0;
				for (int k = 0; k <= Base::width_rem; k += Base::stride, out_k++) {
					patch_offsets[2] = k;
					// Reduce the patches to scalars.
					typename Base::DataBatch patch = in.slice(patch_offsets, patch_extents);
					out(i,out_j,out_k) = reduce(Utils<Scalar>::template map_tensor_to_mat<3>(std::move(patch)), patch_ind++);
				}
			}
		}
		return out;
	};
	inline typename Base::DataBatch pass_back(typename Base::DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<3>(out_grads).demote() == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && Base::rows == out_grads.dimension(0));
		if (Layer<Scalar,2>::is_input_layer())
			return Utils<Scalar>::template get_null_tensor<3>();
		int depth = Base::input_dims(2);
		Dimensions<int,2> patch_dims({ (int) Base::receptor_size, (int) Base::receptor_size });
		typename Base::RankwiseArray patch_offsets({ 0, 0, 0});
		typename Base::RankwiseArray patch_extents({ 1, patch_dims(0), patch_dims(1) });
		typename Base::DataBatch prev_out_grads(Base::rows, Base::input_dims(0), Base::input_dims(1));
		prev_out_grads.setZero();
		int patch_ind = 0;
		for (int i = 0; i < Base::rows; i++) {
			patch_offsets[0] = i;
			int out_grads_j = 0;
			for (int j = 0; j <= Base::height_rem; j += Base::stride, out_grads_j++) {
				patch_offsets[1] = j;
				int out_grads_k = 0;
				for (int k = 0; k <= Base::width_rem; k += Base::stride, out_grads_k++) {
					patch_offsets[2] = k;
					// Expand the scalars back into patches and accumulate the gradients where the patches overlap.
					prev_out_grads.slice(patch_offsets, patch_extents) += Utils<Scalar>::template map_mat_to_tensor<3>(
							d_reduce(out_grads(i,out_grads_j,out_grads_k), patch_ind++), patch_dims);
				}
			}
		}
		return prev_out_grads;
	};
};

// 3D pooling layer.
template<typename Scalar>
class PoolingLayer<Scalar,3> : public PoolingLayerBase<Scalar,3> {
	typedef PoolingLayerBase<Scalar,3> Base;
public:
	PoolingLayer(Dimensions<int,3> input_dims, size_t receptor_size, size_t stride) :
			Base::PoolingLayerBase(input_dims, receptor_size, stride) { };
	virtual ~PoolingLayer() = default;
protected:
	virtual void init_cache() = 0;
	virtual Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) = 0;
	virtual RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) = 0;
	inline typename Base::DataBatch pass_forward(typename Base::DataBatch in, bool training) {
		assert(Utils<Scalar>::template get_dims<4>(in).demote() == Base::input_dims);
		assert(in.dimension(0) > 0);
		Base::rows = in.dimension(0);
		int depth = Base::input_dims(2);
		typename Base::RankwiseArray patch_offsets({ 0, 0, 0, 0 });
		typename Base::RankwiseArray patch_extents({ 1, (int) Base::receptor_size, (int) Base::receptor_size, 1 });
		typename Base::DataBatch out(Base::rows, Base::output_dims(0), Base::output_dims(1), Base::output_dims(2));
		init_cache();
		int patch_ind = 0;
		for (int i = 0; i < Base::rows; i++) {
			patch_offsets[0] = i;
			int out_j = 0;
			for (int j = 0; j <= Base::height_rem; j += Base::stride, out_j++) {
				patch_offsets[1] = j;
				int out_k = 0;
				for (int k = 0; k <= Base::width_rem; k += Base::stride, out_k++) {
					patch_offsets[2] = k;
					for (int l = 0; l < depth; l++) {
						patch_offsets[3] = l;
						typename Base::DataBatch patch = in.slice(patch_offsets, patch_extents);
						out(i,out_j,out_k,l) = reduce(Utils<Scalar>::template map_tensor_to_mat<4>(std::move(patch)), patch_ind++);
					}
				}
			}
		}
		return out;
	};
	inline typename Base::DataBatch pass_back(typename Base::DataBatch out_grads) {
		assert(Utils<Scalar>::template get_dims<4>(out_grads).demote() == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && Base::rows == out_grads.dimension(0));
		if (Layer<Scalar,3>::is_input_layer())
			return Utils<Scalar>::template get_null_tensor<4>();
		int depth = Base::input_dims(2);
		Dimensions<int,3> patch_dims({ (int) Base::receptor_size, (int) Base::receptor_size, 1 });
		typename Base::RankwiseArray patch_offsets({ 0, 0, 0, 0});
		typename Base::RankwiseArray patch_extents({ 1, patch_dims(0), patch_dims(1), patch_dims(2) });
		typename Base::DataBatch prev_out_grads(Base::rows, Base::input_dims(0), Base::input_dims(1), depth);
		prev_out_grads.setZero();
		int patch_ind = 0;
		for (int i = 0; i < Base::rows; i++) {
			patch_offsets[0] = i;
			int out_grads_j = 0;
			for (int j = 0; j <= Base::height_rem; j += Base::stride, out_grads_j++) {
				patch_offsets[1] = j;
				int out_grads_k = 0;
				for (int k = 0; k <= Base::width_rem; k += Base::stride, out_grads_k++) {
					patch_offsets[2] = k;
					for (int l = 0; l < depth; l++) {
						patch_offsets[3] = l;
						prev_out_grads.slice(patch_offsets, patch_extents) +=
								Utils<Scalar>::template map_mat_to_tensor<4>(d_reduce(out_grads(i,out_grads_j,out_grads_k,l),
										patch_ind++), patch_dims);
					}
				}
			}
		}
		return prev_out_grads;
	};
};

template<typename Scalar, size_t Rank>
class SumPoolingLayer : public PoolingLayer<Scalar,Rank> {
public:
	SumPoolingLayer(Dimensions<int,Rank> input_dims, size_t receptor_size = 2, size_t stride = 2) :
			PoolingLayer<Scalar,Rank>::PoolingLayer(input_dims, receptor_size, stride) { };
	Layer<Scalar,Rank>* clone() const {
		return new SumPoolingLayer(*this);
	};
protected:
	void init_cache() { };
	inline Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) {
		return patch.sum();
	};
	inline RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) {
		return RowVector<Scalar>::Constant(PoolingLayerBase<Scalar,Rank>::receptor_area, grad);
	};
	void empty_cache() { };
};

template<typename Scalar, size_t Rank>
class MeanPoolingLayer : public PoolingLayer<Scalar,Rank> {
public:
	MeanPoolingLayer(Dimensions<int,Rank> input_dims, size_t receptor_size = 2, size_t stride = 2) :
			PoolingLayer<Scalar,Rank>::PoolingLayer(input_dims, receptor_size, stride) { };
	Layer<Scalar,Rank>* clone() const {
		return new MeanPoolingLayer(*this);
	};
protected:
	void init_cache() { };
	inline Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) {
		return patch.mean();
	};
	inline RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) {
		return RowVector<Scalar>::Constant(PoolingLayerBase<Scalar,Rank>::receptor_area,
				grad / (Scalar) PoolingLayerBase<Scalar,Rank>::receptor_area);
	};
	void empty_cache() { };
};

template<typename Scalar, size_t Rank>
class MaxPoolingLayer : public PoolingLayer<Scalar,Rank> {
public:
	MaxPoolingLayer(Dimensions<int,Rank> input_dims, unsigned receptor_size = 2, unsigned stride = 2) :
			PoolingLayer<Scalar,Rank>::PoolingLayer(input_dims, receptor_size, stride) { };
	Layer<Scalar,Rank>* clone() const {
		return new MaxPoolingLayer(*this);
	};
protected:
	void init_cache() {
		max_inds = std::vector<unsigned>(PoolingLayerBase<Scalar,Rank>::rows *
				PoolingLayerBase<Scalar,Rank>::output_dims.get_volume());
	};
	inline Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) {
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
	inline RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) {
		RowVector<Scalar> d_patch = RowVector<Scalar>::Zero(PoolingLayerBase<Scalar,Rank>::receptor_area);
		d_patch(max_inds[patch_ind]) = grad;
		return d_patch;
	};
	inline void empty_cache() {
		max_inds = std::vector<unsigned>(0);
	};
private:
	// Cache
	std::vector<unsigned> max_inds;
};

} /* namespace cattle */

#endif /* LAYER_H_ */
