/*
 * Layer.hpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_LAYER_H_
#define CATTL3_LAYER_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>

#include "Dimensions.hpp"
#include "EigenProxy.hpp"
#include "NumericUtils.hpp"
#include "Parameters.hpp"

namespace cattle {

/**
 * An abstract class template representing layers in a neural network.
 */
template<typename Scalar, std::size_t Rank>
class Layer {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal rank");
public:
	// Rank is increased by one to allow for batch training.
	static constexpr std::size_t DATA_RANK = Rank + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
	virtual ~Layer() = default;
	/**
	 * It returns a clone of the layer instance.
	 *
	 * @return A pointer to a copy of the instance. The instance does not take ownership of
	 * the returned pointer (i.e. the caller is responsible for deleting it).
	 */
	virtual Layer<Scalar,Rank>* clone() const = 0;
	/**
	 * It returns a clone of the layer instance using a reference to the original's parameters.
	 * Non-parametric layers do not need to support parameter sharing and thus are just expected
	 * to return a normal clone.
	 *
	 * @return A clone of the original layer instance sharing the same parameters with the
	 * original.
	 */
	virtual Layer<Scalar,Rank>* clone_with_shared_params() = 0;
	/**
	 * It returns a reference to the layer owning the parameters used. If this owner goes out
	 * of scope (in case this one is a clone with shared parameters), the behaviour of the clone
	 * is undefined.
	 *
	 * @return A reference to the layer owning the parameters. If this layer is not using
	 * shared parameters, it returns a reference to itself.
	 */
	virtual const Layer<Scalar,Rank>& get_params_owner() const = 0;
	/**
	 * A simple constant getter method for the input dimensionality of the layer.
	 *
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors accepted by the layer as its input (except for the first rank which denotes
	 * the variable sample size).
	 */
	virtual const Dimensions<std::size_t,Rank>& get_input_dims() const = 0;
	/**
	 * A simple constant getter method for the output dimensionality of the layer.
	 *
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors output by the layer along all ranks except the first one.
	 */
	virtual const Dimensions<std::size_t,Rank>& get_output_dims() const = 0;
	/**
	 * A constant method that returns whether this layer functions as an input layer. An input
	 * layer does not need to propagate the gradients all the way during the backward pass as
	 * it is assumed that no other layer needs them derive the gradient on its parameters. It
	 * is therefore possible for an input layer to simply return a null tensor as the output of
	 * its backward pass.
	 *
	 * @return Whether this layer is the input layer of the neural network that contains it.
	 */
	virtual bool is_input_layer() const = 0;
	/**
	 * Sets this instance's input layer status to the given value.
	 *
	 * @param input_layer Whether this layer is to be an input layer or not.
	 */
	virtual void set_input_layer(bool input_layer) = 0;
	/**
	 * It empties the layer's caches such as those required for the derivation of the function
	 * represented by the layer.
	 */
	virtual void empty_cache() = 0;
	/**
	 * It returns a constant reference to the parameters of the layer.
	 *
	 * @return A constant reference to the parameters of the layer.
	 */
	virtual std::vector<const Parameters<Scalar>*>& get_params() const = 0;
	/**
	 * It returns a reference to the parameters of the layer.
	 *
	 * @return A non-constant reference to the parameters of the layer.
	 */
	virtual std::vector<Parameters<Scalar>*>& get_params() = 0;
	/**
	 * It has the function represented by the layer applied to the input tensor.
	 *
	 * @param in A tensor representing a batch of observations. The observations are of
	 * the rank specified by the layer's template parameter and the input tensors rank is
	 * one greater.
	 * @param training Whether the input is to be processed in training or inference mode.
	 * If the forward pass is performed in inference mode, the backward pass is not
	 * guaranteed to work.
	 * @return The output of the function represented by the layer applied to the input
	 * tensor.
	 */
	virtual Data pass_forward(Data in, bool training) = 0;
	/**
	 * It back-propagates the derivative of the error function w.r.t. the output of the
	 * layer updating the gradient of its learnable parameters along the way if there are
	 * any.
	 *
	 * @param out_grad The derivative of the loss function w.r.t. the output of the
	 * layer
	 * @return The derivative of the loss function w.r.t. the output of the previous layer
	 * or a null tensor if the layer is an input layer.
	 */
	virtual Data pass_back(Data out_grad) = 0;
	/**
	 * It determines whether the layer instance is a clone using the shared parameters of
	 * another instance.
	 *
	 * @return Whether the layer instance is a shared-parameter clone.
	 */
	inline bool is_shared_params_clone() const {
		return this != &get_params_owner();
	}
	/**
	 * A method that returns whether the layer has parameters.
	 *
	 * @return Whether the layer uses parameters.
	 */
	inline bool is_parametric() const {
		return get_params().size() > 0;
	}
};

/**
 * An abstract class template that represents an activation function layer.
 */
template<typename Scalar, std::size_t Rank>
class ActivationLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	virtual ~ActivationLayer() = default;
	virtual Base* clone() const = 0;
	inline Base* clone_with_shared_params() {
		return clone();
	}
	inline const Base& get_params_owner() const {
		return owner;
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return dims;
	}
	inline const Matrix<Scalar>& get_params() const {
		return params_ref;
	}
	inline const Matrix<Scalar>& get_params_grad() const {
		return params_grad;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() {
		params_ref = Matrix<Scalar>(params_rows, params_cols);
		params_grad = Matrix<Scalar>::Zero(params_rows, params_cols);
	}
protected:
	inline ActivationLayer(const Dimensions<std::size_t,Rank>& dims, std::size_t params_rows = 0,
			std::size_t params_cols = 0) :
				dims(dims),
				params_rows(params_rows),
				params_cols(params_cols),
				input_layer(false),
				frozen(false),
				params(),
				params_grad(),
				params_ref(params),
				owner(*this) { }
	inline ActivationLayer(const ActivationLayer<Scalar,Rank>& layer) :
			dims(layer.dims),
			params_rows(layer.params_rows),
			params_cols(layer.params_cols),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			params(layer.params),
			params_grad(layer.params_grad),
			params_ref(layer.is_shared_params_clone() ? layer.params_ref : params),
			owner(layer.is_shared_params_clone() ? layer.owner : *this) { }
	inline ActivationLayer(ActivationLayer<Scalar,Rank>& layer, bool share_params) :
			dims(layer.dims),
			params_rows(layer.params_rows),
			params_cols(layer.params_cols),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			params(share_params ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_params ? layer.params_ref : params),
			owner(share_params ? layer.owner : *this){ }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline Matrix<Scalar>& get_params() {
		return params_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() const {
		return 0;
	}
	inline void enforce_constraints() { }
	inline void empty_cache() { }
	const Dimensions<std::size_t,Rank> dims;
	const std::size_t params_rows;
	const std::size_t params_cols;
	Matrix<Scalar> params_grad;
	Matrix<Scalar>& params_ref;
private:
	bool input_layer;
	bool frozen;
	Matrix<Scalar> params;
	const Base& owner;
};

/**
 * A class template representing an identity activation layer that merely outputs
 * its input.
 *
 * \f$f(x) = x\f$
 */
template<typename Scalar, std::size_t Rank>
class IdentityActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline IdentityActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Root* clone() const {
		return new IdentityActivationLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return in;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		return out_grad;
	}
private:
	std::size_t batch_size;
};

/**
 * A class template that represents a linearly scaling activation layer.
 *
 * \f$f(x) = c x\f$
 */
template<typename Scalar, std::size_t Rank>
class ScaledActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param scale The factor by which the input is to be scaled.
	 */
	inline ScaledActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar scale) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			scale(scale) { }
	inline Root* clone() const {
		return new ScaledActivationLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return in * scale;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		return out_grad * scale;
	}
private:
	const Scalar scale;
	std::size_t batch_size;
};

/**
 * A class template that represents a binary step activation function that outputs either
 * 1 or 0 based on the signum of its input. This function is not differentiable.
 *
 * \f[
 *   f(x) = \begin{cases}
 *     0 & \text{for } x < 0\\
 *     1 & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 */
template<typename Scalar, std::size_t Rank>
class BinaryStepActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline BinaryStepActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Root* clone() const {
		return new BinaryStepActivationLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : .0); });
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		return out_grad.constant(0);
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a sigmoid activation function layer.
 *
 * \f$f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}\f$
 */
template<typename Scalar, std::size_t Rank>
class SigmoidActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SigmoidActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Root* clone() const {
		return new SigmoidActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		out = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		auto out = ((-in).exp() + in.constant(1)).inverse();
		if (training) {
			this->out = out;
			return this->out;
		}
		return out;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && out.dimension(0) == out_grad.dimension(0));
		return (out * (out.constant(1) - out)) * out_grad;
	}
private:
	// Staged computation cache.
	typename Root::Data out;
};

/**
 * A class template representing a hyperbolic tangent activation function layer.
 *
 * \f$f(x) = \text{tanh}(x)\f$
 */
template<typename Scalar, std::size_t Rank>
class TanhActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline TanhActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new TanhActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		out = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		auto out = in.tanh();
		if (training) {
			this->out = out;
			return this->out;
		}
		return out;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && out.dimension(0) == out_grad.dimension(0));
		return (out.constant(1) - out * out) * out_grad;
	}
private:
	typename Root::Data out;
};

/**
 * A class template representing a softsign activation function layer, an alternative to the
 * tanh layer.
 *
 * \f$f(x) = \frac{x}{1 + \left|x\right|}\f$
 */
template<typename Scalar, std::size_t Rank>
class SoftsignActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SoftsignActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SoftsignActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		denominator = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		auto denominator = in.constant(1) + in.abs();
		if (training) {
			this->denominator = denominator;
			return in / this->denominator;
		}
		return in / denominator;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && denominator.dimension(0) == out_grad.dimension(0));
		return denominator.square().inverse() * out_grad;
	}
private:
	// Staged computation cache.
	typename Root::Data denominator;
};

/**
 * A class template representing a softplus activation function layer. The softplus activation function
 * is a differentiable function that approximates the rectified linear unit function.
 *
 * \f$f(x) = \ln(1 + e^x)\f$
 */
template<typename Scalar, std::size_t Rank>
class SoftplusActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SoftplusActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SoftplusActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		in = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		if (training) {
			this->in = std::move(in);
			return (this->in.exp() + this->in.constant(1)).log();
		}
		return (in.exp() + in.constant(1)).log();
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && in.dimension(0) == out_grad.dimension(0));
		return ((-in).exp() + in.constant(1)).inverse() * out_grad;
	}
private:
	// Staged computation cache.
	typename Root::Data in;
};

/**
 * A class template for a softmax activation function layer. Unlike most other activation
 * functions, the softmax layer does not represent a simple coefficient-wise function but
 * a multivariate one. The per-sample sums of the elements of the output tensor of the layer
 * are always 1.
 *
 * \f$f(x_i) = \frac{e^{x_i}}{\epsilon + \sum\limits_{j = 1}^J e^{x_j}}\f$
 */
template<typename Scalar, std::size_t Rank>
class SoftmaxActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param epsilon A small constant to maintain numerical stability.
	 */
	inline SoftmaxActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			epsilon(epsilon),
			conversion_dims(dims.template promote<>()) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SoftmaxActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		out = Matrix<Scalar>(0, 0);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
		/* First subtract the value of the greatest coefficient from each element row-wise
		 * to avoid an overflow due to raising e to great powers. */
		Matrix<Scalar> act = (in_mat.array().colwise() - in_mat.array().rowwise().maxCoeff()).exp();
		act = act.array().colwise() / (act.array().rowwise().sum() + epsilon);
		conversion_dims[0] = rows;
		if (training) {
			out = std::move(act);
			return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
		}
		return TensorMap<Scalar,Root::DATA_RANK>(act.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && out.rows() == out_grad.dimension(0));
		std::size_t rows = out_grad.dimension(0);
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), rows, out_grad.size() / rows);
		Matrix<Scalar> prev_out_grad(rows, out.cols());
		for (int i = 0; i < prev_out_grad.rows(); ++i) {
			RowVector<Scalar> row_i = out.row(i);
			// FIXME Do not evaluate the expressions into a temporary variable.
			Matrix<Scalar> jacobian = row_i.asDiagonal();
			jacobian -= row_i.transpose() * row_i;
			prev_out_grad.row(i) = out_grad_mat.row(i) * jacobian;
		}
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad.data(), conversion_dims);
	}
private:
	const Scalar epsilon;
	RankwiseArray conversion_dims;
	// Staged computation cache matrix.
	Matrix<Scalar> out;
};

/**
 * A class template representing a rectified linear unit (ReLU) activation function. ReLU
 * layers set all negative elements of the input to 0. This function is not differentiable.

 * \f[
 *   f(x) = \begin{cases}
 *     0 & \text{for } x < 0\\
 *     x & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 */
template<typename Scalar, std::size_t Rank>
class ReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline ReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ReLUActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		in = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		if (training)
			this->in = in;
		return in.cwiseMax((Scalar) 0);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && in.dimension(0) == out_grad.dimension(0));
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= 0); }) * out_grad;
	}
private:
	typename Root::Data in;
};

/**
 * A class template representing a leaky rectified linear unit activation function. Unlike
 * traditional ReLU layers leaky ReLU layers do not set negative elements of the input to
 * 0 but scale them by a small constant alpha. This function is not differentiable.
 *
 * \f[
 *   f(x) = \begin{cases}
 *     \alpha x & \text{for } x < 0\\
 *     x & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 *
 * \see https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
 */
template<typename Scalar, std::size_t Rank>
class LeakyReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param alpha The factor by which negative inputs are to be scaled.
	 */
	inline LeakyReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new LeakyReLUActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		in = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		if (training)
			this->in = in;
		return in.cwiseMax(in * alpha);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && in.dimension(0) == out_grad.dimension(0));
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i >= 0 ? 1 : alpha); }) * out_grad;
	}
private:
	const Scalar alpha;
	typename Root::Data in;
};

/**
 * A class template representing an exponential linear unit (ELU) activation function. ELUs
 * apply an exponential (e based) function scaled by alpha to the negative elements of the input.
 * ELU layers are not differentiable.
 *
 * \f[
 *   f(x) = \begin{cases}
 *     \alpha (e^x - 1) & \text{for } x < 0\\
 *     x & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 *
 * \see https://arxiv.org/abs/1511.07289
 */
template<typename Scalar, std::size_t Rank>
class ELUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param alpha The factor by which negative inputs are to be scaled.
	 */
	inline ELUActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha),
			conversion_dims(dims.template promote<>()) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ELUActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		in = Matrix<Scalar>(0, 0);
		out = Matrix<Scalar>(0, 0);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		if (training) {
			this->in = MatrixMap<Scalar>(in.data(), in.dimension(0), Base::dims.get_volume());
			out = this->in.unaryExpr([this](Scalar i) {
				return (Scalar) (i >= 0 ? i : (alpha * (exp(i) - 1)));
			});
			conversion_dims[0] = out.rows();
			return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
		}
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i >= 0 ? i : (alpha * (exp(i) - 1))); });
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && conversion_dims[0] == out_grad.dimension(0));
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), conversion_dims[0], Base::dims.get_volume());
		Matrix<Scalar> prev_out_grad(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); ++i) {
			for (int j = 0; j < in.rows(); ++j)
				prev_out_grad(j,i) = (Scalar) ((in(j,i) >= 0 ? 1 : (out(j,i) + alpha)) * out_grad_mat(j,i));
		}
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad.data(), conversion_dims);
	}
private:
	const Scalar alpha;
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in;
	Matrix<Scalar> out;
};

/**
 * A class template representing a parametric rectified linear unit (PReLU) activation function.
 * PReLU layers are Leaky ReLU activation functions with learnable alphas. PReLU activation
 * functions are not differentiable.
 *
 * \f[
 *   f(x) = \begin{cases}
 *     \alpha x & \text{for } x < 0\\
 *     x & \text{for } x \geq 0
 *   \end{cases}
 * \f]
 *
 * \see https://arxiv.org/abs/1502.01852
 */
template<typename Scalar, std::size_t Rank>
class PReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param param_reg The regularization function to apply to the layer's parameters.
	 * @param init_alpha The initial factor by which negative inputs are to be scaled.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline PReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims, ParamRegSharedPtr<Scalar> param_reg = Root::NO_PARAM_REG,
			Scalar init_alpha = 1e-1, Scalar max_norm_constraint = 0) :
				Base::ActivationLayer(dims, 1, dims.get_volume()),
				param_reg(param_reg),
				init_alpha(init_alpha),
				max_norm_constraint(max_norm_constraint),
				max_norm(NumericUtils<Scalar>::decidedly_greater(max_norm_constraint, (Scalar) 0)),
				conversion_dims(dims.template promote<>()) {
		assert(param_reg != nullptr);
	}
	inline Layer<Scalar,Rank>* clone() const {
		return new PReLUActivationLayer(*this);
	}
	inline Root* clone_with_shared_params() {
		return new PReLUActivationLayer(*this, true);
	}
	inline void init() {
		Base::init();
		Base::params_ref.setConstant(init_alpha);
	}
protected:
	inline PReLUActivationLayer(PReLUActivationLayer<Scalar,Rank>& layer, bool share_params) :
			Base::ActivationLayer(layer, share_params),
			param_reg(layer.param_reg),
			init_alpha(layer.init_alpha),
			max_norm_constraint(layer.max_norm_constraint),
			max_norm(layer.max_norm),
			conversion_dims(layer.conversion_dims) { }
	inline void empty_cache() {
		in = Matrix<Scalar>(0, 0);
	}
	inline void regularize() {
		Base::params_grad += param_reg->d_function(Base::params_ref);
	}
	inline Scalar get_regularization_penalty() const {
		return param_reg->function(Base::params_ref);
	}
	inline void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = Base::params_ref.squaredNorm();
			if (l2_norm > max_norm_constraint)
				Base::params_ref *= (max_norm_constraint / l2_norm);
		}
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		this->in = MatrixMap<Scalar>(in.data(), rows, Base::dims.get_volume());
		Matrix<Scalar> out = this->in.cwiseMax(this->in * Base::params_ref.row(0).asDiagonal());
		conversion_dims[0] = rows;
		return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && conversion_dims[0] == out_grad.dimension(0));
		Base::params_grad.row(0).setZero();
		MatrixMap<Scalar> out_grad_map(out_grad.data(), conversion_dims[0], Base::dims.get_volume());
		Matrix<Scalar> prev_out_grad = Matrix<Scalar>(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); ++i) {
			for (int j = 0; j < in.rows(); ++j) {
				Scalar in_ji = in(j,i);
				if (in_ji >= 0)
					prev_out_grad(j,i) = out_grad_map(j,i);
				else {
					Scalar out_ji = out_grad_map(j,i);
					prev_out_grad(j,i) = Base::params_ref(0,i) * out_ji;
					Base::params_grad(0,i) += in_ji * out_ji;
				}
			}
		}
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad.data(), conversion_dims);
	}
private:
	const ParamRegSharedPtr<Scalar> param_reg;
	const Scalar init_alpha;
	const Scalar max_norm_constraint;
	const bool max_norm;
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in;
};

/**
 * A class template representing the Swish activation function.
 *
 * \f$f(x) = x \sigma(\beta x)\f$
 *
 * \see https://arxiv.org/abs/1710.05941
 */
template<typename Scalar, std::size_t Rank>
class SwishActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param beta The factor by which the input of the sigmoid factor of the Swish
	 * function is to be scaled.
	 */
	inline SwishActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar beta = 1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			beta(beta) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SwishActivationLayer(*this);
	}
protected:
	inline void empty_cache() {
		in = typename Root::Data();
		sig_out = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		auto sig_denom = (-beta * in).exp() + in.constant(1);
		if (training) {
			sig_out = sig_denom.inverse();
			this->in = std::move(in);
			return this->in * sig_out;
		}
		return in / sig_denom;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && sig_out.dimension(0) == out_grad.dimension(0));
		return sig_out * ((sig_out.constant(1) - sig_out) * beta * in + sig_out.constant(1)) * out_grad;
	}
private:
	const Scalar beta;
	// Staged computation cache.
	typename Root::Data in;
	typename Root::Data sig_out;
};

/**
 * A class template representing the parametric Swish activation function with learnable beta
 * values.
 *
 * \f$f(x) = x \sigma(\beta x)\f$
 *
 * \see https://arxiv.org/abs/1710.05941
 */
template<typename Scalar, std::size_t Rank>
class PSwishActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param param_reg The regularization function to apply to the layer's parameters.
	 * @param init_beta The initial factor by which the input of the sigmoid factor of the
	 * Swish function is to be scaled.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline PSwishActivationLayer(const Dimensions<std::size_t,Rank>& dims, ParamRegSharedPtr<Scalar> param_reg = Root::NO_PARAM_REG,
			Scalar init_beta = 1e-1, Scalar max_norm_constraint = 0) :
				Base::ActivationLayer(dims, 1, dims.get_volume()),
				param_reg(param_reg),
				init_beta(init_beta),
				max_norm_constraint(max_norm_constraint),
				max_norm(NumericUtils<Scalar>::decidedly_greater(max_norm_constraint, (Scalar) 0)),
				conversion_dims(dims.template promote<>()) {
		assert(param_reg != nullptr);
	}
	inline Root* clone() const {
		return new PSwishActivationLayer(*this);
	}
	inline Root* clone_with_shared_params() {
		return new PSwishActivationLayer(*this, true);
	}
	inline void init() {
		Base::init();
		Base::params_ref.setConstant(init_beta);
	}
protected:
	inline PSwishActivationLayer(PSwishActivationLayer<Scalar,Rank>& layer, bool share_params) :
			Base::ActivationLayer(layer, share_params),
			param_reg(layer.param_reg),
			init_beta(layer.init_beta),
			max_norm_constraint(layer.max_norm_constraint),
			max_norm(layer.max_norm),
			conversion_dims(layer.conversion_dims) { }
	inline void empty_cache() {
		in = Matrix<Scalar>(0, 0);
		sig_out = Matrix<Scalar>(0, 0);
	}
	inline void regularize() {
		Base::params_grad += param_reg->d_function(Base::params_ref);
	}
	inline Scalar get_regularization_penalty() const {
		return param_reg->function(Base::params_ref);
	}
	inline void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = Base::params_ref.squaredNorm();
			if (l2_norm > max_norm_constraint)
				Base::params_ref *= (max_norm_constraint / l2_norm);
		}
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		conversion_dims[0] = in.dimension(0);
		this->in = MatrixMap<Scalar>(in.data(), conversion_dims[0], in.size() / conversion_dims[0]);
		auto sig_out = ((this->in * (Base::params_ref.row(0).asDiagonal() * -1)).array().exp() + 1);
		Matrix<Scalar> out;
		if (training) {
			this->sig_out = sig_out.inverse();
			out = this->in.cwiseProduct(this->sig_out);
		} else
			out = this->in.array() / sig_out;
		return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == Base::dims);
		assert(out_grad.dimension(0) > 0 && conversion_dims[0] == out_grad.dimension(0));
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), conversion_dims[0], out_grad.size() / conversion_dims[0]);
		Matrix<Scalar> one_min_sig_out = 1 - sig_out.array();
		Base::params_grad = sig_out.cwiseProduct(one_min_sig_out).cwiseProduct(in).cwiseProduct(in)
				.cwiseProduct(out_grad_mat).colwise().sum();
		Matrix<Scalar> prev_out_grad = sig_out.cwiseProduct(((one_min_sig_out * Base::params_ref.row(0).asDiagonal()).array() *
				in.array() + 1).matrix()).cwiseProduct(out_grad_mat);
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grad.data(), conversion_dims);
	}
private:
	const ParamRegSharedPtr<Scalar> param_reg;
	const Scalar init_beta;
	const Scalar max_norm_constraint;
	const bool max_norm;
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in;
	Matrix<Scalar> sig_out;
};

/**
 * An abstract base class template representing a pooling layer.
 */
template<typename Scalar, std::size_t Rank>
class PoolLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	virtual Base* clone() const = 0;
	inline Base* clone_with_shared_params() {
		return clone();
	}
	inline const Base& get_params_owner() const {
		return *this;
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return output_dims;
	}
	inline const Matrix<Scalar>& get_params() const {
		return params;
	}
	inline const Matrix<Scalar>& get_params_grad() const {
		return params_grad;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
protected:
	typedef std::array<std::size_t,4> Array4;
	typedef std::array<std::size_t,2> ReductionRanksArray2D;
	inline PoolLayer(const Dimensions<std::size_t,Rank>& input_dims, std::size_t receptor_height, std::size_t receptor_width,
			std::size_t vertical_stride, std::size_t horizontal_stride) :
				ext_input_dims(input_dims.template extend<3 - Rank>()),
				ext_output_dims(calculate_output_dims(ext_input_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride)),
				input_dims(input_dims),
				output_dims(ext_output_dims.template contract<3 - Rank>()),
				receptor_height(receptor_height),
				receptor_width(receptor_width),
				vertical_stride(vertical_stride),
				horizontal_stride(horizontal_stride),
				height_rem(ext_input_dims(0) - receptor_height),
				width_rem(ext_input_dims(1) - receptor_width),
				input_layer(false),
				frozen(false),
				reduction_ranks({ 1u, 2u }),
				broadcast({ 1u, receptor_height, receptor_width, 1u }),
				patch_offsets({ 0u, 0u, 0u, 0u }),
				patch_extents({ 0u, receptor_height, receptor_width, ext_input_dims(2) }),
				reduced_patch_offsets({ 0u, 0u, 0u, 0u }),
				reduced_patch_extents({ 0u, 1u, 1u, ext_input_dims(2) }),
				params(),
				params_grad() {
		assert(receptor_height > 0 && receptor_width > 0);
		assert(vertical_stride > 0 && horizontal_stride > 0);
		assert(ext_input_dims(0) >= receptor_height && ext_input_dims(1) >= receptor_width);
	}
	/**
	 * Initializes the cache required for back-propagation.
	 */
	virtual void _init_cache() = 0;
	/**
	 * Reduces the input tensor patch along the specified ranks.
	 *
	 * @param patch A tensor representing a spatial patch of the input tensor.
	 * @param patch_ind The index of the patch.
	 * @return The reduced tensor.
	 */
	virtual Tensor<Scalar,4> _reduce(const Tensor<Scalar,4>& patch, std::size_t patch_ind) = 0;
	/**
	 * Differentiates the reduction function and returns the derivative of the loss function
	 * w.r.t. the non-reduced patch.
	 *
	 * @param grad The derivative of the loss function w.r.t. the reduced patch.
	 * @param patch_ind The index of the patch.
	 * @return The derivative of the loss function w.r.t. the non-reduced patch.
	 */
	virtual Tensor<Scalar,4> _d_reduce(const Tensor<Scalar,4>& grad, std::size_t patch_ind) = 0;
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() const {
		return 0;
	}
	inline void enforce_constraints() { }
	inline Tensor<Scalar,4> _pass_forward(Tensor<Scalar,4> in, bool training) {
		std::size_t rows = in.dimension(0);
		patch_extents[0] = rows;
		reduced_patch_extents[0] = rows;
		Tensor<Scalar,4> out(rows, ext_output_dims(0), ext_output_dims(1), ext_output_dims(2));
		_init_cache();
		std::size_t patch_ind = 0;
		std::size_t out_i = 0;
		for (std::size_t i = 0; i <= width_rem; i += horizontal_stride, ++out_i) {
			patch_offsets[2] = i;
			reduced_patch_offsets[2] = out_i;
			std::size_t out_j = 0;
			for (std::size_t j = 0; j <= height_rem; j += vertical_stride, ++out_j) {
				patch_offsets[1] = j;
				reduced_patch_offsets[1] = out_j;
				Tensor<Scalar,4> patch = in.slice(patch_offsets, patch_extents);
				out.slice(reduced_patch_offsets, reduced_patch_extents) = _reduce(patch, patch_ind++);
			}
		}
		return out;
	}
	inline Tensor<Scalar,4> _pass_back(Tensor<Scalar,4> out_grad) {
		if (input_layer)
			return Tensor<Scalar,4>();
		Tensor<Scalar,4> prev_out_grad(patch_extents[0], ext_input_dims(0), ext_input_dims(1),  ext_input_dims(2));
		prev_out_grad.setZero();
		std::size_t patch_ind = 0;
		std::size_t out_grad_i = 0;
		for (std::size_t i = 0; i <= width_rem; i += horizontal_stride, ++out_grad_i) {
			patch_offsets[2] = i;
			reduced_patch_offsets[2] = out_grad_i;
			std::size_t out_grad_j = 0;
			for (std::size_t j = 0; j <= height_rem; j += vertical_stride, ++out_grad_j) {
				patch_offsets[1] = j;
				reduced_patch_offsets[1] = out_grad_j;
				Tensor<Scalar,4> reduced_patch_grad = out_grad.slice(reduced_patch_offsets, reduced_patch_extents);
				// Accumulate the gradients where the patches overlap.
				prev_out_grad.slice(patch_offsets, patch_extents) += _d_reduce(reduced_patch_grad, patch_ind++);
			}
		}
		return prev_out_grad;
	}
	const Dimensions<std::size_t,3> ext_input_dims;
	const Dimensions<std::size_t,3> ext_output_dims;
	const Dimensions<std::size_t,Rank> input_dims;
	const Dimensions<std::size_t,Rank> output_dims;
	const std::size_t receptor_height;
	const std::size_t receptor_width;
	const std::size_t vertical_stride;
	const std::size_t horizontal_stride;
	const std::size_t height_rem;
	const std::size_t width_rem;
	// Arrays for tensor manipulation.
	ReductionRanksArray2D reduction_ranks;
	Array4 broadcast;
	Array4 patch_offsets;
	Array4 patch_extents;
	Array4 reduced_patch_offsets;
	Array4 reduced_patch_extents;
	Array4 dil_strides;
private:
	inline static std::size_t calculate_spatial_output_dim(std::size_t input_dim, std::size_t receptor_size, std::size_t stride) {
		return (input_dim - receptor_size) / stride + 1;
	}
	inline static Dimensions<std::size_t,3> calculate_output_dims(const Dimensions<std::size_t,3>& input_dims,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride) {
		return { calculate_spatial_output_dim(input_dims(0), receptor_height, vertical_stride),
				calculate_spatial_output_dim(input_dims(1), receptor_width, horizontal_stride),
				input_dims(2) };
	}
	bool input_layer;
	bool frozen;
	// No actual parameters.
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
};

namespace {

/**
 * An abstract class template representing a pooling layer that reduces patches of the input by taking their
 * means.
 */
template<typename Scalar, std::size_t Rank>
class MeanPoolLayerBase : public PoolLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef PoolLayer<Scalar,Rank> Base;
public:
	inline MeanPoolLayerBase(const Dimensions<std::size_t,Rank>& input_dims, std::size_t receptor_height,
			std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride) :
				Base::PoolLayer(input_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride),
				receptor_area(receptor_height * receptor_width) { }
protected:
	inline void empty_cache() { }
	inline void _init_cache() { }
	inline Tensor<Scalar,4> _reduce(const Tensor<Scalar,4>& patch, std::size_t patch_ind) {
		Tensor<Scalar,2> reduced_patch = patch.mean(Base::reduction_ranks);
		return TensorMap<Scalar,4>(reduced_patch.data(), Base::reduced_patch_extents);
	}
	inline Tensor<Scalar,4> _d_reduce(const Tensor<Scalar,4>& grad, std::size_t patch_ind) {
		return (grad / (Scalar) receptor_area).broadcast(Base::broadcast);
	}
private:
	std::size_t receptor_area;
};

}

/**
 * A class template representing a 2D mean pooling layer operating on rank-3 data.
 */
template<typename Scalar, std::size_t Rank = 3>
class MeanPoolLayer : public MeanPoolLayerBase<Scalar,Rank> {
	typedef Layer<Scalar,3> Root;
	typedef PoolLayer<Scalar,3> PoolBase;
	typedef MeanPoolLayerBase<Scalar,3> MeanPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 */
	inline MeanPoolLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2) :
				MeanPoolBase::MeanPoolLayerBase(input_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride) { }
	inline Root* clone() const {
		return new MeanPoolLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(std::move(in), training);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,4>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		return PoolBase::_pass_back(std::move(out_grad));
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 2D mean pooling layer operating on rank-2 data.
 */
template<typename Scalar>
class MeanPoolLayer<Scalar,2> : public MeanPoolLayerBase<Scalar,2> {
	typedef Layer<Scalar,2> Root;
	typedef PoolLayer<Scalar,2> PoolBase;
	typedef MeanPoolLayerBase<Scalar,2> MeanPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 */
	inline MeanPoolLayer(const Dimensions<std::size_t,2>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2) :
				MeanPoolBase::MeanPoolLayerBase(input_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride) { }
	inline Root* clone() const {
		return new MeanPoolLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,3>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), in.dimension(2), 1u }),
				training).reshape(std::array<std::size_t,3>({ batch_size, PoolBase::output_dims(0), PoolBase::output_dims(1) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,3>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		Tensor<Scalar,4> prev_out_grad = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grad.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,3>();
		return TensorMap<Scalar,3>(prev_out_grad.data(), { batch_size, PoolBase::input_dims(0), PoolBase::input_dims(1) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 1D mean pooling layer.
 */
template<typename Scalar>
class MeanPoolLayer<Scalar,1> : public MeanPoolLayerBase<Scalar,1> {
	typedef Layer<Scalar,1> Root;
	typedef PoolLayer<Scalar,1> PoolBase;
	typedef MeanPoolLayerBase<Scalar,1> MeanPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_length The length of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the length of the input tensor).
	 */
	inline MeanPoolLayer(const Dimensions<std::size_t,1>& input_dims, std::size_t receptor_length = 2,
			std::size_t stride = 2) :
				MeanPoolBase::MeanPoolLayerBase(input_dims, receptor_length, 1, stride, 1) { }
	inline Root* clone() const {
		return new MeanPoolLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,2>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), 1u, 1u }), training)
				.reshape(std::array<std::size_t,2>({ batch_size, PoolBase::output_dims(0) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,2>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		Tensor<Scalar,4> prev_out_grad = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grad.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,2>();
		return TensorMap<Scalar,2>(prev_out_grad.data(), { batch_size, PoolBase::input_dims(0) });
	}
private:
	std::size_t batch_size;
};

namespace {

/**
 * An abstract class template representing a pooling layer that reduces patches of the input by taking their
 * maxima.
 */
template<typename Scalar, std::size_t Rank>
class MaxPoolLayerBase : public PoolLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef PoolLayer<Scalar,Rank> Base;
public:
	inline MaxPoolLayerBase(const Dimensions<std::size_t,Rank>& input_dims, std::size_t receptor_height,
			std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride) :
				Base::PoolLayer(input_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride) { }
protected:
	inline void empty_cache() {
		max_inds = std::vector<std::vector<unsigned>>(0);
	}
	inline void _init_cache() {
		max_inds = std::vector<std::vector<unsigned>>(Base::ext_output_dims(0) * Base::ext_output_dims(1));
	}
	inline Tensor<Scalar,4> _reduce(const Tensor<Scalar,4>& patch, std::size_t patch_ind) {
		std::size_t rows = patch.dimension(0);
		std::size_t depth = patch.dimension(3);
		std::vector<unsigned> inds(rows * depth);
		Tensor<Scalar,4> reduced_patch(rows, 1u, 1u, depth);
		for (std::size_t i = 0; i < depth; ++i) {
			for (std::size_t j = 0; j < rows; ++j) {
				Scalar max = NumericUtils<Scalar>::MIN;
				unsigned max_height = 0;
				unsigned max_width = 0;
				for (std::size_t k = 0; k < Base::receptor_width; ++k) {
					for (std::size_t l = 0; l < Base::receptor_height; ++l) {
						Scalar val = patch(j,l,k,i);
						if (val > max) {
							max = val;
							max_height = l;
							max_width = k;
						}
					}
				}
				inds[i * rows + j] = max_width * Base::receptor_height + max_height;
				reduced_patch(j,0u,0u,i) = max;
			}
		}
		max_inds[patch_ind] = inds;
		return reduced_patch;
	}
	inline Tensor<Scalar,4> _d_reduce(const Tensor<Scalar,4>& grad, std::size_t patch_ind) {
		std::size_t rows = grad.dimension(0);
		std::size_t depth = grad.dimension(3);
		Tensor<Scalar,4> patch(rows, Base::receptor_height, Base::receptor_width, depth);
		patch.setZero();
		std::vector<unsigned>& inds = max_inds[patch_ind];
		for (std::size_t i = 0; i < depth; ++i) {
			for (std::size_t j = 0; j < rows; ++j) {
				unsigned max_ind = inds[i * rows + j];
				unsigned max_height = max_ind % Base::receptor_height;
				unsigned max_width = max_ind / Base::receptor_height;
				patch(j,max_height,max_width,i) = grad(j,0u,0u,i);
			}
		}
		return patch;
	}
private:
	// Cache
	std::vector<std::vector<unsigned>> max_inds;
};

}

/**
 * A class template representing a 2D max pooling layer operating on rank-3 data.
 *
 * \see http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf
 */
template<typename Scalar, std::size_t Rank = 3>
class MaxPoolLayer : public MaxPoolLayerBase<Scalar,Rank> {
	typedef Layer<Scalar,3> Root;
	typedef PoolLayer<Scalar,3> PoolBase;
	typedef MaxPoolLayerBase<Scalar,3> MaxPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 */
	inline MaxPoolLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2) :
				MaxPoolBase::MaxPoolLayerBase(input_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride) { }
	inline Root* clone() const {
		return new MaxPoolLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(std::move(in), training);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,4>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		return PoolBase::_pass_back(std::move(out_grad));
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 2D max pooling layer operating on rank-2 data.
 *
 * \see http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf
 */
template<typename Scalar>
class MaxPoolLayer<Scalar,2> : public MaxPoolLayerBase<Scalar,2> {
	typedef Layer<Scalar,2> Root;
	typedef PoolLayer<Scalar,2> PoolBase;
	typedef MaxPoolLayerBase<Scalar,2> MaxPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 */
	inline MaxPoolLayer(const Dimensions<std::size_t,2>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2) :
				MaxPoolBase::MaxPoolLayerBase(input_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride) { }
	inline Root* clone() const {
		return new MaxPoolLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,3>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), in.dimension(2), 1u }), training)
				.reshape(std::array<std::size_t,3>({ batch_size, PoolBase::output_dims(0), PoolBase::output_dims(1) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,3>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		Tensor<Scalar,4> prev_out_grad = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grad.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,3>();
		return TensorMap<Scalar,3>(prev_out_grad.data(), { batch_size, PoolBase::input_dims(0), PoolBase::input_dims(1) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 1D max pooling layer.
 *
 * \see http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf
 */
template<typename Scalar>
class MaxPoolLayer<Scalar,1> : public MaxPoolLayerBase<Scalar,1> {
	typedef Layer<Scalar,1> Root;
	typedef PoolLayer<Scalar,1> PoolBase;
	typedef MaxPoolLayerBase<Scalar,1> MaxPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_length The length of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the length of the input tensor).
	 */
	inline MaxPoolLayer(const Dimensions<std::size_t,1>& input_dims, std::size_t receptor_length = 2,
			std::size_t stride = 2) :
				MaxPoolBase::MaxPoolLayerBase(input_dims, receptor_length, 1, stride, 1) { }
	inline Root* clone() const {
		return new MaxPoolLayer(*this);
	}
protected:
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,2>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), 1u, 1u }), training)
				.reshape(std::array<std::size_t,2>({ batch_size, PoolBase::output_dims(0) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grad) {
		assert((Dimensions<std::size_t,2>(out_grad.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grad.dimension(0) > 0 && batch_size == out_grad.dimension(0));
		Tensor<Scalar,4> prev_out_grad = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grad.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,2>();
		return TensorMap<Scalar,2>(prev_out_grad.data(), { batch_size, PoolBase::input_dims(0) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a broadcasting layer that repeats the contents of its input tensors
 * along its ranks.
 */
template<typename Scalar, std::size_t Rank>
class BroadcastLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Rank + 1> RankwiseArray;
public:
	/**
	 * @param input_dims The nominal input dimensions of the layer.
	 * @param broadcast The number of times the input tensor's contents are
	 * repeated along each rank. All elements should be greater than 0.
	 */
	inline BroadcastLayer(const Dimensions<std::size_t,Rank>& input_dims,
			const Dimensions<std::size_t,Rank>& broadcast) :
				input_dims(input_dims),
				output_dims(input_dims * broadcast),
				input_layer(false),
				frozen(false),
				broadcast(broadcast.template promote<>()),
				params(),
				params_grad() {
		slice_offsets.fill(0);
		for (std::size_t i = 0; i < Rank; ++i)
			assert(broadcast(i) > 0);
	}
	inline Base* clone() const {
		return new BroadcastLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return clone();
	}
	inline const Base& get_params_owner() const {
		return *this;
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return output_dims;
	}
	inline const Matrix<Scalar>& get_params() const {
		return params;
	}
	inline const Matrix<Scalar>& get_params_grad() const {
		return params_grad;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
protected:
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void empty_cache() { }
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() const {
		return 0;
	}
	inline void enforce_constraints() { }
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == input_dims);
		assert(in.dimension(0) > 0);
		rows = in.dimension(0);
		return in.broadcast(broadcast);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == output_dims);
		assert(out_grad.dimension(0) > 0 && rows == out_grad.dimension(0));
		typename Base::Data prev_out_grad = std::move(out_grad);
		slice_offsets.fill(0);
		slice_extents = output_dims.template promote<>();
		slice_extents[0] = rows;
		for (std::size_t i = 0; i < Rank; ++i) {
			if (broadcast[i + 1] <= 1)
				continue;
			slice_extents[i + 1] = input_dims(i);
			typename Base::Data work_tensor(slice_extents);
			work_tensor.setZero();
			for (std::size_t j = 0; j < broadcast[i + 1]; ++j) {
				work_tensor += prev_out_grad.slice(slice_offsets, slice_extents);
				slice_offsets[i + 1] += input_dims(i);
			}
			slice_offsets[i + 1] = 0;
			prev_out_grad = std::move(work_tensor);
		}
		return prev_out_grad;
	}
private:
	const Dimensions<std::size_t,Rank> input_dims;
	const Dimensions<std::size_t,Rank> output_dims;
	RankwiseArray broadcast;
	RankwiseArray slice_offsets;
	RankwiseArray slice_extents;
	std::size_t rows;
	bool input_layer;
	bool frozen;
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
};

/**
 * A class template for a per-channel batch normalization layer.
 *
 * \see https://arxiv.org/abs/1502.03167
 */
template<typename Scalar, std::size_t Rank, bool PerLastRank = (Rank == 3)>
class BatchNormLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef BatchNormLayer<Scalar,Rank,PerLastRank> Self;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param gamma_reg The regularization function to apply to the layer's gamma parameters.
	 * @param beta_reg The regularization function to apply to the layer's beta parameters.
	 * @param gamma_max_norm_constraint An optional max-norm constraint to enforce on the
	 * gamma parameters. If it is 0 or less, no constraint is applied.
	 * @param beta_max_norm_constraint An optional max-norm constraint to enforce on the
	 * beta parameters. If it is 0 or less, no constraint is applied.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(const Dimensions<std::size_t,Rank>& dims, ParamRegSharedPtr<Scalar> gamma_reg = Base::NO_PARAM_REG,
			ParamRegSharedPtr<Scalar> beta_reg = Base::NO_PARAM_REG, Scalar gamma_max_norm_constraint = 0,
			Scalar beta_max_norm_constraint = 0, Scalar norm_avg_decay = .1, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				dims(dims),
				gamma_reg(gamma_reg),
				beta_reg(beta_reg),
				gamma_max_norm_constraint(gamma_max_norm_constraint),
				beta_max_norm_constraint(beta_max_norm_constraint),
				gamma_max_norm(NumericUtils<Scalar>::decidedly_greater(gamma_max_norm_constraint, (Scalar) 0)),
				beta_max_norm(NumericUtils<Scalar>::decidedly_greater(beta_max_norm_constraint, (Scalar) 0)),
				norm_avg_decay(norm_avg_decay),
				epsilon(epsilon),
				channels(dims(Rank - 1)),
				input_layer(false),
				frozen(false),
				offsets(),
				extents(dims.template promote<>()),
				avg_means(),
				avg_inv_sds(),
				avgs_init(false),
				params(),
				params_grad(),
				params_ref(params),
				owner(*this),
				cache_vec(channels) {
		assert(gamma_reg != nullptr);
		assert(beta_reg != nullptr);
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
		offsets.fill(0);
		extents[Rank] = 1;
	}
	inline BatchNormLayer(const Self& layer) :
			dims(layer.dims),
			gamma_reg(layer.gamma_reg),
			beta_reg(layer.beta_reg),
			gamma_max_norm_constraint(layer.gamma_max_norm_constraint),
			beta_max_norm_constraint(layer.beta_max_norm_constraint),
			gamma_max_norm(layer.gamma_max_norm),
			beta_max_norm(layer.beta_max_norm),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			channels(layer.channels),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			offsets(layer.offsets),
			extents(layer.extents),
			avg_means(layer.avg_means),
			avg_inv_sds(layer.avg_inv_sds),
			avgs_init(layer.avgs_init),
			params(layer.params),
			params_grad(layer.params_grad),
			params_ref(layer.is_shared_params_clone() ? layer.params_ref : params),
			owner(layer.is_shared_params_clone() ? layer.owner : *this),
			cache_vec(layer.cache_vec) { }
	inline Base* clone() const {
		return new BatchNormLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return new BatchNormLayer(*this, true);
	}
	inline const Base& get_params_owner() const {
		return owner;
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return dims;
	}
	inline const Matrix<Scalar>& get_params() const {
		return params_ref;
	}
	inline const Matrix<Scalar>& get_params_grad() const {
		return params_grad;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() {
		params_ref = Matrix<Scalar>(channels, 2);
		// Gamma.
		params_ref.col(0).setOnes();
		// Beta.
		params_ref.col(1).setZero();
		params_grad = Matrix<Scalar>::Zero(channels, 2);
		avg_means = Matrix<Scalar>(1, channels);
		avg_inv_sds = Matrix<Scalar>(1, channels);
		avgs_init = false;
	}
protected:
	inline BatchNormLayer(Self& layer, bool share_params) :
			dims(layer.dims),
			gamma_reg(layer.gamma_reg),
			beta_reg(layer.beta_reg),
			gamma_max_norm_constraint(layer.gamma_max_norm_constraint),
			beta_max_norm_constraint(layer.beta_max_norm_constraint),
			gamma_max_norm(layer.gamma_max_norm),
			beta_max_norm(layer.beta_max_norm),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			channels(layer.channels),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			offsets(layer.offsets),
			extents(layer.extents),
			avg_means(layer.avg_means),
			avg_inv_sds(layer.avg_inv_sds),
			avgs_init(layer.avgs_init),
			params(share_params ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_params ? layer.params_ref : params),
			owner(share_params ? layer.owner : *this),
			cache_vec(layer.cache_vec) { }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void empty_cache() {
		for (unsigned i = 0; i < cache_vec.size(); ++i)
			cache_vec[i].std_in = Matrix<Scalar>(0, 0);
	}
	inline Matrix<Scalar>& get_params() {
		return params_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() {
		params_grad.col(0) += gamma_reg->d_function(params_ref.col(0));
		params_grad.col(1) += beta_reg->d_function(params_ref.col(1));
	}
	inline Scalar get_regularization_penalty() const {
		return gamma_reg->function(params_ref.col(0)) + beta_reg->function(params_ref.col(1));
	}
	inline void enforce_constraints() {
		Scalar l2_norm;
		if (gamma_max_norm) {
			l2_norm = params_ref.col(0).squaredNorm();
			if (l2_norm > gamma_max_norm_constraint)
				params_ref.col(0) *= (gamma_max_norm_constraint / l2_norm);
		}
		if (beta_max_norm) {
			l2_norm = params_ref.col(1).squaredNorm();
			if (l2_norm > beta_max_norm_constraint)
				params_ref.col(1) *= (beta_max_norm_constraint / l2_norm);
		}
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		extents[0] = rows;
		typename Base::Data out;
		if (channels == 1) {
			MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
			Matrix<Scalar> out_mat = _pass_forward(in_mat, 0, training);
			out = TensorMap<Scalar,Base::DATA_RANK>(out_mat.data(), extents);
		} else {
			out = typename Base::Data(in.dimensions());
			for (std::size_t i = 0; i < channels; ++i) {
				offsets[Rank] = i;
				typename Base::Data in_slice = in.slice(offsets, extents);
				MatrixMap<Scalar> in_slice_mat(in_slice.data(), rows, in_slice.size() / rows);
				Matrix<Scalar> out_slice_mat = _pass_forward(in_slice_mat, i, training);
				out.slice(offsets, extents) = TensorMap<Scalar,Base::DATA_RANK>(out_slice_mat.data(), extents);
			}
		}
		return out;
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == dims);
		assert(out_grad.dimension(0) > 0 && extents[0] == out_grad.dimension(0));
		std::size_t rows = out_grad.dimension(0);
		typename Base::Data prev_out_grad;
		if (channels == 1) {
			MatrixMap<Scalar> out_grad_mat(out_grad.data(), rows, out_grad.size() / rows);
			if (input_layer) {
				_pass_back(out_grad_mat, 0);
				return typename Base::Data();
			} else {
				Matrix<Scalar> prev_out_grad_mat = _pass_back(out_grad_mat, 0);
				prev_out_grad = TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad_mat.data(), extents);
			}
		} else {
			prev_out_grad = input_layer ? typename Base::Data() : typename Base::Data(out_grad.dimensions());
			for (std::size_t i = 0; i < channels; ++i) {
				offsets[Rank] = i;
				typename Base::Data out_grad_slice = out_grad.slice(offsets, extents);
				MatrixMap<Scalar> out_grad_slice_mat(out_grad_slice.data(), rows, out_grad_slice.size() / rows);
				if (input_layer) {
					_pass_back(out_grad_slice_mat, i);
					continue;
				} else {
					Matrix<Scalar> prev_out_grad_slice_mat = _pass_back(out_grad_slice_mat, i);
					prev_out_grad.slice(offsets, extents) =
							TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad_slice_mat.data(), extents);
				}
			}
		}
		return prev_out_grad;
	}
private:
	inline Matrix<Scalar> _pass_forward(MatrixMap<Scalar>& in, std::size_t i, bool training) {
		Matrix<Scalar> out;
		if (training) {
			Cache& cache = cache_vec[i];
			Scalar mean = in.mean();
			Matrix<Scalar> norm_in = in.array() - mean;
			cache.inv_in_sd = 1 / sqrt(norm_in.array().square().mean() + epsilon);
			cache.std_in = norm_in * cache.inv_in_sd;
			out = cache.std_in;
			if (avgs_init) {
				avg_means(i) = (1.0 - norm_avg_decay) * avg_means(i) + norm_avg_decay * mean;
				avg_inv_sds(i) = (1.0 - norm_avg_decay) * avg_inv_sds(i) + norm_avg_decay * cache.inv_in_sd;
			} else {
				avg_means(i) = mean;
				avg_inv_sds(i) = cache.inv_in_sd;
				avgs_init = true;
			}
		} else {
			assert(avgs_init);
			out = (in.array() - avg_means(i)) * avg_inv_sds(i);
		}
		return (out * params_ref(i, 0)).array() + params_ref(i, 1);
	}
	inline Matrix<Scalar> _pass_back(MatrixMap<Scalar>& out_grad, std::size_t i) {
		Cache& cache = cache_vec[i];
		params_grad(i, 0) = out_grad.cwiseProduct(cache.std_in).sum();
		params_grad(i, 1) = out_grad.sum();
		if (input_layer)
			return Matrix<Scalar>();
		std::size_t locations = out_grad.size();
		Matrix<Scalar> std_in_grad = out_grad * params_ref(i, 0);
		return (((locations * std_in_grad).array() - std_in_grad.sum()).matrix() -
				cache.std_in * cache.std_in.cwiseProduct(std_in_grad).sum()) *
				(((Scalar) 1 / locations) * cache.inv_in_sd);
	}
	const Dimensions<std::size_t,Rank> dims;
	const ParamRegSharedPtr<Scalar> gamma_reg;
	const ParamRegSharedPtr<Scalar> beta_reg;
	const Scalar gamma_max_norm_constraint;
	const Scalar beta_max_norm_constraint;
	const bool gamma_max_norm;
	const bool beta_max_norm;
	const Scalar norm_avg_decay;
	const Scalar epsilon;
	const std::size_t channels;
	bool input_layer;
	bool frozen;
	std::array<std::size_t,Rank + 1> offsets;
	std::array<std::size_t,Rank + 1> extents;
	// Dynamic batch normalization parameters.
	RowVector<Scalar> avg_means;
	RowVector<Scalar> avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	Matrix<Scalar> params;
	Matrix<Scalar>& params_ref;
	Matrix<Scalar> params_grad;
	const Base& owner;
	// Staged computation cache vector.
	struct Cache {
		Scalar inv_in_sd;
		Matrix<Scalar> std_in;
	};
	std::vector<Cache> cache_vec;
};

/**
 * A class template for a per-activation batch normalization layer.
 *
 * \see https://arxiv.org/abs/1502.03167
 */
template<typename Scalar, std::size_t Rank>
class BatchNormLayer<Scalar,Rank,false> : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef BatchNormLayer<Scalar,Rank,false> Self;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param gamma_reg The regularization function to apply to the layer's gamma parameters.
	 * @param beta_reg The regularization function to apply to the layer's beta parameters.
	 * @param gamma_max_norm_constraint An optional max-norm constraint to enforce on the
	 * gamma parameters. If it is 0 or less, no constraint is applied.
	 * @param beta_max_norm_constraint An optional max-norm constraint to enforce on the
	 * beta parameters. If it is 0 or less, no constraint is applied.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(const Dimensions<std::size_t,Rank>& dims, ParamRegSharedPtr<Scalar> gamma_reg = Base::NO_PARAM_REG,
			ParamRegSharedPtr<Scalar> beta_reg = Base::NO_PARAM_REG, Scalar gamma_max_norm_constraint = 0,
			Scalar beta_max_norm_constraint = 0, Scalar norm_avg_decay = .1, Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				dims(dims),
				gamma_reg(gamma_reg),
				beta_reg(beta_reg),
				gamma_max_norm_constraint(gamma_max_norm_constraint),
				beta_max_norm_constraint(beta_max_norm_constraint),
				gamma_max_norm(NumericUtils<Scalar>::decidedly_greater(gamma_max_norm_constraint, (Scalar) 0)),
				beta_max_norm(NumericUtils<Scalar>::decidedly_greater(beta_max_norm_constraint, (Scalar) 0)),
				norm_avg_decay(norm_avg_decay),
				epsilon(epsilon),
				input_layer(false),
				frozen(false),
				avg_means(),
				avg_inv_sds(),
				avgs_init(false),
				params(),
				params_grad(),
				params_ref(params),
				owner(*this) {
		assert(gamma_reg != nullptr);
		assert(beta_reg != nullptr);
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline BatchNormLayer(const Self& layer) :
			dims(layer.dims),
			gamma_reg(layer.gamma_reg),
			beta_reg(layer.beta_reg),
			gamma_max_norm_constraint(layer.gamma_max_norm_constraint),
			beta_max_norm_constraint(layer.beta_max_norm_constraint),
			gamma_max_norm(layer.gamma_max_norm),
			beta_max_norm(layer.beta_max_norm),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			avg_means(layer.avg_means),
			avg_inv_sds(layer.avg_inv_sds),
			avgs_init(layer.avgs_init),
			params(layer.params),
			params_grad(layer.params_grad),
			params_ref(layer.is_shared_params_clone() ? layer.params_ref : params),
			owner(layer.is_shared_params_clone() ? layer.owner : *this),
			inv_in_sd(layer.inv_in_sd),
			std_in(layer.std_in) { }
	inline Base* clone() const {
		return new BatchNormLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return new BatchNormLayer(*this, true);
	}
	inline const Base& get_params_owner() const {
		return owner;
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return dims;
	}
	inline const Matrix<Scalar>& get_params() const {
		return params_ref;
	}
	inline const Matrix<Scalar>& get_params_grad() const {
		return params_grad;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() {
		params_ref = Matrix<Scalar>(dims.get_volume(), 2);
		// Gamma.
		params_ref.col(0).setOnes();
		// Beta.
		params_ref.col(1).setZero();
		params_grad = Matrix<Scalar>::Zero(params_ref.rows(), params_ref.cols());
		avg_means = Matrix<Scalar>(1, params_ref.rows());
		avg_inv_sds = Matrix<Scalar>(1, params_ref.rows());
		avgs_init = false;
	}
protected:
	inline BatchNormLayer(Self& layer, bool share_params) :
			dims(layer.dims),
			gamma_reg(layer.gamma_reg),
			beta_reg(layer.beta_reg),
			gamma_max_norm_constraint(layer.gamma_max_norm_constraint),
			beta_max_norm_constraint(layer.beta_max_norm_constraint),
			gamma_max_norm(layer.gamma_max_norm),
			beta_max_norm(layer.beta_max_norm),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			avg_means(layer.avg_means),
			avg_inv_sds(layer.avg_inv_sds),
			avgs_init(layer.avgs_init),
			params(share_params ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_params ? layer.params : params),
			owner(share_params ? layer.owner : *this),
			inv_in_sd(layer.inv_in_sd),
			std_in(layer.std_in) { }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void empty_cache() {
		inv_in_sd = RowVector<Scalar>(0);
		std_in = Matrix<Scalar>(0, 0);
	}
	inline Matrix<Scalar>& get_params() {
		return params_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() {
		params_grad.col(0) += gamma_reg->d_function(params_ref.col(0));
		params_grad.col(1) += beta_reg->d_function(params_ref.col(1));
	}
	inline Scalar get_regularization_penalty() const {
		return gamma_reg->function(params_ref.col(0)) + beta_reg->function(params_ref.col(1));
	}
	inline void enforce_constraints() {
		Scalar l2_norm;
		if (gamma_max_norm) {
			l2_norm = params_ref.col(0).squaredNorm();
			if (l2_norm > gamma_max_norm_constraint)
				params_ref.col(0) *= (gamma_max_norm_constraint / l2_norm);
		}
		if (beta_max_norm) {
			l2_norm = params_ref.col(1).squaredNorm();
			if (l2_norm > beta_max_norm_constraint)
				params_ref.col(1) *= (beta_max_norm_constraint / l2_norm);
		}
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
		if (training) {
			RowVector<Scalar> means = in_mat.colwise().mean();
			Matrix<Scalar> norm_in = in_mat.rowwise() - means;
			inv_in_sd = (norm_in.array().square().colwise().mean() + epsilon).sqrt().inverse();
			std_in = norm_in * inv_in_sd.asDiagonal();
			in_mat = std_in;
			// Maintain a moving average of means and variances for testing.
			if (avgs_init) {
				avg_means = (1.0 - norm_avg_decay) * avg_means + norm_avg_decay * means;
				avg_inv_sds = (1.0 - norm_avg_decay) * avg_inv_sds + norm_avg_decay * inv_in_sd;
			} else {
				avg_means = means;
				avg_inv_sds = inv_in_sd;
				avgs_init = true;
			}
		} else {
			// For testing, use the moving averages.
			assert(avgs_init);
			in_mat = (in_mat.rowwise() - avg_means) * avg_inv_sds.asDiagonal();
		}
		Matrix<Scalar> out = (in_mat * params_ref.col(0).asDiagonal()).rowwise() + params_ref.col(1).transpose();
		return TensorMap<Scalar,Base::DATA_RANK>(out.data(), in.dimensions());
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == dims);
		assert(out_grad.dimension(0) > 0 && std_in.rows() == out_grad.dimension(0));
		std::size_t rows = out_grad.dimension(0);
		/* Back-propagate the gradient through the batch normalization function and also calculate the
		 * gradients of the betas and gammas. */
		MatrixMap<Scalar> out_grad_mat(out_grad.data(), rows, out_grad.size() / rows);
		params_grad.col(0) = out_grad_mat.cwiseProduct(std_in).colwise().sum().transpose();
		params_grad.col(1) = out_grad_mat.colwise().sum().transpose();
		if (input_layer)
			return typename Base::Data();
		Matrix<Scalar> std_in_grad = out_grad_mat * params_ref.col(0).asDiagonal();
		Matrix<Scalar> prev_out_grad = (((rows * std_in_grad).rowwise() - std_in_grad.colwise().sum()) -
				std_in * (std_in.cwiseProduct(std_in_grad).colwise().sum().asDiagonal())) *
				(((Scalar) 1 / rows) * inv_in_sd).asDiagonal();
		return TensorMap<Scalar,Base::DATA_RANK>(prev_out_grad.data(), out_grad.dimensions());
	}
private:
	const Dimensions<std::size_t,Rank> dims;
	const ParamRegSharedPtr<Scalar> gamma_reg;
	const ParamRegSharedPtr<Scalar> beta_reg;
	const Scalar gamma_max_norm_constraint;
	const Scalar beta_max_norm_constraint;
	const bool gamma_max_norm;
	const bool beta_max_norm;
	const Scalar norm_avg_decay;
	const Scalar epsilon;
	bool input_layer;
	bool frozen;
	// Dynamic batch normalization parameters.
	RowVector<Scalar> avg_means;
	RowVector<Scalar> avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	Matrix<Scalar> params;
	Matrix<Scalar>& params_ref;
	Matrix<Scalar> params_grad;
	const Base& owner;
	// Staged computation caches.
	RowVector<Scalar> inv_in_sd;
	Matrix<Scalar> std_in;
};

/**
 * A class template representing a drop-out layer.
 *
 * \see https://arxiv.org/abs/1207.0580
 * \see http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
 */
template<typename Scalar, std::size_t Rank>
class DropoutLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param dropout_prob The probability of an element of the input tensor being set to 0.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline DropoutLayer(const Dimensions<std::size_t,Rank>& dims, Scalar dropout_prob,
			Scalar epsilon = NumericUtils<Scalar>::EPSILON2) :
				dims(dims),
				dropout_prob(dropout_prob),
				epsilon(epsilon),
				input_layer(false),
				frozen(false),
				params(),
				params_grad() {
		assert(dropout_prob > 0 && dropout_prob <= 1 &&
				"dropout probability must be greater than 0 and no greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline Base* clone() const {
		return new DropoutLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return clone();
	}
	inline const Base& get_params_owner() const {
		return *this;
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return dims;
	}
	inline const Matrix<Scalar>& get_params() const {
		return params;
	}
	inline const Matrix<Scalar>& get_params_grad() const {
		return params_grad;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
protected:
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void empty_cache() {
		dropout_mask = typename Base::Data();
	}
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() const {
		return 0;
	}
	inline void enforce_constraints() { }
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		if (training) {
			// Inverted dropout.
			Scalar scaling_factor = (Scalar) 1 / (1 - dropout_prob + epsilon);
			dropout_mask = in.random().unaryExpr([this,scaling_factor](Scalar i) {
				return (Scalar) (i <= dropout_prob ? 0 : scaling_factor);
			});
			return in * dropout_mask;
		}
		return in;
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == dims);
		assert(out_grad.dimension(0) > 0 && dropout_mask.dimension(0) == out_grad.dimension(0));
		if (input_layer)
			return typename Base::Data();
		// The derivative of the dropout function.
		return out_grad * dropout_mask;
	}
private:
	const Dimensions<std::size_t,Rank> dims;
	const Scalar dropout_prob;
	const Scalar epsilon;
	bool input_layer;
	bool frozen;
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
	// Staged computation cache.
	typename Base::Data dropout_mask;
};

/**
 * A class template representing a reshaping layer that outputs a reshaped copy of the input
 * tensor with the same volume. The data backing the tensor is not shifted in any way.
 */
template<typename Scalar, std::size_t Rank>
class ReshapeLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Rank + 1> RankwiseArray;
public:
	/**
	 * @param input_dims The nominal input dimensions of the layer.
	 * @param output_dims The dimensions of the reshaped tensor. The output tensor must have
	 * the same volume as the input tensor.
	 */
	inline ReshapeLayer(const Dimensions<std::size_t,Rank>& input_dims,
			const Dimensions<std::size_t,Rank>& output_dims) :
				input_dims(input_dims),
				output_dims(output_dims),
				input_layer(false),
				frozen(false),
				input_conversion_dims(output_dims.template promote<>()),
				output_conversion_dims(input_dims.template promote<>()),
				params(),
				params_grad() {
		assert(input_dims.get_volume() == output_dims.get_volume());
	}
	inline Base* clone() const {
		return new ReshapeLayer(*this);
	}
	inline Base* clone_with_shared_params() {
		return clone();
	}
	inline const Base& get_params_owner() const {
		return *this;
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return output_dims;
	}
	inline const Matrix<Scalar>& get_params() const {
		return params;
	}
	inline const Matrix<Scalar>& get_params_grad() const {
		return params_grad;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
protected:
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void empty_cache() { }
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() const {
		return 0;
	}
	inline void enforce_constraints() { }
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == input_dims);
		assert(in.dimension(0) > 0);
		input_conversion_dims[0] = in.dimension(0);
		return in.reshape(input_conversion_dims);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grad) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()) == output_dims);
		assert(out_grad.dimension(0) > 0 && input_conversion_dims[0] == out_grad.dimension(0));
		output_conversion_dims[0] = input_conversion_dims[0];
		return out_grad.reshape(output_conversion_dims);
	}
private:
	const Dimensions<std::size_t,Rank> input_dims;
	const Dimensions<std::size_t,Rank> output_dims;
	RankwiseArray input_conversion_dims;
	RankwiseArray output_conversion_dims;
	bool input_layer;
	bool frozen;
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
};

} /* namespace cattle */

#endif /* CATTL3_LAYER_H_ */
