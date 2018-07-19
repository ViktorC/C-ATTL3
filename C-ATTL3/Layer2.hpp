/*
 * Layer.hpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_LAYER_H_
#define CATTL3_LAYER_H_

#ifdef CATTL3_USE_CUDA
#define CATTL3_USE_CUBLAS
#define CATTL3_USE_CUDNN
#endif

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
#include "ParameterRegularization.hpp"
#include "utils/EigenProxy.hpp"
#include "utils/NumericUtils.hpp"
#include "WeightInitialization.hpp"

#ifdef CATTL3_USE_CUBLAS
#include "utils/gpu/CuBLASHandle.hpp"
#endif

#ifdef CATTL3_USE_CUDNN
#include "utils/gpu/CuDNNHandle.hpp"
#endif

namespace cattle {

// TODO FFT and/or Winograd filtering for CPU convolution.

/**
 * An alias for a shared pointer to a WeightInitialization implementation instance of
 * an arbitrary scalar type.
 */
template<typename Scalar>
using WeightInitSharedPtr = std::shared_ptr<WeightInitialization<Scalar>>;

/**
 * An alias for a shared pointer to a regularization penalty of an arbitrary scalar type.
 */
template<typename Scalar>
using ParamRegSharedPtr = std::shared_ptr<ParamaterRegularization<Scalar>>;

// Forward declarations to NeuralNetwork and Optimizer so they can be friended.
template<typename Scalar, std::size_t Rank, bool Sequential> class NeuralNetwork;
template<typename Scalar, std::size_t Rank, bool Sequential> class Optimizer;

/**
 * An abstract class template representing layers in a neural network.
 */
template<typename Scalar>
class Layer {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	typedef Dimensions<std::size_t,3> Dimensions;
	static const ParamRegSharedPtr<Scalar> NO_PARAM_REG;
	virtual ~Layer() = default;
	/**
	 * It returns a clone of the layer instance.
	 *
	 * @return A pointer to a copy of the instance. The instance does not take ownership of
	 * the returned pointer (i.e. the caller is responsible for deleting it).
	 */
	virtual Layer<Scalar>* clone() const = 0;
	/**
	 * A simple constant getter method for the input dimensionality of the layer.
	 *
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors accepted by the layer as its input (except for the first rank which denotes
	 * the variable sample size).
	 */
	virtual const Dimensions& get_input_dims() const = 0;
	/**
	 * A simple constant getter method for the output dimensionality of the layer.
	 *
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors output by the layer along all ranks except the first one.
	 */
	virtual const Dimensions& get_output_dims() const = 0;
	/**
	 * It returns a clone of the layer instance using a reference to the original's parameters.
	 * Non-parametric layers do not need to support parameter sharing and thus are just expected
	 * to return a normal clone.
	 *
	 * @return A clone of the original layer instance sharing the same parameters with the
	 * original.
	 */
	inline virtual Layer<Scalar>* clone_with_shared_params() {
		return clone();
	}
	/**
	 * It initializes the layer and its parameters.
	 */
	inline virtual void init() { }
	/**
	 * A method that returns whether the layer has parameters that can be learned.
	 *
	 * @return Whether the layer uses learnable parameters.
	 */
	inline virtual bool is_parametric() const {
		return false;
	}
	/**
	 * It returns a reference to the layer owning the parameters used. If this owner goes out
	 * of scope (in case this one is a clone with shared parameters), the behavior of the clone
	 * is undefined.
	 *
	 * @return A reference to the layer owning the parameters. If this layer is not using
	 * shared parameters, it returns a reference to itself.
	 */
	inline virtual const Layer<Scalar>& get_params_owner() const {
		return *this;
	}
	/**
	 * It returns a vector of pointers to the parameters of the layer.
	 *
	 * @return A vector of pointers to the parameters of the layer that are to be learned.
	 */
	inline virtual std::vector<const Matrix<Scalar>*> get_params() const {
		return std::vector<const Matrix<Scalar>*>(0);
	}
	/**
	 * It returns a vector of pointers to the gradients of the learnable parameters of the
	 * layer.
	 *
	 * @return A vector of pointers to the gradients of the learnable parameters of the layer.
	 */
	inline virtual std::vector<const Matrix<Scalar>*> get_params_grads() const {
		return std::vector<const Matrix<Scalar>*>(0);
	}
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
	 * It determines whether the parameters of the layer, if there are any, are to be
	 * updated during optimization.
	 *
	 * @return Whether the parameters should not be updated during optimization.
	 */
	inline bool is_frozen() const {
		return frozen;
	}
	/**
	 * It sets whether the parameters of the layer should not be updated during optimization.
	 * Frozen layers are not regularized either.
	 *
	 * @param frozen Whether the parameters of the layer are to be frozen, i.e. not
	 * updatable via optimization.
	 */
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	/**
	 * @return A string representation of the layer.
	 */
	inline virtual std::string to_string() const {
		static const int header_length = 128;
		std::stringstream strm;
		std::stringstream id_num_strm;
		id_num_strm << this;
		std::string id = "<" + std::string(typeid(*this).name()) + id_num_strm.str() + ">";
		strm << id << std::string(std::max(0, (int) (header_length - id.length())), '-') << std::endl;
		strm << "\tinput dims: " << get_input_dims().to_string() << std::endl;
		strm << "\toutput dims: " << get_output_dims().to_string() << std::endl;
		if (is_parametric()) {
			strm << "\tparams:" << std::endl;
			const Matrix<Scalar>& params = get_params();
			for (int j = 0; j < params.rows(); ++j) {
				strm << "\t[ ";
				for (int k = 0; k < params.cols(); ++k) {
					strm << std::setw(11) << std::setprecision(4) << params(j,k);
					if (k != params.cols() - 1)
						strm << ", ";
				}
				strm << " ]" << std::endl;
			}
		}
		return strm.str();
	}
	inline friend std::ostream& operator<<(std::ostream& os, const Layer<Scalar>& layer) {
		return os << layer.to_string() << std::flush;
	}
protected:
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
	virtual UnifiedTensor<Scalar> pass_forward(UnifiedTensor<Scalar> in, bool training) = 0;
	/**
	 * It back-propagates the derivative of the error function w.r.t. the output of the
	 * layer updating the gradient of its learnable parameters along the way if there are
	 * any. If there are, it also calculates the derivative of the regularization penalty
	 * w.r.t. to the layer's parameters and adds it to their gradient.
	 *
	 * @param out_grad The derivative of the loss function w.r.t. the output of the
	 * layer
	 * @return The derivative of the loss function w.r.t. the output of the previous layer
	 * or a null tensor if the layer is an input layer.
	 */
	virtual UnifiedTensor<Scalar> pass_back(UnifiedTensor<Scalar> out_grad) = 0;
	/**
	 * It empties the layer's caches such as those required for the derivation of the function
	 * represented by the layer.
	 */
	inline virtual void empty_cache() { }
	/**
	 * It returns a vector of pointers to the parameters of the layer.
	 *
	 * @return A vector of pointers to the parameters of the layer that are to be learned.
	 */
	inline virtual std::vector<Matrix<Scalar>*> get_params() {
		return std::vector<Matrix<Scalar>*>(0);
	}
	/**
	 * It returns a vector of pointers to the gradients of the learnable parameters of the
	 * layer.
	 *
	 * @return A vector of pointers to the gradients of the learnable parameters of the layer.
	 */
	inline virtual std::vector<Matrix<Scalar>*> get_params_grads() {
		return std::vector<Matrix<Scalar>*>(0);
	}
	/**
	 * It computes the derivative of the regularization function w.r.t. the parameters of the
	 * layer and adds it to their gradient. If the layer is not parametric, calling this
	 * method has no effect.
	 */
	inline virtual void regularize() { }
	/**
	 * It calculates the regularization penalty of the layer's parameters. If the layer is not
	 * parametric, 0 is returned.
	 *
	 * @return A scalar representing the penalty on the magnitude of the layer's parameters.
	 */
	inline virtual Scalar get_regularization_penalty() const {
		return 0;
	}
	/**
	 * It applies constraints such as max-norm to the parameters of the layer (if applicable).
	 */
	inline virtual void enforce_constraints() { }
	/**
	 * A constant method that returns whether this layer functions as an input layer. An input
	 * layer does not need to propagate the gradients all the way during the backward pass as
	 * it is assumed that no other layer needs them derive the gradient on its parameters. It
	 * is therefore possible for an input layer to simply return a null tensor as the output of
	 * its backward pass.
	 *
	 * @return Whether this layer is the input layer of the neural network that contains it.
	 */
	inline bool is_input_layer() const {
		return input_layer;
	}
	/**
	 * Sets this instance's input layer status to the given value.
	 *
	 * @param input_layer Whether this layer is to be an input layer or not.
	 */
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
private:
	bool input_layer = true;
	bool frozen = false;
};

// Initialize the static default regularization penalty.
template<typename Scalar>
const ParamRegSharedPtr<Scalar> Layer<Scalar>::NO_PARAM_REG = std::make_shared<NoParameterRegularization<Scalar>>();


} /* namespace cattle */

#endif /* CATTL3_LAYER_H_ */
