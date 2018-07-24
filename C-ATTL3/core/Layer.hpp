/*
 * Layer.hpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_LAYER_H_
#define CATTL3_LAYER_H_

#include <type_traits>
#include <vector>

#include "Dimensions.hpp"
#include "EigenProxy.hpp"
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
	typedef Dimensions<std::size_t,Rank> Dims;
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
	virtual const Dims& get_input_dims() const = 0;
	/**
	 * A simple constant getter method for the output dimensionality of the layer.
	 *
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors output by the layer along all ranks except the first one.
	 */
	virtual const Dims& get_output_dims() const = 0;
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

} /* namespace cattle */

#endif /* CATTL3_LAYER_H_ */
