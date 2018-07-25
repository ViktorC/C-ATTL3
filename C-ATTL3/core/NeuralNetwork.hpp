/*
 * NeuralNetwork.hpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_NEURALNETWORK_H_
#define CATTL3_NEURALNETWORK_H_

#include "Layer.hpp"

namespace cattle {

/**
 * An abstract neural network class template. It allows for inference and training via
 * back-propagation.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NeuralNetwork {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal neural network rank");
protected:
	static constexpr std::size_t DATA_RANK = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
	typedef Dimensions<std::size_t,Rank> Dims;
public:
	virtual ~NeuralNetwork() = default;
	/**
	 * A constant method implementing the clone pattern.
	 *
	 * @return A pointer to a copy of the instance. The instance does not take ownership of
	 * the returned pointer (i.e. the caller is responsible for deleting it).
	 */
	virtual NeuralNetwork<Scalar,Rank,Sequential>* clone() const = 0;
	/**
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors accepted by the network as its input along (except for the first rank which
	 * denotes the variable sample size and in case of sequential networks the second rank
	 * which denotes the variable time steps).
	 */
	virtual const Dims& get_input_dims() const = 0;
	/**
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors output by the network (except for the first rank which denotes the variable
	 * sample size and in case of sequential networks the second rank which denotes the
	 * variable time steps).
	 */
	virtual const Dims& get_output_dims() const = 0;
	/**
	 * @return A vector of pointers to constant layers constituting the network. The ownership
	 * of the layers remains with the network.
	 */
	virtual std::vector<const Layer<Scalar,Rank>*> get_layers() const = 0;
	/**
	 * @return A vector of pointers to the layers of the network. The ownership of the
	 * layers remains with the network.
	 */
	virtual std::vector<Layer<Scalar,Rank>*> get_layers() = 0;
	/**
	 * @return Whether the instance is a foremost network. If the instance is not a stand-alone
	 * network and it is not the first module of a complex network, it is not a foremost
	 * network. Foremost networks do not need to back-propagate the gradients all the way
	 * given that no other network is expected to depend on them.
	 */
	virtual bool is_foremost() const = 0;
	/**
	 * Sets the foremost status of the network.
	 *
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	virtual void set_foremost(bool foremost) = 0;
	/**
	 * Empties the caches of every layer of the network.
	 */
	virtual void empty_caches() = 0;
	/**
	 * It propagates the input tensor through the network and outputs its prediction.
	 *
	 * @param input The input tensor to propagate through.
	 * @param training Whether the input is to be propagated in training mode or not.
	 * Propagating the input in training mode may be more time and memory consuming, but
	 * is a prerequisite of back-propagation.
	 * @return The output tensor of the network in response to the input.
	 */
	virtual Data propagate(Data input, bool training) = 0;
	/**
	 * It back-propagates the derivative of the loss function w.r.t. the output of the
	 * network through its layers updating the gradients on their parameters.
	 *
	 * @param out_grad The derivative of the loss function w.r.t. the output of the
	 * network.
	 * @return The derivative of the loss function w.r.t. the input of the network or
	 * a null tensor if the network is a foremost network.
	 */
	virtual Data backpropagate(Data out_grad) = 0;
	/**
	 * Invokes the Layer#set_frozen(bool) method of all layers of the network with the
	 * provided argument. A frozen networks parameters are not regularized.
	 *
	 * @param frozen Whether the parameters of all layers should be frozen (i.e. not updatable
	 * via optimization) or active.
	 */
	inline virtual void set_frozen(bool frozen) {
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (std::size_t i = 0; i < layers.size(); ++i) {
			std::vector<Parameters<Scalar>*> params = layers[i]->get_params();
			for (std::size_t j = 0; j < params.size(); ++j)
				params[j]->set_frozen(frozen);
		}
	}
	/**
	 * Initializes all parameters of the network.
	 */
	inline virtual void init() {
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (std::size_t i = 0; i < layers.size(); ++i) {
			std::vector<Parameters<Scalar>*> params = layers[i]->get_params();
			for (std::size_t j = 0; j < params.size(); ++j)
				params[j]->init();
		}
	}
	/**
	 * It propagates the input through the neural network and outputs its prediction
	 * according to its current parameters.
	 *
	 * @param input The input to be mapped.
	 * @return The inference/prediction of the neural network.
	 */
	inline virtual Data infer(Data input) {
		return propagate(std::move(input), false);
	}
};

} /* namespace cattle */

#endif /* CATTL3_NEURALNETWORK_H_ */
