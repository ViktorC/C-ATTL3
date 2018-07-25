/*
 * CompositeNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_COMPOSITENEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_COMPOSITENEURALNETWORK_H_

#include "core/NeuralNetwork.hpp"

namespace cattle {

/**
 * An alias for a unique pointer to a neural network of arbitrary scalar type, rank,
 * and sequentiality.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using NeuralNetPtr = std::unique_ptr<NeuralNetwork<Scalar,Rank,Sequential>>;

/**
 * A class template for composite neural networks consisting of one or more neural
 * network modules.
 */
template<typename Scalar, std::size_t Rank, bool Sequential, typename Module>
class CompositeNeuralNetwork : public NeuralNetwork<Scalar,Rank,Sequential> {
public:
	/**
	 * @return A vector of pointers pointing to the sub-modules of the composite
	 * network instance. The ownership of the modules is not transferred to the
	 * caller of the method.
	 */
	virtual std::vector<Module*> get_modules() = 0;
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_COMPOSITENEURALNETWORK_H_ */
