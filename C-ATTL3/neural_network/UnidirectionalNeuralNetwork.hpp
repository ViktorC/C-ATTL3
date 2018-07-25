/*
 * UnidirectionalNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_UNIDIRECTIONALNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_UNIDIRECTIONALNEURALNETWORK_H_

#include "core/NeuralNetwork.hpp"

namespace cattle {

/**
 * An abstract class template for unidirectional recurrent neural networks.
 */
template<typename Scalar, std::size_t Rank>
class UnidirectionalNeuralNetwork : public NeuralNetwork<Scalar,Rank,true> {
	typedef NeuralNetwork<Scalar,Rank,true> Base;
public:
	virtual ~UnidirectionalNeuralNetwork() = default;
	/**
	 * @return Whether the direction along the time-step rank in which the network processes
	 * its inputs is reversed.
	 */
	virtual bool is_reversed() const;
	/**
	 * Flips the direction along the time-step rank in which the network processes its inputs
	 * is reversed.
	 */
	virtual void reverse();
	/**
	 * Reverses a tensor along its time axis.
	 *
	 * @param tensor The tensor to reverse.
	 */
	inline static void reverse_along_time_axis(typename Base::Data& tensor) {
		std::array<bool,Base::DATA_RANK> reverse;
		reverse.fill(false);
		reverse[1] = true;
		tensor = tensor.reverse(reverse);
	}
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_UNIDIRECTIONALNEURALNETWORK_H_ */
