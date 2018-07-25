/*
 * FeedforwardNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_STACKEDNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_STACKEDNEURALNETWORK_H_

#include <cassert>
#include <utility>

#include "neural_network/CompositeNeuralNetwork.hpp"

namespace cattle {

/**
 * A class template for a composite neural network that consists of a set of
 * serially stacked neural network sub-modules.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class StackedNeuralNetwork :
		public CompositeNeuralNetwork<Scalar,Rank,Sequential,NeuralNetwork<Scalar,Rank,Sequential>> {
	typedef NeuralNetwork<Scalar,Rank,Sequential> Base;
	typedef StackedNeuralNetwork<Scalar,Rank,Sequential> Self;
	typedef NeuralNetPtr<Scalar,Rank,Sequential> Block;
public:
	/**
	 * @param blocks A vector of unique pointers to neural networks.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline StackedNeuralNetwork(std::vector<Block>&& blocks, bool foremost = true) :
			blocks(std::move(blocks)),
			foremost(foremost) {
		assert(this->blocks.size() > 0 && "blocks must contain at least 1 element");
		assert(this->blocks[0] != nullptr && "blocks contains null pointers");
		Base& first_block = *this->blocks[0];
		input_dims = first_block.get_input_dims();
		output_dims = this->blocks[this->blocks.size() - 1]->get_output_dims();
		typename Base::Dims prev_dims = first_block.get_output_dims();
		for (std::size_t i = 1; i < this->blocks.size(); ++i) {
			assert(this->blocks[i] != nullptr && "blocks contains null pointers");
			Base& block = *this->blocks[i];
			assert(prev_dims == block.get_input_dims() && "incompatible network dimensions");
			block.set_foremost(false);
			prev_dims = block.get_output_dims();
		}
		first_block.set_foremost(foremost);
	}
	/**
	 * @param block A unique pointer to a neural network.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline StackedNeuralNetwork(Block&& block, bool foremost = true) :
			StackedNeuralNetwork(create_vector(std::move(block)), foremost) { }
	inline StackedNeuralNetwork(const Self& network) :
			blocks(network.blocks.size()),
			foremost(network.foremost),
			input_dims(network.input_dims),
			output_dims(network.output_dims) {
		for (std::size_t i = 0; i < blocks.size(); ++i)
			blocks[i] = Block(network.blocks[i]->clone());
	}
	inline StackedNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~StackedNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Base* clone() const {
		return new StackedNeuralNetwork(*this);
	}
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
		return output_dims;
	}
	inline std::vector<const Layer<Scalar,Rank>*> get_layers() const {
		std::vector<const Layer<Scalar,Rank>*> layer_ptrs;
		populate_layer_vector<const Layer<Scalar,Rank>*>(layer_ptrs);
		return layer_ptrs;
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layer_ptrs;
		populate_layer_vector<Layer<Scalar,Rank>*>(layer_ptrs);
		return layer_ptrs;
	}
	inline std::vector<Base*> get_modules() {
		std::vector<Base*> modules;
		for (std::size_t i = 0; i < blocks.size(); ++i)
			modules.push_back(blocks[i].get());
		return modules;
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline void set_foremost(bool foremost) {
		blocks[0]->set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		for (std::size_t i = 0; i < blocks.size(); ++i)
			blocks[i]->empty_caches();
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		assert(input_dims == (Dimensions<std::size_t,Base::DATA_RANK>(input.dimensions()).template demote<>()));
		for (std::size_t i = 0; i < blocks.size(); ++i)
			input = blocks[i]->propagate(std::move(input), training);
		return input;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grad) {
		assert(output_dims == (Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()));
		for (int i = blocks.size() - 1; i >= 0; --i)
			out_grad = blocks[i]->backpropagate(std::move(out_grad));
		return out_grad;
	}
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.blocks, network2.blocks);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	}
private:
	inline static std::vector<Block> create_vector(Block&& net) {
		std::vector<Block> vec(1);
		vec[0] = std::move(net);
		return vec;
	}
	template<typename _LayerPtr>
	inline void populate_layer_vector(std::vector<_LayerPtr>& layer_ptrs) const {
		for (std::size_t i = 0; i < blocks.size(); ++i) {
			std::vector<Layer<Scalar,Rank>*> internal_layer_ptrs = blocks[i]->get_layers();
			for (std::size_t j = 0; j < internal_layer_ptrs.size(); ++j)
				layer_ptrs.push_back(internal_layer_ptrs[j]);
		}
	}
	std::vector<Block> blocks;
	bool foremost;
	typename Base::Dims input_dims, output_dims;
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_STACKEDNEURALNETWORK_H_ */
