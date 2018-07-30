/*
 * FeedforwardNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_FEEDFORWARDNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_FEEDFORWARDNEURALNETWORK_H_

#include <cassert>
#include <memory>
#include <utility>

#include "core/NeuralNetwork.hpp"

namespace cattle {

/**
 * An alias for a unique pointer to a layer of arbitrary rank and scalar type.
 */
template<typename Scalar, std::size_t Rank>
using LayerPtr = std::unique_ptr<Layer<Scalar,Rank>>;

/**
 * A class template representing a simple feed-forward neural network.
 */
template<typename Scalar, std::size_t Rank>
class FeedforwardNeuralNetwork : public NeuralNetwork<Scalar,Rank,false> {
	typedef NeuralNetwork<Scalar,Rank,false> Base;
	typedef FeedforwardNeuralNetwork<Scalar,Rank> Self;
public:
	/**
	 * @param layers A vector of unique smart pointers to the layers that constitute the neural network.
	 * @param foremost Whether the network directly receives its input. If it is set to false, back-propagation
	 * returns an empty tensor.
	 */
	inline FeedforwardNeuralNetwork(std::vector<LayerPtr<Scalar,Rank>>&& layers, bool foremost = true) :
			layers(std::move(layers)),
			foremost(foremost) {
		assert(this->layers.size() > 0 && "layers must contain at least 1 element");
		assert(this->layers[0] != nullptr);
		Layer<Scalar,Rank>& first_layer = *this->layers[0];
		input_dims = first_layer.get_input_dims();
		output_dims = this->layers[this->layers.size() - 1]->get_output_dims();
		typename Base::Dims prev_dims = first_layer.get_output_dims();
		for (std::size_t i = 1; i < this->layers.size(); ++i) {
			assert(this->layers[i] != nullptr && "layers contains null pointers");
			assert(prev_dims == this->layers[i]->get_input_dims() && "incompatible layer dimensions");
			prev_dims = this->layers[i]->get_output_dims();
		}
		first_layer.set_input_layer(foremost);
	}
	/**
	 * @param layer A unique pointer to the single layer of the network.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline FeedforwardNeuralNetwork(LayerPtr<Scalar,Rank>&& layer, bool foremost = true) :
			FeedforwardNeuralNetwork(create_vector(std::move(layer)), foremost) { }
	// Copy constructor.
	inline FeedforwardNeuralNetwork(const Self& network) :
			layers(network.layers.size()),
			foremost(network.foremost),
			input_dims(network.input_dims),
			output_dims(network.output_dims) {
		for (std::size_t i = 0; i < layers.size(); ++i)
			layers[i] = LayerPtr<Scalar,Rank>(network.layers[i]->clone());
	}
	// Move constructor.
	inline FeedforwardNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	// The smart pointers take care of deleting the layers.
	~FeedforwardNeuralNetwork() = default;
	/* The assignment uses the move or copy constructor to pass the parameter
	 * based on whether it is an rvalue or an lvalue. */
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Base* clone() const {
		return new FeedforwardNeuralNetwork(*this);
	}
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
		return output_dims;
	}
	inline std::vector<const Layer<Scalar,Rank>*> get_layers() const {
		std::vector<const Layer<Scalar,Rank>*> layer_ptrs(layers.size());
		populate_layer_vector<const Layer<Scalar,Rank>*>(layer_ptrs);
		return layer_ptrs;
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layer_ptrs(layers.size());
		populate_layer_vector<Layer<Scalar,Rank>*>(layer_ptrs);
		return layer_ptrs;
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline void set_foremost(bool foremost) {
		layers[0]->set_input_layer(foremost);
		this->foremost = foremost;
	}
	inline virtual void empty_caches() {
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (std::size_t i = 0; i < layers.size(); ++i)
			layers[i]->empty_cache();
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		assert(input_dims == (Dimensions<std::size_t,Base::DATA_RANK>(input.dimensions()).template demote<>()));
		for (std::size_t i = 0; i < layers.size(); ++i) {
			Layer<Scalar,Rank>& layer = *layers[i];
			input = layer.pass_forward(std::move(input), training);
		}
		return input;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grad) {
		assert(output_dims == (Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()));
		for (int i = layers.size() - 1; i >= 0; --i) {
			Layer<Scalar,Rank>& layer = *layers[i];
			out_grad = layer.pass_back(std::move(out_grad));
		}
		return out_grad;
	}
	// For the copy-and-swap idiom.
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	}
private:
	inline static std::vector<LayerPtr<Scalar,Rank>> create_vector(LayerPtr<Scalar,Rank>&& layer) {
		std::vector<LayerPtr<Scalar,Rank>> vec(1);
		vec[0] = std::move(layer);
		return vec;
	}
	template<typename _LayerPtr>
	inline void populate_layer_vector(std::vector<_LayerPtr>& layer_ptrs) const {
		for (std::size_t i = 0; i < layers.size(); ++i)
			layer_ptrs[i] = layers[i].get();
	}
	std::vector<LayerPtr<Scalar,Rank>> layers;
	bool foremost;
	typename Base::Dims input_dims, output_dims;
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_FEEDFORWARDNEURALNETWORK_H_ */
