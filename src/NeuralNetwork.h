/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <Dimensions.h>
#include <Eigen/Core>
#include <functional>
#include <iomanip>
#include <Layer.h>
#include <pthread.h>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

// TODO Sequential feed forward network.
// TODO LSTM and GRU networks.
// TODO Possibility to add and remove modules (e.g. layers for sequential networks, inception modules for InceptionNets).
// TODO Serialization.

namespace cattle {

template<typename Scalar, size_t Rank, bool Sequential> class Optimizer;
template<typename Scalar, size_t Rank, bool Sequential> class CompositeNeuralNetwork;
template<typename Scalar> class ParallelNeuralNetwork;
template<typename Scalar, size_t Rank> class SequentialNeuralNetwork;

/**
 * A neural network class that consists of a vector of neuron layers. It allows
 * for the calculation of the output of the network given an input vector and
 * it provides a member function for back-propagating the derivative of a loss
 * function with respect to the activated output of the network's last layer.
 * This back-propagation sets the gradients of the parameters associated with each
 * layer.
 */
template<typename Scalar, size_t Rank, bool Sequential>
class NeuralNetwork {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	friend class Optimizer<Scalar,Rank,Sequential>;
	static_assert(Rank > 0 && Rank < 4, "illegal neural network rank");
	friend class CompositeNeuralNetwork<Scalar,Rank,Sequential>;
	friend class ParallelNeuralNetwork<Scalar>;
	friend class SequentialNeuralNetwork<Scalar,Rank>;
protected:
	static constexpr size_t DATA_RANKS = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANKS> Data;
	typedef Dimensions<int,Rank> Dims;
public:
	virtual ~NeuralNetwork() = default;
	virtual NeuralNetwork<Scalar,Rank,Sequential>* clone() const = 0;
	virtual bool is_foremost() const = 0;
	virtual Dims get_input_dims() const = 0;
	virtual Dims get_output_dims() const = 0;
	inline virtual void init() {
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); ++i)
			layers[i]->init();
	}
	inline virtual Data infer(Data input) {
		return propagate(std::move(input), false);
	}
	inline virtual std::string to_string() {
		std::stringstream strm;
		strm << "Neural Net " << this << std::endl;
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); ++i) {
			Layer<Scalar,Rank>* layer = layers[i];
			strm << "\tLayer " << std::setw(3) << std::to_string(i + 1) << std::string(28, '-') << std::endl;
			strm << "\t\tinput dims: " << layer->get_input_dims().to_string() << std::endl;
			strm << "\t\toutput dims: " << layer->get_output_dims().to_string() << std::endl;
			if (layer->is_parametric()) {
				strm << "\t\tparams:" << std::endl;
				Matrix<Scalar>& params = layer->get_params();
				for (int j = 0; j < params.rows(); ++j) {
					strm << "\t\t[ ";
					for (int k = 0; k < params.cols(); ++k) {
						strm << std::setw(11) << std::setprecision(4) << params(j,k);
						if (k != params.cols() - 1)
							strm << ", ";
					}
					strm << " ]" << std::endl;
				}
			}
		}
		return strm.str();
	}
	friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<Scalar,Rank,Sequential>& nn) {
		return os << nn.to_string() << std::endl;
	}
protected:
	virtual void set_foremost(bool foremost) = 0;
	virtual void empty_caches() = 0;
	virtual std::vector<Layer<Scalar,Rank>*> get_layers() = 0;
	virtual Data propagate(Data input, bool training) = 0;
	virtual Data backpropagate(Data out_grads) = 0;
	inline static void set_input_layer(Layer<Scalar,Rank>& layer, bool on) {
		layer.input_layer = on;
	}
	inline static void empty_cache(Layer<Scalar,Rank>& layer) {
		layer.empty_cache();
	}
	inline static Tensor<Scalar,Rank + 1> pass_forward(Layer<Scalar,Rank>& layer, Tensor<Scalar,Rank + 1> prev_out,
			bool training) {
		return layer.pass_forward(std::move(prev_out), training);
	}
	inline static Tensor<Scalar,Rank + 1> pass_back(Layer<Scalar,Rank>& layer, Tensor<Scalar,Rank + 1> out_grads) {
		return layer.pass_back(std::move(out_grads));
	}
	inline static Matrix<Scalar>& get_param_grads(Layer<Scalar,Rank>& layer) {
		return layer.get_param_grads();
	}
};

template<typename Scalar, size_t Rank, bool Sequential>
using NeuralNetPtr = std::unique_ptr<NeuralNetwork<Scalar,Rank,Sequential>>;

template<typename Scalar, size_t Rank> class ResidualNeuralNetwork;
template<typename Scalar> class DenseNeuralNetwork;

template<typename Scalar, size_t Rank, bool Sequential>
class CompositeNeuralNetwork : public NeuralNetwork<Scalar,Rank,Sequential> {
	friend class ResidualNeuralNetwork<Scalar,Rank>;
	friend class DenseNeuralNetwork<Scalar>;
	typedef NeuralNetwork<Scalar,Rank,Sequential> Base;
	typedef CompositeNeuralNetwork<Scalar,Rank,Sequential> Self;
	typedef NeuralNetPtr<Scalar,Rank,Sequential> Block;
public:
	inline CompositeNeuralNetwork(std::vector<Block> blocks, bool foremost = true) :
			blocks(std::move(blocks)),
			foremost(foremost) {
		assert(this->blocks.size() > 0 && "blocks must contain at least 1 element");
		assert(this->blocks[0] != nullptr && "blocks contains null pointers");
		Base& first_block = *(this->blocks[0]);
		input_dims = first_block.get_input_dims();
		output_dims = this->blocks[this->blocks.size() - 1]->get_output_dims();
		typename Base::Dims prev_dims = first_block.get_output_dims();
		for (unsigned i = 1; i < this->blocks.size(); ++i) {
			assert(this->blocks[i] != nullptr && "blocks contains null pointers");
			Base& block = *this->blocks[i];
			assert(prev_dims == block.get_input_dims() && "incompatible network dimensions");
			block.set_foremost(false);
			prev_dims = block.get_output_dims();
		}
		first_block.set_foremost(foremost);
	}
	inline CompositeNeuralNetwork(Block&& block, bool foremost = true) :
			CompositeNeuralNetwork(create_vector(std::move(block)), foremost) { }
	inline CompositeNeuralNetwork(const Self& network) :
			blocks(network.blocks.size()) {
		for (unsigned i = 0; i < blocks.size(); ++i)
			blocks[i] = Block(network.blocks[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	}
	inline CompositeNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~CompositeNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Base* clone() const {
		return new CompositeNeuralNetwork(*this);
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline typename Base::Dims get_input_dims() const {
		return input_dims;
	}
	inline typename Base::Dims get_output_dims() const {
		return output_dims;
	}
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.blocks, network2.blocks);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	}
protected:
	inline void set_foremost(bool foremost) {
		blocks[0]->set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		for (unsigned i = 0; i < blocks.size(); ++i)
			blocks[i]->empty_caches();
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers;
		for (unsigned i = 0; i < blocks.size(); ++i) {
			std::vector<Layer<Scalar,Rank>*> internal_layers = blocks[i]->get_layers();
			for (unsigned j = 0; j < internal_layers.size(); ++j)
				layers.push_back(internal_layers[j]);
		}
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input).template demote<>());
		for (unsigned i = 0; i < blocks.size(); ++i)
			input = blocks[i]->propagate(std::move(input), training);
		return input;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads).template demote<>());
		for (int i = blocks.size() - 1; i >= 0; --i)
			out_grads = blocks[i]->backpropagate(std::move(out_grads));
		return out_grads;
	}
	inline static std::vector<Block> create_vector(Block&& net) {
		std::vector<Block> vec(1);
		vec[0] = std::move(net);
		return vec;
	}
private:
	std::vector<Block> blocks;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
};

template<typename Scalar, size_t Rank>
using LayerPtr = std::unique_ptr<Layer<Scalar,Rank>>;

template<typename Scalar, size_t Rank>
class FeedforwardNeuralNetwork : public NeuralNetwork<Scalar,Rank,false> {
	typedef NeuralNetwork<Scalar,Rank,false> Base;
	typedef FeedforwardNeuralNetwork<Scalar,Rank> Self;
public:
	/**
	 * Constructs the network using the provided layer pointers. It takes ownership of the layer pointers.
	 *
	 * @param layers A vector of unique smart pointers to the layers that constitute the neural network.
	 * @param foremost Whether the network directly receives its input. If it is set to false, back-propagation
	 * returns an empty tensor.
	 */
	inline FeedforwardNeuralNetwork(std::vector<LayerPtr<Scalar,Rank>> layers, bool foremost = true) :
			layers(std::move(layers)),
			foremost(foremost) {
		assert(this->layers.size() > 0 && "layers must contain at least 1 element");
		assert(this->layers[0] != nullptr);
		Layer<Scalar,Rank>& first_layer = *(this->layers[0]);
		input_dims = first_layer.get_input_dims();
		output_dims = this->layers[this->layers.size() - 1]->get_output_dims();
		typename Base::Dims prev_dims = first_layer.get_output_dims();
		for (unsigned i = 1; i < this->layers.size(); ++i) {
			assert(this->layers[i] != nullptr && "layers contains null pointers");
			assert(prev_dims == this->layers[i]->get_input_dims() && "incompatible layer dimensions");
			prev_dims = this->layers[i]->get_output_dims();
		}
		Base::set_input_layer(first_layer, foremost);
	}
	inline FeedforwardNeuralNetwork(LayerPtr<Scalar,Rank> layer, bool foremost = true) :
			FeedforwardNeuralNetwork(create_vector(std::move(layer)), foremost) { }
	// Copy constructor.
	inline FeedforwardNeuralNetwork(const Self& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); ++i)
			layers[i] = LayerPtr<Scalar,Rank>(network.layers[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
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
	inline bool is_foremost() const {
		return foremost;
	}
	inline typename Base::Dims get_input_dims() const {
		return input_dims;
	}
	inline typename Base::Dims get_output_dims() const {
		return output_dims;
	}
	// For the copy-and-swap idiom.
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	}
	inline static std::vector<LayerPtr<Scalar,Rank>> create_vector(LayerPtr<Scalar,Rank> layer) {
		std::vector<LayerPtr<Scalar,Rank>> vec(1);
		vec[0] = std::move(layer);
		return vec;
	}
protected:
	inline void set_foremost(bool foremost) {
		Base::set_input_layer(*layers[0], foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		for (unsigned i = 0; i < layers.size(); ++i)
			Base::empty_cache(*layers[i]);
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers_raw(layers.size());
		for (unsigned i = 0; i < layers.size(); ++i)
			layers_raw[i] = layers[i].get();
		return layers_raw;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input).template demote<>());
		for (unsigned i = 0; i < layers.size(); ++i) {
			Layer<Scalar,Rank>& layer = *layers[i];
			input = Base::pass_forward(layer, std::move(input), training);
			if (!training)
				Base::empty_cache(layer);
		}
		return input;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads).template demote<>());
		for (int i = layers.size() - 1; i >= 0; --i) {
			Layer<Scalar,Rank>& layer = *layers[i];
			out_grads = Base::pass_back(layer, std::move(out_grads));
			Base::empty_cache(layer);
		}
		return out_grads;
	}
private:
	std::vector<LayerPtr<Scalar,Rank>> layers;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
};

/**
 * This class can be used to build InceptionNets, ResNets, Inception-ResNets, or non-residual composite
 * neural networks.
 */
template<typename Scalar, size_t Rank>
class ResidualNeuralNetwork : public NeuralNetwork<Scalar,Rank,false> {
	typedef NeuralNetwork<Scalar,Rank,false> Base;
	typedef CompositeNeuralNetwork<Scalar,Rank,false> Module;
public:
	inline ResidualNeuralNetwork(std::vector<std::pair<Module,bool>> modules, bool foremost = true) :
			modules(modules),
			foremost(foremost) {
		assert(modules.size() > 0 && "modules must contain at least 1 element");
		Module& first_module = this->modules[0].first;
		input_dims = first_module.get_input_dims();
		output_dims = this->modules[modules.size() - 1].first.get_output_dims();
		first_module.set_foremost(foremost);
		typename Base::Dims prev_dims = input_dims;
		for (unsigned i = 0; i < modules.size(); ++i) {
			std::pair<Module,bool>& module = this->modules[i];
			Module& module_net = module.first;
			if (i != 0)
				module_net.set_foremost(false);
			assert((!module.second || module_net.get_input_dims() == module_net.get_output_dims()) &&
					"residual module input-output dimension discrepancy");
			assert(prev_dims == module_net.get_input_dims() && "incompatible module dimensions");
			prev_dims = module_net.get_output_dims();
		}
	}
	inline Base* clone() const {
		return new ResidualNeuralNetwork(*this);
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline typename Base::Dims get_input_dims() const {
		return input_dims;
	}
	inline typename Base::Dims get_output_dims() const {
		return output_dims;
	}
protected:
	inline void set_foremost(bool foremost) {
		modules[0].first.set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		for (unsigned i = 0; i < modules.size(); ++i)
			modules[i].first.empty_caches();
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers;
		for (unsigned i = 0; i < modules.size(); ++i) {
			std::vector<Layer<Scalar,Rank>*> module_layers = modules[i].first.get_layers();
			for (unsigned j = 0; j < module_layers.size(); ++j)
				layers.push_back(module_layers[j]);
		}
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input).template demote<>());
		for (unsigned i = 0; i < modules.size(); ++i) {
			std::pair<Module,bool>& module = modules[i];
			if (module.second) // If it is a residual module, propagate the sum of the input and the output.
				input += module.first.propagate(input, training);
			else
				input = module.first.propagate(std::move(input), training);
		}
		return input;
	}
	inline Tensor<Scalar,Rank + 1> backpropagate(Tensor<Scalar,Rank + 1> out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads).template demote<>());
		for (int i = modules.size() - 1; i >= 0; --i) {
			std::pair<Module,bool>& module = modules[i];
			if (module.second)
				out_grads += module.first.backpropagate(out_grads);
			else
				out_grads = module.first.backpropagate(std::move(out_grads));
		}
		return out_grads;
	}
private:
	std::vector<std::pair<Module,bool>> modules;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
};

template<typename Scalar>
class DenseNeuralNetwork : public NeuralNetwork<Scalar,3,false> {
	typedef NeuralNetwork<Scalar,3,false> Base;
	typedef CompositeNeuralNetwork<Scalar,3,false> Module;
public:
	inline DenseNeuralNetwork(std::vector<Module> modules, bool foremost = true) :
			modules(modules),
			foremost(foremost) {
		assert(this->modules.size() > 0 && "modules must contain at least 1 element");
		Module& first_module = this->modules[0];
		input_dims = first_module.get_input_dims();
		typename Base::Dims output_dims = first_module.get_output_dims();
		output_dims(2) += input_dims(2);
		assert(input_dims(0) == output_dims(0) && input_dims(1) == output_dims(1));
		for (unsigned i = 1; i < this->modules.size(); ++i) {
			Module& module = this->modules[i];
			const typename Base::Dims& module_input_dims = module.get_input_dims();
			assert(module_input_dims == output_dims && "incompatible module dimensions");
			output_dims(2) += module.get_output_dims()(2);
			module.set_foremost(false);
		}
		this->output_dims = output_dims;
		first_module.set_foremost(foremost);
	}
	inline Base* clone() const {
		return new DenseNeuralNetwork(*this);
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline typename Base::Dims get_input_dims() const {
		return input_dims;
	}
	inline typename Base::Dims get_output_dims() const {
		return output_dims;
	}
protected:
	inline void set_foremost(bool foremost) {
		modules[0].set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		for (unsigned i = 0; i < modules.size(); ++i)
			modules[i].empty_caches();
	}
	inline std::vector<Layer<Scalar,3>*> get_layers() {
		std::vector<Layer<Scalar,3>*> layers;
		for (unsigned i = 0; i < modules.size(); ++i) {
			std::vector<Layer<Scalar,3>*> module_layers = modules[i].get_layers();
			for (unsigned j = 0; j < module_layers.size(); ++j)
				layers.push_back(module_layers[j]);
		}
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<4>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<4>(input).template demote<>());
		std::array<int,4> offsets({ 0, 0, 0, 0 });
		std::array<int,4> extents({ input.dimension(0), input_dims(0), input_dims(1), input_dims(2) });
		for (unsigned i = 0; i < modules.size(); ++i) {
			Module& module = modules[i];
			int layer_input_depth = module.get_input_dims()(2);
			int layer_output_depth = module.get_output_dims()(2);
			std::array<int,4> out_i_sizes = extents;
			out_i_sizes[3] = layer_input_depth + layer_output_depth;
			typename Base::Data out_i(out_i_sizes);
			offsets[3] = 0;
			extents[3] = layer_input_depth;
			out_i.slice(offsets, extents) = input;
			offsets[3] = layer_input_depth;
			extents[3] = layer_output_depth;
			out_i.slice(offsets, extents) = module.propagate(std::move(input), training);
			input = typename Base::Data(std::move(out_i));
		}
		return input;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<4>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<4>(out_grads).template demote<>());
		std::array<int,4> offsets({ 0, 0, 0, 0 });
		std::array<int,4> extents({ out_grads.dimension(0), input_dims(0), input_dims(1), input_dims(2) });
		for (int i = modules.size() - 1; i >= 0; --i) {
			Module& module = modules[i];
			int layer_input_depth = module.get_input_dims()(2);
			int layer_output_depth = module.get_output_dims()(2);
			offsets[3] = layer_input_depth;
			extents[3] = layer_output_depth;
			typename Base::Data out_grads_i = out_grads.slice(offsets, extents);
			offsets[3] = 0;
			extents[3] = layer_input_depth;
			out_grads = typename Base::Data(out_grads.slice(offsets, extents) +
					module.backpropagate(std::move(out_grads_i)));
		}
		return out_grads;
	}
	std::vector<Module> modules;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
};

template<typename Scalar>
class ParallelNeuralNetwork : public NeuralNetwork<Scalar,3,false> {
	typedef NeuralNetwork<Scalar,3,false> Base;
	typedef ParallelNeuralNetwork<Scalar> Self;
	typedef NeuralNetPtr<Scalar,3,false> Lane;
public:
	inline ParallelNeuralNetwork(std::vector<Lane> lanes, bool foremost = true) :
			lanes(std::move(lanes)) {
		assert(this->lanes.size() > 0 && "lanes must contain at least 1 element");
		assert(this->lanes[0] != nullptr && "lanes contains null pointers");
		Base& first_lane = *this->lanes[0];
		const typename Base::Dims& input_dims = first_lane.get_input_dims();
		typename Base::Dims output_dims = first_lane.get_output_dims();
		for (unsigned i = 1; i < this->lanes.size(); ++i) {
			assert(this->lanes[i] != nullptr && "lanes contains null pointers");
			Base& lane = *this->lanes[i];
			assert(input_dims == lane.get_input_dims());
			const typename Base::Dims& lane_output_dims = lane.get_output_dims();
			assert(output_dims(0) == lane_output_dims(0) &&
					output_dims(1) == lane_output_dims(1));
			output_dims(2) += lane_output_dims(2);
		}
		set_foremost(foremost);
		this->input_dims = first_lane.get_input_dims();
		this->output_dims = output_dims;
	}
	inline ParallelNeuralNetwork(Base&& lane, bool foremost = true) :
			ParallelNeuralNetwork(create_vector(std::move(lane)), foremost) { }
	inline ParallelNeuralNetwork(const Self& network) :
			lanes(network.lanes.size()) {
		for (unsigned i = 0; i < lanes.size(); ++i)
			lanes[i] = Lane(network.lanes[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	}
	inline ParallelNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~ParallelNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline Base* clone() const {
		return new ParallelNeuralNetwork(*this);
	}
	inline typename Base::Dims get_input_dims() const {
		return input_dims;
	}
	inline typename Base::Dims get_output_dims() const {
		return output_dims;
	}
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.lanes, network2.lanes);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	}
protected:
	inline void set_foremost(bool foremost) {
		for (unsigned i = 0; i < lanes.size(); ++i)
			lanes[i]->set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		for (unsigned i = 0; i < lanes.size(); ++i)
			lanes[i]->empty_caches();
	}
	inline std::vector<Layer<Scalar,3>*> get_layers() {
		std::vector<Layer<Scalar,3>*> layers;
		for (unsigned i = 0; i < lanes.size(); ++i) {
			std::vector<Layer<Scalar,3>*> lane_layers = lanes[i]->get_layers();
			for (unsigned j = 0; j < lane_layers.size(); ++j)
				layers.push_back(lane_layers[j]);
		}
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<4>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<4>(input).template demote<>());
		int rows = input.dimension(0);
		std::array<int,4> offsets({ 0, 0, 0, 0 });
		std::array<int,4> extents({ rows, output_dims(0), output_dims(1), output_dims(2) });
		typename Base::Data out(extents);
		unsigned lane_num = lanes.size();
		unsigned helper_thread_num = lane_num - 1;
		pthread_t threads[helper_thread_num];
		pthread_attr_t attr;
		if (helper_thread_num > 0) {
			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		}
		PropArgs args_arr[lane_num];
		for (int i = helper_thread_num; i >= 0; --i) {
			PropArgs args;
			args.obj = this;
			args.lane_id = i;
			args.training = training;
			args.in = &input;
			args_arr[i] = args;
			// Leave the first lane to the main thread.
			if (i == 0)
				propagate(&args_arr[i]);
			else
				assert(!pthread_create(&threads[i - 1], &attr, propagate, &args_arr[i]));
		}
		for (unsigned i = 0; i < lane_num; ++i) {
			if (i != 0)
				assert(!pthread_join(threads[i - 1], nullptr));
			int depth = lanes[i]->get_output_dims()(2);
			extents[3] = depth;
			out.slice(offsets, extents) = args_arr[i].out;
			offsets[3] += depth;
		}
		if (helper_thread_num > 0)
			pthread_attr_destroy(&attr);
		return out;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<4>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<4>(out_grads).template demote<>());
		typename Base::Data prev_out_grads;
		if (foremost)
			prev_out_grads = Utils<Scalar>::template get_null_tensor<4>();
		else {
			prev_out_grads = typename Base::Data(out_grads.dimension(0), input_dims(0),
					input_dims(1), input_dims(2));
			prev_out_grads.setZero();
		}
		unsigned lane_num = lanes.size();
		unsigned helper_thread_num = lane_num - 1;
		pthread_t threads[helper_thread_num];
		pthread_attr_t attr;
		if (helper_thread_num > 0) {
			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		}
		BackpropArgs args_arr[lane_num];
		int depth_offset = out_grads.dimension(3);
		for (int i = helper_thread_num; i >= 0; --i) {
			depth_offset -= lanes[i]->get_output_dims()(2);
			BackpropArgs args;
			args.obj = this;
			args.lane_id = i;
			args.depth_offset = depth_offset;
			args.out_grads = &out_grads;
			args_arr[i] = args;
			// Leave the first lane to the main thread.
			if (i == 0)
				backpropagate(&args_arr[i]);
			else
				assert(!pthread_create(&threads[i - 1], &attr, backpropagate, &args_arr[i]));
		}
		for (unsigned i = 0; i < lanes.size(); ++i) {
			if (i != 0)
				assert(!pthread_join(threads[i - 1], nullptr));
			if (!foremost)
				prev_out_grads += args_arr[i].prev_out_grads;
		}
		if (helper_thread_num > 0)
			pthread_attr_destroy(&attr);
		return prev_out_grads;
	}
	inline static std::vector<Lane> create_vector(Lane&& net) {
		std::vector<Lane> vec(1);
		vec[0] = std::move(net);
		return vec;
	}
private:
	std::vector<Lane> lanes;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
	struct PropArgs {
		Self* obj;
		int lane_id;
		bool training;
		typename Base::Data* in;
		typename Base::Data out;
	};
	struct BackpropArgs {
		Self* obj;
		int lane_id;
		int depth_offset;
		typename Base::Data* out_grads;
		typename Base::Data prev_out_grads;
	};
	inline static void* propagate(void* args_ptr) {
		PropArgs& args = *((PropArgs*) args_ptr);
		args.out = args.obj->lanes[args.lane_id]->propagate(*args.in, args.training);
		return nullptr;
	}
	inline static void* backpropagate(void* args_ptr) {
		BackpropArgs& args = *((BackpropArgs*) args_ptr);
		Base& lane = *args.obj->lanes[args.lane_id];
		const typename Base::Dims& lane_output_dims = lane.get_output_dims();
		std::array<int,4> offsets({ 0, 0, 0, args.depth_offset });
		std::array<int,4> extents({ args.out_grads->dimension(0), lane_output_dims(0),
				lane_output_dims(1), lane_output_dims(2) });
		typename Base::Data out_grads_slice = args.out_grads->slice(offsets, extents);
		args.prev_out_grads = lane.backpropagate(std::move(out_grads_slice));
		return nullptr;
	}
};

template<typename Scalar, size_t Rank>
using KernelPtr = std::unique_ptr<KernelLayer<Scalar,Rank>>;

template<typename Scalar, size_t Rank>
using ActivationPtr = std::unique_ptr<ActivationLayer<Scalar,Rank>>;

template<typename Scalar, size_t Rank>
class RecurrentNeuralNetwork : public NeuralNetwork<Scalar,Rank,true> {
	typedef NeuralNetwork<Scalar,Rank,true> Base;
	typedef RecurrentNeuralNetwork<Scalar,Rank> Self;
	typedef std::array<int,Base::DATA_RANKS> RankwiseArray;
	typedef std::function<std::pair<size_t,size_t>(size_t)> OutputSeqSizeFunc;
public:
	inline RecurrentNeuralNetwork(KernelPtr<Scalar,Rank> input_kernel, KernelPtr<Scalar,Rank> state_kernel,
			KernelPtr<Scalar,Rank> output_kernel, ActivationPtr<Scalar,Rank> state_act, ActivationPtr<Scalar,Rank> output_act,
			OutputSeqSizeFunc output_seq_size_func, bool stateful = false, bool mul_int = false, bool foremost = true) :
				main_cell(),
				output_seq_size_func(output_seq_size_func),
				stateful(stateful),
				mul_int(mul_int),
				foremost(foremost),
				cells(0),
				batch_size(-1),
				input_seq_length(-1),
				output_seq_length(-1),
				output_seq_delay(-1) {
		assert(input_kernel && state_kernel && output_kernel && state_act && output_act);
		typename Base::Dims input_layer_input_dims = input_kernel->get_input_dims();
		typename Base::Dims input_layer_output_dims = input_kernel->get_output_dims();
		typename Base::Dims output_layer_output_dims = output_kernel->get_output_dims();
		assert(input_layer_output_dims == state_kernel->get_output_dims() &&
				input_layer_output_dims == output_kernel->get_input_dims() &&
				input_layer_output_dims == state_act->get_input_dims() &&
				output_layer_output_dims == output_act->get_input_dims() &&
				state_kernel->get_input_dims() == state_kernel->get_output_dims());
		main_cell.input_kernel = std::move(input_kernel);
		main_cell.state_kernel = std::move(state_kernel);
		main_cell.output_kernel = std::move(output_kernel);
		main_cell.state_act = std::move(state_act);
		main_cell.output_act = std::move(output_act);
		input_dims = std::move(input_layer_input_dims);
		output_dims = std::move(output_layer_output_dims);
		set_foremost(foremost);
	}
	// Copy constructor.
	inline RecurrentNeuralNetwork(const Self& network) {
		main_cell.input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) network.main_cell.input_kernel->clone());
		main_cell.state_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) network.main_cell.state_kernel->clone());
		main_cell.output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) network.main_cell.output_kernel->clone());
		main_cell.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.state_act->clone());
		main_cell.output_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.output_act->clone());
		main_cell.state_cache = network.main_cell.state_cache;
		main_cell.u_cache = network.main_cell.u_cache;
		output_seq_size_func = network.output_seq_size_func;
		stateful = network.stateful;
		mul_int = network.mul_int;
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
		size_t cells_size = network.cells.size();
		cells = std::vector<Cell>(cells_size);
		for (size_t i = 0; i < cells_size; i++) {
			Cell& c1 = cells[i];
			const Cell& c2 = network.cells[i];
			c1.input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.input_kernel->clone());
			c1.state_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.state_kernel->clone());
			c1.output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.output_kernel->clone());
			c1.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.state_act->clone());
			c1.output_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.output_act->clone());
			c1.state_cache = c2.state_cache;
			c1.u_cache = c2.u_cache;
		}
		state = network.state;
		batch_size = network.batch_size;
		input_seq_length = network.input_seq_length;
		output_seq_length = network.output_seq_length;
		output_seq_delay = network.output_seq_delay;
	}
	inline RecurrentNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~RecurrentNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Base* clone() const {
		return new RecurrentNeuralNetwork(*this);
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline typename Base::Dims get_input_dims() const {
		return input_dims;
	}
	inline typename Base::Dims get_output_dims() const {
		return output_dims;
	}
	// For the copy-and-swap idiom.
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.main_cell, network2.main_cell);
		swap(network1.output_seq_size_func, network2.output_seq_size_func);
		swap(network1.stateful, network2.stateful);
		swap(network1.mul_int, network2.mul_int);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
		swap(network1.cells, network2.cells);
		swap(network1.state, network2.state);
		swap(network1.batch_size, network2.batch_size);
		swap(network1.input_seq_length, network2.input_seq_length);
		swap(network1.output_seq_length, network2.output_seq_length);
		swap(network1.output_seq_delay, network2.output_seq_delay);
	}
protected:
	inline void set_foremost(bool foremost) {
		Base::set_input_layer(*main_cell.input_kernel, foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		Base::empty_cache(*main_cell.input_kernel);
		Base::empty_cache(*main_cell.state_kernel);
		Base::empty_cache(*main_cell.output_kernel);
		Base::empty_cache(*main_cell.state_act);
		Base::empty_cache(*main_cell.output_act);
		cells = std::vector<Cell>(0);
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers(5);
		layers[0] = main_cell.input_kernel.get();
		layers[1] = main_cell.state_kernel.get();
		layers[2] = main_cell.output_kernel.get();
		layers[3] = main_cell.state_act.get();
		layers[4] = main_cell.output_act.get();
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		Dimensions<int,Base::DATA_RANKS> data_dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input);
		assert(input_dims == data_dims.template demote<2>());
		int samples = data_dims(0);
		int input_seq_length = data_dims(1);
		// Calculate the output sequence length and delay based on the input sequence length.
		std::pair<size_t,size_t> output_seq_info = output_seq_size_func((size_t) input_seq_length);
		int output_seq_length = (int) output_seq_info.first;
		int output_seq_delay = (int) output_seq_info.second;
		assert(output_seq_length > 0 && output_seq_length + output_seq_delay >= input_seq_length);
		int time_steps = output_seq_length + output_seq_delay;
		// If in training mode, unroll the updated network.
		if (training) {
			cells = std::vector<Cell>(time_steps - 1);
			if (time_steps > 1) {
				// Empty the caches of the main cell to reduce the amount of data to copy.
				Base::empty_cache(*main_cell.input_kernel);
				Base::empty_cache(*main_cell.state_kernel);
				Base::empty_cache(*main_cell.output_kernel);
				Base::empty_cache(*main_cell.state_act);
				Base::empty_cache(*main_cell.output_act);
				// Unroll the network by creating n -1 copies of the main cell;
				for (int j = 1; j < time_steps; ++j) {
					Cell& cell = cells[j - 1];
					cell.state_kernel = KernelPtr<Scalar,Rank>(
							(KernelLayer<Scalar,Rank>*) main_cell.state_kernel->clone());
					cell.state_act = ActivationPtr<Scalar,Rank>(
							(ActivationLayer<Scalar,Rank>*) main_cell.state_act->clone());
					// Only copy the kernels and activations that will actually be used.
					if (j < input_seq_length)
						cell.input_kernel = KernelPtr<Scalar,Rank>(
								(KernelLayer<Scalar,Rank>*) main_cell.input_kernel->clone());
					if (j >= output_seq_delay) {
						cell.output_kernel = KernelPtr<Scalar,Rank>(
								(KernelLayer<Scalar,Rank>*) main_cell.output_kernel->clone());
						cell.output_act = ActivationPtr<Scalar,Rank>(
								(ActivationLayer<Scalar,Rank>*) main_cell.output_act->clone());
					}
				}
			}
		}
		// If the network is stateful and we are in training mode, retain the state.
		if (!training || !stateful || batch_size == -1) {
			Dimensions<int,Rank + 1> dims = main_cell.input_kernel->get_output_dims().template promote<>();
			dims(0) = samples;
			state = Tensor<Scalar,Rank + 1>(dims);
			state.setZero();
		} else if (samples != batch_size) {
			std::array<int,Rank + 1> offsets;
			std::array<int,Rank + 1> extents = main_cell.input_kernel->get_output_dims().template promote<>();
			offsets.fill(0);
			extents[0] = samples;
			Tensor<Scalar,Rank + 1> new_state;
			if (samples > batch_size) {
				new_state = Tensor<Scalar,Rank + 1>(extents);
				new_state.setZero();
				extents[0] = batch_size;
				new_state.slice(offsets, extents) = state;
			} else
				new_state = state.slice(offsets, extents);
			state = std::move(new_state);
		}
		RankwiseArray input_offsets;
		RankwiseArray input_extents = data_dims;
		RankwiseArray output_offsets;
		RankwiseArray output_extents = output_dims.template promote<2>();
		input_offsets.fill(0);
		output_offsets.fill(0);
		input_extents[1] = 1;
		output_extents[0] = samples;
		typename Base::Data out;
		// If the output is a single time step prediction, there is no need to create an output tensor.
		if (output_seq_length > 1) {
			output_extents[1] = output_seq_length;
			out = typename Base::Data(output_extents);
		}
		output_extents[1] = 1;
		Dimensions<int,Rank + 1> input_time_step_dims = input_dims.template promote<>();
		input_time_step_dims(0) = samples;
		for (int i = 0; i < time_steps; ++i) {
			Cell* cell;
			// In inference mode, do not unroll the network.
			if (!training)
				cell = &main_cell;
			else if (i == 0) {
				cell = &main_cell;
			} else
				cell = &cells[i - 1];
			// Always apply the state kernel.
			state = Base::pass_forward(*cell->state_kernel, std::move(state), training);
			// If in inference mode, empty the caches after passing the data through each layer.
			if (!training)
				Base::empty_cache(*cell->state_kernel);
			// If there is an input for the time step...
			if (i < input_seq_length) {
				typename Base::Data in_i_seq;
				if (input_seq_length == 1)
					in_i_seq = std::move(input);
				else {
					in_i_seq = input.slice(input_offsets, input_extents);
					input_offsets[1] += 1;
				}
				if (mul_int) {
					if (training) {
						/* If multiplicative integration is enabled, cache the factors of the multiplication so that
						 * the function can be differentiated in the backward pass. */
						cell->state_cache = state;
						cell->u_cache = Base::pass_forward(*cell->input_kernel,
								Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
										input_time_step_dims), training);
						state *= cell->u_cache;
					} else
						state *= Base::pass_forward(*cell->input_kernel,
								Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
										input_time_step_dims), training);
				} else
					state += Base::pass_forward(*cell->input_kernel,
							Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
								input_time_step_dims), training);
				if (!training)
					Base::empty_cache(*cell->input_kernel);
			}
			state = Base::pass_forward(*cell->state_act, std::move(state), training);
			if (!training)
				Base::empty_cache(*cell->state_act);
			// If there is an output for the time step...
			if (i >= output_seq_delay) {
				Tensor<Scalar,Rank + 1> out_i = Base::pass_forward(*cell->output_kernel, state, training);
				if (!training)
					Base::empty_cache(*cell->output_kernel);
				// If the output is a single time step prediction, just return it.
				if (output_seq_length == 1)
					out = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
							Base::pass_forward(*cell->output_act, std::move(out_i), training),
							output_extents);
				else {
					out.slice(output_offsets, output_extents) = Utils<Scalar>::template map_tensor_to_tensor<
							Rank + 1,Base::DATA_RANKS>(Base::pass_forward(*cell->output_act, std::move(out_i),
									training), output_extents);
					output_offsets[1] += 1;
				}
				if (!training)
					Base::empty_cache(*cell->output_act);
			}
		}
		batch_size = samples;
		this->input_seq_length = input_seq_length;
		this->output_seq_length = output_seq_length;
		this->output_seq_delay = output_seq_delay;
		return out;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		Dimensions<int,Base::DATA_RANKS> data_dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads);
		assert(output_dims == data_dims.template demote<2>() && batch_size == data_dims(0) &&
				output_seq_length == data_dims(1));
		RankwiseArray output_offsets;
		RankwiseArray output_extents = data_dims;
		RankwiseArray input_offsets;
		RankwiseArray input_extents = input_dims.template promote<2>();
		output_offsets.fill(0);
		output_offsets[1] = output_seq_length - 1;
		input_offsets.fill(0);
		input_offsets[1] = input_seq_length - 1;
		output_extents[1] = 1;
		input_extents[0] = batch_size;
		typename Base::Data prev_out_grads;
		if (input_seq_length > 1) {
			input_extents[1] = input_seq_length;
			prev_out_grads = typename Base::Data(input_extents);
		}
		input_extents[1] = 1;
		Tensor<Scalar,Rank + 1> state_grads(state.dimensions());
		state_grads.setZero();
		Dimensions<int,Rank + 1> out_time_step_dims = output_dims.template promote<>();
		out_time_step_dims(0) = batch_size;
		int time_steps = output_seq_length + output_seq_delay;
		for (int i = time_steps - 1; i >= 0; --i) {
			Cell& cell = i == 0 ? main_cell : cells[i - 1];
			// If there was an output at the time step...
			if (i >= output_seq_delay) {
				typename Base::Data out_grads_seq_i;
				if (output_seq_length == 1)
					out_grads_seq_i = std::move(out_grads);
				else {
					out_grads_seq_i = out_grads.slice(output_offsets, output_extents);
					output_offsets[1] -= 1;
				}
				Tensor<Scalar,Rank + 1> out_grads_i = Base::pass_back(*cell.output_act,
						Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_RANKS,Rank + 1>(
								std::move(out_grads_seq_i), out_time_step_dims));
				Base::empty_cache(*cell.output_act);
				state_grads += Base::pass_back(*cell.output_kernel, std::move(out_grads_i));
				Base::empty_cache(*cell.output_kernel);
			}
			// Always back-propagate the state gradient.
			state_grads = Base::pass_back(*cell.state_act, std::move(state_grads));
			Base::empty_cache(*cell.state_act);
			// If there was an input at the time step...
			if (i < input_seq_length) {
				// If it is the foremost layer, the gradients do not need to be propagated further back.
				if (foremost) {
					if (mul_int) { // Multiplicative integration.
						Base::pass_back(*cell.input_kernel, cell.state_cache * state_grads);
						cell.state_cache = Utils<Scalar>::template get_null_tensor<Rank + 1>();
					} else // Additive integration.
						Base::pass_back(*cell.input_kernel, state_grads);
				} else if (input_seq_length == 1) {
					if (mul_int) {
						prev_out_grads = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
								Base::pass_back(*cell.input_kernel, cell.state_cache * state_grads), input_extents);
						cell.state_cache = Utils<Scalar>::template get_null_tensor<Rank + 1>();
					} else
						prev_out_grads = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
								Base::pass_back(*cell.input_kernel, state_grads), input_extents);
				} else {
					if (mul_int) {
						prev_out_grads.slice(input_offsets, input_extents) =
								Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
										Base::pass_back(*cell.input_kernel, cell.state_cache * state_grads), input_extents);
						cell.state_cache = Utils<Scalar>::template get_null_tensor<Rank + 1>();
					} else
						prev_out_grads.slice(input_offsets, input_extents) =
								Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
										Base::pass_back(*cell.input_kernel, state_grads), input_extents);
					input_offsets[1] -= 1;
				}
				Base::empty_cache(*cell.input_kernel);
			}
			// Compute the gradients w.r.t. the state kernel.
			if (mul_int) {
				state_grads = Base::pass_back(*cell.state_kernel, cell.u_cache * state_grads);
				cell.u_cache = Utils<Scalar>::template get_null_tensor<Rank + 1>();
			} else
				state_grads = Base::pass_back(*cell.state_kernel, std::move(state_grads));
			Base::empty_cache(*cell.state_kernel);
		}
		// Roll the network up and accumulate the gradients.
		auto u_param_grads = Base::get_param_grads(*main_cell.input_kernel);
		auto w_param_grads = Base::get_param_grads(*main_cell.state_kernel);
		auto v_param_grads = Base::get_param_grads(*main_cell.output_kernel);
		auto state_act_param_grads = Base::get_param_grads(*main_cell.state_act);
		auto output_act_param_grads = Base::get_param_grads(*main_cell.output_act);
		for (int i = 1; i < time_steps; ++i) {
			Cell& cell = cells[i - 1];
			w_param_grads += Base::get_param_grads(*cell.state_kernel);
			state_act_param_grads += Base::get_param_grads(*cell.state_act);
			if (i < input_seq_length)
				u_param_grads += Base::get_param_grads(*cell.input_kernel);
			if (i >= output_seq_delay) {
				v_param_grads += Base::get_param_grads(*cell.output_kernel);
				output_act_param_grads += Base::get_param_grads(*cell.output_act);
			}
		}
		Base::get_param_grads(*main_cell.input_kernel) = u_param_grads;
		Base::get_param_grads(*main_cell.state_kernel) = w_param_grads;
		Base::get_param_grads(*main_cell.output_kernel) = v_param_grads;
		Base::get_param_grads(*main_cell.state_act) = state_act_param_grads;
		Base::get_param_grads(*main_cell.output_act) = output_act_param_grads;
		cells = std::vector<Cell>(0);
		return prev_out_grads;
	}
private:
	struct Cell {
		KernelPtr<Scalar,Rank> input_kernel;
		KernelPtr<Scalar,Rank> state_kernel;
		KernelPtr<Scalar,Rank> output_kernel;
		ActivationPtr<Scalar,Rank> state_act;
		ActivationPtr<Scalar,Rank> output_act;
		// State and input caches for multiplicative integration.
		Tensor<Scalar,Rank + 1> state_cache;
		Tensor<Scalar,Rank + 1> u_cache;
	};
	Cell main_cell;
	OutputSeqSizeFunc output_seq_size_func;
	bool stateful;
	bool mul_int;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
	// State.
	std::vector<Cell> cells;
	Tensor<Scalar,Rank + 1> state;
	int batch_size;
	int input_seq_length;
	int output_seq_length;
	int output_seq_delay;
};

template<typename Scalar, size_t Rank>
class LSTMNeuralNetwork : public NeuralNetwork<Scalar,Rank,true> {
	typedef NeuralNetwork<Scalar,Rank,true> Base;
	typedef LSTMNeuralNetwork<Scalar,Rank> Self;
	typedef std::array<int,Base::DATA_RANKS> RankwiseArray;
	typedef std::function<std::pair<size_t,size_t>(size_t)> OutputSeqSizeFunc;
public:
	inline LSTMNeuralNetwork(KernelPtr<Scalar,Rank> h_forget_kernel, KernelPtr<Scalar,Rank> x_forget_kernel,
			KernelPtr<Scalar,Rank> h_input_kernel, KernelPtr<Scalar,Rank> x_input_kernel,
			KernelPtr<Scalar,Rank> h_candidate_kernel, KernelPtr<Scalar,Rank> x_candidate_kernel,
			KernelPtr<Scalar,Rank> h_output_kernel, KernelPtr<Scalar,Rank> x_output_kernel,
			ActivationPtr<Scalar,Rank> forget_act, ActivationPtr<Scalar,Rank> input_act,
			ActivationPtr<Scalar,Rank> candidate_act, ActivationPtr<Scalar,Rank> state_act,
			ActivationPtr<Scalar,Rank> output_act, OutputSeqSizeFunc output_seq_size_func,
			bool stateful = false, bool mul_int = false, bool foremost = true) :
				main_cell(),
				output_seq_size_func(output_seq_size_func),
				stateful(stateful),
				mul_int(mul_int),
				foremost(foremost),
				cells(0),
				batch_size(-1),
				input_seq_length(-1),
				output_seq_length(-1),
				output_seq_delay(-1) {
		assert(h_forget_kernel && x_forget_kernel && h_input_kernel && x_input_kernel &&
				h_candidate_kernel && x_candidate_kernel && h_output_kernel && x_output_kernel &&
				forget_act && input_act && candidate_act && state_act && output_act);
		assert(h_forget_kernel->get_input_dims() == output_act->get_output_dims());
		main_cell.h_forget_kernel = std::move(h_forget_kernel);
		main_cell.h_input_kernel = std::move(h_input_kernel);
		main_cell.h_candidate_kernel = std::move(h_candidate_kernel);
		main_cell.h_output_kernel = std::move(h_output_kernel);
		main_cell.x_forget_kernel = std::move(x_forget_kernel);
		main_cell.x_input_kernel = std::move(x_input_kernel);
		main_cell.x_candidate_kernel = std::move(x_candidate_kernel);
		main_cell.x_output_kernel = std::move(x_output_kernel);
		main_cell.forget_act = std::move(forget_act);
		main_cell.input_act = std::move(input_act);
		main_cell.candidate_act = std::move(candidate_act);
		main_cell.state_act = std::move(state_act);
		main_cell.output_act = std::move(output_act);
		input_dims = std::move(u_layer_input_dims);
		output_dims = std::move(v_layer_output_dims);
		set_foremost(foremost);
	}
	// Copy constructor.
	inline LSTMNeuralNetwork(const Self& network) {
		main_cell.h_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.h_forget_kernel->clone());
		main_cell.x_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.x_forget_kernel->clone());
		main_cell.h_input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.h_input_kernel->clone());
		main_cell.x_input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.x_input_kernel->clone());
		main_cell.h_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.h_candidate_kernel->clone());
		main_cell.x_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.x_candidate_kernel->clone());
		main_cell.h_output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.h_output_kernel->clone());
		main_cell.x_output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.x_output_kernel->clone());
		main_cell.forget_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.forget_act->clone());
		main_cell.input_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.input_act->clone());
		main_cell.candidate_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.candidate_act->clone());
		main_cell.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.state_act->clone());
		main_cell.output_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.output_act->clone());
		output_seq_size_func = network.output_seq_size_func;
		stateful = network.stateful;
		mul_int = network.mul_int;
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
		size_t cells_size = network.cells.size();
		cells = std::vector<Cell>(cells_size);
		for (size_t i = 0; i < cells_size; i++) {
			Cell& c1 = cells[i];
			const Cell& c2 = network.cells[i];
			c1.h_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.h_forget_kernel->clone());
			c1.x_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.x_forget_kernel->clone());
			c1.h_input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.h_input_kernel->clone());
			c1.x_input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.x_input_kernel->clone());
			c1.h_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.h_candidate_kernel->clone());
			c1.x_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.x_candidate_kernel->clone());
			c1.h_output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.h_output_kernel->clone());
			c1.x_output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.x_output_kernel->clone());
			c1.input_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.input_act->clone());
			c1.forget_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.forget_act->clone());
			c1.candidate_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.candidate_act->clone());
			c1.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.state_act->clone());
			c1.output_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.output_act->clone());
		}
		state = network.state;
		batch_size = network.batch_size;
		input_seq_length = network.input_seq_length;
		output_seq_length = network.output_seq_length;
		output_seq_delay = network.output_seq_delay;
	}
	inline LSTMNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~LSTMNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Base* clone() const {
		return new LSTMNeuralNetwork(*this);
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline typename Base::Dims get_input_dims() const {
		return input_dims;
	}
	inline typename Base::Dims get_output_dims() const {
		return output_dims;
	}
	// For the copy-and-swap idiom.
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.main_cell, network2.main_cell);
		swap(network1.output_seq_size_func, network2.output_seq_size_func);
		swap(network1.stateful, network2.stateful);
		swap(network1.mul_int, network2.mul_int);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
		swap(network1.cells, network2.cells);
		swap(network1.state, network2.state);
		swap(network1.batch_size, network2.batch_size);
		swap(network1.input_seq_length, network2.input_seq_length);
		swap(network1.output_seq_length, network2.output_seq_length);
		swap(network1.output_seq_delay, network2.output_seq_delay);
	}
protected:
	inline void set_foremost(bool foremost) {
		Base::set_input_layer(*main_cell.h_forget_kernel, foremost);
		Base::set_input_layer(*main_cell.x_forget_kernel, foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		Base::empty_cache(*main_cell.h_forget_kernel);
		Base::empty_cache(*main_cell.x_forget_kernel);
		Base::empty_cache(*main_cell.h_input_kernel);
		Base::empty_cache(*main_cell.x_input_kernel);
		Base::empty_cache(*main_cell.h_candidate_kernel);
		Base::empty_cache(*main_cell.x_candidate_kernel);
		Base::empty_cache(*main_cell.h_output_kernel);
		Base::empty_cache(*main_cell.x_output_kernel);
		Base::empty_cache(*main_cell.input_act);
		Base::empty_cache(*main_cell.forget_act);
		Base::empty_cache(*main_cell.candidate_act);
		Base::empty_cache(*main_cell.state_act);
		Base::empty_cache(*main_cell.output_act);
		cells = std::vector<Cell>(0);
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers(13);
		layers[0] = main_cell.h_forget_kernel.get();
		layers[1] = main_cell.x_forget_kernel.get();
		layers[2] = main_cell.h_input_kernel.get();
		layers[3] = main_cell.x_input_kernel.get();
		layers[4] = main_cell.h_candidate_kernel.get();
		layers[5] = main_cell.x_candidate_kernel.get();
		layers[6] = main_cell.h_output_kernel.get();
		layers[7] = main_cell.x_output_kernel.get();
		layers[8] = main_cell.forget_act.get();
		layers[5] = main_cell.input_act.get();
		layers[6] = main_cell.candidate_act.get();
		layers[7] = main_cell.output_act.get();
		layers[8] = main_cell.state_act.get();
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		Dimensions<int,Base::DATA_RANKS> data_dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input);
		assert(input_dims == data_dims.template demote<2>());
		int samples = data_dims(0);
		int input_seq_length = data_dims(1);
		// Calculate the output sequence length and delay based on the input sequence length.
		std::pair<size_t,size_t> output_seq_info = output_seq_size_func((size_t) input_seq_length);
		int output_seq_length = (int) output_seq_info.first;
		int output_seq_delay = (int) output_seq_info.second;
		assert(output_seq_length > 0 && output_seq_length + output_seq_delay >= input_seq_length);
		int time_steps = output_seq_length + output_seq_delay;
		// If in training mode, unroll the updated network.
		if (training) {
			cells = std::vector<Cell>(time_steps - 1);
			if (time_steps > 1) {
				// Empty the caches of the main cell to reduce the amount of data to copy.
				Base::empty_cache(*main_cell.forget_kernel);
				Base::empty_cache(*main_cell.input_kernel);
				Base::empty_cache(*main_cell.candidate_kernel);
				Base::empty_cache(*main_cell.state_act);
				Base::empty_cache(*main_cell.output_act);
				// Unroll the network by creating n -1 copies of the main cell;
				for (int j = 1; j < time_steps; ++j) {
					Cell& cell = cells[j - 1];
					cell.candidate_kernel = KernelPtr<Scalar,Rank>(
							(KernelLayer<Scalar,Rank>*) main_cell.candidate_kernel->clone());
					cell.state_act = ActivationPtr<Scalar,Rank>(
							(ActivationLayer<Scalar,Rank>*) main_cell.state_act->clone());
					// Only copy the kernels and activations that will actually be used.
					if (j < input_seq_length)
						cell.forget_kernel = KernelPtr<Scalar,Rank>(
								(KernelLayer<Scalar,Rank>*) main_cell.forget_kernel->clone());
					if (j >= output_seq_delay) {
						cell.input_kernel = KernelPtr<Scalar,Rank>(
								(KernelLayer<Scalar,Rank>*) main_cell.input_kernel->clone());
						cell.output_act = ActivationPtr<Scalar,Rank>(
								(ActivationLayer<Scalar,Rank>*) main_cell.output_act->clone());
					}
				}
			}
		}
		// If the network is stateful and we are in training mode, retain the state.
		if (!training || !stateful || batch_size == -1) {
			Dimensions<int,Rank + 1> dims = main_cell.forget_kernel->get_output_dims().template promote<>();
			dims(0) = samples;
			state = Tensor<Scalar,Rank + 1>(dims);
			state.setZero();
		} else if (samples != batch_size) {
			std::array<int,Rank + 1> offsets;
			std::array<int,Rank + 1> extents = main_cell.forget_kernel->get_output_dims().template promote<>();
			offsets.fill(0);
			extents[0] = samples;
			Tensor<Scalar,Rank + 1> new_state;
			if (samples > batch_size) {
				new_state = Tensor<Scalar,Rank + 1>(extents);
				new_state.setZero();
				extents[0] = batch_size;
				new_state.slice(offsets, extents) = state;
			} else
				new_state = state.slice(offsets, extents);
			state = std::move(new_state);
		}
		RankwiseArray input_offsets;
		RankwiseArray input_extents = data_dims;
		RankwiseArray output_offsets;
		RankwiseArray output_extents = output_dims.template promote<2>();
		input_offsets.fill(0);
		output_offsets.fill(0);
		input_extents[1] = 1;
		output_extents[0] = samples;
		typename Base::Data out;
		// If the output is a single time step prediction, there is no need to create an output tensor.
		if (output_seq_length > 1) {
			output_extents[1] = output_seq_length;
			out = typename Base::Data(output_extents);
		}
		output_extents[1] = 1;
		Dimensions<int,Rank + 1> input_time_step_dims = input_dims.template promote<>();
		input_time_step_dims(0) = samples;
		for (int i = 0; i < time_steps; ++i) {
			Cell* cell;
			// In inference mode, do not unroll the network.
			if (!training)
				cell = &main_cell;
			else if (i == 0) {
				cell = &main_cell;
			} else
				cell = &cells[i - 1];
			// Always apply the state kernel.
			state = Base::pass_forward(*cell->candidate_kernel, std::move(state), training);
			// If in inference mode, empty the caches after passing the data through each layer.
			if (!training)
				Base::empty_cache(*cell->candidate_kernel);
			// If there is an input for the time step...
			if (i < input_seq_length) {
				typename Base::Data in_i_seq;
				if (input_seq_length == 1)
					in_i_seq = std::move(input);
				else {
					in_i_seq = input.slice(input_offsets, input_extents);
					input_offsets[1] += 1;
				}
				if (mul_int) {
					if (training) {
						/* If multiplicative integration is enabled, cache the factors of the multiplication so that
						 * the function can be differentiated in the backward pass. */
						cell->state_cache = state;
						cell->u_cache = Base::pass_forward(*cell->forget_kernel,
								Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
										input_time_step_dims), training);
						state *= cell->u_cache;
					} else
						state *= Base::pass_forward(*cell->forget_kernel,
								Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
										input_time_step_dims), training);
				} else
					state += Base::pass_forward(*cell->forget_kernel,
							Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
								input_time_step_dims), training);
				if (!training)
					Base::empty_cache(*cell->forget_kernel);
			}
			state = Base::pass_forward(*cell->state_act, std::move(state), training);
			if (!training)
				Base::empty_cache(*cell->state_act);
			// If there is an output for the time step...
			if (i >= output_seq_delay) {
				Tensor<Scalar,Rank + 1> out_i = Base::pass_forward(*cell->input_kernel, state, training);
				if (!training)
					Base::empty_cache(*cell->input_kernel);
				// If the output is a single time step prediction, just return it.
				if (output_seq_length == 1)
					out = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
							Base::pass_forward(*cell->output_act, std::move(out_i), training),
							output_extents);
				else {
					out.slice(output_offsets, output_extents) = Utils<Scalar>::template map_tensor_to_tensor<
							Rank + 1,Base::DATA_RANKS>(Base::pass_forward(*cell->output_act, std::move(out_i),
									training), output_extents);
					output_offsets[1] += 1;
				}
				if (!training)
					Base::empty_cache(*cell->output_act);
			}
		}
		batch_size = samples;
		this->input_seq_length = input_seq_length;
		this->output_seq_length = output_seq_length;
		this->output_seq_delay = output_seq_delay;
		return out;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		Dimensions<int,Base::DATA_RANKS> data_dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads);
		assert(output_dims == data_dims.template demote<2>() && batch_size == data_dims(0) &&
				output_seq_length == data_dims(1));
		RankwiseArray output_offsets;
		RankwiseArray output_extents = data_dims;
		RankwiseArray input_offsets;
		RankwiseArray input_extents = input_dims.template promote<2>();
		output_offsets.fill(0);
		output_offsets[1] = output_seq_length - 1;
		input_offsets.fill(0);
		input_offsets[1] = input_seq_length - 1;
		output_extents[1] = 1;
		input_extents[0] = batch_size;
		typename Base::Data prev_out_grads;
		if (input_seq_length > 1) {
			input_extents[1] = input_seq_length;
			prev_out_grads = typename Base::Data(input_extents);
		}
		input_extents[1] = 1;
		Tensor<Scalar,Rank + 1> state_grads(state.dimensions());
		state_grads.setZero();
		Dimensions<int,Rank + 1> out_time_step_dims = output_dims.template promote<>();
		out_time_step_dims(0) = batch_size;
		int time_steps = output_seq_length + output_seq_delay;
		for (int i = time_steps - 1; i >= 0; --i) {
			Cell& cell = i == 0 ? main_cell : cells[i - 1];
			// If there was an output at the time step...
			if (i >= output_seq_delay) {
				typename Base::Data out_grads_seq_i;
				if (output_seq_length == 1)
					out_grads_seq_i = std::move(out_grads);
				else {
					out_grads_seq_i = out_grads.slice(output_offsets, output_extents);
					output_offsets[1] -= 1;
				}
				Tensor<Scalar,Rank + 1> out_grads_i = Base::pass_back(*cell.output_act,
						Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_RANKS,Rank + 1>(
								std::move(out_grads_seq_i), out_time_step_dims));
				Base::empty_cache(*cell.output_act);
				state_grads += Base::pass_back(*cell.input_kernel, std::move(out_grads_i));
				Base::empty_cache(*cell.input_kernel);
			}
			// Always back-propagate the state gradient.
			state_grads = Base::pass_back(*cell.state_act, std::move(state_grads));
			Base::empty_cache(*cell.state_act);
			// If there was an input at the time step...
			if (i < input_seq_length) {
				// If it is the foremost layer, the gradients do not need to be propagated further back.
				if (foremost) {
					if (mul_int) { // Multiplicative integration.
						Base::pass_back(*cell.forget_kernel, cell.state_cache * state_grads);
						cell.state_cache = Utils<Scalar>::template get_null_tensor<Rank + 1>();
					} else // Additive integration.
						Base::pass_back(*cell.forget_kernel, state_grads);
				} else if (input_seq_length == 1) {
					if (mul_int) {
						prev_out_grads = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
								Base::pass_back(*cell.forget_kernel, cell.state_cache * state_grads), input_extents);
						cell.state_cache = Utils<Scalar>::template get_null_tensor<Rank + 1>();
					} else
						prev_out_grads = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
								Base::pass_back(*cell.forget_kernel, state_grads), input_extents);
				} else {
					if (mul_int) {
						prev_out_grads.slice(input_offsets, input_extents) =
								Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
										Base::pass_back(*cell.forget_kernel, cell.state_cache * state_grads), input_extents);
						cell.state_cache = Utils<Scalar>::template get_null_tensor<Rank + 1>();
					} else
						prev_out_grads.slice(input_offsets, input_extents) =
								Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_RANKS>(
										Base::pass_back(*cell.forget_kernel, state_grads), input_extents);
					input_offsets[1] -= 1;
				}
				Base::empty_cache(*cell.forget_kernel);
			}
			// Compute the gradients w.r.t. the state kernel.
			if (mul_int) {
				state_grads = Base::pass_back(*cell.candidate_kernel, cell.u_cache * state_grads);
				cell.u_cache = Utils<Scalar>::template get_null_tensor<Rank + 1>();
			} else
				state_grads = Base::pass_back(*cell.candidate_kernel, std::move(state_grads));
			Base::empty_cache(*cell.candidate_kernel);
		}
		// Roll the network up and accumulate the gradients.
		auto u_param_grads = Base::get_param_grads(*main_cell.forget_kernel);
		auto v_param_grads = Base::get_param_grads(*main_cell.input_kernel);
		auto w_param_grads = Base::get_param_grads(*main_cell.candidate_kernel);
		auto state_act_param_grads = Base::get_param_grads(*main_cell.state_act);
		auto output_act_param_grads = Base::get_param_grads(*main_cell.output_act);
		for (int i = 1; i < time_steps; ++i) {
			Cell& cell = cells[i - 1];
			w_param_grads += Base::get_param_grads(*cell.candidate_kernel);
			state_act_param_grads += Base::get_param_grads(*cell.state_act);
			if (i < input_seq_length)
				u_param_grads += Base::get_param_grads(*cell.forget_kernel);
			if (i >= output_seq_delay) {
				v_param_grads += Base::get_param_grads(*cell.input_kernel);
				output_act_param_grads += Base::get_param_grads(*cell.output_act);
			}
		}
		Base::get_param_grads(*main_cell.forget_kernel) = u_param_grads;
		Base::get_param_grads(*main_cell.input_kernel) = v_param_grads;
		Base::get_param_grads(*main_cell.candidate_kernel) = w_param_grads;
		Base::get_param_grads(*main_cell.state_act) = state_act_param_grads;
		Base::get_param_grads(*main_cell.output_act) = output_act_param_grads;
		cells = std::vector<Cell>(0);
		return prev_out_grads;
	}
private:
	struct Cell {
		KernelPtr<Scalar,Rank> h_forget_kernel;
		KernelPtr<Scalar,Rank> x_forget_kernel;
		KernelPtr<Scalar,Rank> h_input_kernel;
		KernelPtr<Scalar,Rank> x_input_kernel;
		KernelPtr<Scalar,Rank> h_candidate_kernel;
		KernelPtr<Scalar,Rank> x_candidate_kernel;
		KernelPtr<Scalar,Rank> h_output_kernel;
		KernelPtr<Scalar,Rank> x_output_kernel;
		ActivationPtr<Scalar,Rank> input_act;
		ActivationPtr<Scalar,Rank> forget_act;
		ActivationPtr<Scalar,Rank> candidate_act;
		ActivationPtr<Scalar,Rank> state_act;
		ActivationPtr<Scalar,Rank> output_act;
	};
	Cell main_cell;
	OutputSeqSizeFunc output_seq_size_func;
	bool stateful;
	bool mul_int;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
	// State.
	std::vector<Cell> cells;
	Tensor<Scalar,Rank + 1> state;
	int batch_size;
	int input_seq_length;
	int output_seq_length;
	int output_seq_delay;
};

/**
 * Enables the use of non-sequential networks on sequential data by joining the 'samples' and 'time steps' ranks of
 * the tensors and splitting them again once the internal, non-sequential network is done processing them.
 */
template<typename Scalar, size_t Rank>
class SequentialNeuralNetwork : public NeuralNetwork<Scalar,Rank,true> {
	typedef NeuralNetwork<Scalar,Rank,true> Base;
	typedef SequentialNeuralNetwork<Scalar,Rank> Self;
	typedef NeuralNetPtr<Scalar,Rank,false> Net;
	typedef std::array<int,Base::DATA_RANKS> RankwiseArray;
public:
	inline SequentialNeuralNetwork(Net network, bool foremost = true) :
			net(std::move(network)),
			foremost(foremost),
			batch_size(-1) {
		assert(net);
		input_dims = net->get_input_dims();
		output_dims = net->get_output_dims();
		set_foremost(foremost);
	}
	inline SequentialNeuralNetwork(const Self& network) {
		net = Net(network.net->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
		batch_size = network.batch_size;
	}
	inline SequentialNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~SequentialNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Base* clone() const {
		return new SequentialNeuralNetwork(*this);
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline typename Base::Dims get_input_dims() const {
		return input_dims;
	}
	inline typename Base::Dims get_output_dims() const {
		return output_dims;
	}
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.net, network2.net);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
		swap(network1.batch_size, network2.batch_size);
	}
protected:
	inline void set_foremost(bool foremost) {
		net->set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		net->empty_caches();
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		return net->get_layers();
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input).template demote<2>());
		batch_size = input.dimension(0);
		int seq_length = input.dimension(1);
		return Utils<Scalar>::template split_first_rank<Base::DATA_RANKS - 1>(
				net->propagate(Utils<Scalar>::template join_first_two_ranks<Base::DATA_RANKS>(std::move(input)), training),
				batch_size, seq_length);
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads).template demote<2>());
		assert(batch_size == out_grads.dimension(0));
		int seq_length = out_grads.dimension(1);
		return Utils<Scalar>::template split_first_rank<Base::DATA_RANKS - 1>(
				net->backpropagate(Utils<Scalar>::template join_first_two_ranks<Base::DATA_RANKS>(std::move(out_grads))),
				batch_size, seq_length);
	}
private:
	Net net;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
	int batch_size;
};

} /* namespace cattle */

#endif /* NEURALNETWORK_H_ */
