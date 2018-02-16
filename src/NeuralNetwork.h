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
#include <iomanip>
#include <Layer.h>
#include <pthread.h>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

// TODO Possibility to add and remove modules (e.g. layers for sequential networks, inception modules for InceptionNets).
// TODO Serialization.

namespace cattle {

// Forward declaration to Optimizer and CompositeNeuralNetwork so they can be friended.
template<typename Scalar, size_t Rank, bool Sequential> class Optimizer;
template<typename Scalar, bool Sequential> class ParallelNeuralNetwork;
template<typename Scalar, size_t Rank, bool Sequential> class CompositeNeuralNetwork;

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
	static_assert(Rank > 0 && Rank < 4, "illegal neural network rank");
	friend class Optimizer<Scalar,Rank,Sequential>;
	friend class ParallelNeuralNetwork<Scalar,Sequential>;
	friend class CompositeNeuralNetwork<Scalar,Rank,Sequential>;
protected:
	static constexpr size_t DATA_DIMS = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_DIMS> Data;
	typedef Dimensions<int,Rank> Dims;
public:
	virtual ~NeuralNetwork() = default;
	virtual NeuralNetwork<Scalar,Rank,Sequential>* clone() const = 0;
	virtual bool is_foremost() const = 0;
	virtual Dims get_input_dims() const = 0;
	virtual Dims get_output_dims() const = 0;
	virtual void init() {
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i]->init();
	};
	inline virtual Data infer(Data input) {
		return propagate(std::move(input), false);
	};
	virtual std::string to_string() {
		std::stringstream strm;
		strm << "Neural Net " << this << std::endl;
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar,Rank>* layer = layers[i];
			strm << "\tLayer " << std::setw(3) << std::to_string(i + 1) << std::string(28, '-') << std::endl;
			strm << "\t\tinput dims: " << layer->get_input_dims().to_string() << std::endl;
			strm << "\t\toutput dims: " << layer->get_output_dims().to_string() << std::endl;
			if (layer->is_parametric()) {
				strm << "\t\tparams:" << std::endl;
				Matrix<Scalar>& params = layer->get_params();
				for (int j = 0; j < params.rows(); j++) {
					strm << "\t\t[ ";
					for (int k = 0; k < params.cols(); k++) {
						strm << std::setw(11) << std::setprecision(4) << params(j,k);
						if (k != params.cols() - 1)
							strm << ", ";
					}
					strm << " ]" << std::endl;
				}
			}
		}
		return strm.str();
	};
	friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<Scalar,Rank,Sequential>& nn) {
		return os << nn.to_string() << std::endl;
	};
protected:
	virtual void set_foremost(bool foremost) = 0;
	virtual std::vector<Layer<Scalar,Rank>*> get_layers() = 0;
	virtual Data propagate(Data input, bool training) = 0;
	virtual Data backpropagate(Data out_grads) = 0;
	inline static void set_input_layer(Layer<Scalar,Rank>& layer, bool on) {
		layer.input_layer = on;
	};
	inline static void empty_cache(Layer<Scalar,Rank>& layer) {
		layer.empty_cache();
	};
	inline static Data pass_forward(Layer<Scalar,Rank>& layer, Data&& prev_out, bool training) {
		return layer.pass_forward(std::move(prev_out), training);
	};
	inline static Data pass_back(Layer<Scalar,Rank>& layer, Data&& out_grads) {
		return layer.pass_back(std::move(out_grads));
	};
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
	FeedforwardNeuralNetwork(std::vector<LayerPtr<Scalar,Rank>> layers, bool foremost = true) :
			layers(std::move(layers)),
			foremost(foremost) {
		assert(this->layers.size() > 0 && "layers must contain at least 1 element");
		assert(this->layers[0] != nullptr);
		Layer<Scalar,Rank>& first_layer = *(this->layers[0]);
		input_dims = first_layer.get_input_dims();
		output_dims = this->layers[this->layers.size() - 1]->get_output_dims();
		typename Base::Dims prev_dims = first_layer.get_output_dims();
		for (unsigned i = 1; i < this->layers.size(); i++) {
			assert(this->layers[i] != nullptr && "layers contains null pointers");
			assert(prev_dims == this->layers[i]->get_input_dims() && "incompatible layer dimensions");
			prev_dims = this->layers[i]->get_output_dims();
		}
		Base::set_input_layer(first_layer, foremost);
	};
	FeedforwardNeuralNetwork(LayerPtr<Scalar,Rank> layer, bool foremost = true) :
			FeedforwardNeuralNetwork(create_vector(std::move(layer)), foremost) { };
	// Copy constructor.
	FeedforwardNeuralNetwork(const Self& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i] = LayerPtr<Scalar,Rank>(network.layers[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	// Move constructor.
	FeedforwardNeuralNetwork(Self&& network) {
		swap(*this, network);
	};
	// The smart pointers take care of deleting the layers.
	~FeedforwardNeuralNetwork() = default;
	/* The assignment uses the move or copy constructor to pass the parameter
	 * based on whether it is an rvalue or an lvalue. */
	Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	};
	Base* clone() const {
		return new FeedforwardNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	typename Base::Dims get_input_dims() const {
		return input_dims;
	};
	typename Base::Dims get_output_dims() const {
		return output_dims;
	};
	// For the copy-and-swap idiom.
	friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	};
protected:
	inline void set_foremost(bool foremost) {
		Base::set_input_layer(*layers[0], foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers_raw(layers.size());
		for (unsigned i = 0; i < layers.size(); i++)
			layers_raw[i] = layers[i].get();
		return layers_raw;
	};
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_tensor_dims<Rank + 1>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Rank + 1>(input).template demote<>());
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar,Rank>& layer = *layers[i];
			input = Base::pass_forward(layer, std::move(input), training);
			if (!training)
				Base::empty_cache(layer);
		}
		return input;
	};
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_tensor_dims<Rank + 1>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Rank + 1>(out_grads).template demote<>());
		for (int i = layers.size() - 1; i >= 0; i--) {
			Layer<Scalar,Rank>& layer = *layers[i];
			out_grads = Base::pass_back(layer, std::move(out_grads));
			Base::empty_cache(layer);
		}
		return out_grads;
	};
	static std::vector<LayerPtr<Scalar,Rank>> create_vector(LayerPtr<Scalar,Rank> layer) {
		std::vector<LayerPtr<Scalar,Rank>> vec(1);
		vec[0] = std::move(layer);
		return vec;
	};
	std::vector<LayerPtr<Scalar,Rank>> layers;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
};

template<typename Scalar, size_t Rank>
using KernelPtr = std::unique_ptr<KernelLayer<Scalar,Rank>>;

template<typename Scalar, size_t Rank>
using ActivationPtr = std::unique_ptr<ActivationLayer<Scalar,Rank>>;

template<typename Scalar, size_t Rank>
class RecurrentNeuralNetwork : public NeuralNetwork<Scalar,Rank,true> {
	typedef NeuralNetwork<Scalar,Rank,true> Base;
	typedef RecurrentNeuralNetwork<Scalar,Rank> Self;
public:
	RecurrentNeuralNetwork(KernelPtr<Scalar,Rank> u_kernel, KernelPtr<Scalar,Rank> v_kernel,
			KernelPtr<Scalar,Rank> w_kernel, ActivationPtr<Scalar,Rank> state_act, ActivationPtr<Scalar,Rank> output_act,
			size_t input_seq_length, size_t output_seq_length, size_t output_seq_delay, bool stateful = false,
			bool mul_integration = false, bool foremost = true) :
				u_kernel(std::move(u_kernel)),
				v_kernel(std::move(v_kernel)),
				w_kernel(std::move(w_kernel)),
				state_act(std::move(state_act)),
				output_act(std::move(output_act)),
				input_seq_length(input_seq_length),
				output_seq_length(output_seq_length),
				output_seq_delay(output_seq_delay),
				stateful(stateful),
				mul_integration(mul_integration),
				foremost(foremost),
				batch_size(-1) {
		assert(input_seq_length > 0 && output_seq_length > 0 && output_seq_length + output_seq_delay >= input_seq_length &&
				"illegal input-output sequence sizes");
		assert(this->u_kernel && this->v_kernel && this->w_kernel && this->state_act && this->output_act);
		typename Base::Dims layer_input_dims = this->u_kernel->get_input_dims();
		typename Base::Dims layer_output_dims = this->u_kernel->get_output_dims();
		assert(layer_output_dims == this->w_kernel->get_output_dims() &&
				layer_output_dims == this->v_kernel->get_input_dims() &&
				layer_output_dims == this->state_act->get_input_dims() &&
				this->w_kernel->get_input_dims() == this->w_kernel->get_output_dims() &&
				this->output_act->get_input_dims() == this->v_kernel->get_output_dims());
		typename Base::Dims input_dims = layer_input_dims.template promote<>();
		typename Base::Dims output_dims = layer_output_dims.template promote<>();
		input_dims(0) = input_seq_length;
		output_dims(0) = output_seq_length;
		this->input_dims = input_dims;
		this->output_dims = output_dims;
		set_foremost(foremost);
	};
	// Copy constructor.
	RecurrentNeuralNetwork(const Self& network) {
		u_kernel = KernelPtr<Scalar,Rank>(network.u_kernel->clone());
		v_kernel = KernelPtr<Scalar,Rank>(network.v_kernel->clone());
		w_kernel = KernelPtr<Scalar,Rank>(network.w_kernel->clone());
		state_act = ActivationPtr<Scalar,Rank>(network.state_act->clone());
		output_act = ActivationPtr<Scalar,Rank>(network.output_act->clone());
		input_seq_length = network.input_seq_length;
		output_seq_length = network.output_seq_length;
		output_seq_delay = network.output_seq_delay;
		stateful = network.stateful;
		mul_integration = network.mul_integration;
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
		state = network.state;
	};
	RecurrentNeuralNetwork(Self&& network) {
		swap(*this, network);
	};
	~RecurrentNeuralNetwork() = default;
	Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	};
	Base* clone() const {
		return new RecurrentNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	typename Base::Dims get_input_dims() const {
		return input_dims;
	};
	typename Base::Dims get_output_dims() const {
		return output_dims;
	};
	// For the copy-and-swap idiom.
	friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.u_kernel, network2.u_kernel);
		swap(network1.v_kernel, network2.v_kernel);
		swap(network1.w_kernel, network2.w_kernel);
		swap(network1.state_act, network2.state_act);
		swap(network1.output_act, network2.output_act);
		swap(network1.input_seq_length, network2.input_seq_length);
		swap(network1.output_seq_length, network2.output_seq_length);
		swap(network1.output_seq_delay, network2.output_seq_delay);
		swap(network1.stateful, network2.stateful);
		swap(network1.mul_integration, network2.mul_integration);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
		swap(network1.state, network2.state);
	};
protected:
	inline void set_foremost(bool foremost) {
		Base::set_input_layer(*w_kernel, foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers_raw(5);
		layers_raw[0] = u_kernel->get();
		layers_raw[1] = v_kernel->get();
		layers_raw[2] = w_kernel->get();
		layers_raw[3] = state_act->get();
		layers_raw[4] = output_act->get();
		return layers_raw;
	};
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_tensor_dims<Base::DATA_DIMS>(input);
		Dimensions<int,Base::DATA_DIMS> data_dims = Utils<Scalar>::template get_dims<Base::DATA_DIMS>(input);
		assert(input_dims == data_dims.template demote<>());
		int samples = data_dims(0);
		if (!training || !stateful || samples != batch_size) {
			Dimensions<int,Rank + 1> dims = u_kernel->get_output_dims().template promote<>();
			dims(0) = samples;
			state = Tensor<Scalar,Rank + 1>(dims);
			state.setZero();
		}
		batch_size == samples;
		std::array<int,Base::DATA_DIMS> offsets;
		std::array<int,Base::DATA_DIMS> input_extents = data_dims;
		std::array<int,Base::DATA_DIMS> output_extents = output_dims.template promote<>();
		offsets.fill(0);
		input_extents(1) = 1;
		output_extents(0) = samples;
		output_extents(1) = 1;
		typename Base::Data out;
		if (output_seq_length > 1) {
			Dimensions<int,Base::DATA_DIMS> out_dims = output_dims.template promote<>();
			out_dims(0) = samples;
			out = typename Base::Data(out_dims);
		}
		int time_steps = output_seq_length + output_seq_delay;
		for (int i = 0; i < time_steps; i++) {
			offsets[1] = i;
			if (i < input_seq_length) {
				typename Base::Data in_i_seq = input.slice(offsets, input_extents);
				Dimensions<int,Rank + 1> dims = input_dims;
				dims(0) = samples;
				Tensor<Scalar,Rank + 1> in_i = Base::pass_forward(*u_kernel,
						Utils<Scalar>::template map_tensor_to_tensor<Base::DATA_DIMS,Rank + 1>(std::move(in_i_seq),
								samples), training);
				if (!training)
					Base::empty_cache(*u_kernel);
				if (mul_integration)
					in_i *= Base::pass_forward(*w_kernel, state, training);
				else
					in_i += Base::pass_forward(*w_kernel, state, training);
				if (!training)
					Base::empty_cache(*w_kernel);
				state = Base::pass_forward(*state_act, std::move(in_i), training);
			} else
				state = Base::pass_forward(*state_act, std::move(state), training);
			if (!training)
				Base::empty_cache(*state_act);
			if (i >= output_seq_delay) {
				Tensor<Scalar,Rank + 1> out_i = Base::pass_forward(*v_kernel, state, training);
				if (!training)
					Base::empty_cache(*v_kernel);
				out_i = Base::pass_forward(*output_act, std::move(out_i), training);
				if (!training)
					Base::empty_cache(*output_act);
				if (output_seq_length == 1 && i == time_steps - 1) {
					Dimensions<int,Base::DATA_DIMS> out_dims = output_dims.template promote<>();
					out_dims(0) = samples;
					return Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Base::DATA_DIMS>(out_i,
							out_dims);
				} else
					out.slice(offsets, output_extents) = out_i;
			}
		}
		return out;
	};
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_tensor_dims<Base::DATA_DIMS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_DIMS>(out_grads).template demote<>());

		return out_grads;
	};
private:
	const KernelPtr<Scalar,Rank> u_kernel;
	const KernelPtr<Scalar,Rank> v_kernel;
	const KernelPtr<Scalar,Rank> w_kernel;
	const ActivationPtr<Scalar,Rank> state_act;
	const ActivationPtr<Scalar,Rank> output_act;
	const size_t input_seq_length;
	const size_t output_seq_length;
	const size_t output_seq_delay;
	const bool stateful;
	const bool mul_integration;
	const bool foremost;
	const typename Base::Dims input_dims;
	const typename Base::Dims output_dims;
	// State.
	Tensor<Scalar,Rank + 1> state;
	int batch_size;
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
	CompositeNeuralNetwork(std::vector<Block> blocks, bool foremost = true) :
			blocks(std::move(blocks)),
			foremost(foremost) {
		assert(this->blocks.size() > 0 && "blocks must contain at least 1 element");
		assert(this->blocks[0] != nullptr && "blocks contains null pointers");
		Base& first_block = *(this->blocks[0]);
		input_dims = first_block.get_input_dims();
		output_dims = this->blocks[this->blocks.size() - 1]->get_output_dims();
		typename Base::Dims prev_dims = first_block.get_output_dims();
		for (unsigned i = 1; i < this->blocks.size(); i++) {
			assert(this->blocks[i] != nullptr && "blocks contains null pointers");
			Base& block = *this->blocks[i];
			assert(prev_dims == block.get_input_dims() && "incompatible network dimensions");
			block.set_foremost(false);
			prev_dims = block.get_output_dims();
		}
		first_block.set_foremost(foremost);
	};
	CompositeNeuralNetwork(Block&& block, bool foremost = true) :
			CompositeNeuralNetwork(create_vector(std::move(block)),
					foremost) { };
	CompositeNeuralNetwork(const Self& network) :
			blocks(network.blocks.size()) {
		for (unsigned i = 0; i < blocks.size(); i++)
			blocks[i] = Block(network.blocks[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	CompositeNeuralNetwork(Self&& network) {
		swap(*this, network);
	};
	~CompositeNeuralNetwork() = default;
	Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	};
	Base* clone() const {
		return new CompositeNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	typename Base::Dims get_input_dims() const {
		return input_dims;
	};
	typename Base::Dims get_output_dims() const {
		return output_dims;
	};
	friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.blocks, network2.blocks);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	};
protected:
	inline void set_foremost(bool foremost) {
		blocks[0]->set_foremost(foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers;
		for (unsigned i = 0; i < blocks.size(); i++) {
			std::vector<Layer<Scalar,Rank>*> internal_layers = blocks[i]->get_layers();
			for (unsigned j = 0; j < internal_layers.size(); j++)
				layers.push_back(internal_layers[j]);
		}
		return layers;
	};
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_tensor_dims<Base::DATA_DIMS>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Base::DATA_DIMS>(input).template demote<>());
		for (unsigned i = 0; i < blocks.size(); i++)
			input = blocks[i]->propagate(std::move(input), training);
		return input;
	};
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_tensor_dims<Base::DATA_DIMS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_DIMS>(out_grads).template demote<>());
		for (int i = blocks.size() - 1; i >= 0; i--)
			out_grads = blocks[i]->backpropagate(std::move(out_grads));
		return out_grads;
	};
	static std::vector<Block> create_vector(Block&& net) {
		std::vector<Block> vec(1);
		vec[0] = std::move(net);
		return vec;
	};
private:
	std::vector<Block> blocks;
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
	ParallelNeuralNetwork(std::vector<Lane> lanes, bool foremost = true) :
			lanes(std::move(lanes)) {
		assert(this->lanes.size() > 0 && "lanes must contain at least 1 element");
		assert(this->lanes[0] != nullptr && "lanes contains null pointers");
		Base& first_lane = *this->lanes[0];
		const typename Base::Dims& input_dims = first_lane.get_input_dims();
		typename Base::Dims output_dims = first_lane.get_output_dims();
		for (unsigned i = 1; i < this->lanes.size(); i++) {
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
	};
	ParallelNeuralNetwork(Base&& lane, bool foremost = true) :
			ParallelNeuralNetwork(create_vector(std::move(lane)), foremost) { };
	ParallelNeuralNetwork(const Self& network) :
			lanes(network.lanes.size()) {
		for (unsigned i = 0; i < lanes.size(); i++)
			lanes[i] = Lane(network.lanes[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	ParallelNeuralNetwork(Self&& network) {
		swap(*this, network);
	};
	~ParallelNeuralNetwork() = default;
	Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	};
	bool is_foremost() const {
		return foremost;
	};
	Base* clone() const {
		return new ParallelNeuralNetwork(*this);
	};
	typename Base::Dims get_input_dims() const {
		return input_dims;
	};
	typename Base::Dims get_output_dims() const {
		return output_dims;
	};
	friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.lanes, network2.lanes);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	};
protected:
	inline void set_foremost(bool foremost) {
		for (unsigned i = 0; i < lanes.size(); i++)
			lanes[i]->set_foremost(foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar,3>*> get_layers() {
		std::vector<Layer<Scalar,3>*> layers;
		for (unsigned i = 0; i < lanes.size(); i++) {
			std::vector<Layer<Scalar,3>*> lane_layers = lanes[i]->get_layers();
			for (unsigned j = 0; j < lane_layers.size(); j++)
				layers.push_back(lane_layers[j]);
		}
		return layers;
	};
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_tensor_dims<4>(input);
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
		for (int i = helper_thread_num; i >= 0; i--) {
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
		for (unsigned i = 0; i < lane_num; i++) {
			if (i != 0)
				assert(!pthread_join(threads[i - 1], nullptr));
			int depth = lanes[i]->get_output_dims()(Sequential + 2);
			extents[3] = depth;
			out.slice(offsets, extents) = args_arr[i].out;
			offsets[3] += depth;
		}
		if (helper_thread_num > 0)
			pthread_attr_destroy(&attr);
		return out;
	};
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_tensor_dims<4>(out_grads);
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
		for (int i = helper_thread_num; i >= 0; i--) {
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
		for (unsigned i = 0; i < lanes.size(); i++) {
			if (i != 0)
				assert(!pthread_join(threads[i - 1], nullptr));
			if (!foremost)
				prev_out_grads += args_arr[i].prev_out_grads;
		}
		if (helper_thread_num > 0)
			pthread_attr_destroy(&attr);
		return prev_out_grads;
	};
	static std::vector<Lane> create_vector(Lane&& net) {
		std::vector<Lane> vec(1);
		vec[0] = std::move(net);
		return vec;
	};
private:
	std::vector<Lane> lanes;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
	static void* propagate(void* args_ptr) {
		PropArgs& args = *((PropArgs*) args_ptr);
		args.out = args.obj->lanes[args.lane_id]->propagate(*args.in, args.training);
		return nullptr;
	};
	static void* backpropagate(void* args_ptr) {
		BackpropArgs& args = *((BackpropArgs*) args_ptr);
		Base& lane = *args.obj->lanes[args.lane_id];
		const Dimensions<int> lane_output_dims = lane.get_output_dims();
		std::array<int,4> offsets({ 0, 0, 0, args.depth_offset });
		std::array<int,4> extents({ args.out_grads->dimension(0), lane_output_dims(0),
				lane_output_dims(1), lane_output_dims(2) });
		typename Base::Data out_grads_slice = args.out_grads->slice(offsets, extents);
		args.prev_out_grads = lane.backpropagate(std::move(out_grads_slice));
		return nullptr;
	};
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
	ResidualNeuralNetwork(std::vector<std::pair<Module,bool>> modules, bool foremost = true) :
			modules(modules),
			foremost(foremost) {
		assert(modules.size() > 0 && "modules must contain at least 1 element");
		Module& first_module = this->modules[0].first;
		input_dims = first_module.get_input_dims();
		output_dims = this->modules[modules.size() - 1].first.get_output_dims();
		first_module.set_foremost(foremost);
		typename Base::Dims prev_dims = input_dims;
		for (unsigned i = 0; i < modules.size(); i++) {
			std::pair<Module,bool>& module = this->modules[i];
			Module& module_net = module.first;
			if (i != 0)
				module_net.set_foremost(false);
			assert((!module.second || module_net.get_input_dims() == module_net.get_output_dims()) &&
					"residual module input-output dimension discrepancy");
			assert(prev_dims == module_net.get_input_dims() && "incompatible module dimensions");
			prev_dims = module_net.get_output_dims();
		}
	};
	Base* clone() const {
		return new ResidualNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	typename Base::Dims get_input_dims() const {
		return input_dims;
	};
	typename Base::Dims get_output_dims() const {
		return output_dims;
	};
protected:
	inline void set_foremost(bool foremost) {
		modules[0].first.set_foremost(foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers;
		for (unsigned i = 0; i < modules.size(); i++) {
			std::vector<Layer<Scalar,Rank>*> module_layers = modules[i].first.get_layers();
			for (unsigned j = 0; j < module_layers.size(); j++)
				layers.push_back(module_layers[j]);
		}
		return layers;
	};
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_tensor_dims<Base::DATA_DIMS>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Base::DATA_DIMS>(input).template demote<>());
		for (unsigned i = 0; i < modules.size(); i++) {
			std::pair<Module,bool>& module = modules[i];
			if (module.second) // If it is a residual module, propagate the sum of the input and the output.
				input += module.first.propagate(input, training);
			else
				input = module.first.propagate(std::move(input), training);
		}
		return input;
	};
	inline Tensor<Scalar,Rank + 1> backpropagate(Tensor<Scalar,Rank + 1> out_grads) {
		Utils<Scalar>::template check_tensor_dims<Base::DATA_DIMS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_DIMS>(out_grads).template demote<>());
		for (int i = modules.size() - 1; i >= 0; i--) {
			std::pair<Module,bool>& module = modules[i];
			if (module.second)
				out_grads += module.first.backpropagate(out_grads);
			else
				out_grads = module.first.backpropagate(std::move(out_grads));
		}
		return out_grads;
	};
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
	DenseNeuralNetwork(std::vector<Module> modules, bool foremost = true) :
			modules(modules),
			foremost(foremost) {
		assert(this->modules.size() > 0 && "modules must contain at least 1 element");
		Module& first_module = this->modules[0];
		input_dims = first_module.get_input_dims();
		typename Base::Dims output_dims = first_module.get_output_dims();
		output_dims(2) += input_dims(2);
		assert(input_dims(0) == output_dims(0) && input_dims(1) == output_dims(1));
		for (unsigned i = 1; i < this->modules.size(); i++) {
			Module& module = this->modules[i];
			const typename Base::Dims& module_input_dims = module.get_input_dims();
			assert(module_input_dims == output_dims && "incompatible module dimensions");
			output_dims(2) += module.get_output_dims()(2);
			module.set_foremost(false);
		}
		this->output_dims = output_dims;
		first_module.set_foremost(foremost);
	};
	Base* clone() const {
		return new DenseNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	typename Base::Dims get_input_dims() const {
		return input_dims;
	};
	typename Base::Dims get_output_dims() const {
		return output_dims;
	};
protected:
	inline void set_foremost(bool foremost) {
		modules[0].set_foremost(foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar,3>*> get_layers() {
		std::vector<Layer<Scalar,3>*> layers;
		for (unsigned i = 0; i < modules.size(); i++) {
			std::vector<Layer<Scalar,3>*> module_layers = modules[i].get_layers();
			for (unsigned j = 0; j < module_layers.size(); j++)
				layers.push_back(module_layers[j]);
		}
		return layers;
	};
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_tensor_dims<4>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<4>(input).template demote<>());
		std::array<int,4> offsets({ 0, 0, 0, 0 });
		std::array<int,4> extents({ input.dimension(0), input_dims(0), input_dims(1), input_dims(2) });
		for (unsigned i = 0; i < modules.size(); i++) {
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
	};
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_tensor_dims<4>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<4>(out_grads).template demote<>());
		std::array<int,4> offsets({ 0, 0, 0, 0 });
		std::array<int,4> extents({ out_grads.dimension(0), input_dims(0), input_dims(1), input_dims(2) });
		for (int i = modules.size() - 1; i >= 0; i--) {
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
	};
	std::vector<Module> modules;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
};

} /* namespace cattle */

#endif /* NEURALNETWORK_H_ */
