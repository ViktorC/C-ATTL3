/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

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
template<typename Scalar, size_t Rank> class Optimizer;
template<typename Scalar, size_t Rank> class ParallelNeuralNetwork;
template<typename Scalar, size_t Rank> class CompositeNeuralNetwork;

/**
 * A neural network class that consists of a vector of neuron layers. It allows
 * for the calculation of the output of the network given an input vector and
 * it provides a member function for back-propagating the derivative of a loss
 * function with respect to the activated output of the network's last layer.
 * This back-propagation sets the gradients of the parameters associated with each
 * layer.
 */
template<typename Scalar, size_t Rank>
class NeuralNetwork {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal neural network rank");
	friend class Optimizer<Scalar,Rank>;
	friend class ParallelNeuralNetwork<Scalar,Rank>;
	friend class CompositeNeuralNetwork<Scalar,Rank>;
public:
	virtual ~NeuralNetwork() = default;
	virtual NeuralNetwork<Scalar,Rank>* clone() const = 0;
	virtual bool is_foremost() const = 0;
	virtual Dimensions<int,Rank> get_input_dims() const = 0;
	virtual Dimensions<int,Rank> get_output_dims() const = 0;
	virtual void init() {
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i]->init();
	};
	inline virtual Tensor<Scalar,Rank + 1> infer(Tensor<Scalar,Rank + 1> input) {
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
	friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<Scalar,Rank>& nn) {
		return os << nn.to_string() << std::endl;
	};
protected:
	virtual void set_foremost(bool foremost) = 0;
	virtual std::vector<Layer<Scalar,Rank>*> get_layers() = 0;
	virtual Tensor<Scalar,Rank + 1> propagate(Tensor<Scalar,Rank + 1> input, bool training) = 0;
	virtual Tensor<Scalar,Rank + 1> backpropagate(Tensor<Scalar,Rank + 1> out_grads) = 0;
	inline static void set_input_layer(Layer<Scalar,Rank>& layer, bool on) {
		layer.input_layer = on;
	};
	inline static void empty_cache(Layer<Scalar,Rank>& layer) {
		layer.empty_cache();
	};
	inline static Tensor<Scalar,Rank + 1> pass_forward(Layer<Scalar,Rank>& layer, Tensor<Scalar,Rank + 1>&& prev_out, bool training) {
		return layer.pass_forward(prev_out, training);
	};
	inline static Tensor<Scalar,Rank + 1> pass_back(Layer<Scalar,Rank>& layer, Tensor<Scalar,Rank + 1>&& out_grads) {
		return layer.pass_back(out_grads);
	};
};

template<typename Scalar, size_t Rank>
using LayerPtr = std::unique_ptr<Layer<Scalar,Rank>>;

template<typename Scalar, size_t Rank>
class SequentialNeuralNetwork : public NeuralNetwork<Scalar,Rank> {
public:
	/**
	 * Constructs the network using the provided layer pointers. It takes ownership of the layer pointers.
	 *
	 * @param layers A vector of unique smart pointers to the layers that constitute the neural network.
	 * @param foremost Whether the network directly receives its input. If it is set to false, back-propagation
	 * returns an empty tensor.
	 */
	SequentialNeuralNetwork(std::vector<LayerPtr<Scalar,Rank>> layers, bool foremost = true) :
			layers(std::move(layers)),
			foremost(foremost) {
		assert(this->layers.size() > 0 && "layers must contain at least 1 element");
		assert(this->layers[0] != nullptr);
		Layer<Scalar,Rank>& first_layer = *(this->layers[0]);
		input_dims = first_layer.get_input_dims();
		output_dims = this->layers[this->layers.size() - 1]->get_output_dims();
		Dimensions<int,Rank> prev_dims = first_layer.get_output_dims();
		for (unsigned i = 1; i < this->layers.size(); i++) {
			assert(this->layers[i] != nullptr && "layers contains null pointers");
			assert(prev_dims.equals(this->layers[i]->get_input_dims()) && "incompatible layer dimensions");
			prev_dims = this->layers[i]->get_output_dims();
		}
		NeuralNetwork<Scalar,Rank>::set_input_layer(first_layer, foremost);
	};
	SequentialNeuralNetwork(LayerPtr<Scalar,Rank> layer, bool foremost = true) :
			SequentialNeuralNetwork(create_vector(std::move(layer)), foremost) { };
	// Copy constructor.
	SequentialNeuralNetwork(const SequentialNeuralNetwork<Scalar,Rank>& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i] = LayerPtr<Scalar,Rank>(network.layers[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	// Move constructor.
	SequentialNeuralNetwork(SequentialNeuralNetwork<Scalar,Rank>&& network) {
		swap(*this, network);
	};
	// The smart pointers take care of deleting the layers.
	~SequentialNeuralNetwork() = default;
	/* The assignment uses the move or copy constructor to pass the parameter
	 * based on whether it is an rvalue or an lvalue. */
	SequentialNeuralNetwork<Scalar,Rank>& operator=(SequentialNeuralNetwork<Scalar,Rank> network) {
		swap(*this, network);
		return *this;
	};
	NeuralNetwork<Scalar,Rank>* clone() const {
		return new SequentialNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	Dimensions<int,Rank> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int,Rank> get_output_dims() const {
		return output_dims;
	};
	// For the copy-and-swap idiom.
	friend void swap(SequentialNeuralNetwork<Scalar,Rank>& network1,
			SequentialNeuralNetwork<Scalar,Rank>& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	};
protected:
	inline void set_foremost(bool foremost) {
		NeuralNetwork<Scalar,Rank>::set_input_layer(*layers[0], foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers_raw(layers.size());
		for (unsigned i = 0; i < layers.size(); i++)
			layers_raw[i] = layers[i].get();
		return layers_raw;
	};
	inline Tensor<Scalar,Rank + 1> propagate(Tensor<Scalar,Rank + 1> input, bool training) {
		assert(input.dimension(0));
		assert(input_dims == Utils<Scalar>::get_dims(input).demote());
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar,Rank>& layer = *layers[i];
			input = NeuralNetwork<Scalar,Rank>::pass_forward(layer, std::move(input), training);
			if (!training)
				NeuralNetwork<Scalar,Rank>::empty_cache(layer);
		}
		return input;
	};
	inline Tensor<Scalar,Rank + 1> backpropagate(Tensor<Scalar,Rank + 1> out_grads) {
		assert(out_grads.dimension(0));
		assert(output_dims == Utils<Scalar>::get_dims(out_grads).demote());
		for (int i = layers.size() - 1; i >= 0; i--) {
			Layer<Scalar,Rank>& layer = *(layers[i]);
			out_grads = NeuralNetwork<Scalar,Rank>::pass_back(layer, std::move(out_grads));
			NeuralNetwork<Scalar,Rank>::empty_cache(layer);
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
	Dimensions<int,Rank> input_dims;
	Dimensions<int,Rank> output_dims;
};

template<typename Scalar, size_t Rank>
using KernelPtr = std::unique_ptr<FCLayer<Scalar>>;

template<typename Scalar, size_t Rank>
using ActivationPtr = std::unique_ptr<ActivationLayer<Scalar>>;

//template<typename Scalar, size_t Rank>
//class RecurrentNeuralNetwork : public NeuralNetwork<Scalar,Rank> {
//	RecurrentNeuralNetwork(KernelPtr<Scalar,Rank> u_kernel, KernelPtr<Scalar,Rank> v_kernel, KernelPtr<Scalar,Rank> w_kernel,
//			ActivationPtr<Scalar,Rank> state_act, ActivationPtr<Scalar,Rank> output_act, size_t seq_length,
//			bool mul_integration = false, bool foremost = true) :
//				u_kernel(std::move(u_kernel)),
//				v_kernel(std::move(v_kernel)),
//				w_kernel(std::move(w_kernel)),
//				state_act(std::move(state_act)),
//				output_act(std::move(output_act)),
//				seq_length(seq_length),
//				mul_integration(mul_integration),
//				foremost(foremost) {
//		assert(seq_length > 0);
//		assert(this->u_kernel != nullptr);
//		assert(this->v_kernel != nullptr);
//		assert(this->w_kernel != nullptr);
//		assert(this->state_act != nullptr);
//		assert(this->output_act != nullptr);
//		Dimensions<int,Rank> layer_input_dims = this->u_kernel->get_input_dims();
//		Dimensions<int,Rank> layer_output_dims = this->u_kernel->get_output_dims();
//		assert(layer_input_dims(0) == 1);
//		assert(layer_output_dims == this->w_kernel->get_output_dims());
//		assert(layer_output_dims == this->v_kernel->get_input_dims());
//		assert(layer_output_dims == this->state_act->get_input_dims());
//		assert(this->w_kernel->get_input_dims() == this->w_kernel->get_output_dims());
//		assert(this->output_act->get_input_dims() == this->v_kernel->get_output_dims());
//		input_dims = Dimensions<int,Rank>(seq_length, layer_input_dims(1), layer_input_dims(2));
//		output_dims = layer_output_dims;
//		set_foremost(foremost);
//	};
//	// Copy constructor.
//	RecurrentNeuralNetwork(const RecurrentNeuralNetwork<Scalar,Rank>& network) {
//		u_kernel = KernelPtr<Scalar,Rank>(network.u_kernel->clone());
//		v_kernel = KernelPtr<Scalar,Rank>(network.v_kernel->clone());
//		w_kernel = KernelPtr<Scalar,Rank>(network.w_kernel->clone());
//		state_act = ActivationPtr<Scalar,Rank>(network.state_act->clone());
//		output_act = ActivationPtr<Scalar,Rank>(network.output_act->clone());
//		seq_length = network.seq_length;
//		mul_integration = network.mul_integration;
//		foremost = network.foremost;
//		input_dims = network.input_dims;
//		output_dims = network.output_dims;
//		state = network.state;
//	};
//	RecurrentNeuralNetwork(RecurrentNeuralNetwork<Scalar,Rank>&& network) {
//		swap(*this, network);
//	};
//	~RecurrentNeuralNetwork() = default;
//	RecurrentNeuralNetwork<Scalar,Rank>& operator=(RecurrentNeuralNetwork<Scalar,Rank> network) {
//		swap(*this, network);
//		return *this;
//	};
//	NeuralNetwork<Scalar,Rank>* clone() const {
//		return new RecurrentNeuralNetwork(*this);
//	};
//	bool is_foremost() const {
//		return foremost;
//	};
//	Dimensions<int> get_input_dims() const {
//		return input_dims;
//	};
//	Dimensions<int> get_output_dims() const {
//		return output_dims;
//	};
//	// For the copy-and-swap idiom.
//	friend void swap(RecurrentNeuralNetwork<Scalar,Rank>& network1,
//			RecurrentNeuralNetwork<Scalar,Rank>& network2) {
//		using std::swap;
//		swap(network1.u_kernel, network2.u_kernel);
//		swap(network1.v_kernel, network2.v_kernel);
//		swap(network1.w_kernel, network2.w_kernel);
//		swap(network1.state_act, network2.state_act);
//		swap(network1.output_act, network2.output_act);
//		swap(network1.seq_length, network2.seq_length);
//		swap(network1.mul_integration, network2.mul_integration);
//		swap(network1.foremost, network2.foremost);
//		swap(network1.input_dims, network2.input_dims);
//		swap(network1.output_dims, network2.output_dims);
//		swap(network1.state, network2.state);
//	};
//protected:
//	inline void set_foremost(bool foremost) {
//		NeuralNetwork<Scalar,Rank>::set_input_layer(*w_kernel, foremost);
//		this->foremost = foremost;
//	};
//	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
//		std::vector<Layer<Scalar,Rank>*> layers_raw(5);
//		layers_raw[0] = u_kernel->get();
//		layers_raw[1] = v_kernel->get();
//		layers_raw[2] = w_kernel->get();
//		layers_raw[3] = state_act->get();
//		layers_raw[4] = output_act->get();
//		return layers_raw;
//	};
//	inline Tensor<Scalar,Rank + 1> propagate(Tensor<Scalar,Rank + 1> input, bool training) {
//		assert(input.dimension(0));
//		assert(input_dims == Utils<Scalar>::get_dims(input).demote());
//		int samples = input.dimension(0);
//		int time_steps = input.dimension(1);
//		assert(time_steps == (int) seq_length);
//		states = std::vector<Matrix<Scalar>>(input.dimension(0));
//		Array4<int> offsets({ 0, 0, 0, 0 });
//		Array4<int> extents({ samples, input_dims(0), input_dims(1),
//				input_dims(2) });
//		for (int i = 0; i < time_steps; i++) {
//			offsets[1] = i;
//			Tensor<Scalar,Rank + 1> time_step_i = input.slice(offsets, extents);
//			input = NeuralNetwork<Scalar>::pass_forward(*u_kernel, std::move(input), training);
//			if (!training)
//				NeuralNetwork<Scalar>::empty_cache(*u_kernel);
//			input += NeuralNetwork<Scalar>::pass_forward(*w_kernel, Utils<Scalar>::map_mat_to_tensor4(std::move(state)),
//					training);
//			if (!training)
//				NeuralNetwork<Scalar>::empty_cache(*w_kernel);
//			state = NeuralNetwork<Scalar>::pass_forward(*state_act, std::move(input), training);
//			if (!training)
//				NeuralNetwork<Scalar>::empty_cache(*state_act);
//			input = NeuralNetwork<Scalar>::pass_forward(*v_kernel, state, training);
//			if (!training)
//				NeuralNetwork<Scalar>::empty_cache(*v_kernel);
//			input = NeuralNetwork<Scalar>::pass_forward(*output_act, std::move(input), training);
//			if (!training)
//				NeuralNetwork<Scalar>::empty_cache(*output_act);
////			return input;
//		}
//	};
//	inline Tensor<Scalar,Rank + 1> backpropagate(Tensor<Scalar,Rank + 1> out_grads) {
//		assert(out_grads.dimension(0));
//		assert(output_dims == Utils<Scalar>::get_dims(out_grads).demote());
//		for (int i = layers.size() - 1; i >= 0; i--) {
//			Layer<Scalar>& layer = *(layers[i]);
//			out_grads = NeuralNetwork<Scalar>::pass_back(layer, std::move(out_grads));
//			NeuralNetwork<Scalar>::empty_cache(layer);
//		}
//		return out_grads;
//	};
//	KernelPtr<Scalar,Rank> u_kernel;
//	KernelPtr<Scalar,Rank> v_kernel;
//	KernelPtr<Scalar,Rank> w_kernel;
//	ActivationPtr<Scalar,Rank> state_act;
//	ActivationPtr<Scalar,Rank> output_act;
//	size_t seq_length;
//	bool mul_integration;
//	bool foremost;
//	Dimensions<int,Rank> input_dims;
//	Dimensions<int,Rank> output_dims;
//	// State.
//	std::vector<Matrix<Scalar>> states;
//};

template<typename Scalar, size_t Rank>
using NeuralNetPtr = std::unique_ptr<NeuralNetwork<Scalar,Rank>>;

template<typename Scalar>
class ParallelNeuralNetwork : public NeuralNetwork<Scalar,3> {
public:
	ParallelNeuralNetwork(std::vector<NeuralNetPtr<Scalar,3>> lanes, bool foremost = true) :
			lanes(std::move(lanes)) {
		assert(this->lanes.size() > 0 && "lanes must contain at least 1 element");
		assert(this->lanes[0] != nullptr && "lanes contains null pointers");
		NeuralNetwork<Scalar,3>& first_lane = *this->lanes[0];
		Dimensions<int,3> input_dims = first_lane.get_input_dims();
		int output_height = first_lane.get_output_dims()(0);
		int output_width = first_lane.get_output_dims()(1);
		int output_depth = first_lane.get_output_dims()(2);
		for (unsigned i = 1; i < this->lanes.size(); i++) {
			assert(this->lanes[i] != nullptr && "lanes contains null pointers");
			NeuralNetwork<Scalar,Rank>& lane = *this->lanes[i];
			assert(input_dims.equals(lane.get_input_dims()) && output_height == lane.get_output_dims()(0) &&
					output_width == lane.get_output_dims()(1) && "incompatible lane dimensions");
			output_depth += lane.get_output_dims()(2);
		}
		set_foremost(foremost);
		this->input_dims = input_dims;
		output_dims = Dimensions<int,3>(output_height, output_width, output_depth);
	};
	ParallelNeuralNetwork(NeuralNetPtr<Scalar,3>&& lane, bool foremost = true) :
			ParallelNeuralNetwork(create_vector(std::move(lane)), foremost) { };
	ParallelNeuralNetwork(const ParallelNeuralNetwork<Scalar>& network) :
			lanes(network.lanes.size()) {
		for (unsigned i = 0; i < lanes.size(); i++)
			lanes[i] = NeuralNetPtr<Scalar,3>(network.lanes[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	ParallelNeuralNetwork(ParallelNeuralNetwork<Scalar>&& network) {
		swap(*this, network);
	};
	~ParallelNeuralNetwork() = default;
	ParallelNeuralNetwork<Scalar>& operator=(ParallelNeuralNetwork<Scalar> network) {
		swap(*this, network);
		return *this;
	};
	bool is_foremost() const {
		return foremost;
	};
	NeuralNetwork<Scalar,3>* clone() const {
		return new ParallelNeuralNetwork(*this);
	};
	Dimensions<int,3> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int,3> get_output_dims() const {
		return output_dims;
	};
	friend void swap(ParallelNeuralNetwork<Scalar>& network1, ParallelNeuralNetwork<Scalar>& network2) {
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
	inline Tensor<Scalar,4> propagate(Tensor<Scalar,4> input, bool training) {
		assert(input.dimension(0));
		assert(input_dims == Utils<Scalar>::get_dims(input).demote());
		int rows = input.dimension(0);
		Array<int,4> offsets({ 0, 0, 0, 0 });
		Array<int,4> extents({ rows, output_dims(1), output_dims(1), 0 });
		Tensor<Scalar,4> out(rows, output_dims(0), output_dims(1), output_dims(2));
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
			int depth = lanes[i]->get_output_dims()(2);
			extents[3] = depth;
			out.slice(offsets, extents) = args_arr[i].out;
			offsets[3] += depth;
		}
		if (helper_thread_num > 0)
			pthread_attr_destroy(&attr);
		return out;
	};
	inline Tensor<Scalar,4> backpropagate(Tensor<Scalar,4> out_grads) {
		assert(out_grads.dimension(0));
		assert(output_dims == Utils<Scalar>::get_dims(out_grads).demote());
		Tensor<Scalar,4> prev_out_grads = foremost ? Utils<Scalar>::get_null_tensor() :
				Tensor<Scalar,4>(out_grads.dimension(0), input_dims(0), input_dims(1), input_dims(2));
		prev_out_grads.setZero();
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
	static std::vector<NeuralNetPtr<Scalar,3>> create_vector(NeuralNetPtr<Scalar,3>&& net) {
		std::vector<NeuralNetPtr<Scalar,3>> vec(1);
		vec[0] = std::move(net);
		return vec;
	};
private:
	std::vector<NeuralNetPtr<Scalar,3>> lanes;
	bool foremost;
	Dimensions<int,3> input_dims;
	Dimensions<int,3> output_dims;
	static void* propagate(void* args_ptr) {
		PropArgs& args = *((PropArgs*) args_ptr);
		args.out = args.obj->lanes[args.lane_id]->propagate(*args.in, args.training);
		return nullptr;
	};
	static void* backpropagate(void* args_ptr) {
		BackpropArgs& args = *((BackpropArgs*) args_ptr);
		NeuralNetwork<Scalar,Rank>& lane = *args.obj->lanes[args.lane_id];
		Array<int,4> offsets({ 0, 0, 0, args.depth_offset });
		Array<int,4> extents({ args.out_grads->dimension(0), lane.get_output_dims()(0),
				lane.get_output_dims()(1), lane.get_output_dims()(2) });
		Tensor<Scalar,4> out_grads_slice = args.out_grads->slice(offsets, extents);
		args.prev_out_grads = lane.backpropagate(std::move(out_grads_slice));
		return nullptr;
	};
	struct PropArgs {
		ParallelNeuralNetwork<Scalar,3>* obj;
		int lane_id;
		bool training;
		Tensor<Scalar,4>* in;
		Tensor<Scalar,4> out;
	};
	struct BackpropArgs {
		ParallelNeuralNetwork<Scalar,3>* obj;
		int lane_id;
		int depth_offset;
		Tensor<Scalar,4>* out_grads;
		Tensor<Scalar,4> prev_out_grads;
	};
};

template<typename Scalar, size_t Rank> class ResidualNeuralNetwork;
template<typename Scalar, size_t Rank> class DenseNeuralNetwork;

template<typename Scalar, size_t Rank>
class CompositeNeuralNetwork : public NeuralNetwork<Scalar,Rank> {
	friend class ResidualNeuralNetwork<Scalar,Rank>;
	friend class DenseNeuralNetwork<Scalar,Rank>;
public:
	CompositeNeuralNetwork(std::vector<NeuralNetPtr<Scalar,Rank>> blocks, bool foremost = true) :
			blocks(std::move(blocks)),
			foremost(foremost) {
		assert(this->blocks.size() > 0 && "blocks must contain at least 1 element");
		assert(this->blocks[0] != nullptr && "blocks contains null pointers");
		NeuralNetwork<Scalar,Rank>& first_block = *(this->blocks[0]);
		input_dims = first_block.get_input_dims();
		output_dims = this->blocks[this->blocks.size() - 1]->get_output_dims();
		Dimensions<int> prev_dims = first_block.get_output_dims();
		for (unsigned i = 1; i < this->blocks.size(); i++) {
			assert(this->blocks[i] != nullptr && "blocks contains null pointers");
			NeuralNetwork<Scalar,Rank>& block = *this->blocks[i];
			assert(prev_dims == block.get_input_dims() && "incompatible network dimensions");
			block.set_foremost(false);
			prev_dims = block.get_output_dims();
		}
		first_block.set_foremost(foremost);
	};
	CompositeNeuralNetwork(NeuralNetPtr<Scalar,Rank>&& block, bool foremost = true) :
			CompositeNeuralNetwork(create_vector(std::move(block)),
					foremost) { };
	CompositeNeuralNetwork(const CompositeNeuralNetwork<Scalar,Rank>& network) :
			blocks(network.blocks.size()) {
		for (unsigned i = 0; i < blocks.size(); i++)
			blocks[i] = NeuralNetPtr<Scalar,Rank>(network.blocks[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	CompositeNeuralNetwork(CompositeNeuralNetwork<Scalar,Rank>&& network) {
		swap(*this, network);
	};
	~CompositeNeuralNetwork() = default;
	CompositeNeuralNetwork<Scalar,Rank>& operator=(CompositeNeuralNetwork<Scalar,Rank> network) {
		swap(*this, network);
		return *this;
	};
	NeuralNetwork<Scalar,Rank>* clone() const {
		return new CompositeNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	Dimensions<int,Rank> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int,Rank> get_output_dims() const {
		return output_dims;
	};
	friend void swap(CompositeNeuralNetwork<Scalar,Rank>& network1, CompositeNeuralNetwork<Scalar,Rank>& network2) {
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
	inline Tensor<Scalar,Rank + 1> propagate(Tensor<Scalar,Rank + 1> input, bool training) {
		assert(input.dimension(0));
		assert(input_dims == Utils<Scalar>::get_dims(input).demote());
		for (unsigned i = 0; i < blocks.size(); i++)
			input = blocks[i]->propagate(std::move(input), training);
		return input;
	};
	inline Tensor<Scalar,Rank + 1> backpropagate(Tensor<Scalar,Rank + 1> out_grads) {
		assert(out_grads.dimension(0));
		assert(output_dims == Utils<Scalar>::get_dims(out_grads).demote());
		for (int i = blocks.size() - 1; i >= 0; i--)
			out_grads = blocks[i]->backpropagate(std::move(out_grads));
		return out_grads;
	};
	static std::vector<NeuralNetPtr<Scalar,Rank>> create_vector(NeuralNetPtr<Scalar,Rank>&& net) {
		std::vector<NeuralNetPtr<Scalar,Rank>> vec(1);
		vec[0] = std::move(net);
		return vec;
	};
private:
	std::vector<NeuralNetPtr<Scalar,Rank>> blocks;
	bool foremost;
	Dimensions<int,Rank> input_dims;
	Dimensions<int,Rank> output_dims;
};

/**
 * This class can be used to build InceptionNets, ResNets, Inception-ResNets, or non-residual composite
 * neural networks.
 */
template<typename Scalar, size_t Rank>
class ResidualNeuralNetwork : public NeuralNetwork<Scalar,Rank> {
	typedef CompositeNeuralNetwork<Scalar,Rank> Module;
public:
	ResidualNeuralNetwork(std::vector<std::pair<Module,bool>> modules, bool foremost = true) :
			modules(modules),
			foremost(foremost) {
		assert(modules.size() > 0 && "modules must contain at least 1 element");
		Module& first_module = this->modules[0].first;
		input_dims = first_module.get_input_dims();
		output_dims = this->modules[modules.size() - 1].first.get_output_dims();
		first_module.set_foremost(foremost);
		Dimensions<int> prev_dims = input_dims;
		for (unsigned i = 0; i < modules.size(); i++) {
			std::pair<Module,bool>& module = this->modules[i];
			Module& module_net = module.first;
			if (i != 0)
				module_net.set_foremost(false);
			assert((!module.second || module_net.get_input_dims().equals(module_net.get_output_dims())) &&
					"residual module input-output dimension discrepancy");
			assert(prev_dims.equals(module_net.get_input_dims()) && "incompatible module dimensions");
			prev_dims = module_net.get_output_dims();
		}
	};
	NeuralNetwork<Scalar,Rank>* clone() const {
		return new ResidualNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	Dimensions<int,Rank> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int,Rank> get_output_dims() const {
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
	inline Tensor<Scalar,Rank + 1> propagate(Tensor<Scalar,Rank + 1> input, bool training) {
		assert(input.dimension(0));
		assert(input_dims == Utils<Scalar>::get_dims(input).demote());
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
		assert(out_grads.dimension(0));
		assert(output_dims == Utils<Scalar>::get_dims(out_grads).demote());
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
	Dimensions<int,Rank> input_dims;
	Dimensions<int,Rank> output_dims;
};

template<typename Scalar>
class DenseNeuralNetwork : public NeuralNetwork<Scalar,3> {
	typedef CompositeNeuralNetwork<Scalar,3> Module;
public:
	DenseNeuralNetwork(std::vector<Module> modules, bool foremost = true) :
			modules(modules),
			foremost(foremost) {
		assert(this->modules.size() > 0 && "modules must contain at least 1 element");
		Module& first_module = this->modules[0];
		input_dims = first_module.get_input_dims();
		Dimensions<int> first_module_output_dims = first_module.get_output_dims();
		int output_height = first_module_output_dims(0);
		int output_width = first_module_output_dims(1);
		int output_depth = first_module_output_dims(2) + input_dims(2);
		assert(input_dims(0) == output_height && input_dims(1) == output_width);
		for (unsigned i = 1; i < this->modules.size(); i++) {
			Module& module = this->modules[i];
			assert(module.get_input_dims().equals(output_height, output_width, output_depth) &&
					"incompatible module dimensions");
			output_depth += module.get_output_dims()(2);
			module.set_foremost(false);
		}
		output_dims = Dimensions<int>(output_height, output_width, output_depth);
		first_module.set_foremost(foremost);
	};
	NeuralNetwork<Scalar,3>* clone() const {
		return new DenseNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	Dimensions<int,3> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int,3> get_output_dims() const {
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
	inline Tensor<Scalar,4> propagate(Tensor<Scalar,4> input, bool training) {
		assert(input.dimension(0));
		assert(input_dims == Utils<Scalar>::get_dims(input).demote());
		int rows = input.dimension(0);
		Array<int,4> offsets({ 0, 0, 0, 0 });
		Array<int,4> extents({ rows, input_dims(0), input_dims(1), 0 });
		for (unsigned i = 0; i < modules.size(); i++) {
			Module& module = modules[i];
			int layer_input_depth = module.get_input_dims()(2);
			int layer_output_depth = module.get_output_dims()(2);
			Tensor<Scalar,4> out_i(rows, input_dims(0), input_dims(1),
					layer_input_depth + layer_output_depth);
			offsets[3] = 0;
			extents[3] = layer_input_depth;
			out_i.slice(offsets, extents) = input;
			offsets[3] = layer_input_depth;
			extents[3] = layer_output_depth;
			out_i.slice(offsets, extents) = module.propagate(std::move(input), training);
			input = Tensor<Scalar,4>(std::move(out_i));
		}
		return input;
	};
	inline Tensor<Scalar,4> backpropagate(Tensor<Scalar,4> out_grads) {
		assert(out_grads.dimension(0));
		assert(output_dims == Utils<Scalar>::get_dims(out_grads).demote());
		Array<int,4> offsets({ 0, 0, 0, 0 });
		Array<int,4> extents({ out_grads.dimension(0), input_dims(0), input_dims(1), 0 });
		for (int i = modules.size() - 1; i >= 0; i--) {
			Module& module = modules[i];
			int layer_input_depth = module.get_input_dims()(2);
			int layer_output_depth = module.get_output_dims()(2);
			offsets[3] = layer_input_depth;
			extents[3] = layer_output_depth;
			Tensor<Scalar,4> out_grads_i = out_grads.slice(offsets, extents);
			offsets[3] = 0;
			extents[3] = layer_input_depth;
			out_grads = Tensor<Scalar,4>(out_grads.slice(offsets, extents) + module.backpropagate(std::move(out_grads_i)));
		}
		return out_grads;
	};
	std::vector<Module> modules;
	bool foremost;
	Dimensions<int,3> input_dims;
	Dimensions<int,3> output_dims;
};

} /* namespace cattle */

#endif /* NEURALNETWORK_H_ */
