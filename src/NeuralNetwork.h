/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <cassert>
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

namespace cppnn {

// Forward declaration to Optimizer and CompositeNeuralNetwork so they can be friended.
template<typename Scalar> class Optimizer;
template<typename Scalar> class CompositeNeuralNetwork;

/**
 * A neural network class that consists of a vector of neuron layers. It allows
 * for the calculation of the output of the network given an input vector and
 * it provides a member function for back-propagating the derivative of a loss
 * function with respect to the activated output of the network's last layer.
 * This back-propagation sets the gradients of the parameters associated with each
 * layer.
 */
template<typename Scalar>
class NeuralNetwork {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	friend class Optimizer<Scalar>;
	friend class CompositeNeuralNetwork<Scalar>;
public:
	virtual ~NeuralNetwork() = default;
	virtual NeuralNetwork<Scalar>* clone() const = 0;
	virtual bool is_foremost() const = 0;
	virtual Dimensions<int> get_input_dims() const = 0;
	virtual Dimensions<int> get_output_dims() const = 0;
	virtual void init() {
		std::vector<Layer<Scalar>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i]->init();
	};
	inline virtual Tensor4<Scalar> infer(Tensor4<Scalar> input) {
		return propagate(std::move(input), false);
	};
	virtual std::string to_string() {
		std::stringstream strm;
		strm << "Neural Net " << this << std::endl;
		std::vector<Layer<Scalar>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar>* layer = layers[i];
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
	friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork<Scalar>& nn) {
		return os << nn.to_string() << std::endl;
	};
protected:
	virtual void set_foremost(bool foremost) = 0;
	virtual std::vector<Layer<Scalar>*> get_layers() = 0;
	virtual Tensor4<Scalar> propagate(Tensor4<Scalar> input, bool training) = 0;
	virtual Tensor4<Scalar> backpropagate(Tensor4<Scalar> out_grads) = 0;
	inline static void set_input_layer(Layer<Scalar>& layer, bool on) {
		layer.input_layer = on;
	};
	inline static void empty_cache(Layer<Scalar>& layer) {
		layer.empty_cache();
	};
	inline static Tensor4<Scalar> pass_forward(Layer<Scalar>& layer, Tensor4<Scalar>&& prev_out, bool training) {
		return layer.pass_forward(prev_out, training);
	};
	inline static Tensor4<Scalar> pass_back(Layer<Scalar>& layer, Tensor4<Scalar>&& out_grads) {
		return layer.pass_back(out_grads);
	};
	inline static void assert_popagation_input_dims(const Tensor4<Scalar>& input, Dimensions<int> expected_dims) {
		assert(input.dimension(0) > 0);
		assert(input.dimension(1) == expected_dims.get_dim1() && input.dimension(2) == expected_dims.get_dim2() &&
				input.dimension(3) == expected_dims.get_dim3());
	};
};

template<typename Scalar>
using LayerPtr = std::unique_ptr<Layer<Scalar>>;

// Forward declaration for friending.
template<typename Scalar> class ParallelNeuralNetwork;

template<typename Scalar>
class SequentialNeuralNetwork : public NeuralNetwork<Scalar> {
	friend class ParallelNeuralNetwork<Scalar>;
public:
	/**
	 * Constructs the network using the provided layer pointers. It takes ownership of the layer pointers.
	 *
	 * @param layers A vector of unique smart pointers to the layers that constitute the neural network.
	 * @param foremost Whether the network directly receives its input. If it is set to false, back-propagation
	 * returns an empty tensor.
	 */
	SequentialNeuralNetwork(std::vector<LayerPtr<Scalar>> layers, bool foremost = true) :
			layers(std::move(layers)),
			foremost(foremost) {
		assert(this->layers.size() > 0 && "layers must contain at least 1 element");
		assert(this->layers[0] != nullptr);
		Layer<Scalar>& first_layer = *(this->layers[0]);
		input_dims = first_layer.get_input_dims();
		output_dims = this->layers[this->layers.size() - 1]->get_output_dims();
		Dimensions<int> prev_dims = first_layer.get_output_dims();
		for (unsigned i = 1; i < this->layers.size(); i++) {
			assert(this->layers[i] != nullptr && "layers contains null pointers");
			assert(prev_dims.equals(this->layers[i]->get_input_dims()) && "incompatible layer dimensions");
			prev_dims = this->layers[i]->get_output_dims();
		}
		NeuralNetwork<Scalar>::set_input_layer(first_layer, foremost);
	};
	SequentialNeuralNetwork(LayerPtr<Scalar> layer, bool foremost = true) :
			SequentialNeuralNetwork(create_vector(std::move(layer)), foremost) { };
	// Copy constructor.
	SequentialNeuralNetwork(const SequentialNeuralNetwork<Scalar>& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i] = LayerPtr<Scalar>(network.layers[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	// Move constructor.
	SequentialNeuralNetwork(SequentialNeuralNetwork<Scalar>&& network) {
		swap(*this, network);
	};
	// The smart pointers take care of deleting the layers.
	~SequentialNeuralNetwork() = default;
	/* The assignment uses the move or copy constructor to pass the parameter
	 * based on whether it is an rvalue or an lvalue. */
	SequentialNeuralNetwork<Scalar>& operator=(SequentialNeuralNetwork<Scalar> network) {
		swap(*this, network);
		return *this;
	};
	NeuralNetwork<Scalar>* clone() const {
		return new SequentialNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	Dimensions<int> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int> get_output_dims() const {
		return output_dims;
	};
	// For the copy-and-swap idiom.
	friend void swap(SequentialNeuralNetwork<Scalar>& network1,
			SequentialNeuralNetwork<Scalar>& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	};
protected:
	inline void set_foremost(bool foremost) {
		NeuralNetwork<Scalar>::set_input_layer(*layers[0], foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar>*> get_layers() {
		std::vector<Layer<Scalar>*> layers_raw(layers.size());
		for (unsigned i = 0; i < layers.size(); i++)
			layers_raw[i] = layers[i].get();
		return layers_raw;
	};
	inline Tensor4<Scalar> propagate(Tensor4<Scalar> input, bool training) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(input, input_dims);
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar>& layer = *layers[i];
			input = NeuralNetwork<Scalar>::pass_forward(layer, std::move(input), training);
			if (!training)
				NeuralNetwork<Scalar>::empty_cache(layer);
		}
		return input;
	};
	inline Tensor4<Scalar> backpropagate(Tensor4<Scalar> out_grads) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(out_grads, output_dims);
		for (int i = layers.size() - 1; i >= 0; i--) {
			Layer<Scalar>& layer = *(layers[i]);
			out_grads = NeuralNetwork<Scalar>::pass_back(layer, std::move(out_grads));
			NeuralNetwork<Scalar>::empty_cache(layer);
		}
		return out_grads;
	};
	static std::vector<LayerPtr<Scalar>> create_vector(LayerPtr<Scalar>&& layer) {
		std::vector<LayerPtr<Scalar>> vec(1);
		vec[0] = std::move(layer);
		return vec;
	};
	std::vector<LayerPtr<Scalar>> layers;
	bool foremost;
	Dimensions<int> input_dims;
	Dimensions<int> output_dims;
};

template<typename Scalar>
class DenseNeuralNetwork : public NeuralNetwork<Scalar> {
public:
	DenseNeuralNetwork(std::vector<LayerPtr<Scalar>> layers, bool foremost = true) :
			layers(std::move(layers)),
			foremost(foremost) {
		assert(this->layers.size() > 0 && "layers must contain at least 1 element");
		assert(this->layers[0] != nullptr);
		Layer<Scalar>& first_layer = *(this->layers[0]);
		input_dims = first_layer.get_input_dims();
		Dimensions<int> first_layer_output_dims = first_layer.get_output_dims();
		int output_height = first_layer_output_dims.get_dim1();
		int output_width = first_layer_output_dims.get_dim2();
		int output_depth = first_layer_output_dims.get_dim3() + input_dims.get_dim3();
		assert(input_dims.get_dim1() == output_height && input_dims.get_dim2() == output_width);
		for (unsigned i = 1; i < this->layers.size(); i++) {
			assert(this->layers[i] != nullptr && "layers contains null pointers");
			Layer<Scalar>& layer = *this->layers[i];
			assert(layer.get_input_dims().equals(output_height, output_width, output_depth) &&
					"incompatible layer dimensions");
			output_depth += layer.get_output_dims().get_dim3();
		}
		output_dims = Dimensions<int>(output_height, output_width, output_depth);
		NeuralNetwork<Scalar>::set_input_layer(first_layer, foremost);
	};
	DenseNeuralNetwork(const DenseNeuralNetwork<Scalar>& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i] = LayerPtr<Scalar>(network.layers[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	DenseNeuralNetwork(DenseNeuralNetwork<Scalar>&& network) {
		swap(*this, network);
	};
	~DenseNeuralNetwork() = default;
	DenseNeuralNetwork<Scalar>& operator=(DenseNeuralNetwork<Scalar> network) {
		swap(*this, network);
		return *this;
	};
	NeuralNetwork<Scalar>* clone() const {
		return new DenseNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	Dimensions<int> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int> get_output_dims() const {
		return output_dims;
	};
	friend void swap(DenseNeuralNetwork<Scalar>& network1,
			DenseNeuralNetwork<Scalar>& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	};
protected:
	inline std::vector<Layer<Scalar>*> get_layers() {
		std::vector<Layer<Scalar>*> layers_raw(layers.size());
		for (unsigned i = 0; i < layers.size(); i++)
			layers_raw[i] = layers[i].get();
		return layers_raw;
	};
	inline Tensor4<Scalar> propagate(Tensor4<Scalar> input, bool training) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(input, input_dims);
		int rows = input.dimension(0);
		Array4<int> offsets({ 0, 0, 0, 0 });
		Array4<int> extents({ rows, input_dims.get_dim1(), input_dims.get_dim2(), 0 });
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar>& layer = *layers[i];
			int input_depth = input.dimension(3);
			int layer_output_depth = layer.get_output_dims().get_dim3();
			Tensor4<Scalar> out_i(rows, input_dims.get_dim1(), input_dims.get_dim2(),
					input_depth + layer_output_depth);
			offsets[4] = 0;
			extents[4] = input_depth;
			out_i.slice(offsets, extents) = input;
			offsets[4] = input_depth;
			extents[4] = layer_output_depth;
			out_i.slice(offsets, extents) = NeuralNetwork<Scalar>::pass_forward(layer, std::move(input), training);
			if (!training)
				NeuralNetwork<Scalar>::empty_cache(layer);
			input = Tensor4<Scalar>(std::move(out_i));
		}
		return input;
	};
	inline Tensor4<Scalar> backpropagate(Tensor4<Scalar> out_grads) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(out_grads, output_dims);
		for (int i = layers.size() - 1; i >= 0; i--) {
			Layer<Scalar>& layer = *(layers[i]);
			out_grads = NeuralNetwork<Scalar>::pass_back(layer, std::move(out_grads));
			NeuralNetwork<Scalar>::empty_cache(layer);
		}
		return out_grads;
	};
	std::vector<LayerPtr<Scalar>> layers;
	bool foremost;
	Dimensions<int> input_dims;
	Dimensions<int> output_dims;
};

template<typename Scalar>
class ParallelNeuralNetwork : public NeuralNetwork<Scalar> {
	typedef SequentialNeuralNetwork<Scalar> Lane;
public:
	ParallelNeuralNetwork(std::vector<Lane> lanes, bool foremost = true) :
			lanes(lanes) {
		assert(lanes.size() > 0 && "lanes must contain at least 1 element");
		Lane& first_lane = this->lanes[0];
		Dimensions<int> input_dims = first_lane.input_dims;
		int output_height = first_lane.output_dims.get_dim1();
		int output_width = first_lane.output_dims.get_dim2();
		int output_depth = first_lane.output_dims.get_dim3();
		for (unsigned i = 1; i < lanes.size(); i++) {
			Lane& lane = this->lanes[i];
			assert(input_dims.equals(lane.input_dims) && output_height == lane.output_dims.get_dim1() &&
					output_width == lane.output_dims.get_dim2() && "incompatible lane dimensions");
			output_depth += lane.output_dims.get_dim3();
		}
		set_foremost(foremost);
		this->input_dims = input_dims;
		output_dims = Dimensions<int>(output_height, output_width, output_depth);
	};
	ParallelNeuralNetwork(Lane lane, bool foremost = true) :
			ParallelNeuralNetwork(std::vector<Lane>({ lane }), foremost) { };
	bool is_foremost() const {
		return foremost;
	};
	NeuralNetwork<Scalar>* clone() const {
		return new ParallelNeuralNetwork(*this);
	};
	Dimensions<int> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int> get_output_dims() const {
		return output_dims;
	};
protected:
	inline void set_foremost(bool foremost) {
		for (unsigned i = 0; i < lanes.size(); i++)
			lanes[i].set_foremost(foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar>*> get_layers() {
		std::vector<Layer<Scalar>*> layers;
		for (unsigned i = 0; i < lanes.size(); i++) {
			std::vector<Layer<Scalar>*> lane_layers = lanes[i].get_layers();
			for (unsigned j = 0; j < lane_layers.size(); j++)
				layers.push_back(lane_layers[j]);
		}
		return layers;
	};
	inline Tensor4<Scalar> propagate(Tensor4<Scalar> input, bool training) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(input, input_dims);
		int rows = input.dimension(0);
		Array4<int> offsets({ 0, 0, 0, 0 });
		Array4<int> extents({ rows, output_dims.get_dim1(), output_dims.get_dim2(), 0 });
		Tensor4<Scalar> out(rows, output_dims.get_dim1(), output_dims.get_dim2(), output_dims.get_dim3());
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
			int depth = lanes[i].output_dims.get_dim3();
			extents[3] = depth;
			out.slice(offsets, extents) = args_arr[i].out;
			offsets[3] += depth;
		}
		if (helper_thread_num > 0)
			pthread_attr_destroy(&attr);
		return out;
	};
	inline Tensor4<Scalar> backpropagate(Tensor4<Scalar> out_grads) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(out_grads, output_dims);
		Tensor4<Scalar> prev_out_grads = foremost ? Utils<Scalar>::NULL_TENSOR :
				Tensor4<Scalar>(out_grads.dimension(0), input_dims.get_dim1(), input_dims.get_dim2(),
						input_dims.get_dim3());
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
			depth_offset -= lanes[i].output_dims.get_dim3();
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
private:
	std::vector<Lane> lanes;
	bool foremost;
	Dimensions<int> input_dims;
	Dimensions<int> output_dims;
	static void* propagate(void* args_ptr) {
		PropArgs& args = *((PropArgs*) args_ptr);
		args.out = args.obj->lanes[args.lane_id].propagate(*args.in, args.training);
		return nullptr;
	};
	static void* backpropagate(void* args_ptr) {
		BackpropArgs& args = *((BackpropArgs*) args_ptr);
		Lane& lane = args.obj->lanes[args.lane_id];
		Array4<int> offsets({ 0, 0, 0, args.depth_offset });
		Array4<int> extents({ args.out_grads->dimension(0), lane.output_dims.get_dim1(),
				lane.output_dims.get_dim2(), lane.output_dims.get_dim3() });
		Tensor4<Scalar> out_grads_slice = args.out_grads->slice(offsets, extents);
		args.prev_out_grads = lane.backpropagate(std::move(out_grads_slice));
		return nullptr;
	};
	struct PropArgs {
		ParallelNeuralNetwork<Scalar>* obj;
		int lane_id;
		bool training;
		Tensor4<Scalar>* in;
		Tensor4<Scalar> out;
	};
	struct BackpropArgs {
		ParallelNeuralNetwork<Scalar>* obj;
		int lane_id;
		int depth_offset;
		Tensor4<Scalar>* out_grads;
		Tensor4<Scalar> prev_out_grads;
	};
};

template<typename Scalar>
using NeuralNetPtr = std::unique_ptr<NeuralNetwork<Scalar>>;

template<typename Scalar> class ResidualNeuralNetwork;

template<typename Scalar>
class CompositeNeuralNetwork : public NeuralNetwork<Scalar> {
	friend class ResidualNeuralNetwork<Scalar>;
public:
	CompositeNeuralNetwork(std::vector<NeuralNetPtr<Scalar>> nets, bool foremost = true) :
			nets(std::move(nets)),
			foremost(foremost) {
		assert(this->nets.size() > 0 && "nets must contain at least 1 element");
		assert(this->nets[0] != nullptr);
		NeuralNetwork<Scalar>& first_net = *(this->nets[0]);
		input_dims = first_net.get_input_dims();
		output_dims = this->nets[this->nets.size() - 1]->get_output_dims();
		Dimensions<int> prev_dims = first_net.get_output_dims();
		for (unsigned i = 1; i < this->nets.size(); i++) {
			assert(this->nets[i] != nullptr && "nets contains null pointers");
			NeuralNetwork<Scalar>& net = *this->nets[i];
			assert(prev_dims.equals(net.get_input_dims()) && "incompatible network dimensions");
			net.set_foremost(false);
			prev_dims = net.get_output_dims();
		}
		first_net.set_foremost(foremost);
	};
	CompositeNeuralNetwork(NeuralNetPtr<Scalar> net, bool foremost = true) :
			CompositeNeuralNetwork(create_vector(std::move(net)), foremost) { };
	CompositeNeuralNetwork(const CompositeNeuralNetwork<Scalar>& network) :
			nets(network.nets.size()) {
		for (unsigned i = 0; i < nets.size(); i++)
			nets[i] = NeuralNetPtr<Scalar>(network.nets[i]->clone());
		foremost = network.foremost;
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	CompositeNeuralNetwork(CompositeNeuralNetwork<Scalar>&& network) {
		swap(*this, network);
	};
	~CompositeNeuralNetwork() = default;
	CompositeNeuralNetwork<Scalar>& operator=(CompositeNeuralNetwork<Scalar> network) {
		swap(*this, network);
		return *this;
	};
	NeuralNetwork<Scalar>* clone() const {
		return new CompositeNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	Dimensions<int> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int> get_output_dims() const {
		return output_dims;
	};
	// For the copy-and-swap idiom.
	friend void swap(CompositeNeuralNetwork<Scalar>& network1, CompositeNeuralNetwork<Scalar>& network2) {
		using std::swap;
		swap(network1.nets, network2.nets);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	};
protected:
	inline void set_foremost(bool foremost) {
		nets[0]->set_foremost(foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar>*> get_layers() {
		std::vector<Layer<Scalar>*> layers;
		for (unsigned i = 0; i < nets.size(); i++) {
			std::vector<Layer<Scalar>*> internal_layers = nets[i]->get_layers();
			for (unsigned j = 0; j < internal_layers.size(); j++)
				layers.push_back(internal_layers[j]);
		}
		return layers;
	};
	inline Tensor4<Scalar> propagate(Tensor4<Scalar> input, bool training) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(input, input_dims);
		for (unsigned i = 0; i < nets.size(); i++)
			input = nets[i]->propagate(std::move(input), training);
		return input;
	};
	inline Tensor4<Scalar> backpropagate(Tensor4<Scalar> out_grads) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(out_grads, output_dims);
		for (int i = nets.size() - 1; i >= 0; i--)
			out_grads = nets[i]->backpropagate(std::move(out_grads));
		return out_grads;
	};
	static std::vector<NeuralNetPtr<Scalar>> create_vector(NeuralNetPtr<Scalar>&& net) {
		std::vector<NeuralNetPtr<Scalar>> vec(1);
		vec[0] = std::move(net);
		return vec;
	};
private:
	std::vector<NeuralNetPtr<Scalar>> nets;
	bool foremost;
	Dimensions<int> input_dims;
	Dimensions<int> output_dims;
};

/**
 * This class can be used to build InceptionNets, ResNets, Inception-ResNets, or non-residual composite
 * neural networks.
 */
template<typename Scalar>
class ResidualNeuralNetwork : public NeuralNetwork<Scalar> {
	typedef CompositeNeuralNetwork<Scalar> Module;
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
	NeuralNetwork<Scalar>* clone() const {
		return new ResidualNeuralNetwork(*this);
	};
	bool is_foremost() const {
		return foremost;
	};
	Dimensions<int> get_input_dims() const {
		return input_dims;
	};
	Dimensions<int> get_output_dims() const {
		return output_dims;
	};
protected:
	inline void set_foremost(bool foremost) {
		modules[0].first.set_foremost(foremost);
		this->foremost = foremost;
	};
	inline std::vector<Layer<Scalar>*> get_layers() {
		std::vector<Layer<Scalar>*> layers;
		for (unsigned i = 0; i < modules.size(); i++) {
			std::vector<Layer<Scalar>*> module_layers = modules[i].first.get_layers();
			for (unsigned j = 0; j < module_layers.size(); j++)
				layers.push_back(module_layers[j]);
		}
		return layers;
	};
	inline Tensor4<Scalar> propagate(Tensor4<Scalar> input, bool training) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(input, input_dims);
		for (unsigned i = 0; i < modules.size(); i++) {
			std::pair<Module,bool>& module = modules[i];
			if (module.second) // If it is a residual module, propagate the sum of the input and the output.
				input += module.first.propagate(input, training);
			else
				input = module.first.propagate(std::move(input), training);
		}
		return input;
	};
	inline Tensor4<Scalar> backpropagate(Tensor4<Scalar> out_grads) {
		NeuralNetwork<Scalar>::assert_popagation_input_dims(out_grads, output_dims);
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
	Dimensions<int> input_dims;
	Dimensions<int> output_dims;
};

} /* namespace cppnn */

#endif /* NEURALNETWORK_H_ */
