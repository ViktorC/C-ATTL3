/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <Layer.h>
#include <cassert>
#include <Dimensions.h>
#include <Eigen/Core>
#include <iomanip>
#include <pthread.h>
#include <sstream>
#include <string>
#include <vector>
#include <WeightInitialization.h>

namespace cppnn {

// Forward declaration to Optimizer so it can be friended.
template<typename Scalar>
class Optimizer;

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
	friend class Optimizer<Scalar>;
public:
	virtual ~NeuralNetwork() = default;
	virtual Dimensions get_input_dims() const = 0;
	virtual Dimensions get_output_dims() const = 0;
	virtual void init() {
		std::vector<Layer<Scalar>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); i++)
			layers[i]->init();
	};
	virtual Tensor4<Scalar> infer(Tensor4<Scalar> input) {
		return propagate(input, false);
	};
	virtual std::string to_string() {
		std::stringstream strm;
		strm << "Neural Net " << this << std::endl;
		for (unsigned i = 0; i < get_layers().size(); i++) {
			Layer<Scalar>* layer = get_layers()[i];
			strm << "\tLayer " << std::setw(3) << std::to_string(i + 1) <<
					"----------------------------" << std::endl;
			strm << "\t\tinput dims: " << layer->get_input_dims().to_string() << std::endl;
			strm << "\t\toutput dims: " << layer->get_output_dims().to_string() << std::endl;
			if (layer->is_parametric()) {
				strm << "\t\tparams:" << std::endl;
				Matrix<Scalar>& params = layer->get_params();
				for (int j = 0; j < params.rows(); j++) {
					strm << "\t\t[ ";
					for (int k = 0; k < params.cols(); k++) {
						strm << std::setw(11) << std::setprecision(4) << params(j,k);
						if (k != params.cols() - 1) {
							strm << ", ";
						}
					}
					strm << " ]" << std::endl;
				}
			}
		}
		return strm.str();
	};
protected:
	virtual std::vector<Layer<Scalar>*>& get_layers() = 0;
	virtual Tensor4<Scalar> propagate(Tensor4<Scalar> input, bool training) = 0;
	virtual void backpropagate(Tensor4<Scalar> out_grads) = 0;
	static void set_input_layer(Layer<Scalar>& layer, bool on) {
		layer.set_input_layer(on);
	};
	static void empty_cache(Layer<Scalar>& layer) {
		layer.empty_cache();
	};
	static Tensor4<Scalar> pass_forward(Layer<Scalar>& layer, Tensor4<Scalar> prev_out, bool training) {
		return layer.pass_forward(prev_out, training);
	};
	static Tensor4<Scalar> pass_back(Layer<Scalar>& layer, Tensor4<Scalar> out_grads) {
		return layer.pass_back(out_grads);
	};
	static void assert_popagation_input(const Tensor4<Scalar>& input, Dimensions expected_dims) {
		assert(input.dimension(0));
		assert(input.dimension(1) == expected_dims.get_dim1() && input.dimension(2) == expected_dims.get_dim2() &&
				input.dimension(3) == expected_dims.get_dim3());
	};
};

template<typename Scalar>
class SequentialNeuralNetwork : public NeuralNetwork<Scalar> {
public:
	/**
	 * Constructs the network using the provided vector of layers. The object
	 * takes ownership of the pointers held in the vector and deletes them when
	 * its destructor is invoked.
	 *
	 * @param layers A vector of Layer pointers to compose the neural network.
	 * The vector must contain at least 1 element and no null pointers. The
	 * the prev_nodes attributes of the Layer objects pointed to by the pointers
	 * must match the nodes attribute of the Layer objects pointed to by their
	 * pointers directly preceding them in the vector.
	 */
	SequentialNeuralNetwork(std::vector<Layer<Scalar>*> layers) : // TODO use smart Layer pointers
			layers(layers) {
		assert(layers.size() > 0 && "layers must contain at least 1 element");
		Layer<Scalar>& first_layer = *(layers[0]);
		input_dims = first_layer.get_input_dims();
		output_dims = layers[layers.size() - 1]->get_output_dims();
		Dimensions prev_dims = first_layer.get_output_dims();
		for (unsigned i = 1; i < layers.size(); i++) {
			assert(layers[i] != nullptr && "layers contains null pointers");
			assert(prev_dims.equals(layers[i]->get_input_dims()) && "incompatible layer dimensions");
			prev_dims = layers[i]->get_output_dims();
		}
		NeuralNetwork<Scalar>::set_input_layer(first_layer, true);
	};
	// Copy constructor.
	SequentialNeuralNetwork(const SequentialNeuralNetwork<Scalar>& network) :
			layers(network.layers.size()) {
		for (unsigned i = 0; i < layers.size(); i++) {
			layers[i] = network.layers[i]->clone();
		}
		input_dims = network.input_dims;
		output_dims = network.output_dims;
	};
	// Move constructor.
	SequentialNeuralNetwork(SequentialNeuralNetwork<Scalar>&& network) {
		swap(*this, network);
	};
	// Take ownership of layer pointers.
	~SequentialNeuralNetwork() {
		for (unsigned i = 0; i < layers.size(); i++) {
			delete layers[i];
		}
	};
	/* The assignment uses the move or copy constructor to pass the parameter
	 * based on whether it is an lvalue or an rvalue. */
	SequentialNeuralNetwork<Scalar>& operator=(SequentialNeuralNetwork<Scalar> network) {
		swap(*this, network);
		return *this;
	};
	Dimensions get_input_dims() const {
		return input_dims;
	};
	Dimensions get_output_dims() const {
		return output_dims;
	};
protected:
	// For the copy-and-swap idiom.
	friend void swap(SequentialNeuralNetwork<Scalar>& network1,
			SequentialNeuralNetwork<Scalar>& network2) {
		using std::swap;
		swap(network1.layers, network2.layers);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	};
	std::vector<Layer<Scalar>*>& get_layers() {
		return layers;
	};
	Tensor4<Scalar> propagate(Tensor4<Scalar> input, bool training) {
		NeuralNetwork<Scalar>::assert_popagation_input(input, input_dims);
		for (unsigned i = 0; i < layers.size(); i++) {
			Layer<Scalar>& layer = *(layers[i]);
			input = NeuralNetwork<Scalar>::pass_forward(layer, input, training);
			if (!training)
				NeuralNetwork<Scalar>::empty_cache(layer);
		}
		return input;
	};
	void backpropagate(Tensor4<Scalar> out_grads) {
		NeuralNetwork<Scalar>::assert_popagation_input(out_grads, output_dims);
		for (int i = layers.size() - 1; i >= 0; i--) {
			Layer<Scalar>& layer = *(layers[i]);
			out_grads = NeuralNetwork<Scalar>::pass_back(*(layers[i]), out_grads);
			NeuralNetwork<Scalar>::empty_cache(layer);
		}
	};
	std::vector<Layer<Scalar>*> layers;
	Dimensions input_dims;
	Dimensions output_dims;
};

template<typename Scalar>
class InceptionNeuralNetwork : public NeuralNetwork<Scalar> {
public:
	InceptionNeuralNetwork(std::vector<InceptionModule<Scalar>> modules) :
			modules(modules) {
		assert(modules.size() > 0 && "modules must contain at least 1 element");
	};
	Dimensions get_input_dims() const {
		return input_dims;
	};
	Dimensions get_output_dims() const {
		return output_dims;
	};
	template<typename _Scalar> class InceptionModule;
	template<typename _Scalar>
	class InceptionLane : private SequentialNeuralNetwork<_Scalar> {
		friend class InceptionModule<_Scalar>;
		friend class InceptionNeuralNetwork<_Scalar>;
	public:
		InceptionLane(std::vector<Layer<_Scalar>*> layers) :
				SequentialNeuralNetwork<_Scalar>::SequentialNeuralNetwork(layers) {
			NeuralNetwork<Scalar>::set_input_layer(*(SequentialNeuralNetwork<_Scalar>::layers[0]), false);
		};
		InceptionLane(const InceptionLane<_Scalar>& network) :
				SequentialNeuralNetwork<_Scalar>::SequentialNeuralNetwork(network) { };
		InceptionLane(InceptionLane<Scalar>&& network) :
				SequentialNeuralNetwork<_Scalar>::SequentialNeuralNetwork(network) { };
		InceptionLane<_Scalar>& operator=(InceptionLane<_Scalar> network) {
			SequentialNeuralNetwork<_Scalar>::swap(*this, network);
			return *this;
		};
	};
	template<typename __Scalar>
	class InceptionModule {
		friend class InceptionNeuralNetwork<__Scalar>;
	public:
		InceptionModule(std::vector<InceptionLane<__Scalar>> lanes) :
				lanes(lanes) {
			assert(lanes.size() > 0 && "lanes must contain at least 1 element");
			InceptionLane<__Scalar>& first_lane = lanes[0];
			Dimensions input_dims = first_lane.input_dims;
			int output_height = first_lane.output_dims.get_dim1();
			int output_width = first_lane.output_dims.get_dim2();
			int output_depth = first_lane.output_dims.get_dim3();
			for (unsigned i = 1; i < lanes.size(); i++) {
				InceptionLane<__Scalar>& lane = lanes[i];
				assert(input_dims.equals(lane.input_dims) && output_height == lane.output_dims.get_dim1() &&
						output_width == lane.output_dims.get_dim2() && "incompatible lane dimensions");
				output_depth += lane.output_dims.get_dim3();
			}
			this->input_dims = input_dims;
			output_dims = Dimensions(output_height, output_width, output_depth);
		};
	private:
		Tensor4<__Scalar> propagate(Tensor4<__Scalar> input, bool training) {
			NeuralNetwork<__Scalar>::assert_popagation_input(input, input_dims);
			int rows = input.dimension(0);
			Array4<int> offsets({ 0, 0, 0, 0 });
			Array4<int> extents({ rows, output_dims.get_dim1(), output_dims.get_dim2(), 0});
			Tensor4<Scalar> out(rows, output_dims.get_dim1(), output_dims.get_dim2(), output_dims.get_dim3());
			for (unsigned i = 0; i < lanes.size(); i++) {
				InceptionLane<__Scalar>& lane = lane[i];
				int depth = lane.output_dims.get_dim3();
				offsets[3] += depth;
				extents[3] = depth;
				out.slice(offsets, extents) = lane.propagate(input, training);
			}
			return out;
		};
		Tensor4<__Scalar> backpropagate(Tensor4<__Scalar> out_grads) {
			NeuralNetwork<__Scalar>::assert_popagation_input(out_grads, output_dims);
			int rows = input.dimension(0);
			Array4<int> offsets({ 0, 0, 0, 0 });
			Array4<int> extents({ rows, output_dims.get_dim1(), output_dims.get_dim2(), 0});
			Tensor4<Scalar> prev_out_grads(rows, input.get_dim1(), input.get_dim2(), input.get_dim3());
			for (unsigned i = 0; i < lanes.size(); i++) {
				InceptionLane<__Scalar>& lane = lane[i];
				int depth = lane.output_dims.get_dim3();
				offsets[3] += depth;
				extents[3] = depth;
				Tensor4<Scalar> out_grads_slice_i = out_grads.slice(offsets, extents);
			}
			return out;
		};
	private:
		std::vector<InceptionLane<ModuleScalar>> lanes;
		Dimensions input_dims;
		Dimensions output_dims;
	};
protected:
	Matrix<Scalar> propagate(Tensor4<Scalar> input, bool training) {
		NeuralNetwork<Scalar>::check_popagation_input(input, input_dims);
		return 0;
	};
	void backpropagate(Tensor4<Scalar> out_grads) {
		NeuralNetwork<Scalar>::check_popagation_input(out_grads, output_dims);
		return;
	};
private:
	std::vector<InceptionModule<Scalar>> modules;
	Dimensions input_dims;
	Dimensions output_dims;
};

template<typename Scalar>
class ResidualNeuralNetwork : public SequentialNeuralNetwork<Scalar> {
public:
	ResidualNeuralNetwork(std::vector<Layer<Scalar>*> layers) :
			SequentialNeuralNetwork<Scalar>::SequentialNeuralNetwork(layers) { };
	ResidualNeuralNetwork(const ResidualNeuralNetwork<Scalar>& network) :
			SequentialNeuralNetwork<Scalar>::SequentialNeuralNetwork(network) { };
	ResidualNeuralNetwork(ResidualNeuralNetwork<Scalar>&& network) :
			SequentialNeuralNetwork<Scalar>::SequentialNeuralNetwork(network) { };
	ResidualNeuralNetwork<Scalar>& operator=(ResidualNeuralNetwork<Scalar> network) {
		SequentialNeuralNetwork<Scalar>::swap(*this, network);
		return *this;
	};
protected:
	Matrix<Scalar> propagate(Tensor4<Scalar> input, bool training) {
		NeuralNetwork<Scalar>::assert_popagation_input(input, input_dims);
		for (unsigned i = 0; i < SequentialNeuralNetwork<Scalar>::layers.size(); i++)
			input = NeuralNetwork<Scalar>::pass_forward(*(SequentialNeuralNetwork<Scalar>::layers[i]), input, training);
		return input;
	};
	void backpropagate(Tensor4<Scalar> out_grads) {
		NeuralNetwork<Scalar>::assert_popagation_input(out_grads, output_dims);
		for (int i = SequentialNeuralNetwork<Scalar>::layers.size() - 1; i >= 0; i--)
			out_grads = NeuralNetwork<Scalar>::pass_back(*(SequentialNeuralNetwork<Scalar>::layers[i]), out_grads);
	};
};

} /* namespace cppnn */

#endif /* NEURALNETWORK_H_ */
