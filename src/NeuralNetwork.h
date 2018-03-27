/*
 * NeuralNetwork.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <memory>
#include <pthread.h>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>
#include "Dimensions.h"
#include "Layer.h"
#include "Utils.h"

// TODO GRU network.
// TODO Possibility to add and remove modules (e.g. layers for sequential networks, inception modules for InceptionNets).
// TODO Serialization.

namespace cattle {

template<typename Scalar, std::size_t Rank, bool Sequential> class Optimizer;

/**
 * An enumeration type for the different ways the outputs of sub-modules of neural networks
 * may be merged.
 */
enum OutputMergeType { CONCAT_LO_RANK, CONCAT_HI_RANK, SUM };

/**
 * An abstract neural network class template. It allows for inference and training via
 * back-propagation.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NeuralNetwork {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal neural network rank");
	friend class Optimizer<Scalar,Rank,Sequential>;
	template<typename _Scalar, std::size_t _Rank, bool _Sequential>
	friend class CompositeNeuralNetwork;
	template<typename _Scalar, std::size_t _Rank, OutputMergeType MergeType>
	friend class ParallelNeuralNetwork;
	template<typename _Scalar, std::size_t _Rank>
	friend class SequentialNeuralNetwork;
protected:
	static constexpr std::size_t DATA_RANKS = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANKS> Data;
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
	 * @return Whether the instance is a foremost network. If the instance is not a stand-alone
	 * network and it is not the first module of a complex network, it is not a foremost
	 * network. Foremost networks do not need to back-propagate the gradients all the way
	 * given that no other network is expected to depend on them.
	 */
	virtual bool is_foremost() const = 0;
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
	inline virtual void init() {
		std::vector<Layer<Scalar,Rank>*> layers = get_layers();
		for (unsigned i = 0; i < layers.size(); ++i)
			layers[i]->init();
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
	/**
	 * @return A string representation of the neural network.
	 */
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
	friend std::ostream& operator<<(std::ostream& os, NeuralNetwork<Scalar,Rank,Sequential>& nn) {
		return os << nn.to_string() << std::endl;
	}
protected:
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
	 * @return A vector of pointers to the layers of the network. The ownership of the
	 * layers remains with the network.
	 */
	virtual std::vector<Layer<Scalar,Rank>*> get_layers() = 0;
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
	 * @param out_grads The derivative of the loss function w.r.t. the output of the
	 * network.
	 * @return The derivative of the loss function w.r.t. the input of the network or
	 * a null tensor if the network is a foremost network.
	 */
	virtual Data backpropagate(Data out_grads) = 0;
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * NeuralNetwork that are not friend classes of Layer.
	 *
	 * \see Layer#clone_with_shared_params()
	 *
	 * It produces a clone of the specified layer that shares the original layer's
	 * parameters.
	 *
	 * @param layer A reference to the layer to clone.
	 * @return A pointer to the clone that uses a reference to the original layer's
	 * parameters.
	 */
	inline static Layer<Scalar,Rank>* clone_with_shared_params(const Layer<Scalar,Rank>& layer) {
		return layer.clone_with_shared_params();
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * NeuralNetwork that are not friend classes of Layer.
	 *
	 * \see Layer#set_input_layer(bool)
	 *
	 * It sets the specified layer's input-layer-status.
	 *
	 * @param layer The layer to modify.
	 * @param on Whether the layer is to function as an input layer.
	 */
	inline static void set_input_layer(Layer<Scalar,Rank>& layer, bool on) {
		layer.set_input_layer(on);
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * NeuralNetwork that are not friend classes of Layer.
	 *
	 * \see Layer#empty_cache()
	 *
	 * It empties the cache of the layer.
	 *
	 * @param layer The layer whose cache is to be emptied.
	 */
	inline static void empty_cache(Layer<Scalar,Rank>& layer) {
		layer.empty_cache();
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * NeuralNetwork that are not friend classes of Layer.
	 *
	 * \see Layer#pass_forward(Tensor<Scalar,Rank + 1>,bool)
	 *
	 * It propagates the input through the specified layer.
	 *
	 * @param layer The layer to which the input is to be fed.
	 * @param prev_out The input tensor.
	 * @param training Whether the input is to propagated through the layer in
	 * training mode.
	 * @return The output of the layer in response to the input.
	 */
	inline static Tensor<Scalar,Rank + 1> pass_forward(Layer<Scalar,Rank>& layer, Tensor<Scalar,Rank + 1> prev_out,
			bool training) {
		return layer.pass_forward(std::move(prev_out), training);
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * NeuralNetwork that are not friend classes of Layer.
	 *
	 * \see Layer#pass_back(Tensor<Scalar,Rank + 1>)
	 *
	 * It back-propagates the derivative of the loss function w.r.t. the output of
	 * the layer through the layer updating the gradients on its parameters along
	 * the way.
	 *
	 * @param layer The layer through which the gradients are to be back-propagated.
	 * @param out_grads The derivative of the loss function w.r.t. the output of
	 * the layer
	 * @return The derivative of the loss function w.r.t. the input of the layer or
	 * a null tensor if the layer is an input layer.
	 */
	inline static Tensor<Scalar,Rank + 1> pass_back(Layer<Scalar,Rank>& layer, Tensor<Scalar,Rank + 1> out_grads) {
		return layer.pass_back(std::move(out_grads));
	}
	/**
	 * A method to expose protected methods of the Layer class to subclasses of
	 * NeuralNetwork that are not friend classes of Layer.
	 *
	 * \see Layer#get_params_grad()
	 *
	 * It returns a non-constant reference to the gradient of the specified
	 * layer's parameters.
	 *
	 * @param layer The layer whose parameters' gradient is to be fetched.
	 * @return A non-constant reference to the gradient of the layer's
	 * parameters.
	 */
	inline static Matrix<Scalar>& get_params_grad(Layer<Scalar,Rank>& layer) {
		return layer.get_params_grad();
	}
};

/**
 * An enumeration type for the different ways the input of a layer in a dense network may be concatenated
 * to its output.
 */
enum DenseConcatType { LOWEST_RANK, HIGHEST_RANK };

/**
 * An alias for a unique pointer to a neural network of arbitrary scalar type, rank,
 * and sequentiality.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using NeuralNetPtr = std::unique_ptr<NeuralNetwork<Scalar,Rank,Sequential>>;

/**
 * A class template for a composite neural network that consists of a set of
 * serially stacked neural network sub-modules.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class CompositeNeuralNetwork : public NeuralNetwork<Scalar,Rank,Sequential> {
	template<typename _Scalar, std::size_t _Rank>
	friend class ResidualNeuralNetwork;
	template<typename _Scalar, std::size_t _Rank, DenseConcatType ConcatType>
	friend class DenseNeuralNetwork;
	typedef NeuralNetwork<Scalar,Rank,Sequential> Base;
	typedef CompositeNeuralNetwork<Scalar,Rank,Sequential> Self;
	typedef NeuralNetPtr<Scalar,Rank,Sequential> Block;
public:
	/**
	 * @param blocks A vector of unique pointers to neural networks.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
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
	/**
	 * @param block A unique pointer to a neural network.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline CompositeNeuralNetwork(Block block, bool foremost = true) :
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
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
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
	/**
	 * Creates a size-1 vector out of the specified unique pointer.
	 *
	 * @param net A unique pointer to the single neural network sub-module.
	 * @return A vector of size 1 containing the unique pointer.
	 */
	inline static std::vector<Block> create_vector(Block net) {
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

/**
 * A class template for a parallel neural network that consists of one or more
 * lanes of non-sequential neural networks with the same input dimensions. Inputs
 * and gradients are propagated through the lanes simultaneously using multithreading.
 * The outputs of the lanes are merged by concatenation (either along the lowest
 * or hightest rank) or summation.
 */
template<typename Scalar, std::size_t Rank, OutputMergeType MergeType = CONCAT_HI_RANK>
class ParallelNeuralNetwork : public NeuralNetwork<Scalar,Rank,false> {
	typedef NeuralNetwork<Scalar,Rank,false> Base;
	typedef ParallelNeuralNetwork<Scalar,Rank,MergeType> Self;
	typedef NeuralNetPtr<Scalar,Rank,false> Lane;
	typedef std::array<std::size_t,Base::DATA_RANKS> RankwiseArray;
	static_assert(MergeType >= CONCAT_LO_RANK && MergeType <= SUM, "illegal merge type value");
	static const std::size_t CONCAT_RANK = MergeType == CONCAT_HI_RANK ? Rank - 1 : 0;
	static const std::size_t CONCAT_BATCH_RANK = CONCAT_RANK + 1;
public:
	/**
	 * @param lanes A vector of unique pointers to non-sequential neural networks.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline ParallelNeuralNetwork(std::vector<Lane> lanes, bool foremost = true) :
			lanes(std::move(lanes)),
			foremost(foremost) {
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
			if (MergeType != SUM) {
				if (MergeType == CONCAT_HI_RANK) {
					for (std::size_t i = 0; i < CONCAT_RANK; ++i)
						assert(output_dims(i) == lane_output_dims(i));
				} else {
					for (std::size_t i = Rank - 1; i > CONCAT_RANK; --i)
						assert(output_dims(i) == lane_output_dims(i));
				}
				output_dims(CONCAT_RANK) += lane_output_dims(CONCAT_RANK);
			} else
				assert(output_dims == lane_output_dims);
		}
		set_foremost(foremost);
		this->input_dims = first_lane.get_input_dims();
		this->output_dims = output_dims;
	}
	/**
	 * @param lane A unique pointer to a non-sequential neural network.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline ParallelNeuralNetwork(Base&& lane, bool foremost = true) :
			ParallelNeuralNetwork(create_vector(std::move(lane)), foremost) { }
	inline ParallelNeuralNetwork(const Self& network) :
			lanes(network.lanes.size()),
			foremost(network.foremost),
			input_dims(network.input_dims),
			output_dims(network.output_dims) {
		for (unsigned i = 0; i < lanes.size(); ++i)
			lanes[i] = Lane(network.lanes[i]->clone());
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
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
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
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers;
		for (unsigned i = 0; i < lanes.size(); ++i) {
			std::vector<Layer<Scalar,Rank>*> lane_layers = lanes[i]->get_layers();
			for (unsigned j = 0; j < lane_layers.size(); ++j)
				layers.push_back(lane_layers[j]);
		}
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input).template demote<>());
		std::size_t rows = input.dimension(0);
		typename Base::Data out;
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
			if (i == 0)
				out = std::move(args_arr[i].out);
			else {
				assert(!pthread_join(threads[i - 1], nullptr));
				if (MergeType == SUM)
					out += args_arr[i].out;
				else {
					// Must be evaluated first due to the dimension difference.
					typename Base::Data concat = out.concatenate(std::move(args_arr[i].out), CONCAT_BATCH_RANK);
					out = std::move(concat);
				}
			}
		}
		if (helper_thread_num > 0)
			pthread_attr_destroy(&attr);
		return out;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads).template demote<>());
		typename Base::Data prev_out_grads;
		if (foremost)
			prev_out_grads = typename Base::Data();
		else {
			RankwiseArray dims = input_dims.template promote<>();
			dims[0] = out_grads.dimension(0);
			prev_out_grads = typename Base::Data(dims);
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
		int concat_rank_offset = out_grads.dimension(CONCAT_BATCH_RANK);
		for (int i = helper_thread_num; i >= 0; --i) {
			concat_rank_offset -= lanes[i]->get_output_dims()(CONCAT_RANK);
			BackpropArgs args;
			args.obj = this;
			args.lane_id = i;
			args.concat_rank_offset = concat_rank_offset;
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
	/**
	 * Creates a size-1 vector out of the specified unique pointer.
	 *
	 * @param net A unique pointer to the single non-sequential neural network lane.
	 * @return A vector of size 1 containing the unique pointer.
	 */
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
	/**
	 * A struct containing the data required for propagation.
	 */
	struct PropArgs {
		Self* obj;
		int lane_id;
		bool training;
		typename Base::Data* in;
		typename Base::Data out;
	};
	/**
	 * A struct containing the data require for back-propagation.
	 */
	struct BackpropArgs {
		Self* obj;
		int lane_id;
		int concat_rank_offset;
		typename Base::Data* out_grads;
		typename Base::Data prev_out_grads;
	};
	/**
	 * The propagation function executed in a different thread for each lane of a
	 * parallel network.
	 *
	 * @param args_ptr The propagation argument struct containing all necessary
	 * information.
	 */
	inline static void* propagate(void* args_ptr) {
		PropArgs& args = *((PropArgs*) args_ptr);
		args.out = args.obj->lanes[args.lane_id]->propagate(*args.in, args.training);
		return nullptr;
	}
	/**
	 * The back-propagation function executed in a different thread for each lane of a
	 * parallel network.
	 *
	 * @param args_ptr The back-propagation argument struct containing all necessary
	 * information.
	 */
	inline static void* backpropagate(void* args_ptr) {
		BackpropArgs& args = *((BackpropArgs*) args_ptr);
		Base& lane = *args.obj->lanes[args.lane_id];
		if (MergeType != SUM) {
			RankwiseArray offsets;
			RankwiseArray extents = lane.get_output_dims().template promote<>();
			offsets.fill(0);
			offsets[CONCAT_BATCH_RANK] = args.concat_rank_offset;
			extents[0] = args.out_grads->dimension(0);
			typename Base::Data out_grads_slice = args.out_grads->slice(offsets, extents);
			args.prev_out_grads = lane.backpropagate(std::move(out_grads_slice));
		} else
			args.prev_out_grads = lane.backpropagate(*args.out_grads);
		return nullptr;
	}
};

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
	/**
	 * @param layer A unique pointer to the single layer of the network.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
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
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
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
 * A class template for ResNets. These networks take vectors of CompositeNeuralNetwork and bool pairs
 * as their sub-modules. The second, boolean value of the pair denotes whether the sub-module is a
 * residual module or not.
 */
template<typename Scalar, std::size_t Rank>
class ResidualNeuralNetwork : public NeuralNetwork<Scalar,Rank,false> {
	typedef NeuralNetwork<Scalar,Rank,false> Base;
	typedef CompositeNeuralNetwork<Scalar,Rank,false> Module;
public:
	/**
	 * @param modules A vector of CompositeNeuralNetwork and bool pairs with the boolean denoting
	 * whether the respective network module is residual.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
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
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
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

/**
 * A class template for DenseNet architectures. These networks consist of sub-modules that are all
 * 'connected' to each other as in the input of each module is concatenated to its output and then
 * propagated to the next module as its input. The input is concatenated to the output either along
 * its lowest or highest rank.
 */
template<typename Scalar, std::size_t Rank, DenseConcatType ConcatType = HIGHEST_RANK>
class DenseNeuralNetwork : public NeuralNetwork<Scalar,Rank,false> {
	typedef NeuralNetwork<Scalar,Rank,false> Base;
	typedef CompositeNeuralNetwork<Scalar,Rank,false> Module;
	typedef std::array<std::size_t,Base::DATA_RANKS> RankwiseArray;
	static_assert(ConcatType >= LOWEST_RANK && ConcatType <= HIGHEST_RANK, "illegal merge type value");
	static const std::size_t CONCAT_RANK = ConcatType == HIGHEST_RANK ? Rank - 1 : 0;
	static const std::size_t CONCAT_BATCH_RANK = CONCAT_RANK + 1;
public:
	/**
	 * @param modules A vector of CompositeNeuralNetwork instances.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline DenseNeuralNetwork(std::vector<Module> modules, bool foremost = true) :
			modules(modules),
			foremost(foremost) {
		assert(this->modules.size() > 0 && "modules must contain at least 1 element");
		Module& first_module = this->modules[0];
		input_dims = first_module.get_input_dims();
		typename Base::Dims output_dims = first_module.get_output_dims();
		output_dims(CONCAT_RANK) += input_dims(CONCAT_RANK);
		if (ConcatType == LOWEST_RANK) {
			for (std::size_t i = Rank - 1; i > CONCAT_RANK; --i)
				assert(input_dims(i) == output_dims(i));
		} else {
			for (std::size_t i = 0; i < CONCAT_RANK; ++i)
				assert(input_dims(i) == output_dims(i));
		}
		for (unsigned i = 1; i < this->modules.size(); ++i) {
			Module& module = this->modules[i];
			const typename Base::Dims& module_input_dims = module.get_input_dims();
			assert(module_input_dims == output_dims && "incompatible module dimensions");
			output_dims(CONCAT_RANK) += module.get_output_dims()(CONCAT_RANK);
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
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
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
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers;
		for (unsigned i = 0; i < modules.size(); ++i) {
			std::vector<Layer<Scalar,Rank>*> module_layers = modules[i].get_layers();
			for (unsigned j = 0; j < module_layers.size(); ++j)
				layers.push_back(module_layers[j]);
		}
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		Dimensions<std::size_t,Base::DATA_RANKS> dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input);
		assert(input_dims == dims.template demote<>());
		RankwiseArray offsets;
		RankwiseArray extents = dims;
		offsets.fill(0);
		for (unsigned i = 0; i < modules.size(); ++i) {
			typename Base::Data concat = input.concatenate(modules[i].propagate(input, training), CONCAT_BATCH_RANK);
			input = std::move(concat);
		}
		return input;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		assert(output_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads).template demote<>());
		RankwiseArray offsets;
		RankwiseArray extents = input_dims.template promote<>();
		offsets.fill(0);
		extents[0] = out_grads.dimension(0);
		for (int i = modules.size() - 1; i >= 0; --i) {
			Module& module = modules[i];
			int layer_input_concat_rank_dim = module.get_input_dims()(CONCAT_RANK);
			int layer_output_concat_rank_dim = module.get_output_dims()(CONCAT_RANK);
			offsets[CONCAT_BATCH_RANK] = layer_input_concat_rank_dim;
			extents[CONCAT_BATCH_RANK] = layer_output_concat_rank_dim;
			typename Base::Data out_grads_i = out_grads.slice(offsets, extents);
			offsets[CONCAT_BATCH_RANK] = 0;
			extents[CONCAT_BATCH_RANK] = layer_input_concat_rank_dim;
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

/**
 * A class template for a wrapper neural network that enables the use of non-sequential networks on
 * sequential data by joining the 'samples' and 'time steps' ranks of the tensors and splitting them
 * again once the internal, non-sequential network is done processing them.
 */
template<typename Scalar, std::size_t Rank>
class SequentialNeuralNetwork : public NeuralNetwork<Scalar,Rank,true> {
	typedef NeuralNetwork<Scalar,Rank,true> Base;
	typedef SequentialNeuralNetwork<Scalar,Rank> Self;
	typedef NeuralNetPtr<Scalar,Rank,false> Net;
	typedef std::array<std::size_t,Base::DATA_RANKS> RankwiseArray;
public:
	/**
	 * @param network A unique pointer to a non-sequential neural network to wrap.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
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
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
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

/**
 * An abstract class template for unidirectional recurrent neural networks.
 */
template<typename Scalar, std::size_t Rank>
class UnidirectionalNeuralNetwork : public NeuralNetwork<Scalar,Rank,true> {
	template<typename _Scalar, std::size_t _Rank, OutputMergeType MergeType>
	friend class BidirectionalNeuralNetwork;
public:
	virtual ~UnidirectionalNeuralNetwork() = default;
protected:
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
};

/**
 * An alias for unidirectional recurrent neural network of arbitrary scalar type and rank.
 */
template<typename Scalar, std::size_t Rank>
using UnidirNeuralNetPtr = std::unique_ptr<UnidirectionalNeuralNetwork<Scalar,Rank>>;

/**
 * A class template for a bidirectional neural network that takes a unidirectional recurrent
 * network, clones it, reverses the clone's processing direction, and uses the two networks
 * as its parallel sub-modules. The outputs of the two sub-networks can be merged by summation
 * or concatenation either along the lowest (the 3rd after the sample and time-step ranks) or
 * highest rank.
 */
template<typename Scalar, std::size_t Rank, OutputMergeType MergeType = CONCAT_LO_RANK>
class BidirectionalNeuralNetwork : public NeuralNetwork<Scalar,Rank,true> {
	typedef NeuralNetwork<Scalar,Rank,true> Base;
	typedef BidirectionalNeuralNetwork<Scalar,Rank,MergeType> Self;
	typedef UnidirNeuralNetPtr<Scalar,Rank> UnidirNet;
	typedef std::array<std::size_t,Base::DATA_RANKS> RankwiseArray;
	static_assert(MergeType >= CONCAT_LO_RANK && MergeType <= SUM, "illegal merge type value");
	static const std::size_t CONCAT_RANK = MergeType == CONCAT_HI_RANK ? Rank - 1 : 0;
	static const std::size_t CONCAT_BATCH_RANK = CONCAT_RANK + 2;
public:
	/**
	 * @param network A unique pointer to a unidirectional recurrent neural network that,
	 * along with its reversed clone, will constitute the bidirectional network.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline BidirectionalNeuralNetwork(UnidirNet network, bool foremost = true) :
				net(std::move(network)),
				foremost(foremost) {
		assert(this->net);
		net_rev = UnidirNet((UnidirectionalNeuralNetwork<Scalar,Rank>*) this->net->clone());
		net_rev->reverse();
		input_dims = this->net->get_input_dims();
		output_dims = this->net->get_output_dims();
		if (MergeType != SUM)
			output_dims(CONCAT_RANK) *= 2;
	}
	inline BidirectionalNeuralNetwork(const Self& network) :
			net(UnidirNet((UnidirectionalNeuralNetwork<Scalar,Rank>*) network.net->clone())),
			net_rev(UnidirNet((UnidirectionalNeuralNetwork<Scalar,Rank>*) network.net_rev->clone())),
			foremost(network.foremost),
			input_dims(network.input_dims),
			output_dims(network.output_dims) { }
	inline BidirectionalNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~BidirectionalNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Base* clone() const {
		return new BidirectionalNeuralNetwork(*this);
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline const typename Base::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Base::Dims& get_output_dims() const {
		return output_dims;
	}
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.net, network2.net);
		swap(network1.net_rev, network2.net_rev);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
	}
protected:
	inline void set_foremost(bool foremost) {
		net->set_foremost(foremost);
		net_rev->set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		net->empty_caches();
		net_rev->empty_caches();
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers;
		std::vector<Layer<Scalar,Rank>*> net_layers = net->get_layers();
		for (std::size_t i = 0; i < net_layers.size(); ++i)
			layers.push_back(net_layers[i]);
		std::vector<Layer<Scalar,Rank>*> net_rev_layers = net_rev->get_layers();
		for (std::size_t i = 0; i < net_rev_layers.size(); ++i)
			layers.push_back(net_rev_layers[i]);
		return layers;
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(input);
		assert(input_dims == Utils<Scalar>::template get_dims<Base::DATA_RANKS>(input).template demote<2>());
		pthread_attr_t attr;
		pthread_t helper_thread;
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		PropArgs args;
		args.obj = this;
		args.training = training;
		args.in = &input;
		assert(!pthread_create(&helper_thread, &attr, propagate, &args));
		typename Base::Data forward_out = net->propagate(input, training);
		assert(!pthread_join(helper_thread, nullptr));
		pthread_attr_destroy(&attr);
		assert(forward_out.dimension(1) == args.out.dimension(1));
		if (MergeType != SUM)
			return forward_out.concatenate(args.out, CONCAT_BATCH_RANK);
		else
			return forward_out + args.out;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(out_grads);
		Dimensions<std::size_t,Base::DATA_RANKS> dims = Utils<Scalar>::template get_dims<Base::DATA_RANKS>(out_grads);
		assert(output_dims == dims.template demote<2>());
		pthread_attr_t attr;
		pthread_t helper_thread;
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		BackpropArgs args;
		args.obj = this;
		typename Base::Data forward_prev_out_grads;
		if (MergeType != SUM) {
			RankwiseArray offsets;
			RankwiseArray extents = dims;
			offsets.fill(0);
			extents[CONCAT_BATCH_RANK] /= 2;
			offsets[CONCAT_BATCH_RANK] += extents[CONCAT_BATCH_RANK];
			typename Base::Data backward_slice = out_grads.slice(offsets, extents);
			args.out_grads = &backward_slice;
			assert(!pthread_create(&helper_thread, &attr, backpropagate, &args));
			offsets[CONCAT_BATCH_RANK] -= extents[CONCAT_BATCH_RANK];
			typename Base::Data forward_slice = out_grads.slice(offsets, extents);
			forward_prev_out_grads = net->backpropagate(std::move(forward_slice));
			// Make sure that backward_slice does not go out of scope before the thread terminates.
			assert(!pthread_join(helper_thread, nullptr));
		} else {
			args.out_grads = &out_grads;
			assert(!pthread_create(&helper_thread, &attr, backpropagate, &args));
			forward_prev_out_grads = net->backpropagate(out_grads);
			assert(!pthread_join(helper_thread, nullptr));
		}
		pthread_attr_destroy(&attr);
		out_grads = typename Base::Data();
		return forward_prev_out_grads + args.prev_out_grads;
	}
private:
	UnidirNet net;
	UnidirNet net_rev;
	bool foremost;
	typename Base::Dims input_dims;
	typename Base::Dims output_dims;
	/**
	 * A struct containing the data required for propagation.
	 */
	struct PropArgs {
		Self* obj;
		bool training;
		typename Base::Data* in;
		typename Base::Data out;
	};
	/**
	 * A struct containing the data require for back-propagation.
	 */
	struct BackpropArgs {
		Self* obj;
		typename Base::Data* out_grads;
		typename Base::Data prev_out_grads;
	};
	/**
	 * The propagation function executed in a different thread for each lane of a
	 * parallel network.
	 *
	 * @param args_ptr The propagation argument struct containing all necessary
	 * information.
	 */
	inline static void* propagate(void* args_ptr) {
		PropArgs& args = *((PropArgs*) args_ptr);
		args.out = args.obj->net_rev->propagate(*args.in, args.training);
		return nullptr;
	}
	/**
	 * The back-propagation function executed in a different thread for each lane of a
	 * parallel network.
	 *
	 * @param args_ptr The back-propagation argument struct containing all necessary
	 * information.
	 */
	inline static void* backpropagate(void* args_ptr) {
		BackpropArgs& args = *((BackpropArgs*) args_ptr);
		args.prev_out_grads = args.obj->net_rev->backpropagate(*args.out_grads);
		return nullptr;
	}
};

/**
 * An alias for a unique pointer to a kernel layer of arbitrary rank and scalar type.
 */
template<typename Scalar, std::size_t Rank>
using KernelPtr = std::unique_ptr<KernelLayer<Scalar,Rank>>;

/**
 * An alias for a unique pointer to an activation layer of arbitrary rank and scalar type.
 */
template<typename Scalar, std::size_t Rank>
using ActivationPtr = std::unique_ptr<ActivationLayer<Scalar,Rank>>;

/**
 * A class template for a simple recurrent neural network (RNN). The network can use multiplicative
 * integration to combine its linearly transformed input and its linearly transformed previous hidden
 * state. A stateful network retains its hidden state across sequences as long as the batch size is
 * constant.
 */
template<typename Scalar, std::size_t Rank, bool MulInt = false, bool Stateful = false>
class RecurrentNeuralNetwork : public UnidirectionalNeuralNetwork<Scalar,Rank> {
	typedef NeuralNetwork<Scalar,Rank,true> Root;
	typedef RecurrentNeuralNetwork<Scalar,Rank> Self;
	typedef std::array<std::size_t,Root::DATA_RANKS> RankwiseIntArray;
	typedef std::array<bool,Root::DATA_RANKS> RankwiseBoolArray;
	typedef std::function<std::pair<std::size_t,std::size_t>(std::size_t)> OutputSeqSizeFunc;
	typedef Tensor<Scalar,Rank + 1> TimeStepData;
public:
	/**
	 * @param input_kernel The linear layer applied to the input of the network at each time step
	 * with an input.
	 * @param state_kernel The linear layer applied to the previous hidden state of the network at
	 * each time step.
	 * @param output_kernel The linear layer applied to the hidden state of the network at each time
	 * step with an output.
	 * @param state_act The activation function applied to the hidden state at each time step.
	 * @param output_act The activation function applied to the linearly transformed hidden state
	 * of the network at each time step with an output.
	 * @param output_seq_size_func A function parameterized by the input sequence length that
	 * determines the output sequence delay and length
	 * @param reversed Whether the network is to reverse its inputs along the time-step rank.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline RecurrentNeuralNetwork(KernelPtr<Scalar,Rank> input_kernel, KernelPtr<Scalar,Rank> state_kernel,
			KernelPtr<Scalar,Rank> output_kernel, ActivationPtr<Scalar,Rank> state_act, ActivationPtr<Scalar,Rank> output_act,
			OutputSeqSizeFunc output_seq_size_func, bool reversed = false, bool foremost = true) :
				main_cell(),
				output_seq_size_func(output_seq_size_func),
				reversed(reversed),
				foremost(foremost),
				cells(0),
				batch_size(-1),
				input_seq_length(-1),
				output_seq_length(-1),
				output_seq_delay(-1) {
		assert(input_kernel && state_kernel && output_kernel && state_act && output_act);
		typename Root::Dims input_layer_input_dims = input_kernel->get_input_dims();
		typename Root::Dims input_layer_output_dims = input_kernel->get_output_dims();
		typename Root::Dims output_layer_output_dims = output_kernel->get_output_dims();
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
	inline RecurrentNeuralNetwork(const Self& network) :
			main_cell(),
			output_seq_size_func(network.output_seq_size_func),
			reversed(network.reversed),
			foremost(network.foremost),
			input_dims(network.input_dims),
			output_dims(network.output_dims),
			cells(network.cells.size()),
			state(network.state),
			batch_size(network.batch_size),
			input_seq_length(network.input_seq_length),
			output_seq_length(network.output_seq_length),
			output_seq_delay(network.output_seq_delay) {
		main_cell.input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) network.main_cell.input_kernel->clone());
		main_cell.state_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) network.main_cell.state_kernel->clone());
		main_cell.output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) network.main_cell.output_kernel->clone());
		main_cell.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.state_act->clone());
		main_cell.output_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.output_act->clone());
		main_cell.state_kernel_cache = network.main_cell.state_kernel_cache;
		main_cell.input_kernel_cache = network.main_cell.input_kernel_cache;
		for (std::size_t i = 0; i < cells.size(); i++) {
			Cell& c1 = cells[i];
			const Cell& c2 = network.cells[i];
			c1.input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.input_kernel->clone());
			c1.state_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.state_kernel->clone());
			c1.output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*) c2.output_kernel->clone());
			c1.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.state_act->clone());
			c1.output_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*) c2.output_act->clone());
			c1.state_kernel_cache = c2.state_kernel_cache;
			c1.input_kernel_cache = c2.input_kernel_cache;
		}
	}
	inline RecurrentNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~RecurrentNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Root* clone() const {
		return new RecurrentNeuralNetwork(*this);
	}
	inline bool is_reversed() const {
		return reversed;
	}
	inline void reverse() {
		this->reversed = !this->reversed;
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline const typename Root::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Root::Dims& get_output_dims() const {
		return output_dims;
	}
	// For the copy-and-swap idiom.
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.main_cell, network2.main_cell);
		swap(network1.output_seq_size_func, network2.output_seq_size_func);
		swap(network1.reversed, network2.reversed);
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
		Root::set_input_layer(*main_cell.input_kernel, foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		TimeStepData null_tensor;
		Root::empty_cache(*main_cell.input_kernel);
		Root::empty_cache(*main_cell.state_kernel);
		Root::empty_cache(*main_cell.output_kernel);
		Root::empty_cache(*main_cell.state_act);
		Root::empty_cache(*main_cell.output_act);
		main_cell.state_kernel_cache = null_tensor;
		main_cell.input_kernel_cache = null_tensor;
		// Clear the state as well.
		batch_size = -1;
		state = null_tensor;
		input_seq_length = -1;
		output_seq_length = -1;
		output_seq_delay = -1;
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
	inline typename Root::Data propagate(typename Root::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Root::DATA_RANKS>(input);
		Dimensions<std::size_t,Root::DATA_RANKS> data_dims = Utils<Scalar>::template get_dims<Root::DATA_RANKS>(input);
		assert(input_dims == data_dims.template demote<2>());
		int samples = data_dims(0);
		int input_seq_length = data_dims(1);
		// Calculate the output sequence length and delay based on the input sequence length.
		std::pair<std::size_t,std::size_t> output_seq_info = output_seq_size_func((std::size_t) input_seq_length);
		int output_seq_length = (int) output_seq_info.first;
		int output_seq_delay = (int) output_seq_info.second;
		assert(output_seq_length > 0);
		if (reversed) {
			RankwiseBoolArray reverse;
			reverse.fill(false);
			reverse[1] = true;
			input = input.reverse(reverse);
		}
		int output_end = output_seq_length + output_seq_delay;
		int time_steps = std::max(input_seq_length, output_end);
		// If in training mode, unroll the network (unless it has already been unrolled for the same alignment).
		if (training && (input_seq_length != this->input_seq_length ||
				output_seq_length != this->output_seq_length || output_seq_delay != this->output_seq_delay)) {
			if (time_steps > 1) {
				// Empty the caches of the main cell to reduce the amount of data to copy.
				empty_caches();
				// Emptying the caches also clears the cell vector, thus it has to be recreated afterwards.
				cells = std::vector<Cell>(time_steps - 1);
				// Unroll the network by creating n -1 copies of the main cell;
				for (int j = 1; j < time_steps; ++j) {
					Cell& cell = cells[j - 1];
					cell.state_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.state_kernel));
					cell.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.state_act));
					// Only copy the kernels and activations that will actually be used.
					if (j < input_seq_length)
						cell.input_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
								Root::clone_with_shared_params(*main_cell.input_kernel));
					if (j >= output_seq_delay && j < output_end) {
						cell.output_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
								Root::clone_with_shared_params(*main_cell.output_kernel));
						cell.output_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
								Root::clone_with_shared_params(*main_cell.output_act));
					}
				}
			} else
				cells = std::vector<Cell>(0);
		}
		// If the network is stateful, retain the state.
		if (!Stateful || batch_size == -1) {
			Dimensions<std::size_t,Rank + 1> dims = main_cell.input_kernel->get_output_dims().template promote<>();
			dims(0) = samples;
			state = Tensor<Scalar,Rank + 1>(dims);
			state.setZero();
		} else if (samples != batch_size) {
			std::array<std::size_t,Rank + 1> offsets;
			std::array<std::size_t,Rank + 1> extents = main_cell.input_kernel->get_output_dims().template promote<>();
			offsets.fill(0);
			extents[0] = samples;
			TimeStepData new_state;
			if (samples > batch_size) {
				new_state = Tensor<Scalar,Rank + 1>(extents);
				new_state.setZero();
				extents[0] = batch_size;
				new_state.slice(offsets, extents) = state;
			} else
				new_state = state.slice(offsets, extents);
			state = std::move(new_state);
		}
		RankwiseIntArray input_offsets;
		RankwiseIntArray input_extents = data_dims;
		RankwiseIntArray output_offsets;
		RankwiseIntArray output_extents = output_dims.template promote<2>();
		input_offsets.fill(0);
		output_offsets.fill(0);
		input_extents[1] = 1;
		output_extents[0] = samples;
		typename Root::Data out;
		// If the output is a single time step prediction, there is no need to create an output tensor.
		if (output_seq_length > 1) {
			output_extents[1] = output_seq_length;
			out = typename Root::Data(output_extents);
		}
		output_extents[1] = 1;
		Dimensions<std::size_t,Rank + 1> input_time_step_dims = input_dims.template promote<>();
		input_time_step_dims(0) = samples;
		for (int i = 0; i < time_steps; ++i) {
			// In inference mode, do not unroll the network.
			Cell& cell = !training || i == 0 ? main_cell : cells[i - 1];
			// Always apply the state kernel.
			state = Root::pass_forward(*cell.state_kernel, std::move(state), training);
			// If in inference mode, empty the caches after passing the data through each layer.
			if (!training)
				Root::empty_cache(*cell.state_kernel);
			// If there is an input for the time step...
			if (i < input_seq_length) {
				typename Root::Data in_i_seq;
				if (input_seq_length == 1)
					in_i_seq = std::move(input);
				else {
					in_i_seq = input.slice(input_offsets, input_extents);
					input_offsets[1] += 1;
				}
				if (MulInt) {
					if (training) {
						/* If multiplicative integration is enabled, cache the factors of the multiplication so that
						 * the function can be differentiated in the backward pass. */
						cell.state_kernel_cache = state;
						cell.input_kernel_cache = Root::pass_forward(*cell.input_kernel,
								Utils<Scalar>::template map_tensor_to_tensor<Root::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
										input_time_step_dims), training);
						state *= cell.input_kernel_cache;
					} else
						state *= Root::pass_forward(*cell.input_kernel,
								Utils<Scalar>::template map_tensor_to_tensor<Root::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
										input_time_step_dims), training);
				} else
					state += Root::pass_forward(*cell.input_kernel,
							Utils<Scalar>::template map_tensor_to_tensor<Root::DATA_RANKS,Rank + 1>(std::move(in_i_seq),
								input_time_step_dims), training);
				if (!training)
					Root::empty_cache(*cell.input_kernel);
			}
			state = Root::pass_forward(*cell.state_act, std::move(state), training);
			if (!training)
				Root::empty_cache(*cell.state_act);
			// If there is an output for the time step...
			if (i >= output_seq_delay && i < output_end) {
				TimeStepData out_i = Root::pass_forward(*cell.output_kernel, state, training);
				if (!training)
					Root::empty_cache(*cell.output_kernel);
				// If the output is a single time step prediction, just return it.
				if (output_seq_length == 1)
					out = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(
							Root::pass_forward(*cell.output_act, std::move(out_i), training),
							output_extents);
				else {
					out.slice(output_offsets, output_extents) = Utils<Scalar>::template map_tensor_to_tensor<
							Rank + 1,Root::DATA_RANKS>(Root::pass_forward(*cell.output_act, std::move(out_i),
									training), output_extents);
					output_offsets[1] += 1;
				}
				if (!training)
					Root::empty_cache(*cell.output_act);
			}
		}
		batch_size = samples;
		this->input_seq_length = input_seq_length;
		this->output_seq_length = output_seq_length;
		this->output_seq_delay = output_seq_delay;
		return out;
	}
	inline typename Root::Data backpropagate(typename Root::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Root::DATA_RANKS>(out_grads);
		Dimensions<std::size_t,Root::DATA_RANKS> data_dims = Utils<Scalar>::template get_dims<Root::DATA_RANKS>(out_grads);
		assert(output_dims == data_dims.template demote<2>() && batch_size == data_dims(0) &&
				output_seq_length == data_dims(1));
		TimeStepData null_tensor;
		RankwiseIntArray output_offsets;
		RankwiseIntArray output_extents = data_dims;
		RankwiseIntArray input_offsets;
		RankwiseIntArray input_extents = input_dims.template promote<2>();
		output_offsets.fill(0);
		output_offsets[1] = output_seq_length - 1;
		input_offsets.fill(0);
		input_offsets[1] = input_seq_length - 1;
		output_extents[1] = 1;
		input_extents[0] = batch_size;
		typename Root::Data prev_out_grads;
		if (input_seq_length > 1) {
			input_extents[1] = input_seq_length;
			prev_out_grads = typename Root::Data(input_extents);
		}
		input_extents[1] = 1;
		TimeStepData state_grads(state.dimensions());
		state_grads.setZero();
		Dimensions<std::size_t,Rank + 1> out_time_step_dims = output_dims.template promote<>();
		out_time_step_dims(0) = batch_size;
		int output_end = output_seq_length + output_seq_delay;
		int time_steps = std::max(input_seq_length, output_end);
		for (int i = time_steps - 1; i >= 0; --i) {
			Cell& cell = i == 0 ? main_cell : cells[i - 1];
			// If there was an output at the time step...
			if (i >= output_seq_delay && i < output_end) {
				typename Root::Data out_grads_seq_i;
				if (output_seq_length == 1)
					out_grads_seq_i = std::move(out_grads);
				else {
					out_grads_seq_i = out_grads.slice(output_offsets, output_extents);
					output_offsets[1] -= 1;
				}
				TimeStepData out_grads_i = Root::pass_back(*cell.output_act,
						Utils<Scalar>::template map_tensor_to_tensor<Root::DATA_RANKS,Rank + 1>(
								std::move(out_grads_seq_i), out_time_step_dims));
				Root::empty_cache(*cell.output_act);
				state_grads += Root::pass_back(*cell.output_kernel, std::move(out_grads_i));
				Root::empty_cache(*cell.output_kernel);
			}
			// Always back-propagate the state gradient.
			state_grads = Root::pass_back(*cell.state_act, std::move(state_grads));
			Root::empty_cache(*cell.state_act);
			// If there was an input at the time step...
			if (i < input_seq_length) {
				// If it is the foremost layer, the gradients do not need to be propagated further back.
				if (foremost) {
					if (MulInt) { // Multiplicative integration.
						Root::pass_back(*cell.input_kernel, cell.state_kernel_cache * state_grads);
						cell.state_kernel_cache = null_tensor;
					} else // Additive integration.
						Root::pass_back(*cell.input_kernel, state_grads);
				} else if (input_seq_length == 1) {
					if (MulInt) {
						prev_out_grads = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(
								Root::pass_back(*cell.input_kernel, cell.state_kernel_cache * state_grads), input_extents);
						cell.state_kernel_cache = null_tensor;
					} else
						prev_out_grads = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(
								Root::pass_back(*cell.input_kernel, state_grads), input_extents);
				} else {
					if (MulInt) {
						prev_out_grads.slice(input_offsets, input_extents) =
								Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(
										Root::pass_back(*cell.input_kernel, cell.state_kernel_cache * state_grads), input_extents);
						cell.state_kernel_cache = null_tensor;
					} else
						prev_out_grads.slice(input_offsets, input_extents) =
								Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(
										Root::pass_back(*cell.input_kernel, state_grads), input_extents);
					input_offsets[1] -= 1;
				}
				Root::empty_cache(*cell.input_kernel);
			}
			// Compute the the state kernel's gradient.
			if (MulInt) {
				state_grads = Root::pass_back(*cell.state_kernel, cell.input_kernel_cache * state_grads);
				cell.input_kernel_cache = null_tensor;
			} else
				state_grads = Root::pass_back(*cell.state_kernel, std::move(state_grads));
			Root::empty_cache(*cell.state_kernel);
		}
		// Roll the network up and accumulate the gradients.
		// FIXME Single evaluation.
		Matrix<Scalar>& u_params_grad = Root::get_params_grad(*main_cell.input_kernel);
		Matrix<Scalar>& w_params_grad = Root::get_params_grad(*main_cell.state_kernel);
		Matrix<Scalar>& v_params_grad = Root::get_params_grad(*main_cell.output_kernel);
		Matrix<Scalar>& state_act_params_grad = Root::get_params_grad(*main_cell.state_act);
		Matrix<Scalar>& output_act_params_grad = Root::get_params_grad(*main_cell.output_act);
		for (int i = 1; i < time_steps; ++i) {
			Cell& cell = cells[i - 1];
			w_params_grad += Root::get_params_grad(*cell.state_kernel);
			state_act_params_grad += Root::get_params_grad(*cell.state_act);
			if (i < input_seq_length)
				u_params_grad += Root::get_params_grad(*cell.input_kernel);
			if (i >= output_seq_delay && i < output_end) {
				v_params_grad += Root::get_params_grad(*cell.output_kernel);
				output_act_params_grad += Root::get_params_grad(*cell.output_act);
			}
		}
		return prev_out_grads;
	}
private:
	/**
	 * A struct representing a cell in the unrolled RNN.
	 */
	struct Cell {
		KernelPtr<Scalar,Rank> input_kernel;
		KernelPtr<Scalar,Rank> state_kernel;
		KernelPtr<Scalar,Rank> output_kernel;
		ActivationPtr<Scalar,Rank> state_act;
		ActivationPtr<Scalar,Rank> output_act;
		// State and input caches for multiplicative integration.
		TimeStepData state_kernel_cache;
		TimeStepData input_kernel_cache;
	};
	Cell main_cell;
	OutputSeqSizeFunc output_seq_size_func;
	bool reversed;
	bool foremost;
	typename Root::Dims input_dims;
	typename Root::Dims output_dims;
	// State.
	std::vector<Cell> cells;
	TimeStepData state;
	int batch_size;
	int input_seq_length;
	int output_seq_length;
	int output_seq_delay;
};

/**
 * A class template representing a long-short term memory (LSTM) recurrent neural network. The network
 * can use multiplicative integration to combine its linearly transformed inputs and its linearly
 * transformed hidden outputs. A stateful network retains its hidden state across sequences as long as
 * the batch size is constant.
 */
template<typename Scalar, std::size_t Rank, bool MulInt = false, bool Stateful = false>
class LSTMNeuralNetwork : public UnidirectionalNeuralNetwork<Scalar,Rank> {
	typedef NeuralNetwork<Scalar,Rank,true> Root;
	typedef LSTMNeuralNetwork<Scalar,Rank> Self;
	typedef std::array<std::size_t,Root::DATA_RANKS> RankwiseIntArray;
	typedef std::array<bool,Root::DATA_RANKS> RankwiseBoolArray;
	typedef std::function<std::pair<std::size_t,std::size_t>(std::size_t)> OutputSeqSizeFunc;
	typedef Tensor<Scalar,Rank + 1> TimeStepData;
public:
	/**
	 * @param input_forget_kernel The forget kernel to apply to the input of the network.
	 * @param output_forget_kernel The forget kernel to apply to the hidden output of the network
	 * at the previous time step.
	 * @param input_write_kernel The write kernel to apply to the input of the network.
	 * @param output_write_kernel The write kernel to apply to the hidden output of the network
	 * at the previous time step.
	 * @param input_candidate_kernel The candidate kernel to apply to the input of the network.
	 * @param output_candidate_kernel The candidate kernel to apply to the hidden output of the
	 * network at the previous time step.
	 * @param input_read_kernel The read kernel to apply to the input of the network.
	 * @param output_read_kernel The read kernel to apply to the hidden output of the network
	 * at the previous time step.
	 * @param forget_act The activation layer of the forget gate. Usually a sigmoid activation
	 * function.
	 * @param write_act The activation layer of the filter of the write gate. Usually a sigmoid
	 * activation function.
	 * @param candidate_act The activation layer of the candidates of the write gate. Usually
	 * a hyperbolic tangent activation function.
	 * @param state_act The activation layer of the state at the read gate. Usually a hyperbolic
	 * tangent activation function.
	 * @param read_act The activation layer of the read filter. Usually a sigmoid activation
	 * function.
	 * @param output_seq_size_func A function parameterized by the input sequence length that
	 * determines the output sequence delay and length.
	 * @param reversed Whether the network is to reverse its inputs along the time-step rank.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline LSTMNeuralNetwork(KernelPtr<Scalar,Rank> input_forget_kernel, KernelPtr<Scalar,Rank> output_forget_kernel,
			KernelPtr<Scalar,Rank> input_write_kernel, KernelPtr<Scalar,Rank> output_write_kernel,
			KernelPtr<Scalar,Rank> input_candidate_kernel, KernelPtr<Scalar,Rank> output_candidate_kernel,
			KernelPtr<Scalar,Rank> input_read_kernel, KernelPtr<Scalar,Rank> output_read_kernel,
			ActivationPtr<Scalar,Rank> forget_act, ActivationPtr<Scalar,Rank> write_act,
			ActivationPtr<Scalar,Rank> candidate_act, ActivationPtr<Scalar,Rank> state_act,
			ActivationPtr<Scalar,Rank> read_act, OutputSeqSizeFunc output_seq_size_func,
			bool reversed = false, bool foremost = true) :
				main_cell(),
				output_seq_size_func(output_seq_size_func),
				reversed(reversed),
				foremost(foremost),
				cells(0),
				batch_size(-1),
				input_seq_length(-1),
				output_seq_length(-1),
				output_seq_delay(-1) {
		assert(output_forget_kernel && input_forget_kernel && output_write_kernel && input_write_kernel &&
				output_candidate_kernel && input_candidate_kernel && output_read_kernel && input_read_kernel &&
				forget_act && write_act && candidate_act && state_act && read_act);
		typename Root::Dims in_forget_kernel_input_dims = input_forget_kernel->get_input_dims();
		typename Root::Dims out_forget_kernel_input_dims = output_forget_kernel->get_input_dims();
		assert(out_forget_kernel_input_dims == input_forget_kernel->get_output_dims() &&
				in_forget_kernel_input_dims == input_write_kernel->get_input_dims() &&
				in_forget_kernel_input_dims == input_candidate_kernel->get_input_dims() &&
				in_forget_kernel_input_dims == input_write_kernel->get_input_dims() &&
				out_forget_kernel_input_dims == output_write_kernel->get_input_dims() &&
				out_forget_kernel_input_dims == output_candidate_kernel->get_input_dims() &&
				out_forget_kernel_input_dims == output_write_kernel->get_input_dims() &&
				out_forget_kernel_input_dims == output_forget_kernel->get_output_dims() &&
				out_forget_kernel_input_dims == input_write_kernel->get_output_dims() &&
				out_forget_kernel_input_dims == output_write_kernel->get_output_dims() &&
				out_forget_kernel_input_dims == input_candidate_kernel->get_output_dims() &&
				out_forget_kernel_input_dims == output_candidate_kernel->get_output_dims() &&
				out_forget_kernel_input_dims == input_read_kernel->get_output_dims() &&
				out_forget_kernel_input_dims == output_read_kernel->get_output_dims() &&
				out_forget_kernel_input_dims == forget_act->get_input_dims() &&
				out_forget_kernel_input_dims == write_act->get_input_dims() &&
				out_forget_kernel_input_dims == candidate_act->get_input_dims() &&
				out_forget_kernel_input_dims == state_act->get_input_dims() &&
				out_forget_kernel_input_dims == read_act->get_input_dims());
		main_cell.input_forget_kernel = std::move(input_forget_kernel);
		main_cell.output_forget_kernel = std::move(output_forget_kernel);
		main_cell.input_write_kernel = std::move(input_write_kernel);
		main_cell.output_write_kernel = std::move(output_write_kernel);
		main_cell.input_candidate_kernel = std::move(input_candidate_kernel);
		main_cell.output_candidate_kernel = std::move(output_candidate_kernel);
		main_cell.input_read_kernel = std::move(input_read_kernel);
		main_cell.output_read_kernel = std::move(output_read_kernel);
		main_cell.forget_act = std::move(forget_act);
		main_cell.write_act = std::move(write_act);
		main_cell.candidate_act = std::move(candidate_act);
		main_cell.state_act = std::move(state_act);
		main_cell.read_act = std::move(read_act);
		input_dims = std::move(in_forget_kernel_input_dims);
		output_dims = std::move(out_forget_kernel_input_dims);
		set_foremost(foremost);
	}
	inline LSTMNeuralNetwork(const Self& network) :
			main_cell(),
			output_seq_size_func(network.output_seq_size_func),
			reversed(network.reversed),
			foremost(network.foremost),
			input_dims(network.input_dims),
			output_dims(network.output_dims),
			cells(network.cells.size()),
			state(network.state),
			batch_size(network.batch_size),
			input_seq_length(network.input_seq_length),
			output_seq_length(network.output_seq_length),
			output_seq_delay(network.output_seq_delay) {
		main_cell.input_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.input_forget_kernel->clone());
		main_cell.output_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.output_forget_kernel->clone());
		main_cell.input_write_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.input_write_kernel->clone());
		main_cell.output_write_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.output_write_kernel->clone());
		main_cell.input_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.input_candidate_kernel->clone());
		main_cell.output_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.output_candidate_kernel->clone());
		main_cell.input_read_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.input_read_kernel->clone());
		main_cell.output_read_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
				network.main_cell.output_read_kernel->clone());
		main_cell.forget_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.forget_act->clone());
		main_cell.write_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.write_act->clone());
		main_cell.candidate_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.candidate_act->clone());
		main_cell.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.state_act->clone());
		main_cell.read_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
				network.main_cell.read_act->clone());
		main_cell.forget_filter_cache = network.main_cell.forget_filter_cache;
		main_cell.prev_state_cache = network.main_cell.prev_state_cache;
		main_cell.write_filter_cache = network.main_cell.write_filter_cache;
		main_cell.candidate_cache = network.main_cell.candidate_cache;
		main_cell.read_filter_cache = network.main_cell.read_filter_cache;
		main_cell.activated_state_cache = network.main_cell.activated_state_cache;
		main_cell.weighted_input_forget_cache = network.main_cell.weighted_input_forget_cache;
		main_cell.weighted_output_forget_cache = network.main_cell.weighted_output_forget_cache;
		main_cell.weighted_input_write_cache = network.main_cell.weighted_input_write_cache;
		main_cell.weighted_output_write_cache = network.main_cell.weighted_output_write_cache;
		main_cell.weighted_input_candidate_cache = network.main_cell.weighted_input_candidate_cache;
		main_cell.weighted_output_candidate_cache = network.main_cell.weighted_output_candidate_cache;
		main_cell.weighted_input_read_cache = network.main_cell.weighted_input_read_cache;
		main_cell.weighted_output_read_cache = network.main_cell.weighted_output_read_cache;
		for (std::size_t i = 0; i < network.cells.size(); i++) {
			Cell& c1 = cells[i];
			const Cell& c2 = network.cells[i];
			c1.input_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
					c2.input_forget_kernel->clone());
			c1.output_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
					c2.output_forget_kernel->clone());
			c1.input_write_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
					c2.input_write_kernel->clone());
			c1.output_write_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
					c2.output_write_kernel->clone());
			c1.input_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
					c2.input_candidate_kernel->clone());
			c1.output_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
					c2.output_candidate_kernel->clone());
			c1.input_read_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
					c2.input_read_kernel->clone());
			c1.output_read_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
					c2.output_read_kernel->clone());
			c1.write_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
					c2.write_act->clone());
			c1.forget_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
					c2.forget_act->clone());
			c1.candidate_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
					c2.candidate_act->clone());
			c1.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
					c2.state_act->clone());
			c1.read_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
					c2.read_act->clone());
			c1.forget_filter_cache = c2.forget_filter_cache;
			c1.prev_state_cache = c2.prev_state_cache;
			c1.write_filter_cache = c2.write_filter_cache;
			c1.candidate_cache = c2.candidate_cache;
			c1.read_filter_cache = c2.read_filter_cache;
			c1.activated_state_cache = c2.activated_state_cache;
			c1.weighted_input_forget_cache = c2.weighted_input_forget_cache;
			c1.weighted_output_forget_cache = c2.weighted_output_forget_cache;
			c1.weighted_input_write_cache = c2.weighted_input_write_cache;
			c1.weighted_output_write_cache = c2.weighted_output_write_cache;
			c1.weighted_input_candidate_cache = c2.weighted_input_candidate_cache;
			c1.weighted_output_candidate_cache = c2.weighted_output_candidate_cache;
			c1.weighted_input_read_cache = c2.weighted_input_read_cache;
			c1.weighted_output_read_cache = c2.weighted_output_read_cache;
		}
	}
	inline LSTMNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~LSTMNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Root* clone() const {
		return new LSTMNeuralNetwork(*this);
	}
	inline bool is_reversed() const {
		return reversed;
	}
	inline void reverse() {
		this->reversed = !this->reversed;
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline const typename Root::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Root::Dims& get_output_dims() const {
		return output_dims;
	}
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.main_cell, network2.main_cell);
		swap(network1.output_seq_size_func, network2.output_seq_size_func);
		swap(network1.reversed, network2.reversed);
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
		Root::set_input_layer(*main_cell.input_forget_kernel, foremost);
		Root::set_input_layer(*main_cell.input_write_kernel, foremost);
		Root::set_input_layer(*main_cell.input_candidate_kernel, foremost);
		Root::set_input_layer(*main_cell.input_read_kernel, foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		TimeStepData null_tensor;
		Root::empty_cache(*main_cell.input_forget_kernel);
		Root::empty_cache(*main_cell.output_forget_kernel);
		Root::empty_cache(*main_cell.input_write_kernel);
		Root::empty_cache(*main_cell.output_write_kernel);
		Root::empty_cache(*main_cell.input_candidate_kernel);
		Root::empty_cache(*main_cell.output_candidate_kernel);
		Root::empty_cache(*main_cell.input_read_kernel);
		Root::empty_cache(*main_cell.output_read_kernel);
		Root::empty_cache(*main_cell.write_act);
		Root::empty_cache(*main_cell.forget_act);
		Root::empty_cache(*main_cell.candidate_act);
		Root::empty_cache(*main_cell.state_act);
		Root::empty_cache(*main_cell.read_act);
		main_cell.forget_filter_cache = null_tensor;
		main_cell.prev_state_cache = null_tensor;
		main_cell.write_filter_cache = null_tensor;
		main_cell.candidate_cache = null_tensor;
		main_cell.read_filter_cache = null_tensor;
		main_cell.activated_state_cache = null_tensor;
		main_cell.weighted_input_forget_cache = null_tensor;
		main_cell.weighted_output_forget_cache = null_tensor;
		main_cell.weighted_input_write_cache = null_tensor;
		main_cell.weighted_output_write_cache = null_tensor;
		main_cell.weighted_input_candidate_cache = null_tensor;
		main_cell.weighted_output_candidate_cache = null_tensor;
		main_cell.weighted_input_read_cache = null_tensor;
		main_cell.weighted_output_read_cache = null_tensor;
		// Clear the state as well.
		batch_size = -1;
		state = null_tensor;
		input_seq_length = -1;
		output_seq_length = -1;
		output_seq_delay = -1;
		cells = std::vector<Cell>(0);
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layers(13);
		layers[0] = main_cell.input_forget_kernel.get();
		layers[1] = main_cell.output_forget_kernel.get();
		layers[2] = main_cell.input_write_kernel.get();
		layers[3] = main_cell.output_write_kernel.get();
		layers[4] = main_cell.input_candidate_kernel.get();
		layers[5] = main_cell.output_candidate_kernel.get();
		layers[6] = main_cell.input_read_kernel.get();
		layers[7] = main_cell.output_read_kernel.get();
		layers[8] = main_cell.forget_act.get();
		layers[9] = main_cell.write_act.get();
		layers[10] = main_cell.candidate_act.get();
		layers[11] = main_cell.read_act.get();
		layers[12] = main_cell.state_act.get();
		return layers;
	}
	inline typename Root::Data propagate(typename Root::Data input, bool training) {
		Utils<Scalar>::template check_dim_validity<Root::DATA_RANKS>(input);
		Dimensions<std::size_t,Root::DATA_RANKS> data_dims = Utils<Scalar>::template get_dims<Root::DATA_RANKS>(input);
		assert(input_dims == data_dims.template demote<2>());
		int samples = data_dims(0);
		int input_seq_length = data_dims(1);
		std::pair<std::size_t,std::size_t> output_seq_info = output_seq_size_func((std::size_t) input_seq_length);
		int output_seq_length = (int) output_seq_info.first;
		int output_seq_delay = (int) output_seq_info.second;
		assert(output_seq_length > 0);
		TimeStepData null_tensor;
		if (reversed) {
			RankwiseBoolArray reverse;
			reverse.fill(false);
			reverse[1] = true;
			input = input.reverse(reverse);
		}
		int output_end = output_seq_length + output_seq_delay;
		int time_steps = std::max(input_seq_length, output_end);
		// Only unroll the network in training mode and if the sequence alignment has changed.
		if (training && (input_seq_length != this->input_seq_length ||
				output_seq_length != this->output_seq_length || output_seq_delay != this->output_seq_delay)) {
			if (time_steps > 1) {
				empty_caches();
				cells = std::vector<Cell>(time_steps - 1);
				for (int j = 1; j < time_steps; ++j) {
					Cell& cell = cells[j - 1];
					cell.output_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.output_forget_kernel));
					cell.output_write_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.output_write_kernel));
					cell.output_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.output_candidate_kernel));
					cell.output_read_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.output_read_kernel));
					cell.write_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.write_act));
					cell.forget_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.forget_act));
					cell.candidate_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.candidate_act));
					cell.state_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.state_act));
					cell.read_act = ActivationPtr<Scalar,Rank>((ActivationLayer<Scalar,Rank>*)
							Root::clone_with_shared_params(*main_cell.read_act));
					if (j < input_seq_length) {
						cell.input_forget_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
								Root::clone_with_shared_params(*main_cell.input_forget_kernel));
						cell.input_write_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
								Root::clone_with_shared_params(*main_cell.input_write_kernel));
						cell.input_candidate_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
								Root::clone_with_shared_params(*main_cell.input_candidate_kernel));
						cell.input_read_kernel = KernelPtr<Scalar,Rank>((KernelLayer<Scalar,Rank>*)
								Root::clone_with_shared_params(*main_cell.input_read_kernel));
					}
				}
			} else
				cells = std::vector<Cell>(0);
		}
		if (!Stateful || batch_size == -1) {
			Dimensions<std::size_t,Rank + 1> dims = main_cell.forget_act->get_output_dims().template promote<>();
			dims(0) = samples;
			state = Tensor<Scalar,Rank + 1>(dims);
			state.setZero();
		} else if (samples != batch_size) {
			std::array<std::size_t,Rank + 1> offsets;
			std::array<std::size_t,Rank + 1> extents = main_cell.forget_act->get_output_dims().template promote<>();
			offsets.fill(0);
			extents[0] = samples;
			TimeStepData new_state;
			if (samples > batch_size) {
				new_state = Tensor<Scalar,Rank + 1>(extents);
				new_state.setZero();
				extents[0] = batch_size;
				new_state.slice(offsets, extents) = state;
			} else
				new_state = state.slice(offsets, extents);
			state = std::move(new_state);
		}
		RankwiseIntArray input_offsets;
		RankwiseIntArray input_extents = data_dims;
		RankwiseIntArray output_offsets;
		RankwiseIntArray output_extents = output_dims.template promote<2>();
		input_offsets.fill(0);
		output_offsets.fill(0);
		input_extents[1] = 1;
		output_extents[0] = samples;
		typename Root::Data out;
		if (output_seq_length > 1) {
			output_extents[1] = output_seq_length;
			out = typename Root::Data(output_extents);
		}
		output_extents[1] = 1;
		Dimensions<std::size_t,Rank + 1> input_time_step_dims = input_dims.template promote<>();
		input_time_step_dims(0) = samples;
		TimeStepData hidden_out;
		for (int i = 0; i < time_steps; ++i) {
			Cell& cell = !training || i == 0 ? main_cell : cells[i - 1];
			TimeStepData input_res;
			// State update.
			if (i < input_seq_length) {
				if (input_seq_length > 1) {
					typename Root::Data input_slice = input.slice(input_offsets, input_extents);
					input_offsets[1] += 1;
					input_res = Utils<Scalar>::template map_tensor_to_tensor<Root::DATA_RANKS,Rank + 1>(std::move(input_slice),
							input_time_step_dims);
				} else
					input_res = Utils<Scalar>::template map_tensor_to_tensor<Root::DATA_RANKS,Rank + 1>(std::move(input),
							input_time_step_dims);
				if (i == 0) {
					// There must be an input at this time step and there cannot be an output from the previous one.
					TimeStepData weighted_input_forget = Root::pass_forward(*cell.input_forget_kernel, input_res, training);
					if (!training)
						Root::empty_cache(*cell.input_forget_kernel);
					// Cache the factors of the multiplication for the backward pass.
					cell.forget_filter_cache = Root::pass_forward(*cell.forget_act, std::move(weighted_input_forget), training);
					cell.prev_state_cache = std::move(state);
					// Selective remembrance.
					// FIXME Use a thread pool to evaluate the tensor operations.
					state = cell.forget_filter_cache * cell.prev_state_cache;
					if (!training)
						Root::empty_cache(*cell.forget_act);
					TimeStepData weighted_input_write = Root::pass_forward(*cell.input_write_kernel, input_res, training);
					if (!training)
						Root::empty_cache(*cell.input_write_kernel);
					cell.write_filter_cache = Root::pass_forward(*cell.write_act, std::move(weighted_input_write), training);
					if (!training)
						Root::empty_cache(*cell.write_act);
					TimeStepData weighted_input_candidates = Root::pass_forward(*cell.input_candidate_kernel, input_res, training);
					if (!training)
						Root::empty_cache(*cell.input_candidate_kernel);
					cell.candidate_cache = Root::pass_forward(*cell.candidate_act, std::move(weighted_input_candidates), training);
					if (!training)
						Root::empty_cache(*cell.candidate_act);
					state += cell.write_filter_cache * cell.candidate_cache;
				} else {
					// There is both an input and an output from the previous time step.
					TimeStepData weighted_input_forget = Root::pass_forward(*cell.input_forget_kernel, input_res, training);
					if (!training)
						Root::empty_cache(*cell.input_forget_kernel);
					TimeStepData weighted_output_forget = Root::pass_forward(*cell.output_forget_kernel, hidden_out, training);
					if (!training)
						Root::empty_cache(*cell.output_forget_kernel);
					TimeStepData weighted_forget;
					if (MulInt) {
						if (training) {
							cell.weighted_input_forget_cache = std::move(weighted_input_forget);
							cell.weighted_output_forget_cache = std::move(weighted_output_forget);
							weighted_forget = cell.weighted_input_forget_cache * cell.weighted_output_forget_cache;
						} else {
							weighted_forget = weighted_input_forget * weighted_output_forget;
							weighted_input_forget = null_tensor;
							weighted_output_forget = null_tensor;
						}
					} else {
						weighted_forget = weighted_input_forget + weighted_output_forget;
						weighted_input_forget = null_tensor;
						weighted_output_forget = null_tensor;
					}
					cell.forget_filter_cache = Root::pass_forward(*cell.forget_act, std::move(weighted_forget), training);
					cell.prev_state_cache = std::move(state);
					state = cell.forget_filter_cache * cell.prev_state_cache;
					if (!training)
						Root::empty_cache(*cell.forget_act);
					TimeStepData weighted_input_write = Root::pass_forward(*cell.input_write_kernel, input_res, training);
					if (!training)
						Root::empty_cache(*cell.input_write_kernel);
					TimeStepData weighted_output_write = Root::pass_forward(*cell.output_write_kernel, hidden_out, training);
					if (!training)
						Root::empty_cache(*cell.output_write_kernel);
					TimeStepData weighted_write;
					if (MulInt) {
						if (training) {
							cell.weighted_input_write_cache = std::move(weighted_input_write);
							cell.weighted_output_write_cache = std::move(weighted_output_write);
							weighted_write = cell.weighted_input_write_cache *
									cell.weighted_output_write_cache;
						} else {
							weighted_write = weighted_input_write * weighted_output_write;
							weighted_input_write = null_tensor;
							weighted_output_write = null_tensor;
						}
					} else {
						weighted_write = weighted_input_write + weighted_output_write;
						weighted_input_write = null_tensor;
						weighted_output_write = null_tensor;
					}
					cell.write_filter_cache = Root::pass_forward(*cell.write_act, std::move(weighted_write), training);
					if (!training)
						Root::empty_cache(*cell.write_act);
					TimeStepData weighted_input_candidates = Root::pass_forward(*cell.input_candidate_kernel, input_res, training);
					if (!training)
						Root::empty_cache(*cell.input_candidate_kernel);
					TimeStepData weighted_output_candidates = Root::pass_forward(*cell.output_candidate_kernel, hidden_out, training);
					if (!training)
						Root::empty_cache(*cell.output_candidate_kernel);
					TimeStepData weighted_candidates;
					if (MulInt) {
						if (training) {
							cell.weighted_input_candidate_cache = std::move(weighted_input_candidates);
							cell.weighted_output_candidate_cache = std::move(weighted_output_candidates);
							weighted_candidates = cell.weighted_input_candidate_cache *
									cell.weighted_output_candidate_cache;
						} else {
							weighted_candidates = weighted_input_candidates * weighted_output_candidates;
							weighted_input_candidates = null_tensor;
							weighted_output_candidates = null_tensor;
						}
					} else {
						weighted_candidates = weighted_input_candidates + weighted_output_candidates;
						weighted_input_candidates = null_tensor;
						weighted_output_candidates = null_tensor;
					}
					cell.candidate_cache = Root::pass_forward(*cell.candidate_act, std::move(weighted_candidates), training);
					if (!training)
						Root::empty_cache(*cell.candidate_act);
					state += cell.write_filter_cache * cell.candidate_cache;
				}
			} else {
				// There is only the output from the previous time step and no new input (i must be greater than 0).
				TimeStepData weighted_output_forget = Root::pass_forward(*cell.output_forget_kernel, hidden_out, training);
				if (!training)
					Root::empty_cache(*cell.output_forget_kernel);
				cell.forget_filter_cache = Root::pass_forward(*cell.forget_act, std::move(weighted_output_forget), training);
				cell.prev_state_cache = std::move(state);
				state = cell.forget_filter_cache * cell.prev_state_cache;
				if (!training)
					Root::empty_cache(*cell.forget_act);
				TimeStepData weighted_output_write = Root::pass_forward(*cell.output_write_kernel, hidden_out, training);
				if (!training)
					Root::empty_cache(*cell.output_write_kernel);
				cell.write_filter_cache = Root::pass_forward(*cell.write_act, std::move(weighted_output_write), training);
				if (!training)
					Root::empty_cache(*cell.write_act);
				TimeStepData weighted_output_candidates = Root::pass_forward(*cell.output_candidate_kernel, hidden_out, training);
				if (!training)
					Root::empty_cache(*cell.output_candidate_kernel);
				cell.candidate_cache = Root::pass_forward(*cell.candidate_act, std::move(weighted_output_candidates), training);
				if (!training)
					Root::empty_cache(*cell.candidate_act);
				state += cell.write_filter_cache * cell.candidate_cache;
			}
			// Output computation.
			TimeStepData weighted_read;
			if (i < input_seq_length) {
				if (i == 0) {
					weighted_read = Root::pass_forward(*cell.input_read_kernel, input_res, training);
					if (!training)
						Root::empty_cache(*cell.input_read_kernel);
				} else {
					TimeStepData weighted_input_read = Root::pass_forward(*cell.input_read_kernel, input_res, training);
					if (!training)
						Root::empty_cache(*cell.input_read_kernel);
					TimeStepData weighted_output_read = Root::pass_forward(*cell.output_read_kernel, hidden_out, training);
					if (!training)
						Root::empty_cache(*cell.output_read_kernel);
					if (MulInt) {
						if (training) {
							cell.weighted_input_read_cache = std::move(weighted_input_read);
							cell.weighted_output_read_cache = std::move(weighted_output_read);
							weighted_read = cell.weighted_input_read_cache *
									cell.weighted_output_read_cache;
						} else {
							weighted_read = weighted_input_read * weighted_output_read;
							weighted_input_read = null_tensor;
							weighted_output_read = null_tensor;
						}
					} else {
						weighted_read = weighted_input_read + weighted_output_read;
						weighted_input_read = null_tensor;
						weighted_output_read = null_tensor;
					}
				}
			} else {
				weighted_read = Root::pass_forward(*cell.output_read_kernel, hidden_out, training);
				if (!training)
					Root::empty_cache(*cell.output_read_kernel);
			}
			cell.read_filter_cache = Root::pass_forward(*cell.read_act, std::move(weighted_read), training);
			if (!training)
				Root::empty_cache(*cell.read_act);
			cell.activated_state_cache = Root::pass_forward(*cell.state_act, state, training);
			if (!training)
				Root::empty_cache(*cell.state_act);
			hidden_out = cell.read_filter_cache * cell.activated_state_cache;
			// If there is a non-hidden output at this time step...
			if (i >= output_seq_delay && i < output_end) {
				if (output_seq_length > 1) {
					out.slice(output_offsets, output_extents) = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(
							hidden_out, output_extents);
					output_offsets[1] += 1;
				} else
					out = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(hidden_out, output_extents);
			}
		}
		batch_size = samples;
		this->input_seq_length = input_seq_length;
		this->output_seq_length = output_seq_length;
		this->output_seq_delay = output_seq_delay;
		return out;
	}
	inline typename Root::Data backpropagate(typename Root::Data out_grads) {
		Utils<Scalar>::template check_dim_validity<Root::DATA_RANKS>(out_grads);
		Dimensions<std::size_t,Root::DATA_RANKS> data_dims = Utils<Scalar>::template get_dims<Root::DATA_RANKS>(out_grads);
		assert(output_dims == data_dims.template demote<2>() && batch_size == data_dims(0) &&
				output_seq_length == data_dims(1));
		TimeStepData null_tensor;
		RankwiseIntArray output_offsets;
		RankwiseIntArray output_extents = data_dims;
		RankwiseIntArray input_offsets;
		RankwiseIntArray input_extents = input_dims.template promote<2>();
		output_offsets.fill(0);
		output_offsets[1] = output_seq_length - 1;
		input_offsets.fill(0);
		input_offsets[1] = input_seq_length - 1;
		output_extents[1] = 1;
		input_extents[0] = batch_size;
		typename Root::Data prev_out_grads;
		if (input_seq_length > 1) {
			input_extents[1] = input_seq_length;
			prev_out_grads = typename Root::Data(input_extents);
		}
		input_extents[1] = 1;
		TimeStepData state_grads(state.dimensions());
		TimeStepData hidden_out_grads(state.dimensions());
		state_grads.setZero();
		hidden_out_grads.setZero();
		Dimensions<std::size_t,Rank + 1> out_time_step_dims = output_dims.template promote<>();
		out_time_step_dims(0) = batch_size;
		int output_end = output_seq_length + output_seq_delay;
		int time_steps = std::max(input_seq_length, output_end);
		for (int i = time_steps - 1; i >= 0; --i) {
			Cell& cell = i == 0 ? main_cell : cells[i - 1];
			// If there was a non-hidden output at the time step, let the gradients flow into the hidden output gradients.
			if (i >= output_seq_delay && i < output_end) {
				if (output_seq_length == 1)
					hidden_out_grads += Utils<Scalar>::template map_tensor_to_tensor<Root::DATA_RANKS,Rank + 1>(std::move(out_grads),
							out_time_step_dims);
				else {
					typename Root::Data out_grads_seq = out_grads.slice(output_offsets, output_extents);
					output_offsets[1] -= 1;
					hidden_out_grads += Utils<Scalar>::template map_tensor_to_tensor<Root::DATA_RANKS,Rank + 1>(std::move(out_grads_seq),
							out_time_step_dims);
				}
			}
			state_grads += Root::pass_back(*cell.state_act, cell.read_filter_cache * hidden_out_grads);
			Root::empty_cache(*cell.state_act);
			TimeStepData weighted_read_grads = Root::pass_back(*cell.read_act, cell.activated_state_cache * hidden_out_grads);
			Root::empty_cache(*cell.read_act);
			TimeStepData candidate_grads = Root::pass_back(*cell.candidate_act, cell.write_filter_cache * state_grads);
			Root::empty_cache(*cell.candidate_act);
			TimeStepData weighted_write_grads = Root::pass_back(*cell.write_act, cell.candidate_cache * state_grads);
			Root::empty_cache(*cell.write_act);
			TimeStepData weighted_forget_grads = Root::pass_back(*cell.forget_act, cell.prev_state_cache * state_grads);
			Root::empty_cache(*cell.forget_act);
			state_grads *= cell.forget_filter_cache;
			if (i < input_seq_length) {
				TimeStepData prev_out_grads_i;
				if (MulInt) {
					if (i != 0) {
						// Calculate the previous hidden output gradients.
						hidden_out_grads = Root::pass_back(*cell.output_read_kernel, cell.weighted_input_read_cache * weighted_read_grads);
						Root::empty_cache(*cell.output_read_kernel);
						hidden_out_grads += Root::pass_back(*cell.output_candidate_kernel, cell.weighted_input_candidate_cache * candidate_grads);
						Root::empty_cache(*cell.output_candidate_kernel);
						hidden_out_grads += Root::pass_back(*cell.output_write_kernel, cell.weighted_input_write_cache * weighted_write_grads);
						Root::empty_cache(*cell.output_write_kernel);
						hidden_out_grads += Root::pass_back(*cell.output_forget_kernel, cell.weighted_input_forget_cache * weighted_forget_grads);
						Root::empty_cache(*cell.output_forget_kernel);
						// Calculate the input gradients.
						prev_out_grads_i = Root::pass_back(*cell.input_read_kernel, cell.weighted_output_read_cache * weighted_read_grads);
						Root::empty_cache(*cell.input_read_kernel);
						weighted_read_grads = null_tensor;
						prev_out_grads_i += Root::pass_back(*cell.input_candidate_kernel, cell.weighted_output_candidate_cache * candidate_grads);
						Root::empty_cache(*cell.input_candidate_kernel);
						candidate_grads = null_tensor;
						prev_out_grads_i += Root::pass_back(*cell.input_write_kernel, cell.weighted_output_write_cache * weighted_write_grads);
						Root::empty_cache(*cell.input_write_kernel);
						weighted_write_grads = null_tensor;
						prev_out_grads_i += Root::pass_back(*cell.input_forget_kernel, cell.weighted_output_forget_cache * weighted_forget_grads);
						Root::empty_cache(*cell.input_forget_kernel);
						weighted_forget_grads = null_tensor;
					} else {
						prev_out_grads_i = Root::pass_back(*cell.input_read_kernel, std::move(weighted_read_grads));
						Root::empty_cache(*cell.input_read_kernel);
						prev_out_grads_i += Root::pass_back(*cell.input_candidate_kernel, std::move(candidate_grads));
						Root::empty_cache(*cell.input_candidate_kernel);
						prev_out_grads_i += Root::pass_back(*cell.input_write_kernel, std::move(weighted_write_grads));
						Root::empty_cache(*cell.input_write_kernel);
						prev_out_grads_i += Root::pass_back(*cell.input_forget_kernel, std::move(weighted_forget_grads));
						Root::empty_cache(*cell.input_forget_kernel);
					}
				} else {
					if (i != 0) {
						hidden_out_grads = Root::pass_back(*cell.output_read_kernel, weighted_read_grads);
						Root::empty_cache(*cell.output_read_kernel);
						hidden_out_grads += Root::pass_back(*cell.output_candidate_kernel, candidate_grads);
						Root::empty_cache(*cell.output_candidate_kernel);
						hidden_out_grads += Root::pass_back(*cell.output_write_kernel, weighted_write_grads);
						Root::empty_cache(*cell.output_write_kernel);
						hidden_out_grads += Root::pass_back(*cell.output_forget_kernel, weighted_forget_grads);
						Root::empty_cache(*cell.output_forget_kernel);
					}
					prev_out_grads_i = Root::pass_back(*cell.input_read_kernel, std::move(weighted_read_grads)) +
							Root::pass_back(*cell.input_candidate_kernel, std::move(candidate_grads)) +
							Root::pass_back(*cell.input_write_kernel, std::move(weighted_write_grads)) +
							Root::pass_back(*cell.input_forget_kernel, std::move(weighted_forget_grads));
					Root::empty_cache(*cell.input_read_kernel);
					Root::empty_cache(*cell.input_candidate_kernel);
					Root::empty_cache(*cell.input_write_kernel);
					Root::empty_cache(*cell.input_forget_kernel);
				}
				if (!foremost) {
					if (input_seq_length > 1) {
						prev_out_grads.slice(input_offsets, input_extents) =
								Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(
										std::move(prev_out_grads_i), input_extents);
						input_offsets[1] -= 1;
					} else
						prev_out_grads = Utils<Scalar>::template map_tensor_to_tensor<Rank + 1,Root::DATA_RANKS>(
								std::move(prev_out_grads_i), input_extents);
				}
			} else {
				hidden_out_grads = Root::pass_back(*cell.output_read_kernel, std::move(weighted_read_grads));
				Root::empty_cache(*cell.output_read_kernel);
				hidden_out_grads += Root::pass_back(*cell.output_candidate_kernel, std::move(candidate_grads));
				Root::empty_cache(*cell.output_candidate_kernel);
				hidden_out_grads += Root::pass_back(*cell.output_write_kernel, std::move(weighted_write_grads));
				Root::empty_cache(*cell.output_write_kernel);
				hidden_out_grads += Root::pass_back(*cell.output_forget_kernel, std::move(weighted_forget_grads));
				Root::empty_cache(*cell.output_forget_kernel);
			}
		}
		// Roll-up the network.
		Matrix<Scalar>& input_forget_kernel_params_grad = Root::get_params_grad(*main_cell.input_forget_kernel);
		Matrix<Scalar>& output_forget_kernel_params_grad = Root::get_params_grad(*main_cell.output_forget_kernel);
		Matrix<Scalar>& input_write_kernel_params_grad = Root::get_params_grad(*main_cell.input_write_kernel);
		Matrix<Scalar>& output_write_kernel_params_grad = Root::get_params_grad(*main_cell.output_write_kernel);
		Matrix<Scalar>& input_candidate_kernel_params_grad = Root::get_params_grad(*main_cell.input_candidate_kernel);
		Matrix<Scalar>& output_candidate_kernel_params_grad = Root::get_params_grad(*main_cell.output_candidate_kernel);
		Matrix<Scalar>& input_read_kernel_params_grad = Root::get_params_grad(*main_cell.input_read_kernel);
		Matrix<Scalar>& output_read_kernel_params_grad = Root::get_params_grad(*main_cell.output_read_kernel);
		Matrix<Scalar>& forget_act_params_grad = Root::get_params_grad(*main_cell.forget_act);
		Matrix<Scalar>& write_act_params_grad = Root::get_params_grad(*main_cell.write_act);
		Matrix<Scalar>& candidate_act_params_grad = Root::get_params_grad(*main_cell.candidate_act);
		Matrix<Scalar>& state_act_params_grad = Root::get_params_grad(*main_cell.state_act);
		Matrix<Scalar>& read_act_params_grad = Root::get_params_grad(*main_cell.read_act);
		for (int i = 1; i < time_steps; ++i) {
			Cell& cell = cells[i - 1];
			forget_act_params_grad += Root::get_params_grad(*cell.forget_act);
			write_act_params_grad += Root::get_params_grad(*cell.write_act);
			candidate_act_params_grad += Root::get_params_grad(*cell.candidate_act);
			state_act_params_grad += Root::get_params_grad(*cell.state_act);
			read_act_params_grad += Root::get_params_grad(*cell.read_act);
			output_forget_kernel_params_grad += Root::get_params_grad(*cell.output_forget_kernel);
			output_write_kernel_params_grad += Root::get_params_grad(*cell.output_write_kernel);
			output_candidate_kernel_params_grad += Root::get_params_grad(*cell.output_candidate_kernel);
			output_read_kernel_params_grad += Root::get_params_grad(*cell.output_read_kernel);
			if (i < input_seq_length) {
				input_forget_kernel_params_grad += Root::get_params_grad(*cell.input_forget_kernel);
				input_write_kernel_params_grad += Root::get_params_grad(*cell.input_write_kernel);
				input_candidate_kernel_params_grad += Root::get_params_grad(*cell.input_candidate_kernel);
				input_read_kernel_params_grad += Root::get_params_grad(*cell.input_read_kernel);
			}
		}
		return prev_out_grads;
	}
private:
	/**
	 * A struct representing a cell in the unrolled LSTM.
	 */
	struct Cell {
		KernelPtr<Scalar,Rank> input_forget_kernel;
		KernelPtr<Scalar,Rank> output_forget_kernel;
		KernelPtr<Scalar,Rank> input_write_kernel;
		KernelPtr<Scalar,Rank> output_write_kernel;
		KernelPtr<Scalar,Rank> input_candidate_kernel;
		KernelPtr<Scalar,Rank> output_candidate_kernel;
		KernelPtr<Scalar,Rank> input_read_kernel;
		KernelPtr<Scalar,Rank> output_read_kernel;
		ActivationPtr<Scalar,Rank> forget_act;
		ActivationPtr<Scalar,Rank> write_act;
		ActivationPtr<Scalar,Rank> candidate_act;
		ActivationPtr<Scalar,Rank> state_act;
		ActivationPtr<Scalar,Rank> read_act;
		// Caches for the derivation of multiplicative filtering operations.
		TimeStepData forget_filter_cache;
		TimeStepData prev_state_cache;
		TimeStepData write_filter_cache;
		TimeStepData candidate_cache;
		TimeStepData read_filter_cache;
		TimeStepData activated_state_cache;
		// Caches for the derivation of multiplicative integration operations.
		TimeStepData weighted_input_forget_cache;
		TimeStepData weighted_output_forget_cache;
		TimeStepData weighted_input_write_cache;
		TimeStepData weighted_output_write_cache;
		TimeStepData weighted_input_candidate_cache;
		TimeStepData weighted_output_candidate_cache;
		TimeStepData weighted_input_read_cache;
		TimeStepData weighted_output_read_cache;
	};
	Cell main_cell;
	OutputSeqSizeFunc output_seq_size_func;
	bool reversed;
	bool foremost;
	typename Root::Dims input_dims;
	typename Root::Dims output_dims;
	// State.
	std::vector<Cell> cells;
	TimeStepData state;
	int batch_size;
	int input_seq_length;
	int output_seq_length;
	int output_seq_delay;
};

} /* namespace cattle */

#endif /* NEURALNETWORK_H_ */
