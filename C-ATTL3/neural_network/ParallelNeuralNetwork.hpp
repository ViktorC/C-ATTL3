/*
 * ParallelNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_PARALLELNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_PARALLELNEURALNETWORK_H_

#include <array>
#include <cassert>
#include <pthread.h>
#include <utility>

#include "neural_network/CompositeNeuralNetwork.hpp"

namespace cattle {

/**
 * An enumeration type for the different ways the outputs of sub-modules of neural
 * networks may be merged.
 */
enum ParallelOutputMergeType { PARALLEL_CONCAT_LO_RANK, PARALLEL_CONCAT_HI_RANK, PARALLEL_SUM, PARALLEL_MUL };

/**
 * A class template for a parallel neural network that consists of one or more
 * lanes of non-sequential neural networks with the same input dimensions. Inputs
 * and gradients are propagated through the lanes simultaneously using multithreading.
 * The outputs of the lanes are merged by concatenation (either along the lowest
 * or hightest rank), summation, or multiplication.
 *
 * \see https://arxiv.org/abs/1409.4842
 */
template<typename Scalar, std::size_t Rank, ParallelOutputMergeType MergeType = PARALLEL_CONCAT_HI_RANK>
class ParallelNeuralNetwork :
		public CompositeNeuralNetwork<Scalar,Rank,false,NeuralNetwork<Scalar,Rank,false>> {
	typedef NeuralNetwork<Scalar,Rank,false> Base;
	typedef ParallelNeuralNetwork<Scalar,Rank,MergeType> Self;
	typedef NeuralNetPtr<Scalar,Rank,false> Lane;
	typedef std::array<std::size_t,Base::DATA_RANK> RankwiseArray;
	static_assert(MergeType >= PARALLEL_CONCAT_LO_RANK && MergeType <= PARALLEL_MUL, "illegal merge type value");
	static constexpr std::size_t CONCAT_RANK = MergeType == PARALLEL_CONCAT_HI_RANK ? Rank - 1 : 0;
	static constexpr std::size_t CONCAT_BATCH_RANK = CONCAT_RANK + 1;
public:
	/**
	 * @param lanes A vector of unique pointers to non-sequential neural networks.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline ParallelNeuralNetwork(std::vector<Lane>&& lanes, bool foremost = true) :
			lanes(std::move(lanes)),
			foremost(foremost),
			outputs(this->lanes.size()) {
		assert(this->lanes.size() > 0 && "lanes must contain at least 1 element");
		assert(this->lanes[0] != nullptr && "lanes contains null pointers");
		Base& first_lane = *this->lanes[0];
		const typename Base::Dims& input_dims = first_lane.get_input_dims();
		typename Base::Dims output_dims = first_lane.get_output_dims();
		for (std::size_t i = 1; i < this->lanes.size(); ++i) {
			assert(this->lanes[i] != nullptr && "lanes contains null pointers");
			Base& lane = *this->lanes[i];
			assert(input_dims == lane.get_input_dims());
			const typename Base::Dims& lane_output_dims = lane.get_output_dims();
			if (MergeType == PARALLEL_CONCAT_HI_RANK || MergeType == PARALLEL_CONCAT_LO_RANK) {
				if (MergeType == PARALLEL_CONCAT_HI_RANK) {
					for (std::size_t i = 0; i < +CONCAT_RANK; ++i)
						assert(output_dims(i) == lane_output_dims(i));
				} else {
					for (std::size_t i = Rank - 1; i > +CONCAT_RANK; --i)
						assert(output_dims(i) == lane_output_dims(i));
				}
				output_dims(+CONCAT_RANK) += lane_output_dims(+CONCAT_RANK);
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
			output_dims(network.output_dims),
			outputs(network.outputs) {
		for (std::size_t i = 0; i < lanes.size(); ++i)
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
	inline Base* clone() const {
		return new ParallelNeuralNetwork(*this);
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
		for (std::size_t i = 0; i < lanes.size(); ++i)
			modules.push_back(lanes[i].get());
		return modules;
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline void set_foremost(bool foremost) {
		for (std::size_t i = 0; i < lanes.size(); ++i)
			lanes[i]->set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		for (std::size_t i = 0; i < lanes.size(); ++i) {
			lanes[i]->empty_caches();
			outputs[i] = typename Base::Data();
		}
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		assert(input_dims == (Dimensions<std::size_t,Base::DATA_RANK>(input.dimensions()).template demote<>()));
		std::size_t rows = input.dimension(0);
		int pthread_state;
		typename Base::Data out;
		std::size_t lane_num = lanes.size();
		std::size_t helper_thread_num = lane_num - 1;
		pthread_t threads[helper_thread_num];
		pthread_attr_t attr;
		if (helper_thread_num > 0) {
			pthread_state = pthread_attr_init(&attr);
			assert(!pthread_state);
			pthread_state = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
			assert(!pthread_state);
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
			else {
				pthread_state = pthread_create(&threads[i - 1], &attr, propagate, &args_arr[i]);
				assert(!pthread_state);
			}
		}
		for (std::size_t i = 0; i < lane_num; ++i) {
			if (i == 0) {
				out = std::move(args_arr[i].out);
				if (MergeType == PARALLEL_MUL)
					outputs[i] = out;
			} else {
				pthread_state = pthread_join(threads[i - 1], nullptr);
				assert(!pthread_state);
				if (MergeType == PARALLEL_SUM)
					out += args_arr[i].out;
				else if (MergeType == PARALLEL_MUL) {
					outputs[i] = std::move(args_arr[i].out);
					out *= outputs[i];
				} else {
					// Must be evaluated first due to the dimension difference.
					typename Base::Data concat = out.concatenate(std::move(args_arr[i].out), +CONCAT_BATCH_RANK);
					out = std::move(concat);
				}
			}
		}
		if (helper_thread_num > 0) {
			pthread_state = pthread_attr_destroy(&attr);
			assert(!pthread_state);
		}
		return out;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grad) {
		assert(output_dims == (Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()));
		typename Base::Data prev_out_grad;
		if (foremost)
			prev_out_grad = typename Base::Data();
		else {
			RankwiseArray dims = input_dims.template promote<>();
			dims[0] = out_grad.dimension(0);
			prev_out_grad = typename Base::Data(dims);
			prev_out_grad.setZero();
		}
		int pthread_state;
		std::size_t lane_num = lanes.size();
		std::size_t helper_thread_num = lane_num - 1;
		pthread_t threads[helper_thread_num];
		pthread_attr_t attr;
		if (helper_thread_num > 0) {
			pthread_state = pthread_attr_init(&attr);
			assert(!pthread_state);
			pthread_state = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
			assert(!pthread_state);
		}
		BackpropArgs args_arr[lane_num];
		int concat_rank_offset = out_grad.dimension(+CONCAT_BATCH_RANK);
		for (int i = helper_thread_num; i >= 0; --i) {
			concat_rank_offset -= lanes[i]->get_output_dims()(+CONCAT_RANK);
			BackpropArgs args;
			args.obj = this;
			args.lane_id = i;
			args.concat_rank_offset = concat_rank_offset;
			args.out_grad = &out_grad;
			args_arr[i] = args;
			// Leave the first lane to the main thread.
			if (i == 0)
				backpropagate(&args_arr[i]);
			else {
				pthread_state = pthread_create(&threads[i - 1], &attr, backpropagate, &args_arr[i]);
				assert(!pthread_state);
			}
		}
		for (std::size_t i = 0; i < lanes.size(); ++i) {
			if (i != 0) {
				pthread_state = pthread_join(threads[i - 1], nullptr);
				assert(!pthread_state);
			}
			if (!foremost)
				prev_out_grad += args_arr[i].prev_out_grad;
		}
		if (helper_thread_num > 0) {
			pthread_state = pthread_attr_destroy(&attr);
			assert(!pthread_state);
		}
		return prev_out_grad;
	}
	inline friend void swap(Self& network1, Self& network2) {
		using std::swap;
		swap(network1.lanes, network2.lanes);
		swap(network1.foremost, network2.foremost);
		swap(network1.input_dims, network2.input_dims);
		swap(network1.output_dims, network2.output_dims);
		swap(network1.outputs, network2.outputs);
	}
private:
	inline static std::vector<Lane> create_vector(Lane&& net) {
		std::vector<Lane> vec(1);
		vec[0] = std::move(net);
		return vec;
	}
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
		if (MergeType == PARALLEL_SUM)
			args.prev_out_grad = lane.backpropagate(*args.out_grad);
		else if (MergeType == PARALLEL_MUL) {
			typename Base::Data out_grad = *args.out_grad;
			for (std::size_t i = 0; i < args.obj->lanes.size(); ++i) {
				if (i != (std::size_t) args.lane_id)
					out_grad *= args.obj->outputs[i];
			}
			args.prev_out_grad = lane.backpropagate(std::move(out_grad));
		} else {
			RankwiseArray offsets;
			RankwiseArray extents = lane.get_output_dims().template promote<>();
			offsets.fill(0);
			offsets[+CONCAT_BATCH_RANK] = args.concat_rank_offset;
			extents[0] = args.out_grad->dimension(0);
			typename Base::Data out_grad_slice = args.out_grad->slice(offsets, extents);
			args.prev_out_grad = lane.backpropagate(std::move(out_grad_slice));
		}
		return nullptr;
	}
	template<typename _LayerPtr>
	inline void populate_layer_vector(std::vector<_LayerPtr>& layer_ptrs) const {
		for (std::size_t i = 0; i < lanes.size(); ++i) {
			std::vector<Layer<Scalar,Rank>*> internal_layer_ptrs = lanes[i]->get_layers();
			for (std::size_t j = 0; j < internal_layer_ptrs.size(); ++j)
				layer_ptrs.push_back(internal_layer_ptrs[j]);
		}
	}
	std::vector<Lane> lanes;
	bool foremost;
	typename Base::Dims input_dims, output_dims;
	std::vector<typename Base::Data> outputs;
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
		typename Base::Data* out_grad;
		typename Base::Data prev_out_grad;
	};
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_PARALLELNEURALNETWORK_H_ */
