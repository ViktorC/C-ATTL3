/*
 * RecurrentNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_RECURRENTNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_RECURRENTNEURALNETWORK_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <utility>

#include "layer/ActivationLayer.hpp"
#include "layer/KernelLayer.hpp"
#include "neural_network/UnidirectionalNeuralNetwork.hpp"

namespace cattle {

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
	typedef UnidirectionalNeuralNetwork<Scalar,Rank> Base;
	typedef RecurrentNeuralNetwork<Scalar,Rank,MulInt,Stateful> Self;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
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
	 * determines the output sequence delay and length. The output of the function is a pair of unsigned
	 * integers where the first element is the sequence length and the second element is the sequence
	 * delay.
	 * @param reversed Whether the network is to reverse its inputs along the time-step rank.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline RecurrentNeuralNetwork(KernelPtr<Scalar,Rank>&& input_kernel, KernelPtr<Scalar,Rank>&& state_kernel,
			KernelPtr<Scalar,Rank>&& output_kernel, ActivationPtr<Scalar,Rank>&& state_act,
			ActivationPtr<Scalar,Rank>&& output_act, OutputSeqSizeFunc output_seq_size_func, bool reversed = false,
			bool foremost = true) :
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
			main_cell(network.main_cell),
			output_seq_size_func(network.output_seq_size_func),
			reversed(network.reversed),
			foremost(network.foremost),
			input_dims(network.input_dims),
			output_dims(network.output_dims),
			cells(0),
			state(network.state),
			batch_size(network.batch_size),
			input_seq_length(network.input_seq_length),
			output_seq_length(network.output_seq_length),
			output_seq_delay(network.output_seq_delay) {
		for (std::size_t i = 0; i < network.cells.size(); i++)
			cells.push_back(Cell(network.cells[i]));
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
		reversed = !reversed;
	}
	inline const typename Root::Dims& get_input_dims() const {
		return input_dims;
	}
	inline const typename Root::Dims& get_output_dims() const {
		return output_dims;
	}
	inline std::vector<const Layer<Scalar,Rank>*> get_layers() const {
		std::vector<const Layer<Scalar,Rank>*> layer_ptrs(5);
		populate_layer_vector<const Layer<Scalar,Rank>*>(layer_ptrs);
		return layer_ptrs;
	}
	inline std::vector<Layer<Scalar,Rank>*> get_layers() {
		std::vector<Layer<Scalar,Rank>*> layer_ptrs(5);
		populate_layer_vector<Layer<Scalar,Rank>*>(layer_ptrs);
		return layer_ptrs;
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline void set_foremost(bool foremost) {
		main_cell.input_kernel->set_input_layer(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		main_cell.input_kernel->empty_cache();
		main_cell.state_kernel->empty_cache();
		main_cell.output_kernel->empty_cache();
		main_cell.state_act->empty_cache();
		main_cell.output_act->empty_cache();
		main_cell.state_kernel_cache = TimeStepData();
		main_cell.input_kernel_cache = TimeStepData();
		// Clear the state as well.
		batch_size = -1;
		state = TimeStepData();
		input_seq_length = -1;
		output_seq_length = -1;
		output_seq_delay = -1;
		cells = std::vector<Cell>(0);
	}
	inline typename Root::Data propagate(typename Root::Data input, bool training) {
		Dimensions<std::size_t,Root::DATA_RANK> data_dims = input.dimensions();
		assert(input_dims == data_dims.template demote<2>());
		int samples = data_dims(0);
		int input_seq_length = data_dims(1);
		// Calculate the output sequence length and delay based on the input sequence length.
		std::pair<std::size_t,std::size_t> output_seq_info = output_seq_size_func((std::size_t) input_seq_length);
		int output_seq_length = (int) output_seq_info.first;
		int output_seq_delay = (int) output_seq_info.second;
		assert(output_seq_length > 0);
		if (reversed)
			Base::reverse_along_time_axis(input);
		int output_end = output_seq_length + output_seq_delay;
		int time_steps = std::max(input_seq_length, output_end);
		// If in training mode, unroll the network (unless it has already been unrolled for the same alignment).
		if (training && (input_seq_length != this->input_seq_length ||
				output_seq_length != this->output_seq_length || output_seq_delay != this->output_seq_delay))
			unroll_network(time_steps, input_seq_length, output_seq_delay, output_end);
		setup_hidden_state(samples);
		RankwiseArray input_offsets, output_offsets;
		RankwiseArray input_extents = data_dims;
		RankwiseArray output_extents = output_dims.template promote<2>();
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
			state = cell.state_kernel->pass_forward(std::move(state), training);
			// If there is an input for the time step...
			if (i < input_seq_length) {
				typename Root::Data in_i_seq;
				if (input_seq_length == 1)
					in_i_seq = std::move(input);
				else {
					in_i_seq = input.slice(input_offsets, input_extents);
					input_offsets[1] += 1;
				}
				TensorMap<Scalar,Rank + 1> in_i(in_i_seq.data(), input_time_step_dims);
				if (MulInt) {
					if (training) {
						/* If multiplicative integration is enabled, cache the factors of the multiplication so that
						 * the function can be differentiated in the backward pass. */
						cell.state_kernel_cache = state;
						cell.input_kernel_cache = cell.input_kernel->pass_forward(in_i, training);
						state *= cell.input_kernel_cache;
					} else
						state *= cell.input_kernel->pass_forward(in_i, training);
				} else
					state += cell.input_kernel->pass_forward(in_i, training);
			}
			state = cell.state_act->pass_forward(std::move(state), training);
			// If there is an output for the time step...
			if (i >= output_seq_delay && i < output_end) {
				// If the output is a single time step prediction, just return it.
				TimeStepData act_out_i = cell.output_act->pass_forward(
						cell.output_kernel->pass_forward(state, training), training);
				TensorMap<Scalar,Root::DATA_RANK> out_i_seq(act_out_i.data(), output_extents);
				if (output_seq_length == 1)
					out = out_i_seq;
				else {
					out.slice(output_offsets, output_extents) = out_i_seq;
					output_offsets[1] += 1;
				}
			}
		}
		batch_size = samples;
		this->input_seq_length = input_seq_length;
		this->output_seq_length = output_seq_length;
		this->output_seq_delay = output_seq_delay;
		return out;
	}
	inline typename Root::Data backpropagate(typename Root::Data out_grad) {
		Dimensions<std::size_t,Root::DATA_RANK> data_dims = out_grad.dimensions();
		assert(output_dims == data_dims.template demote<2>() && batch_size == data_dims(0) &&
				output_seq_length == data_dims(1));
		RankwiseArray output_offsets, input_offsets;
		RankwiseArray output_extents = data_dims;
		RankwiseArray input_extents = input_dims.template promote<2>();
		output_offsets.fill(0);
		input_offsets.fill(0);
		output_offsets[1] = output_seq_length - 1;
		input_offsets[1] = input_seq_length - 1;
		output_extents[1] = 1;
		input_extents[0] = batch_size;
		typename Root::Data prev_out_grad;
		if (input_seq_length > 1) {
			input_extents[1] = input_seq_length;
			prev_out_grad = typename Root::Data(input_extents);
		}
		input_extents[1] = 1;
		TimeStepData state_grad(state.dimensions());
		state_grad.setZero();
		Dimensions<std::size_t,Rank + 1> out_time_step_dims = output_dims.template promote<>();
		out_time_step_dims(0) = batch_size;
		int output_end = output_seq_length + output_seq_delay;
		int time_steps = std::max(input_seq_length, output_end);
		for (int i = time_steps - 1; i >= 0; --i) {
			Cell& cell = i == 0 ? main_cell : cells[i - 1];
			// If there was an output at the time step...
			if (i >= output_seq_delay && i < output_end) {
				typename Root::Data out_grad_seq_i;
				if (output_seq_length == 1)
					out_grad_seq_i = std::move(out_grad);
				else {
					out_grad_seq_i = out_grad.slice(output_offsets, output_extents);
					output_offsets[1] -= 1;
				}
				state_grad += cell.output_kernel->pass_back(cell.output_act->pass_back(
						TensorMap<Scalar,Rank + 1>(out_grad_seq_i.data(), out_time_step_dims)));
			}
			// Always back-propagate the state gradient.
			state_grad = cell.state_act->pass_back(std::move(state_grad));
			// If there was an input at the time step...
			if (i < input_seq_length) {
				// If it is the foremost layer, the gradients do not need to be propagated further back.
				if (foremost) {
					if (MulInt) // Multiplicative integration.
						cell.input_kernel->pass_back(cell.state_kernel_cache * state_grad);
					else // Additive integration.
						cell.input_kernel->pass_back(state_grad);
				} else if (input_seq_length == 1) {
					TimeStepData input_i;
					if (MulInt)
						input_i = cell.input_kernel->pass_back(cell.state_kernel_cache * state_grad);
					else
						input_i = cell.input_kernel->pass_back(state_grad);
					prev_out_grad = TensorMap<Scalar,Root::DATA_RANK>(input_i.data(), input_extents);
				} else {
					TimeStepData input_i;
					if (MulInt)
						input_i = cell.input_kernel->pass_back(cell.state_kernel_cache * state_grad);
					else
						input_i = cell.input_kernel->pass_back(state_grad);
					prev_out_grad.slice(input_offsets, input_extents) =
							TensorMap<Scalar,Root::DATA_RANK>(input_i.data(), input_extents);
					input_offsets[1] -= 1;
				}
				// Compute the the state kernel's gradient.
				if (MulInt)
					state_grad = cell.state_kernel->pass_back(cell.input_kernel_cache * state_grad);
				else
					state_grad = cell.state_kernel->pass_back(std::move(state_grad));
			} else
				state_grad = cell.state_kernel->pass_back(std::move(state_grad));
		}
		return prev_out_grad;
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
private:
	template<typename _LayerPtr>
	inline void populate_layer_vector(std::vector<_LayerPtr>& layer_ptrs) const {
		layer_ptrs[0] = main_cell.input_kernel.get();
		layer_ptrs[1] = main_cell.state_kernel.get();
		layer_ptrs[2] = main_cell.output_kernel.get();
		layer_ptrs[3] = main_cell.state_act.get();
		layer_ptrs[4] = main_cell.output_act.get();
	}
	inline void unroll_network(std::size_t time_steps, std::size_t input_seq_length,
			std::size_t output_seq_delay, std::size_t output_end) {
		if (time_steps > 1) {
			// Empty the caches of the main cell to reduce the amount of data to copy.
			empty_caches();
			// Emptying the caches also clears the cell vector, thus it has to be recreated afterwards.
			cells = std::vector<Cell>(time_steps - 1);
			// Unroll the network by creating n -1 copies of the main cell;
			for (int j = 1; j < time_steps; ++j) {
				Cell& cell = cells[j - 1];
				cell.state_kernel = KernelPtr<Scalar,Rank>(static_cast<KernelLayer<Scalar,Rank>*>(
						main_cell.state_kernel->clone_with_shared_params()));
				cell.state_act = ActivationPtr<Scalar,Rank>(static_cast<ActivationLayer<Scalar,Rank>*>(
						main_cell.state_act->clone_with_shared_params()));
				// Only copy the kernels and activations that will actually be used.
				if (j < input_seq_length) {
					cell.input_kernel = KernelPtr<Scalar,Rank>(static_cast<KernelLayer<Scalar,Rank>*>(
							main_cell.input_kernel->clone_with_shared_params()));
				}
				if (j >= output_seq_delay && j < output_end) {
					cell.output_kernel = KernelPtr<Scalar,Rank>(static_cast<KernelLayer<Scalar,Rank>*>(
							main_cell.output_kernel->clone_with_shared_params()));
					cell.output_act = ActivationPtr<Scalar,Rank>(static_cast<ActivationLayer<Scalar,Rank>*>(
							main_cell.output_act->clone_with_shared_params()));
				}
			}
		} else
			cells = std::vector<Cell>(0);
	}
	inline void setup_hidden_state(std::size_t samples) {
		if (!Stateful || batch_size == -1) {
			Dimensions<std::size_t,Rank + 1> dims = main_cell.input_kernel->get_output_dims().template promote<>();
			dims(0) = samples;
			state = TimeStepData(dims);
			state.setZero();
		} else if (samples != batch_size) {
			// If the network is stateful, retain the state.
			std::array<std::size_t,Rank + 1> offsets;
			std::array<std::size_t,Rank + 1> extents = main_cell.input_kernel->get_output_dims().template promote<>();
			offsets.fill(0);
			extents[0] = samples;
			TimeStepData new_state;
			if (samples > batch_size) {
				new_state = TimeStepData(extents);
				new_state.setZero();
				extents[0] = batch_size;
				new_state.slice(offsets, extents) = state;
			} else
				new_state = state.slice(offsets, extents);
			state = std::move(new_state);
		}
	}
	/**
	 * A struct representing a cell in the unrolled RNN.
	 */
	struct Cell {
		inline Cell() { }
		inline Cell(const Cell& cell) :
				input_kernel(KernelPtr<Scalar,Rank>(static_cast<KernelLayer<Scalar,Rank>*>(
						cell.input_kernel->clone()))),
				state_kernel(KernelPtr<Scalar,Rank>(static_cast<KernelLayer<Scalar,Rank>*>(
						cell.state_kernel->clone()))),
				output_kernel(KernelPtr<Scalar,Rank>(static_cast<KernelLayer<Scalar,Rank>*>(
						cell.output_kernel->clone()))),
				state_act(ActivationPtr<Scalar,Rank>(static_cast<ActivationLayer<Scalar,Rank>*>(
						cell.state_act->clone()))),
				output_act(ActivationPtr<Scalar,Rank>(static_cast<ActivationLayer<Scalar,Rank>*>(
						cell.output_act->clone()))),
				state_kernel_cache(cell.state_kernel_cache),
				input_kernel_cache(cell.input_kernel_cache) { }
		KernelPtr<Scalar,Rank> input_kernel, state_kernel, output_kernel;
		ActivationPtr<Scalar,Rank> state_act, output_act;
		// State and input caches for multiplicative integration.
		TimeStepData state_kernel_cache, input_kernel_cache;
	};
	Cell main_cell;
	OutputSeqSizeFunc output_seq_size_func;
	bool reversed, foremost;
	typename Root::Dims input_dims, output_dims;
	// State.
	std::vector<Cell> cells;
	TimeStepData state;
	int batch_size, input_seq_length, output_seq_length, output_seq_delay;
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_RECURRENTNEURALNETWORK_H_ */
