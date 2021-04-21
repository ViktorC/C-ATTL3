/*
 * LSTMNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_LSTMNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_LSTMNEURALNETWORK_H_

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
 * An alias for a unique pointer to a kernel layer of arbitrary rank and scalar
 * type.
 */
template <typename Scalar, std::size_t Rank>
using KernelPtr = std::unique_ptr<KernelLayer<Scalar, Rank>>;

/**
 * An alias for a unique pointer to an activation layer of arbitrary rank and
 * scalar type.
 */
template <typename Scalar, std::size_t Rank>
using ActivationPtr = std::unique_ptr<ActivationLayer<Scalar, Rank>>;

/**
 * A class template representing a long-short term memory (LSTM) recurrent
 * neural network. The network can use multiplicative integration to combine its
 * linearly transformed inputs and its linearly transformed hidden outputs. A
 * stateful network retains its hidden state across sequences as long as the
 * batch size is constant.
 *
 * \see http://www.bioinf.jku.at/publications/older/2604.pdf
 */
template <typename Scalar, std::size_t Rank, bool MulInt = false, bool Stateful = false>
class LSTMNeuralNetwork : public UnidirectionalNeuralNetwork<Scalar, Rank> {
  typedef NeuralNetwork<Scalar, Rank, true> Root;
  typedef UnidirectionalNeuralNetwork<Scalar, Rank> Base;
  typedef LSTMNeuralNetwork<Scalar, Rank, MulInt, Stateful> Self;
  typedef std::array<std::size_t, Root::DATA_RANK> RankwiseArray;
  typedef std::function<std::pair<std::size_t, std::size_t>(std::size_t)> OutputSeqSizeFunc;
  typedef Tensor<Scalar, Rank + 1> TimeStepData;

 public:
  /**
   * @param input_forget_kernel The forget kernel to apply to the input of the
   * network.
   * @param output_forget_kernel The forget kernel to apply to the hidden output
   * of the network at the previous time step.
   * @param input_write_kernel The write kernel to apply to the input of the
   * network.
   * @param output_write_kernel The write kernel to apply to the hidden output
   * of the network at the previous time step.
   * @param input_candidate_kernel The candidate kernel to apply to the input of
   * the network.
   * @param output_candidate_kernel The candidate kernel to apply to the hidden
   * output of the network at the previous time step.
   * @param input_read_kernel The read kernel to apply to the input of the
   * network.
   * @param output_read_kernel The read kernel to apply to the hidden output of
   * the network at the previous time step.
   * @param forget_act The activation layer of the forget gate. Usually a
   * sigmoid activation function.
   * @param write_act The activation layer of the filter of the write gate.
   * Usually a sigmoid activation function.
   * @param candidate_act The activation layer of the candidates of the write
   * gate. Usually a hyperbolic tangent activation function.
   * @param state_act The activation layer of the state at the read gate.
   * Usually a hyperbolic tangent activation function.
   * @param read_act The activation layer of the read filter. Usually a sigmoid
   * activation function.
   * @param output_seq_size_func A function parameterized by the input sequence
   * length that determines the output sequence delay and length. The output of
   * the function is a pair of unsigned integers where the first element is the
   * sequence length and the second element is the sequence delay.
   * @param reversed Whether the network is to reverse its inputs along the
   * time-step rank.
   * @param foremost Whether the network is to function as a foremost network.
   */
  inline LSTMNeuralNetwork(KernelPtr<Scalar, Rank>&& input_forget_kernel,
                           KernelPtr<Scalar, Rank>&& output_forget_kernel, KernelPtr<Scalar, Rank>&& input_write_kernel,
                           KernelPtr<Scalar, Rank>&& output_write_kernel,
                           KernelPtr<Scalar, Rank>&& input_candidate_kernel,
                           KernelPtr<Scalar, Rank>&& output_candidate_kernel,
                           KernelPtr<Scalar, Rank>&& input_read_kernel, KernelPtr<Scalar, Rank>&& output_read_kernel,
                           ActivationPtr<Scalar, Rank>&& forget_act, ActivationPtr<Scalar, Rank>&& write_act,
                           ActivationPtr<Scalar, Rank>&& candidate_act, ActivationPtr<Scalar, Rank>&& state_act,
                           ActivationPtr<Scalar, Rank>&& read_act, OutputSeqSizeFunc output_seq_size_func,
                           bool reversed = false, bool foremost = true)
      : main_cell(),
        output_seq_size_func(output_seq_size_func),
        reversed(reversed),
        foremost(foremost),
        cells(0),
        batch_size(-1),
        input_seq_length(-1),
        output_seq_length(-1),
        output_seq_delay(-1) {
    assert(output_forget_kernel && input_forget_kernel && output_write_kernel && input_write_kernel &&
           output_candidate_kernel && input_candidate_kernel && output_read_kernel && input_read_kernel && forget_act &&
           write_act && candidate_act && state_act && read_act);
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
  inline LSTMNeuralNetwork(const Self& network)
      : main_cell(network.main_cell),
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
    for (std::size_t i = 0; i < network.cells.size(); i++) cells.push_back(Cell(network.cells[i]));
  }
  inline LSTMNeuralNetwork(Self&& network) { swap(*this, network); }
  ~LSTMNeuralNetwork() = default;
  inline Self& operator=(Self network) {
    swap(*this, network);
    return *this;
  }
  inline Root* clone() const { return new LSTMNeuralNetwork(*this); }
  inline bool is_reversed() const { return reversed; }
  inline void reverse() { reversed = !reversed; }
  inline const typename Root::Dims& get_input_dims() const { return input_dims; }
  inline const typename Root::Dims& get_output_dims() const { return output_dims; }
  inline std::vector<const Layer<Scalar, Rank>*> get_layers() const {
    std::vector<const Layer<Scalar, Rank>*> layer_ptrs(13);
    populate_layer_vector<const Layer<Scalar, Rank>*>(layer_ptrs);
    return layer_ptrs;
  }
  inline std::vector<Layer<Scalar, Rank>*> get_layers() {
    std::vector<Layer<Scalar, Rank>*> layer_ptrs(13);
    populate_layer_vector<Layer<Scalar, Rank>*>(layer_ptrs);
    return layer_ptrs;
  }
  inline bool is_foremost() const { return foremost; }
  inline void set_foremost(bool foremost) {
    main_cell.input_forget_kernel->set_input_layer(foremost);
    main_cell.input_write_kernel->set_input_layer(foremost);
    main_cell.input_candidate_kernel->set_input_layer(foremost);
    main_cell.input_read_kernel->set_input_layer(foremost);
    this->foremost = foremost;
  }
  inline void empty_caches() {
    main_cell.input_read_kernel->empty_cache();
    main_cell.input_candidate_kernel->empty_cache();
    main_cell.input_write_kernel->empty_cache();
    main_cell.input_forget_kernel->empty_cache();
    main_cell.output_read_kernel->empty_cache();
    main_cell.output_candidate_kernel->empty_cache();
    main_cell.output_write_kernel->empty_cache();
    main_cell.output_forget_kernel->empty_cache();
    main_cell.write_act->empty_cache();
    main_cell.forget_act->empty_cache();
    main_cell.candidate_act->empty_cache();
    main_cell.state_act->empty_cache();
    main_cell.read_act->empty_cache();
    main_cell.forget_filter_cache = TimeStepData();
    main_cell.prev_state_cache = TimeStepData();
    main_cell.write_filter_cache = TimeStepData();
    main_cell.candidate_cache = TimeStepData();
    main_cell.read_filter_cache = TimeStepData();
    main_cell.activated_state_cache = TimeStepData();
    main_cell.weighted_input_forget_cache = TimeStepData();
    main_cell.weighted_output_forget_cache = TimeStepData();
    main_cell.weighted_input_write_cache = TimeStepData();
    main_cell.weighted_output_write_cache = TimeStepData();
    main_cell.weighted_input_candidate_cache = TimeStepData();
    main_cell.weighted_output_candidate_cache = TimeStepData();
    main_cell.weighted_input_read_cache = TimeStepData();
    main_cell.weighted_output_read_cache = TimeStepData();
    // Clear the state as well.
    batch_size = -1;
    state = TimeStepData();
    input_seq_length = -1;
    output_seq_length = -1;
    output_seq_delay = -1;
    cells = std::vector<Cell>(0);
  }
  inline typename Root::Data propagate(typename Root::Data input, bool training) {
    Dimensions<std::size_t, Root::DATA_RANK> data_dims = input.dimensions();
    assert(input_dims == data_dims.template demote<2>());
    int samples = data_dims(0);
    int input_seq_length = data_dims(1);
    std::pair<std::size_t, std::size_t> output_seq_info = output_seq_size_func((std::size_t)input_seq_length);
    int output_seq_length = (int)output_seq_info.first;
    int output_seq_delay = (int)output_seq_info.second;
    assert(output_seq_length > 0);
    if (reversed) Base::reverse_along_time_axis(input);
    int output_end = output_seq_length + output_seq_delay;
    int time_steps = std::max(input_seq_length, output_end);
    // Only unroll the network in training mode and if the sequence alignment
    // has changed.
    if (training && (input_seq_length != this->input_seq_length || output_seq_length != this->output_seq_length ||
                     output_seq_delay != this->output_seq_delay))
      unroll_network(time_steps, input_seq_length);
    setup_hidden_state(samples);
    RankwiseArray input_offsets, output_offsets;
    RankwiseArray input_extents = data_dims;
    RankwiseArray output_extents = output_dims.template promote<2>();
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
    Dimensions<std::size_t, Rank + 1> input_time_step_dims = input_dims.template promote<>();
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
          input_res = TensorMap<Scalar, Rank + 1>(input_slice.data(), input_time_step_dims);
        } else
          input_res = TensorMap<Scalar, Rank + 1>(input.data(), input_time_step_dims);
        if (i == 0) {
          // There must be an input at this time step and there cannot be an
          // output from the previous one.
          TimeStepData weighted_input_forget = cell.input_forget_kernel->pass_forward(input_res, training);
          // Cache the factors of the multiplication for the backward pass.
          cell.forget_filter_cache = cell.forget_act->pass_forward(std::move(weighted_input_forget), training);
          cell.prev_state_cache = std::move(state);
          // Selective remembrance.
          state = cell.forget_filter_cache * cell.prev_state_cache;
          TimeStepData weighted_input_write = cell.input_write_kernel->pass_forward(input_res, training);
          cell.write_filter_cache = cell.write_act->pass_forward(std::move(weighted_input_write), training);
          TimeStepData weighted_input_candidates = cell.input_candidate_kernel->pass_forward(input_res, training);
          cell.candidate_cache = cell.candidate_act->pass_forward(std::move(weighted_input_candidates), training);
          state += cell.write_filter_cache * cell.candidate_cache;
        } else {
          // There is both an input and an output from the previous time step.
          TimeStepData weighted_input_forget = cell.input_forget_kernel->pass_forward(input_res, training);
          TimeStepData weighted_output_forget = cell.output_forget_kernel->pass_forward(hidden_out, training);
          TimeStepData weighted_forget;
          if (MulInt) {
            if (training) {
              cell.weighted_input_forget_cache = std::move(weighted_input_forget);
              cell.weighted_output_forget_cache = std::move(weighted_output_forget);
              weighted_forget = cell.weighted_input_forget_cache * cell.weighted_output_forget_cache;
            } else
              weighted_forget = weighted_input_forget * weighted_output_forget;
          } else
            weighted_forget = weighted_input_forget + weighted_output_forget;
          cell.forget_filter_cache = cell.forget_act->pass_forward(std::move(weighted_forget), training);
          cell.prev_state_cache = std::move(state);
          state = cell.forget_filter_cache * cell.prev_state_cache;
          TimeStepData weighted_input_write = cell.input_write_kernel->pass_forward(input_res, training);
          TimeStepData weighted_output_write = cell.output_write_kernel->pass_forward(hidden_out, training);
          TimeStepData weighted_write;
          if (MulInt) {
            if (training) {
              cell.weighted_input_write_cache = std::move(weighted_input_write);
              cell.weighted_output_write_cache = std::move(weighted_output_write);
              weighted_write = cell.weighted_input_write_cache * cell.weighted_output_write_cache;
            } else
              weighted_write = weighted_input_write * weighted_output_write;
          } else
            weighted_write = weighted_input_write + weighted_output_write;
          cell.write_filter_cache = cell.write_act->pass_forward(std::move(weighted_write), training);
          TimeStepData weighted_input_candidates = cell.input_candidate_kernel->pass_forward(input_res, training);
          TimeStepData weighted_output_candidates = cell.output_candidate_kernel->pass_forward(hidden_out, training);
          TimeStepData weighted_candidates;
          if (MulInt) {
            if (training) {
              cell.weighted_input_candidate_cache = std::move(weighted_input_candidates);
              cell.weighted_output_candidate_cache = std::move(weighted_output_candidates);
              weighted_candidates = cell.weighted_input_candidate_cache * cell.weighted_output_candidate_cache;
            } else
              weighted_candidates = weighted_input_candidates * weighted_output_candidates;
          } else
            weighted_candidates = weighted_input_candidates + weighted_output_candidates;
          cell.candidate_cache = cell.candidate_act->pass_forward(std::move(weighted_candidates), training);
          state += cell.write_filter_cache * cell.candidate_cache;
        }
      } else {
        // There is only the output from the previous time step and no new input
        // (i must be greater than 0).
        TimeStepData weighted_output_forget = cell.output_forget_kernel->pass_forward(hidden_out, training);
        cell.forget_filter_cache = cell.forget_act->pass_forward(std::move(weighted_output_forget), training);
        cell.prev_state_cache = std::move(state);
        state = cell.forget_filter_cache * cell.prev_state_cache;
        TimeStepData weighted_output_write = cell.output_write_kernel->pass_forward(hidden_out, training);
        cell.write_filter_cache = cell.write_act->pass_forward(std::move(weighted_output_write), training);
        TimeStepData weighted_output_candidates = cell.output_candidate_kernel->pass_forward(hidden_out, training);
        cell.candidate_cache = cell.candidate_act->pass_forward(std::move(weighted_output_candidates), training);
        state += cell.write_filter_cache * cell.candidate_cache;
      }
      // Output computation.
      TimeStepData weighted_read;
      if (i < input_seq_length) {
        if (i == 0)
          weighted_read = cell.input_read_kernel->pass_forward(input_res, training);
        else {
          TimeStepData weighted_input_read = cell.input_read_kernel->pass_forward(input_res, training);
          TimeStepData weighted_output_read = cell.output_read_kernel->pass_forward(hidden_out, training);
          if (MulInt) {
            if (training) {
              cell.weighted_input_read_cache = std::move(weighted_input_read);
              cell.weighted_output_read_cache = std::move(weighted_output_read);
              weighted_read = cell.weighted_input_read_cache * cell.weighted_output_read_cache;
            } else
              weighted_read = weighted_input_read * weighted_output_read;
          } else
            weighted_read = weighted_input_read + weighted_output_read;
        }
      } else
        weighted_read = cell.output_read_kernel->pass_forward(hidden_out, training);
      cell.read_filter_cache = cell.read_act->pass_forward(std::move(weighted_read), training);
      cell.activated_state_cache = cell.state_act->pass_forward(state, training);
      hidden_out = cell.read_filter_cache * cell.activated_state_cache;
      // If there is a non-hidden output at this time step...
      if (i >= output_seq_delay && i < output_end) {
        TensorMap<Scalar, Root::DATA_RANK> out_i_seq(hidden_out.data(), output_extents);
        if (output_seq_length > 1) {
          out.slice(output_offsets, output_extents) = out_i_seq;
          output_offsets[1] += 1;
        } else
          out = out_i_seq;
      }
    }
    batch_size = samples;
    this->input_seq_length = input_seq_length;
    this->output_seq_length = output_seq_length;
    this->output_seq_delay = output_seq_delay;
    return out;
  }
  inline typename Root::Data backpropagate(typename Root::Data out_grad) {
    Dimensions<std::size_t, Root::DATA_RANK> data_dims = out_grad.dimensions();
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
    typename Root::Data prev_out_grad;
    if (input_seq_length > 1) {
      input_extents[1] = input_seq_length;
      prev_out_grad = typename Root::Data(input_extents);
    }
    input_extents[1] = 1;
    TimeStepData state_grad(state.dimensions());
    TimeStepData hidden_out_grad(state.dimensions());
    state_grad.setZero();
    hidden_out_grad.setZero();
    Dimensions<std::size_t, Rank + 1> out_time_step_dims = output_dims.template promote<>();
    out_time_step_dims(0) = batch_size;
    int output_end = output_seq_length + output_seq_delay;
    int time_steps = std::max(input_seq_length, output_end);
    for (int i = time_steps - 1; i >= 0; --i) {
      Cell& cell = i == 0 ? main_cell : cells[i - 1];
      // If there was a non-hidden output at the time step, let the gradients
      // flow into the hidden output gradients.
      if (i >= output_seq_delay && i < output_end) {
        if (output_seq_length == 1) {
          hidden_out_grad += TensorMap<Scalar, Rank + 1>(out_grad.data(), out_time_step_dims);
        } else {
          typename Root::Data out_grad_seq = out_grad.slice(output_offsets, output_extents);
          output_offsets[1] -= 1;
          hidden_out_grad += TensorMap<Scalar, Rank + 1>(out_grad_seq.data(), out_time_step_dims);
        }
      }
      state_grad += cell.state_act->pass_back(cell.read_filter_cache * hidden_out_grad);
      TimeStepData weighted_read_grad = cell.read_act->pass_back(cell.activated_state_cache * hidden_out_grad);
      TimeStepData candidate_grad = cell.candidate_act->pass_back(cell.write_filter_cache * state_grad);
      TimeStepData weighted_write_grad = cell.write_act->pass_back(cell.candidate_cache * state_grad);
      TimeStepData weighted_forget_grad = cell.forget_act->pass_back(cell.prev_state_cache * state_grad);
      state_grad *= cell.forget_filter_cache;
      if (i < input_seq_length) {
        TimeStepData prev_out_grad_i;
        if (MulInt) {
          if (i != 0) {
            // Calculate the previous hidden output gradients.
            hidden_out_grad =
                cell.output_read_kernel->pass_back(cell.weighted_input_read_cache * weighted_read_grad) +
                cell.output_candidate_kernel->pass_back(cell.weighted_input_candidate_cache * candidate_grad) +
                cell.output_write_kernel->pass_back(cell.weighted_input_write_cache * weighted_write_grad) +
                cell.output_forget_kernel->pass_back(cell.weighted_input_forget_cache * weighted_forget_grad);
            // Calculate the input gradients.
            prev_out_grad_i =
                cell.input_read_kernel->pass_back(cell.weighted_output_read_cache * weighted_read_grad) +
                cell.input_candidate_kernel->pass_back(cell.weighted_output_candidate_cache * candidate_grad) +
                cell.input_write_kernel->pass_back(cell.weighted_output_write_cache * weighted_write_grad) +
                cell.input_forget_kernel->pass_back(cell.weighted_output_forget_cache * weighted_forget_grad);
          } else {
            prev_out_grad_i = cell.input_read_kernel->pass_back(std::move(weighted_read_grad)) +
                              cell.input_candidate_kernel->pass_back(std::move(candidate_grad)) +
                              cell.input_write_kernel->pass_back(std::move(weighted_write_grad)) +
                              cell.input_forget_kernel->pass_back(std::move(weighted_forget_grad));
          }
        } else {
          if (i != 0) {
            hidden_out_grad = cell.output_read_kernel->pass_back(weighted_read_grad) +
                              cell.output_candidate_kernel->pass_back(candidate_grad) +
                              cell.output_write_kernel->pass_back(weighted_write_grad) +
                              cell.output_forget_kernel->pass_back(weighted_forget_grad);
          }
          prev_out_grad_i = cell.input_read_kernel->pass_back(std::move(weighted_read_grad)) +
                            cell.input_candidate_kernel->pass_back(std::move(candidate_grad)) +
                            cell.input_write_kernel->pass_back(std::move(weighted_write_grad)) +
                            cell.input_forget_kernel->pass_back(std::move(weighted_forget_grad));
        }
        if (!foremost) {
          TensorMap<Scalar, Root::DATA_RANK> prev_out_grad_i_seq(prev_out_grad_i.data(), input_extents);
          if (input_seq_length > 1) {
            prev_out_grad.slice(input_offsets, input_extents) = prev_out_grad_i_seq;
            input_offsets[1] -= 1;
          } else
            prev_out_grad = prev_out_grad_i_seq;
        }
      } else {
        hidden_out_grad = cell.output_read_kernel->pass_back(std::move(weighted_read_grad)) +
                          cell.output_candidate_kernel->pass_back(std::move(candidate_grad)) +
                          cell.output_write_kernel->pass_back(std::move(weighted_write_grad)) +
                          cell.output_forget_kernel->pass_back(std::move(weighted_forget_grad));
      }
    }
    return prev_out_grad;
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

 private:
  template <typename _LayerPtr>
  inline void populate_layer_vector(std::vector<_LayerPtr>& layer_ptrs) const {
    layer_ptrs[0] = main_cell.input_forget_kernel.get();
    layer_ptrs[1] = main_cell.output_forget_kernel.get();
    layer_ptrs[2] = main_cell.input_write_kernel.get();
    layer_ptrs[3] = main_cell.output_write_kernel.get();
    layer_ptrs[4] = main_cell.input_candidate_kernel.get();
    layer_ptrs[5] = main_cell.output_candidate_kernel.get();
    layer_ptrs[6] = main_cell.input_read_kernel.get();
    layer_ptrs[7] = main_cell.output_read_kernel.get();
    layer_ptrs[8] = main_cell.forget_act.get();
    layer_ptrs[9] = main_cell.write_act.get();
    layer_ptrs[10] = main_cell.candidate_act.get();
    layer_ptrs[11] = main_cell.read_act.get();
    layer_ptrs[12] = main_cell.state_act.get();
  }
  inline void unroll_network(std::size_t time_steps, std::size_t input_seq_length) {
    if (time_steps > 1) {
      empty_caches();
      cells = std::vector<Cell>(time_steps - 1);
      for (int j = 1; j < time_steps; ++j) {
        Cell& cell = cells[j - 1];
        cell.output_forget_kernel = KernelPtr<Scalar, Rank>(
            static_cast<KernelLayer<Scalar, Rank>*>(main_cell.output_forget_kernel->clone_with_shared_params()));
        cell.output_write_kernel = KernelPtr<Scalar, Rank>(
            static_cast<KernelLayer<Scalar, Rank>*>(main_cell.output_write_kernel->clone_with_shared_params()));
        cell.output_candidate_kernel = KernelPtr<Scalar, Rank>(
            static_cast<KernelLayer<Scalar, Rank>*>(main_cell.output_candidate_kernel->clone_with_shared_params()));
        cell.output_read_kernel = KernelPtr<Scalar, Rank>(
            static_cast<KernelLayer<Scalar, Rank>*>(main_cell.output_read_kernel->clone_with_shared_params()));
        cell.write_act = ActivationPtr<Scalar, Rank>(
            static_cast<ActivationLayer<Scalar, Rank>*>(main_cell.write_act->clone_with_shared_params()));
        cell.forget_act = ActivationPtr<Scalar, Rank>(
            static_cast<ActivationLayer<Scalar, Rank>*>(main_cell.forget_act->clone_with_shared_params()));
        cell.candidate_act = ActivationPtr<Scalar, Rank>(
            static_cast<ActivationLayer<Scalar, Rank>*>(main_cell.candidate_act->clone_with_shared_params()));
        cell.state_act = ActivationPtr<Scalar, Rank>(
            static_cast<ActivationLayer<Scalar, Rank>*>(main_cell.state_act->clone_with_shared_params()));
        cell.read_act = ActivationPtr<Scalar, Rank>(
            static_cast<ActivationLayer<Scalar, Rank>*>(main_cell.read_act->clone_with_shared_params()));
        if (j < input_seq_length) {
          cell.input_forget_kernel = KernelPtr<Scalar, Rank>(
              static_cast<KernelLayer<Scalar, Rank>*>(main_cell.input_forget_kernel->clone_with_shared_params()));
          cell.input_write_kernel = KernelPtr<Scalar, Rank>(
              static_cast<KernelLayer<Scalar, Rank>*>(main_cell.input_write_kernel->clone_with_shared_params()));
          cell.input_candidate_kernel = KernelPtr<Scalar, Rank>(
              static_cast<KernelLayer<Scalar, Rank>*>(main_cell.input_candidate_kernel->clone_with_shared_params()));
          cell.input_read_kernel = KernelPtr<Scalar, Rank>(
              static_cast<KernelLayer<Scalar, Rank>*>(main_cell.input_read_kernel->clone_with_shared_params()));
        }
      }
    } else
      cells = std::vector<Cell>(0);
  }
  inline void setup_hidden_state(std::size_t samples) {
    if (!Stateful || batch_size == -1) {
      Dimensions<std::size_t, Rank + 1> dims = main_cell.forget_act->get_output_dims().template promote<>();
      dims(0) = samples;
      state = TimeStepData(dims);
      state.setZero();
    } else if (samples != batch_size) {
      std::array<std::size_t, Rank + 1> offsets;
      std::array<std::size_t, Rank + 1> extents = main_cell.forget_act->get_output_dims().template promote<>();
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
   * A struct representing a cell in the unrolled LSTM.
   */
  struct Cell {
    inline Cell() {}
    inline Cell(const Cell& cell)
        : input_forget_kernel(
              KernelPtr<Scalar, Rank>(static_cast<KernelLayer<Scalar, Rank>*>(cell.input_forget_kernel->clone()))),
          output_forget_kernel(
              KernelPtr<Scalar, Rank>(static_cast<KernelLayer<Scalar, Rank>*>(cell.output_forget_kernel->clone()))),
          input_write_kernel(
              KernelPtr<Scalar, Rank>(static_cast<KernelLayer<Scalar, Rank>*>(cell.input_write_kernel->clone()))),
          output_write_kernel(
              KernelPtr<Scalar, Rank>(static_cast<KernelLayer<Scalar, Rank>*>(cell.output_write_kernel->clone()))),
          input_candidate_kernel(
              KernelPtr<Scalar, Rank>(static_cast<KernelLayer<Scalar, Rank>*>(cell.input_candidate_kernel->clone()))),
          output_candidate_kernel(
              KernelPtr<Scalar, Rank>(static_cast<KernelLayer<Scalar, Rank>*>(cell.output_candidate_kernel->clone()))),
          input_read_kernel(
              KernelPtr<Scalar, Rank>(static_cast<KernelLayer<Scalar, Rank>*>(cell.input_read_kernel->clone()))),
          output_read_kernel(
              KernelPtr<Scalar, Rank>(static_cast<KernelLayer<Scalar, Rank>*>(cell.output_read_kernel->clone()))),
          write_act(ActivationPtr<Scalar, Rank>(static_cast<ActivationLayer<Scalar, Rank>*>(cell.write_act->clone()))),
          forget_act(
              ActivationPtr<Scalar, Rank>(static_cast<ActivationLayer<Scalar, Rank>*>(cell.forget_act->clone()))),
          candidate_act(
              ActivationPtr<Scalar, Rank>(static_cast<ActivationLayer<Scalar, Rank>*>(cell.candidate_act->clone()))),
          state_act(ActivationPtr<Scalar, Rank>(static_cast<ActivationLayer<Scalar, Rank>*>(cell.state_act->clone()))),
          read_act(ActivationPtr<Scalar, Rank>(static_cast<ActivationLayer<Scalar, Rank>*>(cell.read_act->clone()))),
          forget_filter_cache(cell.forget_filter_cache),
          prev_state_cache(cell.prev_state_cache),
          write_filter_cache(cell.write_filter_cache),
          candidate_cache(cell.candidate_cache),
          read_filter_cache(cell.read_filter_cache),
          activated_state_cache(cell.activated_state_cache),
          weighted_input_forget_cache(cell.weighted_input_forget_cache),
          weighted_output_forget_cache(cell.weighted_output_forget_cache),
          weighted_input_write_cache(cell.weighted_input_write_cache),
          weighted_output_write_cache(cell.weighted_output_write_cache),
          weighted_input_candidate_cache(cell.weighted_input_candidate_cache),
          weighted_output_candidate_cache(cell.weighted_output_candidate_cache),
          weighted_input_read_cache(cell.weighted_input_read_cache),
          weighted_output_read_cache(cell.weighted_output_read_cache) {}
    KernelPtr<Scalar, Rank> input_forget_kernel, output_forget_kernel, input_write_kernel, output_write_kernel,
        input_candidate_kernel, output_candidate_kernel, input_read_kernel, output_read_kernel;
    ActivationPtr<Scalar, Rank> forget_act, write_act, candidate_act, state_act, read_act;
    // Caches for the derivation of multiplicative filtering operations.
    TimeStepData forget_filter_cache, prev_state_cache, write_filter_cache, candidate_cache, read_filter_cache,
        activated_state_cache;
    // Caches for the derivation of multiplicative integration operations.
    TimeStepData weighted_input_forget_cache, weighted_output_forget_cache, weighted_input_write_cache,
        weighted_output_write_cache, weighted_input_candidate_cache, weighted_output_candidate_cache,
        weighted_input_read_cache, weighted_output_read_cache;
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

#endif /* C_ATTL3_NEURAL_NETWORK_LSTMNEURALNETWORK_H_ */
