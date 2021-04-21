/*
 * SequentialNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_SEQUENTIALNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_SEQUENTIALNEURALNETWORK_H_

#include <array>
#include <cassert>
#include <utility>

#include "neural_network/CompositeNeuralNetwork.hpp"

namespace cattle {

/**
 * A class template for a wrapper neural network that enables the use of
 * non-sequential networks on sequential data by joining the 'samples' and 'time
 * steps' ranks of the tensors and splitting them again once the internal,
 * non-sequential network is done processing them.
 */
template <typename Scalar, std::size_t Rank>
class SequentialNeuralNetwork : public CompositeNeuralNetwork<Scalar, Rank, true, NeuralNetwork<Scalar, Rank, false>> {
  typedef NeuralNetwork<Scalar, Rank, true> Base;
  typedef SequentialNeuralNetwork<Scalar, Rank> Self;
  typedef NeuralNetPtr<Scalar, Rank, false> Net;
  typedef std::array<std::size_t, Base::DATA_RANK> RankwiseArray;

 public:
  /**
   * @param network A unique pointer to a non-sequential neural network to wrap.
   * @param foremost Whether the network is to function as a foremost network.
   */
  inline SequentialNeuralNetwork(Net&& network, bool foremost = true) : net(std::move(network)), foremost(foremost) {
    assert(net);
    input_dims = net->get_input_dims();
    output_dims = net->get_output_dims();
    joint_input_dims = input_dims.template promote<>();
    joint_output_dims = output_dims.template promote<>();
    split_input_dims = input_dims.template promote<2>();
    split_output_dims = output_dims.template promote<2>();
    set_foremost(foremost);
  }
  inline SequentialNeuralNetwork(const Self& network)
      : net(Net(network.net->clone())),
        foremost(network.foremost),
        input_dims(network.input_dims),
        output_dims(network.output_dims),
        joint_input_dims(network.joint_input_dims),
        joint_output_dims(network.joint_output_dims),
        split_input_dims(network.split_input_dims),
        split_output_dims(network.split_output_dims) {}
  inline SequentialNeuralNetwork(Self&& network) { swap(*this, network); }
  ~SequentialNeuralNetwork() = default;
  inline Self& operator=(Self network) {
    swap(*this, network);
    return *this;
  }
  inline Base* clone() const { return new SequentialNeuralNetwork(*this); }
  inline const typename Base::Dims& get_input_dims() const { return input_dims; }
  inline const typename Base::Dims& get_output_dims() const { return output_dims; }
  inline std::vector<const Layer<Scalar, Rank>*> get_layers() const {
    return ((const NeuralNetwork<Scalar, Rank, false>&)*net).get_layers();
  }
  inline std::vector<Layer<Scalar, Rank>*> get_layers() { return net->get_layers(); }
  inline std::vector<NeuralNetwork<Scalar, Rank, false>*> get_modules() {
    std::vector<NeuralNetwork<Scalar, Rank, false>*> modules;
    modules.push_back(net.get());
    return modules;
  }
  inline bool is_foremost() const { return foremost; }
  inline void set_foremost(bool foremost) {
    net->set_foremost(foremost);
    this->foremost = foremost;
  }
  inline void empty_caches() { net->empty_caches(); }
  inline typename Base::Data propagate(typename Base::Data input, bool training) {
    assert(input_dims == (Dimensions<std::size_t, Base::DATA_RANK>(input.dimensions()).template demote<2>()));
    std::size_t batch_size = input.dimension(0);
    std::size_t seq_length = input.dimension(1);
    joint_input_dims[0] = batch_size * seq_length;
    split_output_dims[0] = batch_size;
    split_output_dims[1] = seq_length;
    TensorMap<Scalar, Rank + 1> joint_input(input.data(), joint_input_dims);
    Tensor<Scalar, Rank + 1> out = net->propagate(joint_input, training);
    return TensorMap<Scalar, Rank + 2>(out.data(), split_output_dims);
  }
  inline typename Base::Data backpropagate(typename Base::Data out_grad) {
    assert(output_dims == (Dimensions<std::size_t, Base::DATA_RANK>(out_grad.dimensions()).template demote<2>()));
    assert(split_output_dims[0] == out_grad.dimension(0));
    std::size_t batch_size = out_grad.dimension(0);
    std::size_t seq_length = out_grad.dimension(1);
    joint_output_dims[0] = batch_size * seq_length;
    TensorMap<Scalar, Rank + 1> joint_out_grad(out_grad.data(), joint_output_dims);
    if (foremost) {
      net->backpropagate(joint_out_grad);
      return typename Base::Data();
    } else {
      Tensor<Scalar, Rank + 1> prev_out_grad = net->backpropagate(joint_out_grad);
      split_input_dims[0] = batch_size;
      split_input_dims[1] = seq_length;
      return TensorMap<Scalar, Rank + 2>(prev_out_grad.data(), split_input_dims);
    }
  }
  inline friend void swap(Self& network1, Self& network2) {
    using std::swap;
    swap(network1.net, network2.net);
    swap(network1.foremost, network2.foremost);
    swap(network1.input_dims, network2.input_dims);
    swap(network1.output_dims, network2.output_dims);
    swap(network1.joint_input_dims, network2.joint_input_dims);
    swap(network1.joint_output_dims, network2.joint_output_dims);
    swap(network1.split_input_dims, network2.split_input_dims);
    swap(network1.split_output_dims, network2.split_output_dims);
  }

 private:
  Net net;
  bool foremost;
  typename Base::Dims input_dims, output_dims;
  std::array<std::size_t, Rank + 1> joint_input_dims, joint_output_dims;
  std::array<std::size_t, Rank + 2> split_input_dims, split_output_dims;
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_SEQUENTIALNEURALNETWORK_H_ */
