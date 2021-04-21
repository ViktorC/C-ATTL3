/*
 * BidirectionalNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_BIDIRECTIONALNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_BIDIRECTIONALNEURALNETWORK_H_

#include <pthread.h>

#include <array>
#include <cassert>
#include <utility>

#include "neural_network/CompositeNeuralNetwork.hpp"
#include "neural_network/UnidirectionalNeuralNetwork.hpp"

namespace cattle {

/**
 * An enumeration type for the different ways the outputs of sub-modules of
 * neural networks may be merged.
 */
enum BidirectionalOutputMergeType {
  BIDIRECTIONAL_CONCAT_LO_RANK,
  BIDIRECTIONAL_CONCAT_HI_RANK,
  BIDIRECTIONAL_SUM,
  BIDIRECTIONAL_MUL
};

/**
 * An alias for unidirectional recurrent neural network of arbitrary scalar type
 * and rank.
 */
template <typename Scalar, std::size_t Rank>
using UnidirNeuralNetPtr = std::unique_ptr<UnidirectionalNeuralNetwork<Scalar, Rank>>;

/**
 * A class template for a bidirectional neural network that takes a
 * unidirectional recurrent network, clones it, reverses the clone's processing
 * direction, and uses the two networks as its parallel sub-modules. The outputs
 * of the two sub-networks can be merged by summation or concatenation either
 * along the lowest (the 3rd after the sample and time-step ranks) or highest
 * rank.
 *
 * \see
 * https://pdfs.semanticscholar.org/4b80/89bc9b49f84de43acc2eb8900035f7d492b2.pdf
 */
template <typename Scalar, std::size_t Rank, BidirectionalOutputMergeType MergeType = BIDIRECTIONAL_CONCAT_LO_RANK>
class BidirectionalNeuralNetwork
    : public CompositeNeuralNetwork<Scalar, Rank, true, UnidirectionalNeuralNetwork<Scalar, Rank>> {
  typedef NeuralNetwork<Scalar, Rank, true> Base;
  typedef BidirectionalNeuralNetwork<Scalar, Rank, MergeType> Self;
  typedef UnidirNeuralNetPtr<Scalar, Rank> UnidirNet;
  typedef std::array<std::size_t, Base::DATA_RANK> RankwiseArray;
  static_assert(MergeType >= BIDIRECTIONAL_CONCAT_LO_RANK && MergeType <= BIDIRECTIONAL_MUL,
                "illegal merge type value");
  static constexpr std::size_t CONCAT_RANK = MergeType == BIDIRECTIONAL_CONCAT_HI_RANK ? Rank - 1 : 0;
  static constexpr std::size_t CONCAT_BATCH_RANK = CONCAT_RANK + 2;

 public:
  /**
   * @param network A unique pointer to a unidirectional recurrent neural
   * network that, along with its reversed clone, will constitute the
   * bidirectional network.
   * @param foremost Whether the network is to function as a foremost network.
   */
  inline BidirectionalNeuralNetwork(UnidirNet&& network, bool foremost = true)
      : net(std::move(network)), foremost(foremost) {
    assert(this->net);
    net_rev = UnidirNet(static_cast<UnidirectionalNeuralNetwork<Scalar, Rank>*>(this->net->clone()));
    net_rev->reverse();
    input_dims = this->net->get_input_dims();
    output_dims = this->net->get_output_dims();
    if (MergeType == BIDIRECTIONAL_CONCAT_LO_RANK || MergeType == BIDIRECTIONAL_CONCAT_HI_RANK)
      output_dims(+CONCAT_RANK) *= 2;
  }
  inline BidirectionalNeuralNetwork(const Self& network)
      : net(UnidirNet(static_cast<UnidirectionalNeuralNetwork<Scalar, Rank>*>(network.net->clone()))),
        net_rev(UnidirNet(static_cast<UnidirectionalNeuralNetwork<Scalar, Rank>*>(network.net_rev->clone()))),
        foremost(network.foremost),
        input_dims(network.input_dims),
        output_dims(network.output_dims),
        output(network.output),
        output_rev(network.output_rev) {}
  inline BidirectionalNeuralNetwork(Self&& network) { swap(*this, network); }
  ~BidirectionalNeuralNetwork() = default;
  inline Self& operator=(Self network) {
    swap(*this, network);
    return *this;
  }
  inline Base* clone() const { return new BidirectionalNeuralNetwork(*this); }
  inline const typename Base::Dims& get_input_dims() const { return input_dims; }
  inline const typename Base::Dims& get_output_dims() const { return output_dims; }
  inline std::vector<const Layer<Scalar, Rank>*> get_layers() const {
    std::vector<const Layer<Scalar, Rank>*> layer_ptrs;
    populate_layer_vector<const Layer<Scalar, Rank>*>(layer_ptrs);
    return layer_ptrs;
  }
  inline std::vector<Layer<Scalar, Rank>*> get_layers() {
    std::vector<Layer<Scalar, Rank>*> layer_ptrs;
    populate_layer_vector<Layer<Scalar, Rank>*>(layer_ptrs);
    return layer_ptrs;
  }
  inline std::vector<UnidirectionalNeuralNetwork<Scalar, Rank>*> get_modules() {
    std::vector<UnidirectionalNeuralNetwork<Scalar, Rank>*> modules;
    modules.push_back(net.get());
    modules.push_back(net_rev.get());
    return modules;
  }
  inline bool is_foremost() const { return foremost; }
  inline void set_foremost(bool foremost) {
    net->set_foremost(foremost);
    net_rev->set_foremost(foremost);
    this->foremost = foremost;
  }
  inline void empty_caches() {
    net->empty_caches();
    net_rev->empty_caches();
    output = typename Base::Data();
    output_rev = typename Base::Data();
  }
  inline typename Base::Data propagate(typename Base::Data input, bool training) {
    assert(input_dims == (Dimensions<std::size_t, Base::DATA_RANK>(input.dimensions()).template demote<2>()));
    pthread_attr_t attr;
    pthread_t helper_thread;
    int pthread_state;
    pthread_state = pthread_attr_init(&attr);
    assert(!pthread_state);
    pthread_state = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    assert(!pthread_state);
    PropArgs args;
    args.obj = this;
    args.training = training;
    args.in = &input;
    pthread_state = pthread_create(&helper_thread, &attr, propagate, &args);
    assert(!pthread_state);
    typename Base::Data forward_out = net->propagate(input, training);
    pthread_state = pthread_join(helper_thread, nullptr);
    assert(!pthread_state);
    pthread_state = pthread_attr_destroy(&attr);
    assert(!pthread_state);
    assert(forward_out.dimension(1) == args.out.dimension(1));
    if (MergeType == BIDIRECTIONAL_SUM)
      return forward_out + args.out;
    else if (MergeType == BIDIRECTIONAL_MUL) {
      output = std::move(forward_out);
      output_rev = std::move(args.out);
      return output * output_rev;
    } else
      return forward_out.concatenate(args.out, +CONCAT_BATCH_RANK);
  }
  inline typename Base::Data backpropagate(typename Base::Data out_grad) {
    Dimensions<std::size_t, Base::DATA_RANK> dims(out_grad.dimensions());
    assert(output_dims == dims.template demote<2>());
    pthread_attr_t attr;
    pthread_t helper_thread;
    int pthread_state;
    pthread_state = pthread_attr_init(&attr);
    assert(!pthread_state);
    pthread_state = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    assert(!pthread_state);
    BackpropArgs args;
    args.obj = this;
    typename Base::Data forward_prev_out_grad;
    if (MergeType == BIDIRECTIONAL_SUM) {
      args.out_grad = &out_grad;
      pthread_state = pthread_create(&helper_thread, &attr, backpropagate, &args);
      assert(!pthread_state);
      forward_prev_out_grad = net->backpropagate(out_grad);
      pthread_state = pthread_join(helper_thread, nullptr);
      assert(!pthread_state);
      out_grad = typename Base::Data();
    } else if (MergeType == BIDIRECTIONAL_MUL) {
      typename Base::Data out_grad_rev = output * out_grad;
      args.out_grad = &out_grad_rev;
      pthread_state = pthread_create(&helper_thread, &attr, backpropagate, &args);
      assert(!pthread_state);
      out_grad *= output_rev;
      forward_prev_out_grad = net->backpropagate(std::move(out_grad));
      pthread_state = pthread_join(helper_thread, nullptr);
      assert(!pthread_state);
    } else {
      RankwiseArray offsets;
      RankwiseArray extents = dims;
      offsets.fill(0);
      extents[+CONCAT_BATCH_RANK] /= 2;
      offsets[+CONCAT_BATCH_RANK] += extents[+CONCAT_BATCH_RANK];
      typename Base::Data backward_slice = out_grad.slice(offsets, extents);
      args.out_grad = &backward_slice;
      pthread_state = pthread_create(&helper_thread, &attr, backpropagate, &args);
      assert(!pthread_state);
      offsets[+CONCAT_BATCH_RANK] -= extents[+CONCAT_BATCH_RANK];
      typename Base::Data forward_slice = out_grad.slice(offsets, extents);
      out_grad = typename Base::Data();
      forward_prev_out_grad = net->backpropagate(std::move(forward_slice));
      // Make sure that backward_slice does not go out of scope before the
      // thread terminates.
      pthread_state = pthread_join(helper_thread, nullptr);
      assert(!pthread_state);
    }
    pthread_state = pthread_attr_destroy(&attr);
    assert(!pthread_state);
    return forward_prev_out_grad + args.prev_out_grad;
  }
  inline friend void swap(Self& network1, Self& network2) {
    using std::swap;
    swap(network1.net, network2.net);
    swap(network1.net_rev, network2.net_rev);
    swap(network1.foremost, network2.foremost);
    swap(network1.input_dims, network2.input_dims);
    swap(network1.output_dims, network2.output_dims);
    swap(network1.output, network2.output);
    swap(network1.output_rev, network2.output_rev);
  }

 private:
  /**
   * The propagation function executed in a different thread for each lane of a
   * parallel network.
   *
   * @param args_ptr The propagation argument struct containing all necessary
   * information.
   */
  inline static void* propagate(void* args_ptr) {
    PropArgs& args = *((PropArgs*)args_ptr);
    args.out = args.obj->net_rev->propagate(*args.in, args.training);
    return nullptr;
  }
  /**
   * The back-propagation function executed in a different thread for each lane
   * of a parallel network.
   *
   * @param args_ptr The back-propagation argument struct containing all
   * necessary information.
   */
  inline static void* backpropagate(void* args_ptr) {
    BackpropArgs& args = *((BackpropArgs*)args_ptr);
    args.prev_out_grad = args.obj->net_rev->backpropagate(*args.out_grad);
    return nullptr;
  }
  template <typename _LayerPtr>
  inline void populate_layer_vector(std::vector<_LayerPtr>& layer_ptrs) const {
    std::vector<Layer<Scalar, Rank>*> net_layer_ptrs = net->get_layers();
    for (std::size_t i = 0; i < net_layer_ptrs.size(); ++i) layer_ptrs.push_back(net_layer_ptrs[i]);
    std::vector<Layer<Scalar, Rank>*> net_rev_layer_ptrs = net_rev->get_layers();
    for (std::size_t i = 0; i < net_rev_layer_ptrs.size(); ++i) layer_ptrs.push_back(net_rev_layer_ptrs[i]);
  }
  UnidirNet net, net_rev;
  bool foremost;
  typename Base::Dims input_dims, output_dims;
  typename Base::Data output, output_rev;
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
    typename Base::Data* out_grad;
    typename Base::Data prev_out_grad;
  };
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_BIDIRECTIONALNEURALNETWORK_H_ */
