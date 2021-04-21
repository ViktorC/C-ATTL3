/*
 * DenseNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_DENSENEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_DENSENEURALNETWORK_H_

#include <array>
#include <cassert>
#include <utility>

#include "neural_network/CompositeNeuralNetwork.hpp"

namespace cattle {

/**
 * An enumeration type for the different ways the input of a layer in a dense
 * network may be concatenated to its output.
 */
enum DenseConcatType { DENSE_LOWEST_RANK, DENSE_HIGHEST_RANK };

/**
 * A class template for DenseNet architectures. These networks consist of
 * sub-modules that are all 'connected' to each other as in the input of each
 * module is concatenated to its output and then propagated to the next module
 * as its input. The input is concatenated to the output either along its lowest
 * or highest rank.
 *
 * \see https://arxiv.org/abs/1608.06993
 */
template <typename Scalar, std::size_t Rank, DenseConcatType ConcatType = DENSE_HIGHEST_RANK>
class DenseNeuralNetwork : public CompositeNeuralNetwork<Scalar, Rank, false, NeuralNetwork<Scalar, Rank, false>> {
  typedef NeuralNetwork<Scalar, Rank, false> Base;
  typedef NeuralNetPtr<Scalar, Rank, false> Module;
  typedef DenseNeuralNetwork<Scalar, Rank, ConcatType> Self;
  typedef std::array<std::size_t, Base::DATA_RANK> RankwiseArray;
  static_assert(ConcatType >= DENSE_LOWEST_RANK && ConcatType <= DENSE_HIGHEST_RANK, "illegal merge type value");
  static constexpr std::size_t CONCAT_RANK = ConcatType == DENSE_HIGHEST_RANK ? Rank - 1 : 0;
  static constexpr std::size_t CONCAT_BATCH_RANK = CONCAT_RANK + 1;

 public:
  /**
   * @param modules A vector of dense modules.
   * @param foremost Whether the network is to function as a foremost network.
   */
  inline DenseNeuralNetwork(std::vector<Module>&& modules, bool foremost = true)
      : modules(std::move(modules)), foremost(foremost) {
    assert(this->modules.size() > 0 && "modules must contain at least 1 element");
    Base& first_module = *this->modules[0];
    input_dims = first_module.get_input_dims();
    typename Base::Dims output_dims = first_module.get_output_dims();
    output_dims(+CONCAT_RANK) += input_dims(+CONCAT_RANK);
    if (ConcatType == DENSE_LOWEST_RANK) {
      for (std::size_t i = Rank - 1; i > +CONCAT_RANK; --i) assert(input_dims(i) == output_dims(i));
    } else {
      for (std::size_t i = 0; i < +CONCAT_RANK; ++i) assert(input_dims(i) == output_dims(i));
    }
    for (std::size_t i = 1; i < this->modules.size(); ++i) {
      Base& module = *this->modules[i];
      const typename Base::Dims& module_input_dims = module.get_input_dims();
      assert(module_input_dims == output_dims && "incompatible module dimensions");
      output_dims(+CONCAT_RANK) += module.get_output_dims()(+CONCAT_RANK);
      module.set_foremost(false);
    }
    this->output_dims = output_dims;
    first_module.set_foremost(foremost);
  }
  /**
   * @param module A single dense module.
   * @param foremost Whether the network is to function as a foremost network.
   */
  inline DenseNeuralNetwork(Module&& module, bool foremost = true)
      : DenseNeuralNetwork(create_vector(std::move(module), foremost)) {}
  inline DenseNeuralNetwork(const Self& network)
      : modules(network.modules.size()),
        foremost(network.foremost),
        input_dims(network.input_dims),
        output_dims(network.output_dims) {
    for (std::size_t i = 0; i < modules.size(); ++i) modules[i] = Module(network.modules[i]->clone());
  }
  inline DenseNeuralNetwork(Self&& network) { swap(*this, network); }
  ~DenseNeuralNetwork() = default;
  inline Self& operator=(Self network) {
    swap(*this, network);
    return *this;
  }
  inline Base* clone() const { return new DenseNeuralNetwork(*this); }
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
  inline std::vector<Base*> get_modules() {
    std::vector<Base*> modules;
    for (std::size_t i = 0; i < this->modules.size(); ++i) modules.push_back(this->modules[i].get());
    return modules;
  }
  inline bool is_foremost() const { return foremost; }
  inline void set_foremost(bool foremost) {
    modules[0]->set_foremost(foremost);
    this->foremost = foremost;
  }
  inline void empty_caches() {
    for (std::size_t i = 0; i < modules.size(); ++i) modules[i]->empty_caches();
  }
  inline typename Base::Data propagate(typename Base::Data input, bool training) {
    assert(input_dims == (Dimensions<std::size_t, Base::DATA_RANK>(input.dimensions()).template demote<>()));
    for (std::size_t i = 0; i < modules.size(); ++i) {
      typename Base::Data concat = input.concatenate(modules[i]->propagate(input, training), +CONCAT_BATCH_RANK);
      input = std::move(concat);
    }
    return input;
  }
  inline typename Base::Data backpropagate(typename Base::Data out_grad) {
    assert(output_dims == (Dimensions<std::size_t, Base::DATA_RANK>(out_grad.dimensions()).template demote<>()));
    RankwiseArray offsets;
    RankwiseArray extents = input_dims.template promote<>();
    offsets.fill(0);
    extents[0] = out_grad.dimension(0);
    for (int i = modules.size() - 1; i >= 0; --i) {
      Base& module = *modules[i];
      int layer_input_concat_rank_dim = module.get_input_dims()(+CONCAT_RANK);
      int layer_output_concat_rank_dim = module.get_output_dims()(+CONCAT_RANK);
      offsets[+CONCAT_BATCH_RANK] = layer_input_concat_rank_dim;
      extents[+CONCAT_BATCH_RANK] = layer_output_concat_rank_dim;
      typename Base::Data out_grad_i = out_grad.slice(offsets, extents);
      offsets[+CONCAT_BATCH_RANK] = 0;
      extents[+CONCAT_BATCH_RANK] = layer_input_concat_rank_dim;
      if (foremost && i == 0)
        module.backpropagate(std::move(out_grad_i));
      else
        out_grad = typename Base::Data(out_grad.slice(offsets, extents) + module.backpropagate(std::move(out_grad_i)));
    }
    return out_grad;
  }
  inline friend void swap(Self& network1, Self& network2) {
    using std::swap;
    swap(network1.modules, network2.modules);
    swap(network1.foremost, network2.foremost);
    swap(network1.input_dims, network2.input_dims);
    swap(network1.output_dims, network2.output_dims);
  }

 private:
  inline static std::vector<Module> create_vector(Module&& module) {
    std::vector<Module> vec(1);
    vec[0] = std::move(module);
    return vec;
  }
  template <typename _LayerPtr>
  inline void populate_layer_vector(std::vector<_LayerPtr>& layer_ptrs) const {
    for (std::size_t i = 0; i < modules.size(); ++i) {
      std::vector<Layer<Scalar, Rank>*> internal_layer_ptrs = modules[i]->get_layers();
      for (std::size_t j = 0; j < internal_layer_ptrs.size(); ++j) layer_ptrs.push_back(internal_layer_ptrs[j]);
    }
  }
  std::vector<Module> modules;
  bool foremost;
  typename Base::Dims input_dims, output_dims;
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_DENSENEURALNETWORK_H_ */
