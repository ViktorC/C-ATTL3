/*
 * ResidualNeuralNetwork.hpp
 *
 *  Created on: 25 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_NEURAL_NETWORK_RESIDUALNEURALNETWORK_H_
#define C_ATTL3_NEURAL_NETWORK_RESIDUALNEURALNETWORK_H_

#include <cassert>
#include <utility>

#include "neural_network/CompositeNeuralNetwork.hpp"

namespace cattle {

/**
 * A class template for ResNets. These networks take vectors of neural networks as their
 * sub-modules.
 *
 * \see https://arxiv.org/abs/1512.03385
 */
template<typename Scalar, std::size_t Rank>
class ResidualNeuralNetwork :
		public CompositeNeuralNetwork<Scalar,Rank,false,NeuralNetwork<Scalar,Rank,false>> {
	typedef NeuralNetwork<Scalar,Rank,false> Base;
	typedef NeuralNetPtr<Scalar,Rank,false> Module;
	typedef ResidualNeuralNetwork<Scalar,Rank> Self;
public:
	/**
	 * @param modules A vector of residual modules.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline ResidualNeuralNetwork(std::vector<Module>&& modules, bool foremost = true) :
			modules(std::move(modules)),
			foremost(foremost) {
		assert(this->modules.size() > 0 && "modules must contain at least 1 element");
		Base& first_module = *this->modules[0];
		input_dims = first_module.get_input_dims();
		output_dims = this->modules[this->modules.size() - 1]->get_output_dims();
		first_module.set_foremost(foremost);
		typename Base::Dims prev_dims = input_dims;
		for (std::size_t i = 0; i < this->modules.size(); ++i) {
			Base& module = *this->modules[i];
			if (i != 0)
				module.set_foremost(false);
			assert(module.get_input_dims() == module.get_output_dims() &&
					"residual module input-output dimension discrepancy");
			assert(prev_dims == module.get_input_dims() && "incompatible module dimensions");
			prev_dims = module.get_output_dims();
		}
	}
	/**
	 * @param module A single residual module.
	 * @param foremost Whether the network is to function as a foremost network.
	 */
	inline ResidualNeuralNetwork(Module&& module, bool foremost = true) :
			ResidualNeuralNetwork(create_vector(std::move(module), foremost)) { }
	inline ResidualNeuralNetwork(const Self& network) :
			modules(network.modules.size()),
			foremost(network.foremost),
			input_dims(network.input_dims),
			output_dims(network.output_dims) {
		for (std::size_t i = 0; i < modules.size(); ++i)
			modules[i] = Module((Base*) network.modules[i]->clone());
	}
	inline ResidualNeuralNetwork(Self&& network) {
		swap(*this, network);
	}
	~ResidualNeuralNetwork() = default;
	inline Self& operator=(Self network) {
		swap(*this, network);
		return *this;
	}
	inline Base* clone() const {
		return new ResidualNeuralNetwork(*this);
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
		for (std::size_t i = 0; i < this->modules.size(); ++i)
			modules.push_back(this->modules[i].get());
		return modules;
	}
	inline bool is_foremost() const {
		return foremost;
	}
	inline void set_foremost(bool foremost) {
		modules[0]->set_foremost(foremost);
		this->foremost = foremost;
	}
	inline void empty_caches() {
		for (std::size_t i = 0; i < modules.size(); ++i)
			modules[i]->empty_caches();
	}
	inline typename Base::Data propagate(typename Base::Data input, bool training) {
		assert(input_dims == (Dimensions<std::size_t,Base::DATA_RANK>(input.dimensions()).template demote<>()));
		for (std::size_t i = 0; i < modules.size(); ++i)
			input += modules[i]->propagate(input, training);
		return input;
	}
	inline typename Base::Data backpropagate(typename Base::Data out_grad) {
		assert(output_dims == (Dimensions<std::size_t,Base::DATA_RANK>(out_grad.dimensions()).template demote<>()));
		for (int i = modules.size() - 1; i >= 0; --i) {
			if (foremost && i == 0)
				return modules[i]->backpropagate(std::move(out_grad));
			else
				out_grad += modules[i]->backpropagate(out_grad);
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
	template<typename _LayerPtr>
	inline void populate_layer_vector(std::vector<_LayerPtr>& layer_ptrs) const {
		for (std::size_t i = 0; i < modules.size(); ++i) {
			std::vector<Layer<Scalar,Rank>*> internal_layer_ptrs = modules[i]->get_layers();
			for (std::size_t j = 0; j < internal_layer_ptrs.size(); ++j)
				layer_ptrs.push_back(internal_layer_ptrs[j]);
		}
	}
	std::vector<Module> modules;
	bool foremost;
	typename Base::Dims input_dims, output_dims;
};

} /* namespace cattle */

#endif /* C_ATTL3_NEURAL_NETWORK_RESIDUALNEURALNETWORK_H_ */
