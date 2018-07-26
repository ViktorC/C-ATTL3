/*
 * SGDOptimizer.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_OPTIMIZER_SGDOPTIMIZER_H_
#define C_ATTL3_OPTIMIZER_SGDOPTIMIZER_H_

#include "core/Optimizer.hpp"

namespace cattle {

/**
 * An abstract class template for stochastic gradient descent (SGD) optimizers.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class SGDOptimizer : public Optimizer<Scalar,Rank,Sequential> {
	typedef Optimizer<Scalar,Rank,Sequential> Base;
public:
	inline SGDOptimizer(LossSharedPtr<Scalar,Rank,Sequential> loss, std::size_t batch_size) :
			Base::Optimizer(loss),
				batch_size(batch_size) {
		assert(batch_size > 0);
	}
	virtual ~SGDOptimizer() = default;
protected:
	inline Scalar _train(typename Base::Net& net, typename Base::Provider& training_prov, std::size_t epoch, bool verbose) {
		Scalar training_loss = 0;
		Scalar instances = 0;
		std::vector<Layer<Scalar,Rank>*> layers = Base::get_layers(net);
		// Perform an entire training epoch.
		while (training_prov.has_more()) {
			DataPair<Scalar,Rank,Sequential> data_pair = training_prov.get_data(batch_size);
			instances += data_pair.first.dimension(0);
			typename Base::Data out = net.propagate(std::move(data_pair.first), false);
			training_loss += Base::loss->function(out, data_pair.second).sum();
			/* Divide the gradient by the batch size to decouple the learning rate and the batch
			 * size hyper-parameters. Use the nominal batch size as the denominator even if the
			 * actual batch size is different (in case the number of samples in the data set is
			 * not divisible by the batch size and the last batch of the epoch contains fewer
			 * instances than the others) to make sure that the magnitude of the gradient is
			 * proportional to the batch size (just like its 'accuracy' is). */
			net.backpropagate(Base::loss->d_function(std::move(out), std::move(data_pair.second)) / (Scalar) batch_size);
			std::size_t i = 0;
			for (auto layer_ptr : layers) {
				if (!layer_ptr)
					continue;
				for (auto params_ptr : layer_ptr->get_params()) {
					if (!params_ptr || !params_ptr->are_optimizable() || params_ptr->are_frozen())
						continue;
					params_ptr->regularize();
					_update_params(*params_ptr, i, epoch - 1);
				}
			}
		}
		return training_loss / instances;
	}
	inline Scalar _test(typename Base::Net& net, typename Base::Provider& test_prov, std::size_t epoch, bool verbose) {
		Scalar obj_loss = 0;
		Scalar instances = 0;
		std::vector<Layer<Scalar,Rank>*> layers = Base::get_layers(net);
		// Perform an entire test epoch.
		while (test_prov.has_more()) {
			DataPair<Scalar,Rank,Sequential> data_pair = test_prov.get_data(batch_size);
			instances += data_pair.first.dimension(0);
			obj_loss += Base::loss->function(net.infer(std::move(data_pair.first)), std::move(data_pair.second)).sum();
		}
		Scalar mean_obj_loss = obj_loss / instances;
		Scalar reg_loss = 0;
		for (std::size_t j = 0; j < layers.size(); ++j) {
			Layer<Scalar,Rank>& layer = *(layers[j]);
			if (Base::is_parametric(layer) && !layer.is_frozen())
				reg_loss += Base::get_regularization_penalty(layer);
		}
		if (verbose) {
			std::cout << "\tobj loss: " << std::to_string(mean_obj_loss) << std::endl;
			std::cout << "\treg loss: " << std::to_string(reg_loss) << std::endl;
		}
		return mean_obj_loss + reg_loss;
	}
	/**
	 * It updates the parameters of based on their gradients after back-propagation.
	 *
	 * @param params A reference to the parameters to be updated.
	 * @param i The index of the parameters.
	 * @param epoch The index of the epoch.
	 */
	virtual void _update_params(Parameters<Scalar>& params, std::size_t i, std::size_t epoch) = 0;
	const std::size_t batch_size;
private:
	inline void update_all_params(std::vector<Layer<Scalar,Rank>*>& layers, std::size_t epoch) {
		std::size_t i = 0;
		for (auto layer_ptr : layers) {
			if (!layer_ptr)
				continue;
			for (auto params_ptr : layer_ptr->get_params()) {
				if (!params_ptr || !params_ptr->are_optimizable() || params_ptr->are_frozen())
					continue;
				params_ptr->regularize();
				_update_params(*params_ptr, i++, epoch - 1);
			}
		}
	}
	inline void reset_all_grads(std::vector<Layer<Scalar,Rank>*>& layers) {
		for (auto layer_ptr : layers) {
			if (!layer_ptr)
				continue;
			for (auto params_ptr : layer_ptr->get_params()) {
				if (params_ptr && params_ptr->are_optimizable())
					params_ptr->reset_grad();
			}
		}
	}
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_SGDOPTIMIZER_H_ */
