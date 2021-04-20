/*
 * SGDOptimizer.hpp
 *
 *  Created on: 26 Jul 2018
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
	inline void fit(typename Base::Net& net) {
		timestep = 0;
		target_net_ptr = &net;
		_fit(net.get_all_unique_params());
	}
protected:
	inline Scalar _train(typename Base::Net& net, typename Base::Provider& training_prov, std::size_t epoch,
			bool verbose) {
		assert(target_net_ptr == &net);
		Scalar obj_loss = 0;
		Scalar reg_loss = 0;
		std::size_t instances = 0;
		std::size_t updates = 0;
		// Get all the optimizable parameters.
		std::vector<Parameters<Scalar>*> params_vec = net.get_all_unique_params();
		std::vector<Parameters<Scalar>*> optimizable_params_vec;
		for (auto params_ptr : params_vec) {
			if (params_ptr->are_optimizable() || !params_ptr->are_frozen())
				optimizable_params_vec.push_back(params_ptr);
		}
		// Perform an entire training epoch.
		while (training_prov.has_more()) {
			DataPair<Scalar,Rank,Sequential> data_pair = training_prov.get_data(batch_size);
			instances += data_pair.first.dimension(0);
			typename Base::Data out = net.propagate(std::move(data_pair.first), true);
			obj_loss += Base::loss->function(out, data_pair.second).sum();
			/* Divide the gradient by the batch size to decouple the learning rate and the batch
			 * size hyper-parameters. Use the nominal batch size as the denominator even if the
			 * actual batch size is different (in case the number of samples in the data set is
			 * not divisible by the batch size and the last batch of the epoch contains fewer
			 * instances than the others) to make sure that the magnitude of the gradient is
			 * proportional to the batch size (just like its 'accuracy' is). */
			net.backpropagate(Base::loss->d_function(std::move(out),
					std::move(data_pair.second)) / (Scalar) batch_size);
			// Update the values of the parameters.
			std::size_t i = 0;
			for (auto params_ptr : optimizable_params_vec) {
				reg_loss += params_ptr->get_regularization_penalty();
				params_ptr->regularize();
			}
			_update_params(optimizable_params_vec, epoch - 1, timestep);
			++updates;
			++timestep;
			// Reset the gradients of the optimizable parameters.
			for (auto params_ptr : params_vec)
				params_ptr->reset_grad();
		}
		Scalar mean_obj_loss = obj_loss / instances;
		Scalar mean_reg_loss = reg_loss / updates;
		if (verbose) {
			std::cout << std::left << std::setw(20) << "\ttraining obj loss: " << std::right <<
					std::to_string(mean_obj_loss) << std::endl;
			std::cout << std::left << std::setw(20) << "\ttraining reg loss: " << std::right <<
					std::to_string(mean_reg_loss) << std::endl;
		}
		return mean_obj_loss + mean_reg_loss;
	}
	inline Scalar _test(typename Base::Net& net, typename Base::Provider& test_prov, std::size_t epoch,
			bool verbose) {
		assert(target_net_ptr == &net);
		Scalar obj_loss = 0;
		Scalar instances = 0;
		std::vector<Parameters<Scalar>*> params_vec = net.get_all_unique_params();
		// Perform an entire test epoch.
		while (test_prov.has_more()) {
			DataPair<Scalar,Rank,Sequential> data_pair = test_prov.get_data(batch_size);
			instances += data_pair.first.dimension(0);
			obj_loss += Base::loss->function(net.infer(std::move(data_pair.first)),
					std::move(data_pair.second)).sum();
		}
		Scalar mean_obj_loss = obj_loss / instances;
		Scalar reg_loss = 0;
		for (auto params_ptr : params_vec) {
			if (params_ptr->are_optimizable() && !params_ptr->are_frozen())
				reg_loss += params_ptr->get_regularization_penalty();
		}
		if (verbose) {
			std::cout << std::left << std::setw(20) << "\ttest obj loss: " << std::right <<
					std::to_string(mean_obj_loss) << std::endl;
			std::cout << std::left << std::setw(20) << "\ttest reg loss: " << std::right <<
					std::to_string(reg_loss) << std::endl;
		}
		return mean_obj_loss + reg_loss;
	}
	/**
	 * It fits the optimizer to the provided parameters.
	 *
	 * @param params_vec All the unique parameters of the network.
	 */
	virtual void _fit(const std::vector<Parameters<Scalar>*>& params_vec) = 0;
	/**
	 * It updates the parameters based on their gradients after back-propagation.
	 *
	 * @param params_vec All the unique optimizable parameters of the network.
	 * @param epoch The index of the epoch.
	 * @param timestep The index of the update (time step).
	 */
	virtual void _update_params(
			const std::vector<Parameters<Scalar>*>& params_vec,
			std::size_t epoch,
			std::size_t timestep) = 0;
	const std::size_t batch_size;
private:
	std::size_t timestep;
	const typename Base::Net* target_net_ptr;
};

} /* namespace cattle */

#endif /* C_ATTL3_OPTIMIZER_SGDOPTIMIZER_H_ */
