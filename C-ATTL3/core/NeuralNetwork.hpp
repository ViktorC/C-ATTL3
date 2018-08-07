/*
 * NeuralNetwork.hpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_NEURALNETWORK_H_
#define C_ATTL3_CORE_NEURALNETWORK_H_

#include <set>

#include "Layer.hpp"

namespace cattle {

/**
 * An abstract neural network class template. It allows for inference and training via
 * back-propagation.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class NeuralNetwork {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal neural network rank");
protected:
	static constexpr std::size_t DATA_RANK = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
	typedef Dimensions<std::size_t,Rank> Dims;
	static std::string PARAM_SERIAL_PREFIX;
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
	/**
	 * @return A vector of pointers to constant layers constituting the network. The ownership
	 * of the layers remains with the network.
	 */
	virtual std::vector<const Layer<Scalar,Rank>*> get_layers() const = 0;
	/**
	 * @return A vector of pointers to the layers of the network. The ownership of the
	 * layers remains with the network.
	 */
	virtual std::vector<Layer<Scalar,Rank>*> get_layers() = 0;
	/**
	 * @return Whether the instance is a foremost network. If the instance is not a stand-alone
	 * network and it is not the first module of a complex network, it is not a foremost
	 * network. Foremost networks do not need to back-propagate the gradients all the way
	 * given that no other network is expected to depend on them.
	 */
	virtual bool is_foremost() const = 0;
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
	 * @param out_grad The derivative of the loss function w.r.t. the output of the
	 * network.
	 * @return The derivative of the loss function w.r.t. the input of the network or
	 * a null tensor if the network is a foremost network.
	 */
	virtual Data backpropagate(Data out_grad) = 0;
	/**
	 * @return A vector of pointers to the constant parameters of the network. The pointers
	 * are not necessarily unique.
	 */
	inline std::vector<const Parameters<Scalar>*> get_all_params() const {
		std::vector<const Parameters<Scalar>*> params_vec;
		for (auto layer_ptr : get_layers()) {
			if (!layer_ptr)
				continue;
			for (auto params_ptr : layer_ptr->get_params()) {
				if (params_ptr)
					params_vec.push_back(params_ptr);
			}
		}
		return params_vec;
	}
	/**
	 * @return A vector of pointers to the parameters of the network. The pointers are
	 * not necessarily unique.
	 */
	inline std::vector<Parameters<Scalar>*> get_all_params() {
		std::vector<Parameters<Scalar>*> params_vec;
		for (auto layer_ptr : get_layers()) {
			if (!layer_ptr)
				continue;
			for (auto params_ptr : layer_ptr->get_params()) {
				if (params_ptr)
					params_vec.push_back(params_ptr);
			}
		}
		return params_vec;
	}
	/**
	 * @return A vector of pointers to the constant, unique parameters of the network.
	 */
	inline std::vector<const Parameters<Scalar>*> get_all_unique_params() const {
		std::vector<const Parameters<Scalar>*> params_vec;
		std::set<const Parameters<Scalar>*> params_set;
		for (auto params_ptr : get_all_params()) {
			if (params_set.find(params_ptr) == params_set.end()) {
				params_set.insert(params_ptr);
				params_vec.push_back(params_ptr);
			}
		}
		return params_vec;
	}
	/**
	 * @return A vector of pointers to the unique parameters of the network.
	 */
	inline std::vector<Parameters<Scalar>*> get_all_unique_params() {
		std::vector<Parameters<Scalar>*> params_vec;
		std::set<Parameters<Scalar>*> params_set;
		for (auto params_ptr : get_all_params()) {
			if (params_set.find(params_ptr) == params_set.end()) {
				params_set.insert(params_ptr);
				params_vec.push_back(params_ptr);
			}
		}
		return params_vec;
	}
	/**
	 * Sets all parameters of the network to the specified frozens state.
	 *
	 * @param frozen Whether the parameters of the network should be frozen i.e. temporarily
	 * not optimizable.
	 */
	inline virtual void set_frozen(bool frozen) {
		for (auto params_ptr : get_all_unique_params())
			params_ptr->set_frozen(frozen);
	}
	/**
	 * Initializes all parameters of the network.
	 */
	inline virtual void init() {
		for (auto params_ptr : get_all_unique_params())
			params_ptr->init();
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
	 * It serializes the values of the unique parameters of the network into files in a
	 * specified folder.
	 *
	 * @param dir_path The path to the directory.
	 * @param binary Whether the parameters are to be serialized into a binary format.
	 * @param file_name_prefix A prefix to the names of the serialized parameter files.
	 */
	inline void save_all_unique_params_values(const std::string& dir_path, bool binary = true,
			const std::string& file_name_prefix = PARAM_SERIAL_PREFIX) const {
		std::vector<const Parameters<Scalar>*> params_vec = get_all_unique_params();
		for (std::size_t i = 0; i < params_vec.size(); ++i) {
			const Matrix<Scalar>& values = params_vec[i]->get_values();
			std::string file_path = dir_path + "/" + file_name_prefix + std::to_string(i) + ".prms";
			if (binary)
				serialize_binary<Scalar>(values, file_path);
			else
				serialize<Scalar>(values, file_path);
		}
	}
	/**
	 * It sets the values of the unique parameters of the network from files containing
	 * serialized parameter values.
	 *
	 * @param dir_path The path to the directory containing the parameter files.
	 * @param binary Whether the parameter files binary.
	 * @param file_name_prefix The prefix of the names of the parameter files.
	 */
	inline void load_all_unique_params_values(const std::string& dir_path, bool binary = true,
			const std::string& file_name_prefix = PARAM_SERIAL_PREFIX) {
		std::vector<Parameters<Scalar>*> params_vec = get_all_unique_params();
		for (std::size_t i = 0; i < params_vec.size(); ++i) {
			std::string file_path = dir_path + "/" + file_name_prefix + std::to_string(i) + ".prms";
			params_vec[i]->set_values(binary ? deserialize_binary<Scalar>(file_path) :
					deserialize<Scalar>(file_path));
		}
	}
};

template<typename Scalar, std::size_t Rank, bool Sequential>
std::string NeuralNetwork<Scalar,Rank,Sequential>::PARAM_SERIAL_PREFIX =
		"c-attl3_neural_net_params_";

} /* namespace cattle */

#endif /* C_ATTL3_CORE_NEURALNETWORK_H_ */
