/*
 * DataProvider.h
 *
 *  Created on: 18.01.2018
 *      Author: Viktor Csomor
 */

#ifndef DATAPROVIDER_H_
#define DATAPROVIDER_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include "Dimensions.h"
#include "Utils.h"

namespace cattle {

// TODO Specialized data providers for the MNIST, CIFAR, and ImageNet data sets.

/**
 * An alias for a unique tensor pointer.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using TensorPtr = std::unique_ptr<Tensor<Scalar,Rank + Sequential + 1>>;

/**
 * An alias for a pair of two tensors of the same rank. It represents observation-objective pairs.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using DataPair = std::pair<Tensor<Scalar,Rank + Sequential + 1>,Tensor<Scalar,Rank + Sequential + 1>>;

/**
 * A class template for fetching data from memory or disk.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class DataProvider {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal data provider rank");
public:
	virtual ~DataProvider() = default;
	/**
	 * A simple constant getter method for the dimensions of the observations.
	 *
	 * @return A constant reference to the dimensions of the observations.
	 */
	virtual const Dimensions<int,Rank>& get_obs_dims() const = 0;
	/**
	 * A simple constant getter method for the dimensions of the objectives.
	 *
	 * @return A constant reference to the dimensions of the objectives.
	 */
	virtual const Dimensions<int,Rank>& get_obj_dims() const = 0;
	/**
	 * A simple constant method that returns the number of observations and objectives
	 * handled by the data provider instance.
	 *
	 * @return The number of observation-objective pairs provided by this instance.
	 */
	virtual std::size_t instances() const = 0;
	/**
	 * A simple constant method that returns whether the data provider instance has more
	 * data to provide.
	 *
	 * @return Whether there are more observation-objective pairs to read from the
	 * instance.
	 */
	virtual bool has_more() const = 0;
	/**
	 * Reads and returns the specified number of observation-objective pairs. It also
	 * offsets the reader by the specified number.
	 *
	 * @param batch_size The maximum number of observation-objective pairs to read and
	 * return.
	 * @return At most batch_size number of observation-objective pairs. If the number
	 * of unread pairs is less than batch_size, the number of returned pairs is that
	 * of the unread ones.
	 */
	virtual DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) = 0;
	/**
	 * It resets the reader head to the beginning of the data storage.
	 */
	virtual void reset() = 0;
protected:
	static constexpr std::size_t DATA_RANKS = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANKS> Data;
	typedef TensorPtr<Scalar,Rank,Sequential> DataPtr;
};

/**
 * A wrapper class template for data providers associated with continuous partitions of other data
 * providers. It enables the partitioning of a data provider into training and test data providers
 * by mapping two contiguous blocks of its data to two PartitionedDataProvider instances.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class PartitionedDataProvider : DataProvider<Scalar,Rank,Sequential> {
	typedef DataProvider<Scalar,Rank,Sequential> Base;
public:
	PartitionedDataProvider(Base& orig_provider, std::size_t offset, std::size_t length) :
			orig_provider(orig_provider),
			offset(offset),
			length(length) {
		assert(length > 0);
		reset();
	}
	inline const Dimensions<int,Rank>& get_obs_dims() const {
		return orig_provider.get_obs_dims();
	}
	inline const Dimensions<int,Rank>& get_obj_dims() const {
		return orig_provider.get_obj_dims();
	}
	inline std::size_t instances() const {
		return std::min(length, orig_provider.instances());
	}
	inline bool has_more() const {
		return instances_read < length && orig_provider.has_more();
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		std::size_t instances_to_read = std::min(batch_size, length - instances_read);
		instances_read += instances_to_read;
		return orig_provider.get_data(instances_to_read);
	}
	void reset() {
		orig_provider.reset();
		orig_provider.get_data(offset);
		instances_read = 0;
	}
private:
	Base& orig_provider;
	const std::size_t offset;
	const std::size_t length;
	std::size_t instances_read;
};


/**
 * A data provider that reads from the memory. It requires the entire observation and
 * objective data sets to be loaded into memory, but it fetches pairs faster.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class MemoryDataProvider : public DataProvider<Scalar,Rank,Sequential> {
	typedef DataProvider<Scalar,Rank,Sequential> Base;
public:
	/**
	 * It constructs a data provider backed by the specified tensors.
	 *
	 * @param obs A unique pointer to the tensor containing the observations.
	 * @param obj A unique pointer to the tensor containing the objectives.
	 * @param shuffle Whether the 'rows' (first ranks) of the tensors should be randomly
	 * shuffled.
	 */
	inline MemoryDataProvider(typename Base::DataPtr obs, typename Base::DataPtr obj,
			bool shuffle = true) :
				obs(std::move(obs)),
				obj(std::move(obj)),
				offsets() {
		assert(this->obs != nullptr);
		assert(this->obj != nullptr);
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(*this->obs);
		Utils<Scalar>::template check_dim_validity<Base::DATA_RANKS>(*this->obj);
		assert(this->obs->dimension(0) == this->obj->dimension(0) &&
				"mismatched data and obj tensor row numbers");
		Dimensions<int,Base::DATA_RANKS> obs_dims =
				Utils<Scalar>::template get_dims<Base::DATA_RANKS>(*this->obs);
		Dimensions<int,Base::DATA_RANKS> obj_dims =
				Utils<Scalar>::template get_dims<Base::DATA_RANKS>(*this->obj);
		this->obs_dims = obs_dims.template demote<Sequential + 1>();
		this->obj_dims = obj_dims.template demote<Sequential + 1>();
		rows = (std::size_t) this->obs->dimension(0);
		offsets.fill(0);
		data_extents = obs_dims;
		obj_extents = obj_dims;
		if (shuffle) {
			Utils<Scalar>::template shuffle_tensor_rows<Base::DATA_RANKS>(*this->obs);
			Utils<Scalar>::template shuffle_tensor_rows<Base::DATA_RANKS>(*this->obj);
		}
	}
	inline const Dimensions<int,Rank>& get_obs_dims() const {
		return obs_dims;
	}
	inline const Dimensions<int,Rank>& get_obj_dims() const {
		return obj_dims;
	}
	inline std::size_t instances() const {
		return rows;
	}
	inline bool has_more() const {
		return offsets[0] < (int) rows;
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		int max_batch_size = std::min((int) batch_size, (int) (rows - offsets[0]));
		data_extents[0] = max_batch_size;
		obj_extents[0] = max_batch_size;
		typename Base::Data data_batch = obs->slice(offsets, data_extents);
		typename Base::Data obj_batch = obj->slice(offsets, obj_extents);
		offsets[0] = std::min((int) rows, (int) offsets[0] + max_batch_size);
		return std::make_pair(data_batch, obj_batch);
	}
	void reset() {
		offsets[0] = 0;
	}
private:
	typename Base::DataPtr obs;
	typename Base::DataPtr obj;
	Dimensions<int,Rank> obs_dims;
	Dimensions<int,Rank> obj_dims;
	std::size_t rows;
	std::array<int,Base::DATA_RANKS> offsets;
	std::array<int,Base::DATA_RANKS> data_extents;
	std::array<int,Base::DATA_RANKS> obj_extents;
};

/**
 * An abstract class template for a data provider backed by data on disk in the form of a single
 * file containing both the observations and the objectives. Implementations are responsible for
 * specifying the he number of data points, the dimensions of both the observations and the
 * objectives, and for reading batches of observation-objective pairs from the file.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class JointFileDataProvider : public DataProvider<Scalar,Rank,Sequential> {
public:
	inline bool has_more() const {
		return input_stream.eof();
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		return _get_data(input_stream, batch_size);
	}
	void reset() {
		input_stream.seekg(0, std::ios::beg);
	}
protected:
	JointFileDataProvider(std::string dataset_path) :
			input_stream(dataset_path) {
		assert(input_stream.is_open());
	}
	/**
	 * It reads at most the specified number of observation-objective pairs from the provided
	 * file stream.
	 *
	 * @param input_stream A reference to the file stream of the data set.
	 * @param batch_size The number of data points to return.
	 * @return A pair of tensors containing the data batch.
	 */
	virtual DataPair<Scalar,Rank,Sequential> _get_data(std::ifstream& input_stream,
			std::size_t batch_size) = 0;
private:
	std::ifstream input_stream;
};

/**
 * An abstract class template for a data provider backed by two files containing the
 * observations and the objectives respectively. Implementations are responsible for
 * specifying the he number of data points, the dimensions of both the observations and the
 * objectives, and for reading batches of observation-objective pairs from the file.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class SeparatedFileDataProvider : public DataProvider<Scalar,Rank,Sequential> {
public:
	inline bool has_more() const {
		return obs_input_stream.eof() && obj_input_stream.eof();
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		return _get_data(obs_input_stream, obj_input_stream, batch_size);
	}
	void reset() {
		obs_input_stream.seekg(0, std::ios::beg);
		obj_input_stream.seekg(0, std::ios::beg);
	}
protected:
	SeparatedFileDataProvider(std::string obs_path, std::string obj_path) :
			obs_input_stream(obs_path),
			obj_input_stream(obj_path) {
		assert(obs_input_stream.is_open() && obj_input_stream.is_open());
	}
	/**
	 * It reads at most the specified number of observations from the observation-file and
	 * at most the specified number of objectives from the objective-file.
	 *
	 * @param obs_input_stream A reference to the file stream to the file containing the
	 * observations.
	 * @param obj_input_stream A reference to the file stream to the file containing the
	 * objectives.
	 * @param batch_size The number of data points to read.
	 * @return The paired observations and objectives.
	 */
	virtual DataPair<Scalar,Rank,Sequential> _get_data(std::ifstream& obs_input_stream,
			std::ifstream& obj_input_stream, std::size_t batch_size) = 0;
private:
	std::ifstream obs_input_stream;
	std::ifstream obj_input_stream;
};

} /* namespace cattle */

#endif /* DATAPROVIDER_H_ */
