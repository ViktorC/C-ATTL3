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
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include "Dimensions.h"
#include "Utils.h"

namespace cattle {

// TODO OnDiskDataProvider.
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
	virtual unsigned instances() const = 0;
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
	virtual DataPair<Scalar,Rank,Sequential> get_data(unsigned batch_size) = 0;
	/**
	 * It resets the reader head to the beginning of the data storage.
	 */
	virtual void reset() = 0;
};

/**
 * A data provider that reads from the memory. It requires the entire observation and
 * objective data sets to be loaded into memory, but it fetches pairs faster.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class InMemoryDataProvider : public DataProvider<Scalar,Rank,Sequential> {
	static constexpr std::size_t DATA_RANKS = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANKS> Data;
	typedef TensorPtr<Scalar,Rank,Sequential> DataPtr;
public:
	/**
	 * It constructs a data provider backed by the specified tensors.
	 *
	 * @param obs A unique pointer to the tensor containing the observations.
	 * @param obj A unique pointer to the tensor containing the objectives.
	 * @param shuffle Whether the 'rows' (first ranks) of the tensors should be randomly
	 * shuffled.
	 */
	inline InMemoryDataProvider(DataPtr obs, DataPtr obj, bool shuffle = true) :
			obs(std::move(obs)),
			obj(std::move(obj)),
			offsets() {
		assert(this->obs != nullptr);
		assert(this->obj != nullptr);
		Utils<Scalar>::template check_dim_validity<DATA_RANKS>(*this->obs);
		Utils<Scalar>::template check_dim_validity<DATA_RANKS>(*this->obj);
		assert(this->obs->dimension(0) == this->obj->dimension(0) && "mismatched data and obj tensor row numbers");
		Dimensions<int,DATA_RANKS> obs_dims = Utils<Scalar>::template get_dims<DATA_RANKS>(*this->obs);
		Dimensions<int,DATA_RANKS> obj_dims = Utils<Scalar>::template get_dims<DATA_RANKS>(*this->obj);
		this->obs_dims = obs_dims.template demote<Sequential + 1>();
		this->obj_dims = obj_dims.template demote<Sequential + 1>();
		rows = (std::size_t) this->obs->dimension(0);
		offsets.fill(0);
		data_extents = obs_dims;
		obj_extents = obj_dims;
		if (shuffle) {
			Utils<Scalar>::template shuffle_tensor_rows<DATA_RANKS>(*this->obs);
			Utils<Scalar>::template shuffle_tensor_rows<DATA_RANKS>(*this->obj);
		}
	}
	inline const Dimensions<int,Rank>& get_obs_dims() const {
		return obs_dims;
	}
	inline const Dimensions<int,Rank>& get_obj_dims() const {
		return obj_dims;
	}
	inline unsigned instances() const {
		return rows;
	}
	inline bool has_more() const {
		return offsets[0] < (int) rows;
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(unsigned batch_size) {
		int max_batch_size = std::min((int) batch_size, (int) (rows - offsets[0]));
		data_extents[0] = max_batch_size;
		obj_extents[0] = max_batch_size;
		Data data_batch = obs->slice(offsets, data_extents);
		Data obj_batch = obj->slice(offsets, obj_extents);
		offsets[0] = std::min((int) rows, (int) offsets[0] + max_batch_size);
		return std::make_pair(data_batch, obj_batch);
	}
	void reset() {
		offsets[0] = 0;
	}
private:
	DataPtr obs;
	DataPtr obj;
	Dimensions<int,Rank> obs_dims;
	Dimensions<int,Rank> obj_dims;
	std::size_t rows;
	std::array<int,DATA_RANKS> offsets;
	std::array<int,DATA_RANKS> data_extents;
	std::array<int,DATA_RANKS> obj_extents;
};

} /* namespace cattle */

#endif /* DATAPROVIDER_H_ */
