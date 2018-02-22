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
#include <Dimensions.h>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <Utils.h>

namespace cattle {

// TODO OnDiskDataProvider.
// TODO Specialized data providers for the MNIST, CIFAR, and ImageNet data sets.

template<typename Scalar, size_t Rank, bool Sequential>
using TensorPtr = std::unique_ptr<Tensor<Scalar,Rank + Sequential + 1>>;

template<typename Scalar, size_t Rank, bool Sequential>
using DataPair = std::pair<Tensor<Scalar,Rank + Sequential + 1>,Tensor<Scalar,Rank + Sequential + 1>>;

/**
 * A class template for fetching data from memory or disk.
 */
template<typename Scalar, size_t Rank, bool Sequential>
class DataProvider {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal data provider rank");
public:
	virtual ~DataProvider() = default;
	virtual const Dimensions<int,Rank>& get_obs_dims() const = 0;
	virtual const Dimensions<int,Rank>& get_obj_dims() const = 0;
	virtual unsigned instances() const = 0;
	virtual bool has_more() const = 0;
	virtual DataPair<Scalar,Rank,Sequential> get_data(unsigned batch_size) = 0;
	virtual void reset() = 0;
};

template<typename Scalar, size_t Rank, bool Sequential>
class InMemoryDataProvider : public DataProvider<Scalar,Rank,Sequential> {
	static constexpr size_t DATA_DIMS = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_DIMS> Data;
	typedef TensorPtr<Scalar,Rank,Sequential> DataPtr;
public:
	inline InMemoryDataProvider(DataPtr obs, DataPtr obj, bool shuffle = true) :
			obs(std::move(obs)),
			obj(std::move(obj)),
			offsets() {
		assert(this->obs != nullptr);
		assert(this->obj != nullptr);
		Utils<Scalar>::template check_tensor_validity<DATA_DIMS>(*this->obs);
		Utils<Scalar>::template check_tensor_validity<DATA_DIMS>(*this->obj);
		assert(this->obs->dimension(0) == this->obj->dimension(0) && "mismatched data and obj tensor row numbers");
		Dimensions<int,DATA_DIMS> obs_dims = Utils<Scalar>::template get_dims<DATA_DIMS>(*this->obs);
		Dimensions<int,DATA_DIMS> obj_dims = Utils<Scalar>::template get_dims<DATA_DIMS>(*this->obj);
		this->obs_dims = obs_dims.template demote<Sequential + 1>();
		this->obj_dims = obj_dims.template demote<Sequential + 1>();
		rows = (size_t) this->obs->dimension(0);
		offsets.fill(0);
		data_extents = obs_dims;
		obj_extents = obj_dims;
		if (shuffle) {
			Utils<Scalar>::template shuffle_tensor_rows<DATA_DIMS>(*this->obs);
			Utils<Scalar>::template shuffle_tensor_rows<DATA_DIMS>(*this->obj);
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
	size_t rows;
	std::array<int,DATA_DIMS> offsets;
	std::array<int,DATA_DIMS> data_extents;
	std::array<int,DATA_DIMS> obj_extents;
};

} /* namespace cattle */

#endif /* DATAPROVIDER_H_ */
