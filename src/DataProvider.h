/*
 * DataProvider.h
 *
 *  Created on: 18.01.2018
 *      Author: Viktor Csomor
 */

#ifndef DATAPROVIDER_H_
#define DATAPROVIDER_H_

#include <algorithm>
#include <cstddef>
#include <Dimensions.h>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <Utils.h>

namespace cattle {

template<typename Scalar, size_t Rank>
using TensorPtr = std::unique_ptr<Tensor<Scalar,Rank>>;

template<typename Scalar, size_t Rank>
using DataPair = std::pair<Tensor<Scalar,Rank>,Tensor<Scalar,Rank>>;

/**
 * A class template for fetching data from memory or disk.
 */
template<typename Scalar, size_t Rank>
class DataProvider {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal data provider rank");
public:
	virtual ~DataProvider() = default;
	virtual Dimensions<int,Rank> get_obs_dims() const = 0;
	virtual Dimensions<int,Rank> get_obj_dims() const = 0;
	virtual unsigned instances() const = 0;
	virtual bool has_more() const = 0;
	virtual DataPair<Scalar,Rank + 1> get_data(unsigned batch_size) = 0;
	virtual void reset() = 0;
};

template<typename Scalar, size_t Rank>
class InMemoryDataProvider : public DataProvider<Scalar,Rank> {
public:
	InMemoryDataProvider(TensorPtr<Scalar,Rank> obs, TensorPtr<Scalar,Rank> obj, bool shuffle = true) :
			obs(std::move(obs)),
			obj(std::move(obj)),
			offsets() {
		assert(this->obs != nullptr);
		assert(this->obj != nullptr);
		assert(this->obs->dimension(0) > 0);
		assert(this->obs->dimension(0) == this->obj->dimension(0) && "mismatched data and obj tensor row numbers");
		rows = (unsigned) this->obs->dimension(0);
		offsets.fill(0);
		data_extents = Utils<Scalar>::get_dims(this->obs).promote();
		obj_extents = Utils<Scalar>::get_dims(this->obj).promote();
		if (shuffle) {
			Utils<Scalar>::shuffle_tensor_rows(*this->obs);
			Utils<Scalar>::shuffle_tensor_rows(*this->obj);
		}
	};
	Dimensions<int,Rank> get_obs_dims() const {
		return Utils<Scalar>::get_dims(*obs);
	};
	Dimensions<int,Rank> get_obj_dims() const {
		return Utils<Scalar>::get_dims(*obj);
	};
	unsigned instances() const {
		return rows;
	};
	bool has_more() const {
		return offsets[0] < (int) rows;
	};
	DataPair<Scalar,Rank + 1> get_data(unsigned batch_size) {
		int max_batch_size = std::min((int) batch_size, (int) (rows - offsets[0]));
		data_extents[0] = max_batch_size;
		obj_extents[0] = max_batch_size;
		Tensor<Scalar,Rank + 1> data_batch = obs->slice(offsets, data_extents);
		Tensor<Scalar,Rank + 1> obj_batch = obj->slice(offsets, obj_extents);
		offsets[0] = std::min((int) rows, (int) offsets[0] + max_batch_size);
		return std::make_pair(data_batch, obj_batch);
	};
	void reset() {
		offsets[0] = 0;
	};
private:
	TensorPtr<Scalar,Rank + 1> obs;
	TensorPtr<Scalar,Rank + 1> obj;
	unsigned rows;
	Array<int,Rank + 1> offsets;
	Array<int,Rank + 1> data_extents;
	Array<int,Rank + 1> obj_extents;
};

//template<typename Scalar, size_t Rank>
//class OnDiskDataProvider : public DataProvider<Scalar,Rank> {
//public:
//	OnDiskDataProvider(std::string obs_path, std::string obj_path) {
//
//	};
//	Dimensions<int,Rank> get_obs_dims() const {
//
//	};
//	Dimensions<int,Rank> get_obj_dims() const {
//
//	};
//	unsigned instances() const {
//
//	};
//	bool has_more() const {
//
//	};
//	DataPair<Scalar,Rank + 1> get_data(unsigned batch_size) {
//
//	};
//	void reset() {
//
//	};
//private:
//	std::string obs_path;
//	std::string obj_path;
//	std::ifstream obs_stream;
//	std::ifstream obj_stream;
//};

} /* namespace cattle */

#endif /* DATAPROVIDER_H_ */
