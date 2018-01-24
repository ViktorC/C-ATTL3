/*
 * DataProvider.h
 *
 *  Created on: 18.01.2018
 *      Author: Viktor Csomor
 */

#ifndef DATAPROVIDER_H_
#define DATAPROVIDER_H_

#include <algorithm>
#include <Dimensions.h>
#include <memory>
#include <type_traits>
#include <utility>
#include <Utils.h>

namespace cppnn {

template<typename Scalar>
using Tensor4Ptr = std::unique_ptr<Tensor4<Scalar>>;

template<typename Scalar>
using DataPair = std::pair<Tensor4<Scalar>,Tensor4<Scalar>>;

/**
 * A class template for fetching data from memory or disk.
 */
template<typename Scalar>
class DataProvider {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~DataProvider() = default;
	virtual Dimensions<int> get_obs_dims() const = 0;
	virtual Dimensions<int> get_obj_dims() const = 0;
	virtual unsigned instances() const = 0;
	virtual bool has_more() const = 0;
	virtual DataPair<Scalar> get_data(unsigned batch_size) = 0;
	virtual void reset() = 0;
};

template<typename Scalar>
class InMemoryDataProvider : public DataProvider<Scalar> {
public:
	InMemoryDataProvider(Tensor4Ptr<Scalar> obs, Tensor4Ptr<Scalar> obj, bool shuffle = true) :
			obs(std::move(obs)),
			obj(std::move(obj)),
			offsets({ 0, 0, 0, 0 }) {
		assert(this->obs != nullptr);
		assert(this->obj != nullptr);
		assert(this->obs->dimension(0) > 0);
		assert(this->obs->dimension(0) == this->obj->dimension(0) && "mismatched data and obj tensor row numbers");
		rows = (unsigned) this->obs->dimension(0);
		data_extents = Array4<int>({ 0, this->obs->dimension(1), this->obs->dimension(2),
				this->obs->dimension(3) });
		obj_extents = Array4<int>({ 0, this->obj->dimension(1), this->obj->dimension(2),
				this->obj->dimension(3) });
		if (shuffle) {
			Utils<Scalar>::shuffle_tensor_rows(*this->obs);
			Utils<Scalar>::shuffle_tensor_rows(*this->obj);
		}
	};
	Dimensions<int> get_obs_dims() const {
		return Utils<Scalar>::get_dims(*obs);
	};
	Dimensions<int> get_obj_dims() const {
		return Utils<Scalar>::get_dims(*obj);
	};
	unsigned instances() const {
		return rows;
	};
	bool has_more() const {
		return offsets[0] < (int) rows;
	};
	DataPair<Scalar> get_data(unsigned batch_size) {
		int max_batch_size = std::min((int) batch_size, (int) (rows - offsets[0]));
		data_extents[0] = max_batch_size;
		obj_extents[0] = max_batch_size;
		Tensor4<Scalar> data_batch = obs->slice(offsets, data_extents);
		Tensor4<Scalar> obj_batch = obj->slice(offsets, obj_extents);
		offsets[0] = std::min((int) rows, (int) offsets[0] + max_batch_size);
		return std::make_pair(data_batch, obj_batch);
	};
	void reset() {
		offsets[0] = 0;
	};
private:
	Tensor4Ptr<Scalar> obs;
	Tensor4Ptr<Scalar> obj;
	unsigned rows;
	Array4<int> offsets;
	Array4<int> data_extents;
	Array4<int> obj_extents;
};



} /* namespace cppnn */

#endif /* DATAPROVIDER_H_ */
