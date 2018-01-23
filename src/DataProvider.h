/*
 * DataProvider.h
 *
 *  Created on: 18.01.2018
 *      Author: Viktor Csomor
 */

#ifndef DATAPROVIDER_H_
#define DATAPROVIDER_H_

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <Utils.h>

namespace cppnn {

template<typename Scalar>
using Tensor4Ptr = std::unique_ptr<Tensor4<Scalar>>;

/**
 * A class template for fetching data from memory or disk.
 */
template<typename Scalar>
class DataProvider {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~DataProvider() = default;
	virtual unsigned observations() const = 0;
	virtual std::pair<Tensor4<Scalar>,Tensor4<Scalar>> get_data(unsigned batch_size) = 0;
	virtual bool reached_eol() const = 0;
	virtual void reset() = 0;
};

template<typename Scalar>
class InMemoryDataProvider : public DataProvider<Scalar> {
public:
	InMemoryDataProvider(Tensor4Ptr<Scalar> data, Tensor4Ptr<Scalar> obj, bool shuffle = true) :
			data(data),
			obj(obj),
			offsets({ 0, 0, 0, 0 }) {
		assert(data.get() != nullptr);
		assert(obj.get() != nullptr);
		assert(rows == obj.get()->dimension(0) && "mismatched data and obj tensor row numbers");
		rows = (unsigned) data.get()->dimension(0);
		data_extents = Array4<Scalar>({ 0, data.get()->dimension(1), data.get()->dimension(2), data.get()->dimension(3) });
		obj_extents = Array4<Scalar>({ 0, obj.get()->dimension(1), obj.get()->dimension(2), obj.get()->dimension(3) });
		if (shuffle) {
			Utils<Scalar>::shuffle_tensor_rows(*data.get());
			Utils<Scalar>::shuffle_tensor_rows(*obj.get());
		}
	};
	unsigned observations() {
		return rows;
	};
	std::pair<Tensor4<Scalar>,Tensor4<Scalar>> get_data(unsigned batch_size) {
		int max_batch_size = std::min((int) batch_size, row - offsets[0]);
		data_extents[0] = max_batch_size;
		obj_extents[0] = max_batch_size;
		Tensor4<Scalar> data_batch = data.get()->slice(offsets, data_extents);
		Tensor4<Scalar> obj_batch = obj.get()->slice(offsets, obj_extents);
		offsets[0] = std::min((int) rows, offsets[0] + max_batch_size);
		return std::make_pair(data_batch, obj_batch);
	};
	bool reached_eol() {
		return offsets[0] == rows;
	};
	void reset() {
		offsets[0] = 0;
	};
private:
	Tensor4Ptr<Scalar> data;
	Tensor4Ptr<Scalar> obj;
	unsigned rows;
	Array4<int> offsets;
	Array4<int> data_extents;
	Array4<int> obj_extents;
};

} /* namespace cppnn */

#endif /* DATAPROVIDER_H_ */
