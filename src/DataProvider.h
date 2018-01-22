/*
 * DataProvider.h
 *
 *  Created on: 18.01.2018
 *      Author: Viktor Csomor
 */

#ifndef DATAPROVIDER_H_
#define DATAPROVIDER_H_

#include <type_traits>
#include <utility>
#include <Utils.h>

namespace cppnn {

/**
 * A class template for fetching data from memory or disk.
 */
template<typename Scalar>
class DataProvider {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	virtual ~DataProvider() = default;
	virtual std::pair<Tensor4<Scalar>,Tensor4<Scalar>> get_training_data(unsigned batch_size) = 0;
	virtual bool has_reached_eol();
	virtual void reset();
};

template<typename Scalar>
class InMemoryDataProvider : public DataProvider<Scalar> {
public:
	InMemoryDataProvider(Tensor4<Scalar>* data, Tensor4<Scalar>* obj, bool shuffle = true) {

	};
	std::pair<Tensor4<Scalar>,Tensor4<Scalar>> get_data(unsigned batch_size) {

	};
	bool has_reached_eol() {

	};
	void reset() {

	};
private:
	Tensor4<Scalar>* data;
	Tensor4<Scalar>* obj;
};

} /* namespace cppnn */

#endif /* DATAPROVIDER_H_ */
