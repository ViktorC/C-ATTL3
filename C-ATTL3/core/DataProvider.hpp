/*
 * DataProvider.hpp
 *
 *  Created on: 18.01.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_DATAPROVIDER_H_
#define C_ATTL3_CORE_DATAPROVIDER_H_

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "Dimensions.hpp"
#include "EigenProxy.hpp"

namespace cattle {

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
protected:
	static constexpr std::size_t DATA_RANK = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
	typedef Dimensions<std::size_t,Rank> Dims;
public:
	virtual ~DataProvider() = default;
	/**
	 * A simple constant getter method for the dimensions of the observations.
	 *
	 * @return A constant reference to the dimensions of the observations.
	 */
	virtual const Dims& get_obs_dims() const = 0;
	/**
	 * A simple constant getter method for the dimensions of the objectives.
	 *
	 * @return A constant reference to the dimensions of the objectives.
	 */
	virtual const Dims& get_obj_dims() const = 0;
	/**
	 * A method that returns whether the data provider instance has more data to provide.
	 * It should always be called before calling get_data(std::size_t).
	 *
	 * @return Whether there are more observation-objective pairs to read from the
	 * instance.
	 */
	virtual bool has_more() = 0;
	/**
	 * Reads and returns the specified number of observation-objective pairs. It also
	 * offsets the reader by the specified number. If has_more() returns false, the
	 * invocation of this method results in the throwing of a std::out_of_range exception.
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
	/**
	 * It skips the specified number of data points. If has_more() returns false,
	 * the invocation of the method has no effect.
	 *
	 * @param instances The number of instances to skip.
	 */
	virtual void skip(std::size_t instances) = 0;
};

} /* namespace cattle */

#endif /* C_ATTL3_CORE_DATAPROVIDER_H_ */
