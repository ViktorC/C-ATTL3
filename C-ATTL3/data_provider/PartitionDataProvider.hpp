/*
 * PartitionDataProvider.hpp
 *
 *  Created on: 20 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_DATA_PROVIDER_PARTITIONDATAPROVIDER_H_
#define C_ATTL3_DATA_PROVIDER_PARTITIONDATAPROVIDER_H_

#include <algorithm>
#include <array>
#include <cassert>

#include "core/DataProvider.hpp"

namespace cattle {

/**
 * A wrapper class template for data providers associated with continuous partitions of other data
 * providers. It enables the partitioning of a data provider into training and test data providers
 * by mapping two contiguous blocks of its data to two PartitionedDataProvider instances.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class PartitionDataProvider : public DataProvider<Scalar,Rank,Sequential> {
	typedef DataProvider<Scalar,Rank,Sequential> Base;
public:
	inline PartitionDataProvider(Base& orig_provider, std::size_t offset, std::size_t length) :
			orig_provider(orig_provider),
			offset(offset),
			length(length) {
		assert(length > 0);
		reset();
	}
	inline const Dimensions<std::size_t,Rank>& get_obs_dims() const {
		return orig_provider.get_obs_dims();
	}
	inline const Dimensions<std::size_t,Rank>& get_obj_dims() const {
		return orig_provider.get_obj_dims();
	}
	inline bool has_more() {
		return instances_read < length && orig_provider.has_more();
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		std::size_t instances_to_read = std::min(batch_size, length - instances_read);
		instances_read += instances_to_read;
		return orig_provider.get_data(instances_to_read);
	}
	inline void reset() {
		orig_provider.reset();
		orig_provider.skip(offset);
		instances_read = 0;
	}
	inline void skip(std::size_t instances) {
		orig_provider.skip(instances);
	}
private:
	Base& orig_provider;
	const std::size_t offset;
	const std::size_t length;
	std::size_t instances_read;
};

} /* namespace cattle */

#endif /* C_ATTL3_DATA_PROVIDER_PARTITIONDATAPROVIDER_H_ */
