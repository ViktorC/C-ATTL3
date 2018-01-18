/*
 * DataProvider.h
 *
 *  Created on: 18.01.2018
 *      Author: Viktor Csomor
 */

#ifndef DATAPROVIDER_H_
#define DATAPROVIDER_H_

namespace cppnn {

/**
 * A class template for fetching data from the disk or the memory.
 */
template<typename Scalar>
class DataProvider {
public:
	DataProvider();
	virtual ~DataProvider();
};

} /* namespace cppnn */

#endif /* DATAPROVIDER_H_ */
