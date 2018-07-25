/*
 * CIFARDataProvider.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_DATA_PROVIDER_CIFARDATAPROVIDER_H_
#define C_ATTL3_DATA_PROVIDER_CIFARDATAPROVIDER_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include "JointFileDataProvider.hpp"

namespace cattle {

/**
 * An enum denoting different CIFAR data set types.
 */
enum CIFARType { CIFAR_10, CIFAR_100 };

/**
 * A data provider template for the CIFAR-10 and CIFAR-100 data sets.
 *
 * \see https://www.cs.toronto.edu/~kriz/cifar.html
 */
template<typename Scalar, CIFARType CIFARType = CIFAR_10>
class CIFARDataProvider : public JointFileDataProvider<Scalar,3,false,true> {
	static_assert(CIFARType == CIFAR_10 || CIFARType == CIFAR_100, "invalid CIFAR type");
	typedef DataProvider<Scalar,3,false> Root;
	typedef JointFileDataProvider<Scalar,3,false,true> Base;
	typedef std::array<std::size_t,4> RankwiseArray;
	static constexpr std::size_t INSTANCE_LENGTH = CIFARType == CIFAR_10 ? 3073 : 3074;
	static constexpr std::size_t NUM_LABELS = CIFARType == CIFAR_10 ? 10 : 100;
public:
	/**
	 * @param file_paths The paths to the data set files.
	 */
	inline CIFARDataProvider(std::vector<std::string> file_paths) :
			Base::JointFileDataProvider({ 32u, 32u, 3u }, { NUM_LABELS, 1u, 1u }, file_paths),
			offsets({ 0u, 0u, 0u, 0u }),
			obs_extents(Base::obs_dims.template promote<>()),
			obj_extents(Base::obj_dims.template promote<>()) {
		Base::reset();
	}
	/**
	 * @param file_path The path to the data set file.
	 */
	inline CIFARDataProvider(std::string file_path) :
			CIFARDataProvider(std::vector<std::string>({ file_path })) { }
protected:
	inline DataPair<Scalar,3,false> _get_data(const std::string& file_name, std::ifstream& file_stream,
			std::size_t batch_size) {
		typename Root::Data obs(batch_size, Base::obs_dims(0), Base::obs_dims(1), Base::obs_dims(2));
		typename Root::Data obj(batch_size, Base::obj_dims(0), Base::obj_dims(1), Base::obj_dims(2));
		obj.setZero();
		std::size_t i;
		for (i = 0; i < batch_size && file_stream.read(buffer, INSTANCE_LENGTH); ++i) {
			unsigned char* u_buffer = reinterpret_cast<unsigned char*>(buffer);
			std::size_t buffer_ind = 0;
			// Set the label.
			if (CIFARType == CIFAR_100)
				buffer_ind++;
			obj(i,u_buffer[buffer_ind++],0u,0u) = (Scalar) 1;
			// Set the image.
			for (std::size_t channel = 0; channel < Base::obs_dims(2); ++channel) {
				for (std::size_t height = 0; height < Base::obs_dims(1); ++height) {
					for (std::size_t width = 0; width < Base::obj_dims(0); ++width)
						obs(i,height,width,channel) = (Scalar) u_buffer[buffer_ind++];
				}
			}
			assert(buffer_ind == INSTANCE_LENGTH);
		}
		if (i == batch_size)
			return std::make_pair(obs, obj);
		obs_extents[0] = i;
		obj_extents[0] = i;
		return std::make_pair(obs.slice(offsets, obs_extents), obj.slice(offsets, obj_extents));
	}
	inline std::size_t _skip(std::ifstream& file_stream, std::size_t instances) {
		std::streampos curr_pos = file_stream.tellg();
		file_stream.seekg(0, std::ios::end);
		std::size_t skip_extent = file_stream.tellg() - curr_pos;
		file_stream.seekg(curr_pos);
		file_stream.ignore(instances * INSTANCE_LENGTH);
		return std::min(instances, skip_extent / INSTANCE_LENGTH);
	}
private:
	char buffer[INSTANCE_LENGTH];
	RankwiseArray offsets, obs_extents, obj_extents;
};

}

#endif /* C_ATTL3_DATA_PROVIDER_CIFARDATAPROVIDER_H_ */
