/*
 * SplitFileDataProvider.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_DATA_PROVIDER_SPLITFILEDATAPROVIDER_H_
#define C_ATTL3_DATA_PROVIDER_SPLITFILEDATAPROVIDER_H_

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/DataProvider.hpp"

namespace cattle {

/**
 * An abstract class template for a data provider backed by an arbitrary number of file pairs
 * containing the separated observations and the objectives. Implementations are responsible for
 * specifying the dimensions of both the observations and the objectives, for reading batches of
 * observation-objective pairs from the file, and for skipping arbitrary number of data instances.
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool ObsBinary = false, bool ObjBinary = false>
class SplitFileDataProvider : public DataProvider<Scalar,Rank,Sequential> {
	typedef DataProvider<Scalar,Rank,Sequential> Base;
	typedef std::pair<std::string,std::string> FilePair;
	typedef std::pair<std::ifstream,std::ifstream> FileStreamPair;
public:
	virtual ~SplitFileDataProvider() = default;
	inline bool has_more() {
		if (current_file_stream_pair_has_more())
			return true;
		++current_file_pair_ind;
		for (; current_file_pair_ind < file_pairs.size(); ++current_file_pair_ind) {
			init_current_file_stream_pair();
			if (current_file_stream_pair_has_more())
				return true;
		}
		return false;
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		if (!has_more())
			throw std::out_of_range("no more data left to fetch");
		const FilePair& current_file_pair = file_pairs[current_file_pair_ind];
		DataPair<Scalar,Rank,Sequential> data_pair = _get_data(current_file_pair.first,
				current_file_stream_pair.first, current_file_pair.second, current_file_stream_pair.second,
				batch_size);
		assert(data_pair.first.dimension(0) == data_pair.second.dimension(0));
		while (data_pair.first.dimension(0) < batch_size && has_more()) {
			const FilePair& new_current_file_pair = file_pairs[current_file_pair_ind];
			DataPair<Scalar,Rank,Sequential> add_data_pair = _get_data(new_current_file_pair.first,
					current_file_stream_pair.first, new_current_file_pair.second, current_file_stream_pair.second,
					batch_size - data_pair.first.dimension(0));
			assert(add_data_pair.first.dimension(0) == add_data_pair.second.dimension(0));
			typename Base::Data obs_concat = data_pair.first.concatenate(std::move(add_data_pair.first), 0);
			data_pair.first = std::move(obs_concat);
			typename Base::Data obj_concat = data_pair.second.concatenate(std::move(add_data_pair.second), 0);
			data_pair.second = std::move(obj_concat);
		}
		return data_pair;
	}
	inline void reset() {
		current_file_pair_ind = 0;
		init_current_file_stream_pair();
		_set_to_beg(current_file_stream_pair.first, current_file_stream_pair.second);
	}
	inline void skip(std::size_t instances) {
		if (!has_more())
			return;
		std::size_t skipped = _skip(current_file_stream_pair.first,current_file_stream_pair.second,
				instances);
		while (skipped < instances && has_more())
			skipped += _skip(current_file_stream_pair.first, current_file_stream_pair.second,
					instances - skipped);
	}
protected:
	inline SplitFileDataProvider(const std::vector<FilePair>& dataset_path_pairs) :
			file_pairs(dataset_path_pairs),
			current_file_pair_ind(0) {
		assert(!dataset_path_pairs.empty());
		init_current_file_stream_pair();
	}
	inline SplitFileDataProvider(FilePair dataset_path_pair) :
			SplitFileDataProvider(std::vector<FilePair>({ dataset_path_pair })) { }
	/**
	 * It sets the positions of the file streams to the beginning of the observation data set and
	 * the objective data set respectively.
	 *
	 * @param obs_file_stream A reference to the file stream to a file containing observations.
	 * @param obj_file_stream A reference to the file stream to a file containing objectives.
	 */
	virtual inline void _set_to_beg(std::ifstream& obs_file_stream, std::ifstream& obj_file_stream) {
		obs_file_stream.seekg(0, std::ios::beg);
		obj_file_stream.seekg(0, std::ios::beg);
	}
	/**
	 * It reads at most the specified number of observations from the observation-file and at
	 * most the specified number of objectives from the objective-file. The file streams can
	 * be expected not to have any of their fail flags set initially and to have at least 1
	 * more character left to read in each.
	 *
	 * @param obs_file The name of the observation source file.
	 * @param obs_file_stream The input stream of the observation file.
	 * @param obj_file The name of the objective source file.
	 * @param obj_file_stream The input stream of the objective file.
	 * @param batch_size The number of data points to read.
	 * @return The paired observations and objectives.
	 */
	virtual DataPair<Scalar,Rank,Sequential> _get_data(const std::string& obs_file,
			std::ifstream& obs_file_stream, const std::string& obj_file,
			std::ifstream& obj_file_stream, std::size_t batch_size) = 0;
	/**
	 * Skips at most the specified number of instances in the data streams. The file streams can
	 * be expected not to have any of their fail flags set initially.
	 *
	 * @param obs_file_stream A reference to the file stream to a file containing observations.
	 * @param obj_file_stream A reference to the file stream to a file containing objectives.
	 * @param instances The number of data points to skip.
	 * @return The number of actual data points skipped. It may be less than the specified
	 * amount if there are fewer remaining instances in the data streams.
	 */
	virtual std::size_t _skip(std::ifstream& obs_file_stream, std::ifstream& obj_file_stream,
			std::size_t instances) = 0;
private:
	bool current_file_stream_pair_has_more() {
		return current_file_stream_pair.first && current_file_stream_pair.first.peek() != EOF &&
				current_file_stream_pair.second && current_file_stream_pair.second.peek() != EOF;
	}
	void init_current_file_stream_pair() {
		const FilePair& file_pair = file_pairs[current_file_pair_ind];
		std::ifstream obs_stream(file_pair.first, ObsBinary ? std::ios::binary : std::ios::in);
		assert(obs_stream.is_open());
		std::ifstream obj_stream(file_pair.second, ObjBinary ? std::ios::binary : std::ios::in);
		assert(obj_stream.is_open());
		current_file_stream_pair = std::make_pair(std::move(obs_stream), std::move(obj_stream));
	}
	std::vector<FilePair> file_pairs;
	std::size_t current_file_pair_ind;
	FileStreamPair current_file_stream_pair;
};

}

#endif /* C_ATTL3_DATA_PROVIDER_SPLITFILEDATAPROVIDER_H_ */
