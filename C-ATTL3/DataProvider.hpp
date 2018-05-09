/*
 * DataProvider.hpp
 *
 *  Created on: 18.01.2018
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_DATAPROVIDER_H_
#define CATTL3_DATAPROVIDER_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Dimensions.hpp"
#include "utils/EigenProxy.hpp"

// TODO Pre-fetch batches from the files.

namespace cattle {

/**
 * An alias for a unique tensor pointer.
 */
template<typename Scalar, std::size_t Rank>
using TensorPtr = std::unique_ptr<Tensor<Scalar,Rank>>;

/**
 * An alias for a pair of two tensors of the same rank. It represents observation-objective pairs.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
using DataPair = std::pair<Tensor<Scalar,Rank + Sequential + 1>,Tensor<Scalar,Rank + Sequential + 1>>;

template<typename Scalar, std::size_t Rank, bool Sequential> class PartitionDataProvider;

/**
 * A class template for fetching data from memory or disk.
 */
template<typename Scalar, std::size_t Rank, bool Sequential>
class DataProvider {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal data provider rank");
	friend class PartitionDataProvider<Scalar,Rank,Sequential>;
public:
	virtual ~DataProvider() = default;
	/**
	 * A simple constant getter method for the dimensions of the observations.
	 *
	 * @return A constant reference to the dimensions of the observations.
	 */
	virtual const Dimensions<std::size_t,Rank>& get_obs_dims() const = 0;
	/**
	 * A simple constant getter method for the dimensions of the objectives.
	 *
	 * @return A constant reference to the dimensions of the objectives.
	 */
	virtual const Dimensions<std::size_t,Rank>& get_obj_dims() const = 0;
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
protected:
	/**
	 * It skips the specified number of data points. If has_more() returns false,
	 * the invocation of the method has no effect.
	 *
	 * @param instances The number of instances to skip.
	 */
	virtual void skip(std::size_t instances) = 0;
protected:
	static constexpr std::size_t DATA_RANK = Rank + Sequential + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
	typedef TensorPtr<Scalar,DATA_RANK> DataPtr;
};

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
protected:
	inline void skip(std::size_t instances) {
		orig_provider.skip(instances);
	}
private:
	Base& orig_provider;
	const std::size_t offset;
	const std::size_t length;
	std::size_t instances_read;
};


/**
 * A data provider that reads from the memory. It requires the entire observation and
 * objective data sets to be loaded into memory, but it fetches pairs faster.
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool Shuffle = true>
class MemoryDataProvider : public DataProvider<Scalar,Rank,Sequential> {
	typedef DataProvider<Scalar,Rank,Sequential> Base;
public:
	/**
	 * It constructs a data provider backed by the specified tensors.
	 *
	 * @param obs A unique pointer to the tensor containing the observations.
	 * @param obj A unique pointer to the tensor containing the objectives.
	 * shuffled.
	 */
	inline MemoryDataProvider(typename Base::DataPtr obs, typename Base::DataPtr obj) :
			obs(std::move(obs)),
			obj(std::move(obj)),
			offsets() {
		assert(this->obs != nullptr);
		assert(this->obj != nullptr);
		assert(this->obs->dimension(0) == this->obj->dimension(0) &&
				"mismatched data and obj tensor row numbers");
		Dimensions<std::size_t,Base::DATA_RANK> obs_dims = this->obs->dimensions();
		Dimensions<std::size_t,Base::DATA_RANK> obj_dims = this->obj->dimensions();
		this->obs_dims = obs_dims.template demote<Sequential + 1>();
		this->obj_dims = obj_dims.template demote<Sequential + 1>();
		instances = (std::size_t) this->obs->dimension(0);
		offsets.fill(0);
		data_extents = obs_dims;
		obj_extents = obj_dims;
		if (Shuffle)
			shuffle_tensor_rows();
	}
	inline const Dimensions<std::size_t,Rank>& get_obs_dims() const {
		return obs_dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_obj_dims() const {
		return obj_dims;
	}
	inline bool has_more() {
		return offsets[0] < (int) instances;
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		if (!has_more())
			throw std::out_of_range("no more data left to fetch");
		std::size_t max_batch_size = std::min(batch_size, instances - offsets[0]);
		data_extents[0] = max_batch_size;
		obj_extents[0] = max_batch_size;
		typename Base::Data data_batch = obs->slice(offsets, data_extents);
		typename Base::Data obj_batch = obj->slice(offsets, obj_extents);
		offsets[0] = std::min(instances, offsets[0] + max_batch_size);
		return std::make_pair(std::move(data_batch), std::move(obj_batch));
	}
	inline void reset() {
		offsets[0] = 0;
	}
protected:
	inline void skip(std::size_t instances) {
		offsets[0] = std::min((int) this->instances, (int) (offsets[0] + instances));
	}
	inline void shuffle_tensor_rows() {
		std::size_t rows = obs->dimension(0);
		MatrixMap<Scalar> obs_mat(obs->data(), rows, obs->size() / rows);
		MatrixMap<Scalar> obj_mat(obj->data(), rows, obj->size() / rows);
		// Create an identity matrix.
		internal::PermMatrix perm(rows);
		perm.setIdentity();
		// Shuffle its indices.
		std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
		// And apply the same row permutation transformation to both the observations and the objectives.
		obs_mat = perm * obs_mat;
		obj_mat = perm * obj_mat;
	}
private:
	typename Base::DataPtr obs;
	typename Base::DataPtr obj;
	Dimensions<std::size_t,Rank> obs_dims;
	Dimensions<std::size_t,Rank> obj_dims;
	std::size_t instances;
	std::array<std::size_t,Base::DATA_RANK> offsets;
	std::array<std::size_t,Base::DATA_RANK> data_extents;
	std::array<std::size_t,Base::DATA_RANK> obj_extents;
};

/**
 * An abstract class template for a data provider backed by data on disk in the form of an arbitrary
 * number of files containing both the observations and the objectives. Implementations are responsible
 * for specifying the dimensions of both the observations and the objectives, for reading batches of
 * observation-objective pairs from the file, and for skipping arbitrary number of data instances.
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool Binary = false>
class JointFileDataProvider : public DataProvider<Scalar,Rank,Sequential> {
	typedef DataProvider<Scalar,Rank,Sequential> Base;
public:
	virtual ~JointFileDataProvider() = default;
	inline bool has_more() {
		for (; current_stream < data_streams.size(); ++current_stream) {
			std::ifstream& data_stream = data_streams[current_stream];
			if (data_stream && data_stream.peek() != EOF)
				return true;
		}
		return false;
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		if (!has_more())
			throw std::out_of_range("no more data left to fetch");
		DataPair<Scalar,Rank,Sequential> data_pair = _get_data(data_streams[current_stream],
				batch_size);
		assert(data_pair.first.dimension(0) == data_pair.second.dimension(0));
		/* If the data contains fewer batches than expected, the end of the file has been reached and the
		 * rest of the data should be read from the next file. */
		while (data_pair.first.dimension(0) < batch_size && has_more()) {
			DataPair<Scalar,Rank,Sequential> add_data_pair = _get_data(data_streams[current_stream],
					batch_size - data_pair.first.dimension(0));
			assert(add_data_pair.first.dimension(0) == add_data_pair.second.dimension(0));
			// It has to be evaluated into a temporary due to the dimension incompatibility.
			typename Base::Data obs_concat =
					data_pair.first.concatenate(std::move(add_data_pair.first),0);
			data_pair.first = std::move(obs_concat);
			typename Base::Data obj_concat =
					data_pair.second.concatenate(std::move(add_data_pair.second), 0);
			data_pair.second = std::move(obj_concat);
		}
		return data_pair;
	}
	inline void reset() {
		for (std::size_t i = 0; i < data_streams.size(); ++i) {
			std::ifstream& data_stream = data_streams[i];
			data_stream.clear();
			_set_to_beg(data_stream);
		}
		current_stream = 0;
	}
protected:
	inline JointFileDataProvider(std::vector<std::string> dataset_paths) :
			data_streams(dataset_paths.size()),
			current_stream(0) {
		assert(!dataset_paths.empty());
		for (std::size_t i = 0; i < dataset_paths.size(); ++i) {
			data_streams[i] = std::ifstream(dataset_paths[i], Binary ? std::ios::binary : std::ios::in);
			std::ifstream& data_stream = data_streams[i];
			assert(data_stream.is_open());
		}
	}
	inline JointFileDataProvider(std::string dataset_path) :
			JointFileDataProvider({ dataset_path }) { }
	/**
	 * It sets the position of the file stream to the beginning of the data set.
	 *
	 * @param data_stream A reference to the file stream of the data set.
	 */
	virtual inline void _set_to_beg(std::ifstream& data_stream) {
		data_stream.seekg(0, std::ios::beg);
	}
	/**
	 * It reads at most the specified number of observation-objective pairs from the provided
	 * file stream. The file stream can be expected not to have any of its fail flags set
	 * initially and to have at least 1 more character left to read.
	 *
	 * @param data_stream A reference to the file stream of the data set.
	 * @param batch_size The number of data points to return.
	 * @return A pair of tensors containing the data batch.
	 */
	virtual DataPair<Scalar,Rank,Sequential> _get_data(std::ifstream& data_stream,
			std::size_t batch_size) = 0;
	/**
	 * Skips at most the specified number of instances in the data stream. The file stream can
	 * be expected not to have any of its fail flags set initially.
	 *
	 * @param data_stream A reference to the file stream of the data set.
	 * @param instances The number of instances to skip.
	 * @return The number of instances actually skipped. It may be less than the specified
	 * amount if there are fewer remaining instances in the data stream.
	 */
	virtual std::size_t _skip(std::ifstream& data_stream, std::size_t instances) = 0;
	inline void skip(std::size_t instances) {
		if (!has_more())
			return;
		std::size_t skipped = _skip(data_streams[current_stream], instances);
		while (skipped < instances && has_more())
			skipped += _skip(data_streams[current_stream], instances - skipped);
	}
private:
	std::vector<std::ifstream> data_streams;
	std::size_t current_stream;
};

/**
 * An abstract class template for a data provider backed by an arbitrary number of file pairs
 * containing the separated observations and the objectives. Implementations are responsible for
 * specifying the dimensions of both the observations and the objectives, for reading batches of
 * observation-objective pairs from the file, and for skipping arbitrary number of data instances.
 */
template<typename Scalar, std::size_t Rank, bool Sequential, bool ObsBinary = false, bool ObjBinary = false>
class SplitFileDataProvider : public DataProvider<Scalar,Rank,Sequential> {
	typedef DataProvider<Scalar,Rank,Sequential> Base;
public:
	virtual ~SplitFileDataProvider() = default;
	inline bool has_more() {
		for (; current_stream_pair < data_stream_pairs.size(); ++current_stream_pair) {
			std::pair<std::ifstream,std::ifstream>& stream_pair = data_stream_pairs[current_stream_pair];
			if (stream_pair.first && (stream_pair.first.peek() != EOF) &&
					stream_pair.second && (stream_pair.second.peek() != EOF))
				return true;
		}
		return false;
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		if (!has_more())
			throw std::out_of_range("no more data left to fetch");
		std::pair<std::ifstream,std::ifstream>& first_stream_pair = data_stream_pairs[current_stream_pair];
		DataPair<Scalar,Rank,Sequential> data_pair = _get_data(first_stream_pair.first,
				first_stream_pair.second, batch_size);
		assert(data_pair.first.dimension(0) == data_pair.second.dimension(0));
		while (data_pair.first.dimension(0) < batch_size && has_more()) {
			std::pair<std::ifstream,std::ifstream>& stream_pair = data_stream_pairs[current_stream_pair];
			DataPair<Scalar,Rank,Sequential> add_data_pair = _get_data(stream_pair.first,
					stream_pair.second, batch_size - data_pair.first.dimension(0));
			assert(add_data_pair.first.dimension(0) == add_data_pair.second.dimension(0));
			typename Base::Data obs_concat =
					data_pair.first.concatenate(std::move(add_data_pair.first),0);
			data_pair.first = std::move(obs_concat);
			typename Base::Data obj_concat =
					data_pair.second.concatenate(std::move(add_data_pair.second), 0);
			data_pair.second = std::move(obj_concat);
		}
		return data_pair;
	}
	inline void reset() {
		for (std::size_t i = 0; i < data_stream_pairs.size(); ++i) {
			std::pair<std::ifstream,std::ifstream>& stream_pair = data_stream_pairs[i];
			stream_pair.first.clear();
			stream_pair.second.clear();
			_set_to_beg(stream_pair.first, stream_pair.second);
		}
		current_stream_pair = 0;
	}
protected:
	inline SplitFileDataProvider(std::vector<std::pair<std::string,std::string>> dataset_path_pairs) :
			data_stream_pairs(dataset_path_pairs.size()),
			current_stream_pair(0) {
		assert(!dataset_path_pairs.empty());
		for (std::size_t i = 0; i < dataset_path_pairs.size(); ++i) {
			std::pair<std::string,std::string>& path_pair = dataset_path_pairs[i];
			std::ifstream obs_stream(path_pair.first, ObsBinary ? std::ios::binary : std::ios::in);
			assert(obs_stream.is_open());
			std::ifstream obj_stream(path_pair.second, ObjBinary ? std::ios::binary : std::ios::in);
			assert(obj_stream.is_open());
			data_stream_pairs[i] = std::make_pair(std::move(obs_stream), std::move(obj_stream));
		}
	}
	inline SplitFileDataProvider(std::pair<std::string,std::string> dataset_path_pair) :
			SplitFileDataProvider(std::vector<std::pair<std::string,std::string>>({ dataset_path_pair })) { }
	/**
	 * It sets the positions of the file streams to the beginning of the observation data set and
	 * the objective data set respectively.
	 *
	 * @param obs_input_stream A reference to the file stream to a file containing observations.
	 * @param obj_input_stream A reference to the file stream to a file containing objectives.
	 */
	virtual inline void _set_to_beg(std::ifstream& obs_input_stream, std::ifstream& obj_input_stream) {
		obs_input_stream.seekg(0, std::ios::beg);
		obj_input_stream.seekg(0, std::ios::beg);
	}
	/**
	 * It reads at most the specified number of observations from the observation-file and at
	 * most the specified number of objectives from the objective-file. The file streams can
	 * be expected not to have any of their fail flags set initially and to have at least 1
	 * more character left to read in each.
	 *
	 * @param obs_input_stream A reference to the file stream to a file containing observations.
	 * @param obj_input_stream A reference to the file stream to a file containing objectives.
	 * @param batch_size The number of data points to read.
	 * @return The paired observations and objectives.
	 */
	virtual DataPair<Scalar,Rank,Sequential> _get_data(std::ifstream& obs_input_stream,
			std::ifstream& obj_input_stream, std::size_t batch_size) = 0;
	/**
	 * Skips at most the specified number of instances in the data streams. The file streams can
	 * be expected not to have any of their fail flags set initially.
	 *
	 * @param obs_input_stream A reference to the file stream to a file containing observations.
	 * @param obj_input_stream A reference to the file stream to a file containing objectives.
	 * @param instances The number of data points to skip.
	 * @return The number of actual data points skipped. It may be less than the specified
	 * amount if there are fewer remaining instances in the data streams.
	 */
	virtual std::size_t _skip(std::ifstream& obs_input_stream, std::ifstream& obj_input_stream,
			std::size_t instances) = 0;
	inline void skip(std::size_t instances) {
		if (!has_more())
			return;
		std::pair<std::ifstream,std::ifstream>& first_stream_pair = data_stream_pairs[current_stream_pair];
		std::size_t skipped = _skip(first_stream_pair.first, first_stream_pair.second, instances);
		while (skipped < instances && has_more()) {
			std::pair<std::ifstream,std::ifstream>& stream_pair = data_stream_pairs[current_stream_pair];
			skipped += _skip(stream_pair.first, stream_pair.second, instances - skipped);
		}
	}
private:
	std::vector<std::pair<std::ifstream,std::ifstream>> data_stream_pairs;
	std::size_t current_stream_pair;
};

/**
 * A data provider template for the MNIST data set.
 */
template<typename Scalar>
class MNISTDataProvider : public SplitFileDataProvider<Scalar,3,false,true,true> {
	typedef DataProvider<Scalar,3,false> Root;
	typedef SplitFileDataProvider<Scalar,3,false,true,true> Base;
	static constexpr std::size_t OBS_OFFSET = 16;
	static constexpr std::size_t LABEL_OFFSET = 8;
	static constexpr std::size_t OBS_INSTANCE_LENGTH = 784;
	static constexpr std::size_t LABEL_INSTANCE_LENGTH = 1;
public:
	/**
	 * @param obs_path The path to the file containing the observations.
	 * @param labels_path The path to the file containing the corresponding labels.
	 */
	MNISTDataProvider(std::string obs_path, std::string labels_path) :
			Base::SplitFileDataProvider(std::make_pair(obs_path, labels_path)),
			obs({ 28u, 28u, 1u }),
			obj({ 10u, 1u, 1u }),
			offsets({ 0u, 0u, 0u, 0u }),
			obs_extents({ 0u, 28u, 28u, 1u }),
			obj_extents({ 0u, 10u, 1u, 1u }) {
		Base::reset();
	}
	inline const Dimensions<std::size_t,3>& get_obs_dims() const {
		return obs;
	}
	inline const Dimensions<std::size_t,3>& get_obj_dims() const {
		return obj;
	}
protected:
	inline void _set_to_beg(std::ifstream& obs_input_stream, std::ifstream& obj_input_stream) {
		Base::_set_to_beg(obs_input_stream, obj_input_stream);
		obs_input_stream.ignore(OBS_OFFSET);
		obj_input_stream.ignore(LABEL_OFFSET);
	}
	inline DataPair<Scalar,3,false> _get_data(std::ifstream& obs_input_stream,
				std::ifstream& obj_input_stream, std::size_t batch_size) {
		Tensor<Scalar,4> obs(batch_size, 28u, 28u, 1u);
		Tensor<Scalar,4> obj(batch_size, 10u, 1u, 1u);
		obj.setZero();
		std::size_t i;
		for (i = 0; i < batch_size && obs_input_stream.read(obs_buffer, OBS_INSTANCE_LENGTH); ++i) {
			// Read and set the label.
			char label;
			obj_input_stream.read(&label, LABEL_INSTANCE_LENGTH);
			obj(i,static_cast<std::size_t>(label),0u,0u) = (Scalar) 1;
			// Set the image.
			unsigned char* u_buffer = reinterpret_cast<unsigned char*>(obs_buffer);
			std::size_t buffer_ind = 0;
			for (std::size_t height = 0; height < 28; ++height) {
				for (std::size_t width = 0; width < 28; ++width)
					obs(i,height,width,0u) = (Scalar) u_buffer[buffer_ind++];
			}
			assert(buffer_ind == OBS_INSTANCE_LENGTH);
		}
		if (i == batch_size)
			return std::make_pair(obs, obj);
		obs_extents[0] = i;
		obj_extents[0] = i;
		return std::make_pair(obs.slice(offsets, obs_extents), obj.slice(offsets, obj_extents));
	}
	inline std::size_t _skip(std::ifstream& obs_input_stream, std::ifstream& obj_input_stream,
				std::size_t instances) {
		// Skip observations.
		std::streampos curr_obs_pos = obs_input_stream.tellg();
		obs_input_stream.seekg(0, std::ios::end);
		std::size_t obs_skip_extent = obs_input_stream.tellg() - curr_obs_pos;
		obs_input_stream.seekg(curr_obs_pos);
		obs_input_stream.ignore(instances * OBS_INSTANCE_LENGTH);
		std::size_t skipped_obs = std::min(instances, obs_skip_extent / OBS_INSTANCE_LENGTH);
		// Skip labels.
		std::streampos curr_label_pos = obj_input_stream.tellg();
		obj_input_stream.seekg(0, std::ios::end);
		std::size_t label_skip_extent = obj_input_stream.tellg() - curr_label_pos;
		obj_input_stream.seekg(curr_label_pos);
		obj_input_stream.ignore(instances * LABEL_INSTANCE_LENGTH);
		std::size_t skipped_labels = std::min(instances, label_skip_extent / LABEL_INSTANCE_LENGTH);
		assert(skipped_obs == skipped_labels);
		return skipped_obs;
	}
private:
	const Dimensions<std::size_t,3> obs;
	const Dimensions<std::size_t,3> obj;
	char obs_buffer[OBS_INSTANCE_LENGTH];
	std::array<std::size_t,4> offsets;
	std::array<std::size_t,4> obs_extents;
	std::array<std::size_t,4> obj_extents;
};

/**
 * An enum denoting different CIFAR data set types.
 */
enum CIFARType { CIFAR_10, CIFAR_100 };

/**
 * A data provider template for the CIFAR-10 and CIFAR-100 data sets.
 */
template<typename Scalar, CIFARType CIFARType = CIFAR_10>
class CIFARDataProvider : public JointFileDataProvider<Scalar,3,false,true> {
	typedef DataProvider<Scalar,3,false> Root;
	typedef JointFileDataProvider<Scalar,3,false,true> Base;
	static_assert(CIFARType == CIFAR_10 || CIFARType == CIFAR_100, "invalid CIFAR type");
	static constexpr std::size_t INSTANCE_LENGTH = CIFARType == CIFAR_10 ? 3073 : 3074;
	static constexpr std::size_t NUM_LABELS = CIFARType == CIFAR_10 ? 10 : 100;
public:
	/**
	 * @param file_paths The paths to the data set files.
	 */
	inline CIFARDataProvider(std::vector<std::string> file_paths) :
			Base::JointFileDataProvider(file_paths),
			obs({ 32u, 32u, 3u }),
			obj({ NUM_LABELS, 1u, 1u }),
			offsets({ 0u, 0u, 0u, 0u }),
			obs_extents({ 0u, 32u, 32u, 3u }),
			obj_extents({ 0u, NUM_LABELS, 1u, 1u }) {
		Base::reset();
	}
	/**
	 * @param file_path The path to the data set file.
	 */
	inline CIFARDataProvider(std::string file_path) :
			CIFARDataProvider(std::vector<std::string>({ file_path })) { }
	inline const Dimensions<std::size_t,3>& get_obs_dims() const {
		return obs;
	}
	inline const Dimensions<std::size_t,3>& get_obj_dims() const {
		return obj;
	}
protected:
	inline DataPair<Scalar,3,false> _get_data(std::ifstream& data_stream,
				std::size_t batch_size) {
		Tensor<Scalar,4> obs(batch_size, 32u, 32u, 3u);
		Tensor<Scalar,4> obj(batch_size, NUM_LABELS, 1u, 1u);
		obj.setZero();
		std::size_t i;
		for (i = 0; i < batch_size && data_stream.read(buffer, INSTANCE_LENGTH); ++i) {
			unsigned char* u_buffer = reinterpret_cast<unsigned char*>(buffer);
			std::size_t buffer_ind = 0;
			// Set the label.
			if (CIFARType == CIFAR_100)
				buffer_ind++;
			obj(i,u_buffer[buffer_ind++],0u,0u) = (Scalar) 1;
			// Set the image.
			for (std::size_t channel = 0; channel < 3; ++channel) {
				for (std::size_t height = 0; height < 32; ++height) {
					for (std::size_t width = 0; width < 32; ++width)
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
	inline std::size_t _skip(std::ifstream& data_stream, std::size_t instances) {
		std::streampos curr_pos = data_stream.tellg();
		data_stream.seekg(0, std::ios::end);
		std::size_t skip_extent = data_stream.tellg() - curr_pos;
		data_stream.seekg(curr_pos);
		data_stream.ignore(instances * INSTANCE_LENGTH);
		return std::min(instances, skip_extent / INSTANCE_LENGTH);
	}
private:
	const Dimensions<std::size_t,3> obs;
	const Dimensions<std::size_t,3> obj;
	char buffer[INSTANCE_LENGTH];
	std::array<std::size_t,4> offsets;
	std::array<std::size_t,4> obs_extents;
	std::array<std::size_t,4> obj_extents;
};

} /* namespace cattle */

#endif /* CATTL3_DATAPROVIDER_H_ */
