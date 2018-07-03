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
#include <cctype>
#include <cstddef>
#include <dirent.h>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <utility>
#include <vector>

#include "Dimensions.hpp"
#include "utils/EigenProxy.hpp"

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
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size = std::numeric_limits<std::size_t>::max()) {
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
 * An alias for an input file stream paired with file path string.
 */
typedef std::pair<std::string,std::ifstream> InputFile;

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
		for (; current_file < files.size(); ++current_file) {
			std::ifstream& file_stream = files[current_file].second;
			if (file_stream && file_stream.peek() != EOF)
				return true;
		}
		return false;
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		if (!has_more())
			throw std::out_of_range("no more data left to fetch");
		DataPair<Scalar,Rank,Sequential> data_pair = _get_data(files[current_file],
				batch_size);
		assert(data_pair.first.dimension(0) == data_pair.second.dimension(0));
		/* If the data contains fewer batches than expected, the end of the file has been reached and the
		 * rest of the data should be read from the next file. */
		while (data_pair.first.dimension(0) < batch_size && has_more()) {
			DataPair<Scalar,Rank,Sequential> add_data_pair = _get_data(files[current_file],
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
		for (std::size_t i = 0; i < files.size(); ++i) {
			std::ifstream& file_stream = files[i].second;
			file_stream.clear();
			_set_to_beg(file_stream);
		}
		current_file = 0;
	}
protected:
	inline JointFileDataProvider(std::vector<std::string> dataset_paths) :
			files(dataset_paths.size()),
			current_file(0) {
		assert(!dataset_paths.empty());
		for (std::size_t i = 0; i < dataset_paths.size(); ++i) {
			std::string dataset_path = dataset_paths[i];
			auto file = std::make_pair(dataset_path, std::ifstream(dataset_path,
					Binary ? std::ios::binary : std::ios::in));
			assert(file.second.is_open());
			files[i] = std::move(file);
		}
	}
	inline JointFileDataProvider(std::string dataset_path) :
			JointFileDataProvider({ dataset_path }) { }
	/**
	 * It sets the position of the file stream to the beginning of the data set.
	 *
	 * @param file_stream A reference to the file stream of the data set.
	 */
	virtual inline void _set_to_beg(std::ifstream& file_stream) {
		file_stream.seekg(0, std::ios::beg);
	}
	/**
	 * It reads at most the specified number of observation-objective pairs from the provided
	 * file stream. The file stream can be expected not to have any of its fail flags set
	 * initially and to have at least 1 more character left to read.
	 *
	 * @param file A reference to the file stream of the data set paired with its file path.
	 * @param batch_size The number of data points to return.
	 * @return A pair of tensors containing the data batch.
	 */
	virtual DataPair<Scalar,Rank,Sequential> _get_data(InputFile& file, std::size_t batch_size) = 0;
	/**
	 * Skips at most the specified number of instances in the data stream. The file stream can
	 * be expected not to have any of its fail flags set initially.
	 *
	 * @param file_stream A reference to the file stream of the data set.
	 * @param instances The number of instances to skip.
	 * @return The number of instances actually skipped. It may be less than the specified
	 * amount if there are fewer remaining instances in the data stream.
	 */
	virtual std::size_t _skip(std::ifstream& file_stream, std::size_t instances) = 0;
	inline void skip(std::size_t instances) {
		if (!has_more())
			return;
		std::size_t skipped = _skip(files[current_file].second, instances);
		while (skipped < instances && has_more())
			skipped += _skip(files[current_file].second, instances - skipped);
	}
private:
	std::vector<InputFile> files;
	std::size_t current_file;
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
	typedef std::pair<InputFile,InputFile> InputFilePair;
public:
	virtual ~SplitFileDataProvider() = default;
	inline bool has_more() {
		for (; current_file_pair < file_pairs.size(); ++current_file_pair) {
			InputFilePair& file_pair = file_pairs[current_file_pair];
			if (file_pair.first.second && (file_pair.first.second.peek() != EOF) &&
					file_pair.second.second && (file_pair.second.second.peek() != EOF))
				return true;
		}
		return false;
	}
	inline DataPair<Scalar,Rank,Sequential> get_data(std::size_t batch_size) {
		if (!has_more())
			throw std::out_of_range("no more data left to fetch");
		InputFilePair& first_file_pair = file_pairs[current_file_pair];
		DataPair<Scalar,Rank,Sequential> data_pair = _get_data(first_file_pair.first, first_file_pair.second,
				batch_size);
		assert(data_pair.first.dimension(0) == data_pair.second.dimension(0));
		while (data_pair.first.dimension(0) < batch_size && has_more()) {
			InputFilePair& file_pair = file_pairs[current_file_pair];
			DataPair<Scalar,Rank,Sequential> add_data_pair = _get_data(file_pair.first, file_pair.second,
					batch_size - data_pair.first.dimension(0));
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
		for (std::size_t i = 0; i < file_pairs.size(); ++i) {
			InputFilePair& file_pair = file_pairs[i];
			file_pair.first.second.clear();
			file_pair.second.second.clear();
			_set_to_beg(file_pair.first.second, file_pair.second.second);
		}
		current_file_pair = 0;
	}
protected:
	inline SplitFileDataProvider(std::vector<std::pair<std::string,std::string>> dataset_path_pairs) :
			file_pairs(dataset_path_pairs.size()),
			current_file_pair(0) {
		assert(!dataset_path_pairs.empty());
		for (std::size_t i = 0; i < dataset_path_pairs.size(); ++i) {
			std::pair<std::string,std::string>& path_pair = dataset_path_pairs[i];
			std::ifstream obs_stream(path_pair.first, ObsBinary ? std::ios::binary : std::ios::in);
			assert(obs_stream.is_open());
			std::ifstream obj_stream(path_pair.second, ObjBinary ? std::ios::binary : std::ios::in);
			assert(obj_stream.is_open());
			file_pairs[i] = std::make_pair(std::make_pair(path_pair.first, std::move(obs_stream)),
					std::make_pair(path_pair.second, std::move(obj_stream)));
		}
	}
	inline SplitFileDataProvider(std::pair<std::string,std::string> dataset_path_pair) :
			SplitFileDataProvider(std::vector<std::pair<std::string,std::string>>({ dataset_path_pair })) { }
	/**
	 * It sets the positions of the file streams to the beginning of the observation data set and
	 * the objective data set respectively.
	 *
	 * @param obj_file_stream A reference to the file stream to a file containing observations.
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
	 * @param obs_file A reference to the file stream of the data set paired with its file path.
	 * @param obj_file A reference to the file stream of the data set paired with its file path.
	 * @param batch_size The number of data points to read.
	 * @return The paired observations and objectives.
	 */
	virtual DataPair<Scalar,Rank,Sequential> _get_data(InputFile& obs_file, InputFile& obj_file,
			std::size_t batch_size) = 0;
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
	inline void skip(std::size_t instances) {
		if (!has_more())
			return;
		InputFilePair& first_file_pair = file_pairs[current_file_pair];
		std::size_t skipped = _skip(first_file_pair.first.second, first_file_pair.second.second,
				instances);
		while (skipped < instances && has_more()) {
			InputFilePair& file_pair = file_pairs[current_file_pair];
			skipped += _skip(file_pair.first.second, file_pair.second.second, instances - skipped);
		}
	}
private:
	std::vector<InputFilePair> file_pairs;
	std::size_t current_file_pair;
};

/**
 * A data provider template for the MNIST data set.
 *
 * \see http://yann.lecun.com/exdb/mnist/
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
			obs_dims({ 28u, 28u, 1u }),
			obj_dims({ 10u, 1u, 1u }),
			offsets({ 0u, 0u, 0u, 0u }),
			obs_extents({ 0u, 28u, 28u, 1u }),
			obj_extents({ 0u, 10u, 1u, 1u }) {
		Base::reset();
	}
	inline const Dimensions<std::size_t,3>& get_obs_dims() const {
		return obs_dims;
	}
	inline const Dimensions<std::size_t,3>& get_obj_dims() const {
		return obj_dims;
	}
protected:
	inline void _set_to_beg(std::ifstream& obs_file_stream, std::ifstream& obj_file_stream) {
		Base::_set_to_beg(obs_file_stream, obj_file_stream);
		obs_file_stream.ignore(OBS_OFFSET);
		obj_file_stream.ignore(LABEL_OFFSET);
	}
	inline DataPair<Scalar,3,false> _get_data(InputFile& obs_file, InputFile& obj_file,
			std::size_t batch_size) {
		Tensor<Scalar,4> obs(batch_size, 28u, 28u, 1u);
		Tensor<Scalar,4> obj(batch_size, 10u, 1u, 1u);
		obj.setZero();
		std::size_t i;
		for (i = 0; i < batch_size && obs_file.second.read(obs_buffer, OBS_INSTANCE_LENGTH); ++i) {
			// Read and set the label.
			char label;
			obj_file.second.read(&label, LABEL_INSTANCE_LENGTH);
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
	inline std::size_t _skip(std::ifstream& obs_file_stream, std::ifstream& obj_file_stream,
			std::size_t instances) {
		// Skip observations.
		std::streampos curr_obs_pos = obs_file_stream.tellg();
		obs_file_stream.seekg(0, std::ios::end);
		std::size_t obs_skip_extent = obs_file_stream.tellg() - curr_obs_pos;
		obs_file_stream.seekg(curr_obs_pos);
		obs_file_stream.ignore(instances * OBS_INSTANCE_LENGTH);
		std::size_t skipped_obs = std::min(instances, obs_skip_extent / OBS_INSTANCE_LENGTH);
		// Skip labels.
		std::streampos curr_label_pos = obj_file_stream.tellg();
		obj_file_stream.seekg(0, std::ios::end);
		std::size_t label_skip_extent = obj_file_stream.tellg() - curr_label_pos;
		obj_file_stream.seekg(curr_label_pos);
		obj_file_stream.ignore(instances * LABEL_INSTANCE_LENGTH);
		std::size_t skipped_labels = std::min(instances, label_skip_extent / LABEL_INSTANCE_LENGTH);
		assert(skipped_obs == skipped_labels);
		return skipped_obs;
	}
private:
	const Dimensions<std::size_t,3> obs_dims;
	const Dimensions<std::size_t,3> obj_dims;
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
 *
 * \see https://www.cs.toronto.edu/~kriz/cifar.html
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
			obs_dims({ 32u, 32u, 3u }),
			obj_dims({ NUM_LABELS, 1u, 1u }),
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
		return obs_dims;
	}
	inline const Dimensions<std::size_t,3>& get_obj_dims() const {
		return obj_dims;
	}
protected:
	inline DataPair<Scalar,3,false> _get_data(InputFile& file, std::size_t batch_size) {
		Tensor<Scalar,4> obs(batch_size, 32u, 32u, 3u);
		Tensor<Scalar,4> obj(batch_size, NUM_LABELS, 1u, 1u);
		obj.setZero();
		std::size_t i;
		for (i = 0; i < batch_size && file.second.read(buffer, INSTANCE_LENGTH); ++i) {
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
	inline std::size_t _skip(std::ifstream& file_stream, std::size_t instances) {
		std::streampos curr_pos = file_stream.tellg();
		file_stream.seekg(0, std::ios::end);
		std::size_t skip_extent = file_stream.tellg() - curr_pos;
		file_stream.seekg(curr_pos);
		file_stream.ignore(instances * INSTANCE_LENGTH);
		return std::min(instances, skip_extent / INSTANCE_LENGTH);
	}
private:
	const Dimensions<std::size_t,3> obs_dims;
	const Dimensions<std::size_t,3> obj_dims;
	char buffer[INSTANCE_LENGTH];
	std::array<std::size_t,4> offsets;
	std::array<std::size_t,4> obs_extents;
	std::array<std::size_t,4> obj_extents;
};

///**
// * An alias for a read-only dictionary mapping words to indices.
// */
//typedef std::shared_ptr<const std::map<std::string,std::size_t>> VocabSharedPtr;
//
///**
// * An enumeration for the different objective types to use for the IMDB data set.
// */
//enum IMDBObjType { BINARY, SMOOTH, CATEGORICAL };
//
///**
// * A data provider template for the IMDB Large Movie Review Dataset.
// *
// * \see http://ai.stanford.edu/~amaas/data/sentiment/
// */
//template<typename Scalar, IMDBObjType ObjType>
//class IMDBDataProvider : JointFileDataProvider<Scalar,1,true> {
//	typedef JointFileDataProvider<Scalar,1,true> Base;
//	static_assert(ObjType >= BINARY && ObjType <= CATEGORICAL, "invalid IMDB objective type");
//public:
//	/**
//	 * @param file_paths The paths to the data set files.
//	 */
//	inline IMDBDataProvider(std::string pos_reviews_folder_path, std::string neg_reviews_folder_path,
//			VocabSharedPtr vocab, std::size_t seq_length = 100) :
//				Base::JointFileDataProvider(resolve_review_files(pos_reviews_folder_path,
//						neg_reviews_folder_path)),
//				vocab(vocab),
//				seq_length(seq_length) {
//		assert(vocab);
//		obs_dims = Dimensions<std::size_t,1>({ vocab->size() });
//		obj_dims = Dimensions<std::size_t,1>({ CATEGORICAL ? 10 : 1 });
//		Base::reset();
//	}
//	inline const Dimensions<std::size_t,1>& get_obs_dims() const {
//		return obs_dims;
//	}
//	inline const Dimensions<std::size_t,1>& get_obj_dims() const {
//		return obj_dims;
//	}
//	/**
//	 *
//	 * @param vocab_path
//	 * @return
//	 */
//	inline static VocabSharedPtr build_vocab(std::string vocab_path) {
//		std::ifstream vocab_stream(vocab_path);
//		assert(vocab_stream.is_open());
//		std::map<std::string,std::size_t> vocab;
//		// Reserve the first index for padding.
//		std::size_t index = 1;
//		std::string word;
//		while (std::getline(vocab_stream, word))
//			vocab.emplace(std::make_pair(word, index++));
//		return std::make_shared<const std::map<std::string,std::size_t>>(std::move(vocab));
//	}
//protected:
//	/**
//	 *
//	 * @param dir_path
//	 * @param file_names
//	 */
//	inline static void read_files_in_dir(std::string dir_path, std::vector<std::string>& file_names) {
//		auto dir_ptr = opendir(dir_path.c_str());
//		struct dirent* dir_ent_ptr;
//		while ((dir_ent_ptr = readdir(dir_ptr)))
//			file_names.push_back(dir_path + std::string(dir_ent_ptr->d_name));
//		closedir(dir_ptr);
//	}
//	/**
//	 *
//	 * @param pos_reviews_folder_path
//	 * @param neg_reviews_folder_path
//	 * @return
//	 */
//	inline static std::vector<std::string> resolve_review_files(std::string pos_reviews_folder_path,
//			std::string neg_reviews_folder_path) {
//		std::vector<std::string> file_names;
//		read_files_in_dir(pos_reviews_folder_path, file_names);
//		read_files_in_dir(neg_reviews_folder_path, file_names);
//		std::random_shuffle(file_names.begin(), file_names.end());
//		return file_names;
//	}
//	inline DataPair<Scalar,1,true> _get_data(InputFile& file, std::size_t batch_size) {
//		TensorPtr<Scalar,3> obs_ptr(new Tensor<Scalar,3>(1, seq_length, obs_dims(0u)));
//		TensorPtr<Scalar,3> obj_ptr(new Tensor<Scalar,3>(1, 1, obj_dims(0u)));
//		obs_ptr->setZero();
//		std::size_t last_under_score = file.first.find_last_of('_');
//		std::size_t last_period = file.first.find_last_of('.');
//		std::size_t rating_string = file.first.substr(last_under_score + 1,
//				last_period - last_under_score - 1);
//		unsigned rating = (unsigned) std::stoi(rating_string);
//		switch (ObjType) {
//			case BINARY:
//				obj_ptr(0u,0u,0u) = (Scalar) (rating > 5);
//				break;
//			case SMOOTH:
//				obj_ptr(0u,0u,0u) = ((Scalar) (rating - 1)) / 9;
//				break;
//			case CATEGORICAL:
//				obj_ptr.setZero();
//				obj_ptr(0u,0u,rating) = (Scalar) 1;
//				break;
//			default:
//				assert(false);
//		}
//		std::size_t time_step = 0;
//		std::string word;
//		while (file.second >> word && time_step < seq_length) {
//			std::transform(word.begin(), word.end(), word.begin(), std::tolower);
//			auto val = vocab->find(word);
//			if (val != vocab->end()) {
//
//			}
//			time_step++;
//		}
//	}
//	inline std::size_t _skip(std::ifstream& file_stream, std::size_t instances) {
//		file_stream.seekg(0, std::ios::end);
//		return instances - 1;
//	}
//private:
//	const VocabSharedPtr vocab;
//	const std::size_t seq_length;
//	const Dimensions<std::size_t,1> obs_dims;
//	const Dimensions<std::size_t,1> obj_dims;
//};

} /* namespace cattle */

#endif /* CATTL3_DATAPROVIDER_H_ */
