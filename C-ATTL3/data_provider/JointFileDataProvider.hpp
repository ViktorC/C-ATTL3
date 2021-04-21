/*
 * JointFileDataProvider.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_DATA_PROVIDER_JOINTFILEDATAPROVIDER_H_
#define C_ATTL3_DATA_PROVIDER_JOINTFILEDATAPROVIDER_H_

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/DataProvider.hpp"

namespace cattle {

/**
 * An abstract class template for a data provider backed by data on disk in the
 * form of an arbitrary number of files containing both the observations and the
 * objectives. Implementations are responsible for specifying the dimensions of
 * both the observations and the objectives, for reading batches of
 * observation-objective pairs from the file, and for skipping arbitrary number
 * of data instances.
 */
template <typename Scalar, std::size_t Rank, bool Sequential, bool Binary = false>
class JointFileDataProvider : public DataProvider<Scalar, Rank, Sequential> {
  typedef DataProvider<Scalar, Rank, Sequential> Base;

 public:
  virtual ~JointFileDataProvider() = default;
  inline const typename Base::Dims& get_obs_dims() const { return obs_dims; }
  inline const typename Base::Dims& get_obj_dims() const { return obj_dims; }
  inline bool has_more() {
    if (current_file_stream_has_more()) return true;
    ++current_file_ind;
    for (; current_file_ind < files.size(); ++current_file_ind) {
      init_current_file_stream();
      if (current_file_stream_has_more()) return true;
    }
    return false;
  }
  inline DataPair<Scalar, Rank, Sequential> get_data(std::size_t batch_size) {
    if (!has_more()) throw std::out_of_range("no more data left to fetch");
    DataPair<Scalar, Rank, Sequential> data_pair = _get_data(files[current_file_ind], current_file_stream, batch_size);
    assert(data_pair.first.dimension(0) == data_pair.second.dimension(0));
    /* If the data contains fewer batches than expected, the end of the file has
     * been reached and the
     * rest of the data should be read from the next file. */
    while (data_pair.first.dimension(0) < batch_size && has_more()) {
      DataPair<Scalar, Rank, Sequential> add_data_pair =
          _get_data(files[current_file_ind], current_file_stream, batch_size - data_pair.first.dimension(0));
      assert(add_data_pair.first.dimension(0) == add_data_pair.second.dimension(0));
      // It has to be evaluated into a temporary due to the dimension
      // incompatibility.
      typename Base::Data obs_concat = data_pair.first.concatenate(std::move(add_data_pair.first), 0);
      data_pair.first = std::move(obs_concat);
      typename Base::Data obj_concat = data_pair.second.concatenate(std::move(add_data_pair.second), 0);
      data_pair.second = std::move(obj_concat);
    }
    return data_pair;
  }
  inline void reset() {
    current_file_ind = 0;
    init_current_file_stream();
    _set_to_beg(current_file_stream);
  }
  inline void skip(std::size_t instances) {
    if (!has_more()) return;
    std::size_t skipped = _skip(current_file_stream, instances);
    while (skipped < instances && has_more()) skipped += _skip(current_file_stream, instances - skipped);
  }

 protected:
  inline JointFileDataProvider(const typename Base::Dims& obs_dims, const typename Base::Dims& obj_dims,
                               const std::vector<std::string>& dataset_paths)
      : obs_dims(obs_dims), obj_dims(obj_dims), files(dataset_paths), current_file_ind(0) {
    assert(!files.empty());
    init_current_file_stream();
  }
  inline JointFileDataProvider(const typename Base::Dims& obs_dims, const typename Base::Dims& obj_dims,
                               std::string dataset_path)
      : JointFileDataProvider(obs_dims, obj_dims, {dataset_path}) {}
  /**
   * It sets the position of the file stream to the beginning of the data set.
   *
   * @param file_stream A reference to the file stream of the data set.
   */
  virtual inline void _set_to_beg(std::ifstream& file_stream) { file_stream.seekg(0, std::ios::beg); }
  /**
   * It reads at most the specified number of observation-objective pairs from
   * the provided file stream. The file stream can be expected not to have any
   * of its fail flags set initially and to have at least 1 more character left
   * to read.
   *
   * @param file_name The name of the data source file.
   * @param file_stream The input stream of the file.
   * @param batch_size The number of data points to return.
   * @return A pair of tensors containing the data batch.
   */
  virtual DataPair<Scalar, Rank, Sequential> _get_data(const std::string& file_name, std::ifstream& file_stream,
                                                       std::size_t batch_size) = 0;
  /**
   * Skips at most the specified number of instances in the data stream. The
   * file stream can be expected not to have any of its fail flags set
   * initially.
   *
   * @param file_stream A reference to the file stream of the data set.
   * @param instances The number of instances to skip.
   * @return The number of instances actually skipped. It may be less than the
   * specified amount if there are fewer remaining instances in the data stream.
   */
  virtual std::size_t _skip(std::ifstream& file_stream, std::size_t instances) = 0;
  const typename Base::Dims obj_dims, obs_dims;

 private:
  bool current_file_stream_has_more() { return current_file_stream && current_file_stream.peek() != EOF; }
  void init_current_file_stream() {
    current_file_stream = std::ifstream(files[current_file_ind], Binary ? std::ios::binary : std::ios::in);
    assert(current_file_stream.is_open());
  }
  std::vector<std::string> files;
  std::size_t current_file_ind;
  std::ifstream current_file_stream;
};

}  // namespace cattle

#endif /* C_ATTL3_DATA_PROVIDER_JOINTFILEDATAPROVIDER_H_ */
