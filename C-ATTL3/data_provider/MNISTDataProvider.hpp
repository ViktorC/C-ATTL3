/*
 * MNISTDataProvider.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_DATA_PROVIDER_MNISTDATAPROVIDER_H_
#define C_ATTL3_DATA_PROVIDER_MNISTDATAPROVIDER_H_

#include <algorithm>
#include <array>

#include "SplitFileDataProvider.hpp"

namespace cattle {

/**
 * A data provider template for the MNIST data set.
 *
 * \see http://yann.lecun.com/exdb/mnist/
 */
template <typename Scalar>
class MNISTDataProvider : public SplitFileDataProvider<Scalar, 3, false, true, true> {
  typedef DataProvider<Scalar, 3, false> Root;
  typedef SplitFileDataProvider<Scalar, 3, false, true, true> Base;
  typedef std::array<std::size_t, 4> RankwiseArray;
  static constexpr std::size_t OBS_OFFSET = 16;
  static constexpr std::size_t LABEL_OFFSET = 8;
  static constexpr std::size_t OBS_INSTANCE_LENGTH = 784;
  static constexpr std::size_t LABEL_INSTANCE_LENGTH = 1;

 public:
  /**
   * @param obs_path The path to the file containing the observations.
   * @param labels_path The path to the file containing the corresponding
   * labels.
   */
  MNISTDataProvider(std::string obs_path, std::string labels_path)
      : Base::SplitFileDataProvider({28u, 28u, 1u}, {10u, 1u, 1u}, std::make_pair(obs_path, labels_path)),
        offsets({0u, 0u, 0u, 0u}),
        obs_extents(Base::obs_dims.template promote<>()),
        obj_extents(Base::obs_dims.template promote<>()) {
    Base::reset();
  }

 protected:
  inline void _set_to_beg(std::ifstream& obs_file_stream, std::ifstream& obj_file_stream) {
    Base::_set_to_beg(obs_file_stream, obj_file_stream);
    obs_file_stream.ignore(OBS_OFFSET);
    obj_file_stream.ignore(LABEL_OFFSET);
  }
  inline DataPair<Scalar, 3, false> _get_data(const std::string& obs_file, std::ifstream& obs_file_stream,
                                              const std::string& obj_file, std::ifstream& obj_file_stream,
                                              std::size_t batch_size) {
    typename Root::Data obs(batch_size, Base::obs_dims(0), Base::obs_dims(1), Base::obs_dims(2));
    typename Root::Data obj(batch_size, Base::obj_dims(0), Base::obj_dims(1), Base::obj_dims(2));
    obj.setZero();
    std::size_t i;
    for (i = 0; i < batch_size && obs_file_stream.read(obs_buffer, OBS_INSTANCE_LENGTH); ++i) {
      // Read and set the label.
      char label;
      obj_file_stream.read(&label, LABEL_INSTANCE_LENGTH);
      obj(i, static_cast<std::size_t>(label), 0u, 0u) = (Scalar)1;
      // Set the image.
      unsigned char* u_buffer = reinterpret_cast<unsigned char*>(obs_buffer);
      std::size_t buffer_ind = 0;
      for (std::size_t height = 0; height < Base::obs_dims(0); ++height) {
        for (std::size_t width = 0; width < Base::obs_dims(1); ++width)
          obs(i, height, width, 0u) = (Scalar)u_buffer[buffer_ind++];
      }
      assert(buffer_ind == OBS_INSTANCE_LENGTH);
    }
    if (i == batch_size) return std::make_pair(obs, obj);
    obs_extents[0] = i;
    obj_extents[0] = i;
    return std::make_pair(obs.slice(offsets, obs_extents), obj.slice(offsets, obj_extents));
  }
  inline std::size_t _skip(std::ifstream& obs_file_stream, std::ifstream& obj_file_stream, std::size_t instances) {
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
  char obs_buffer[OBS_INSTANCE_LENGTH];
  RankwiseArray offsets, obs_extents, obj_extents;
};

}  // namespace cattle

#endif /* C_ATTL3_DATA_PROVIDER_MNISTDATAPROVIDER_H_ */
