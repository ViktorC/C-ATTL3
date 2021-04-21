/*
 * MemoryDataProvider.hpp
 *
 *  Created on: 20 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_DATA_PROVIDER_MEMORYDATAPROVIDER_H_
#define C_ATTL3_DATA_PROVIDER_MEMORYDATAPROVIDER_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <utility>

#include "core/DataProvider.hpp"

namespace cattle {

/**
 * An alias for a unique pointer to a tensor.
 */
template <typename Scalar, std::size_t Rank>
using TensorPtr = std::unique_ptr<Tensor<Scalar, Rank>>;

/**
 * A data provider that reads from the memory. It requires the entire
 * observation and objective data sets to be loaded into memory, but it fetches
 * pairs faster.
 */
template <typename Scalar, std::size_t Rank, bool Sequential, bool Shuffle = true>
class MemoryDataProvider : public DataProvider<Scalar, Rank, Sequential> {
  typedef DataProvider<Scalar, Rank, Sequential> Base;
  typedef TensorPtr<Scalar, Base::DATA_RANK> DataPtr;
  typedef std::array<std::size_t, Base::DATA_RANK> RankwiseArray;

 public:
  /**
   * It constructs a data provider backed by the specified tensors.
   *
   * @param obs A unique pointer to the tensor containing the observations.
   * @param obj A unique pointer to the tensor containing the objectives.
   * shuffled.
   */
  inline MemoryDataProvider(DataPtr obs, DataPtr obj) : obs(std::move(obs)), obj(std::move(obj)), offsets() {
    assert(this->obs != nullptr);
    assert(this->obj != nullptr);
    assert(this->obs->dimension(0) == this->obj->dimension(0) && "mismatched data and obj tensor row numbers");
    Dimensions<std::size_t, Base::DATA_RANK> obs_dims = this->obs->dimensions();
    Dimensions<std::size_t, Base::DATA_RANK> obj_dims = this->obj->dimensions();
    this->obs_dims = obs_dims.template demote<Sequential + 1>();
    this->obj_dims = obj_dims.template demote<Sequential + 1>();
    instances = (std::size_t)this->obs->dimension(0);
    offsets.fill(0);
    data_extents = obs_dims;
    obj_extents = obj_dims;
    if (Shuffle) shuffle_tensor_rows();
  }
  inline const typename Base::Dims& get_obs_dims() const { return obs_dims; }
  inline const typename Base::Dims& get_obj_dims() const { return obj_dims; }
  inline bool has_more() { return offsets[0] < (int)instances; }
  inline DataPair<Scalar, Rank, Sequential> get_data(std::size_t batch_size = std::numeric_limits<std::size_t>::max()) {
    if (!has_more()) throw std::out_of_range("no more data left to fetch");
    std::size_t max_batch_size = std::min(batch_size, instances - offsets[0]);
    data_extents[0] = max_batch_size;
    obj_extents[0] = max_batch_size;
    typename Base::Data data_batch = obs->slice(offsets, data_extents);
    typename Base::Data obj_batch = obj->slice(offsets, obj_extents);
    offsets[0] = std::min(instances, offsets[0] + max_batch_size);
    return std::make_pair(std::move(data_batch), std::move(obj_batch));
  }
  inline void reset() { offsets[0] = 0; }
  inline void skip(std::size_t instances) {
    offsets[0] = std::min((int)this->instances, (int)(offsets[0] + instances));
  }

 protected:
  inline void shuffle_tensor_rows() {
    std::size_t rows = obs->dimension(0);
    MatrixMap<Scalar> obs_mat(obs->data(), rows, obs->size() / rows);
    MatrixMap<Scalar> obj_mat(obj->data(), rows, obj->size() / rows);
    // Create an identity matrix.
    PermMatrix perm(rows);
    perm.setIdentity();
    // Shuffle its indices.
    std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
    // And apply the same row permutation transformation to both the
    // observations and the objectives.
    obs_mat = perm * obs_mat;
    obj_mat = perm * obj_mat;
  }

 private:
  DataPtr obs, obj;
  typename Base::Dims obs_dims, obj_dims;
  std::size_t instances;
  RankwiseArray offsets, data_extents, obj_extents;
};

} /* namespace cattle */

#endif /* C_ATTL3_DATA_PROVIDER_MEMORYDATAPROVIDER_H_ */
