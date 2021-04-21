/*
 * CuDNNTensor.hpp
 *
 *  Created on: 8 Jul 2018
 *      Author: Viktor Csomor
 */

#include <cudnn.h>

#include <cmath>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "CuDNNHandle.hpp"
#include "core/EigenProxy.hpp"
#include "core/gpu/cuda/CUDAArray.hpp"

#ifndef C_ATTL3_CORE_GPU_CUDNN_CUDNNTENSOR_H_
#define C_ATTL3_CORE_GPU_CUDNN_CUDNNTENSOR_H_

namespace cattle {
namespace gpu {

/**
 * A template class for representing row-major cuDNN device tensors of different
 * data types.
 */
template <typename Scalar>
class CuDNNTensor : public CUDAArray<Scalar> {
  static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
  typedef CUDAArray<Scalar> Base;
  typedef CuDNNTensor<Scalar> Self;

 public:
  static constexpr cudnnDataType_t DATA_TYPE =
      std::is_same<Scalar, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;
  static constexpr cudnnTensorFormat_t TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
  static constexpr cudnnNanPropagation_t NAN_PROP = CUDNN_PROPAGATE_NAN;
  /**
   * @param data The device array to wrap in a CuDNNTensor. The ownership of the
   * pointer is not transfered to the tensor.
   * @param samples The batch size.
   * @param height The height.
   * @param width The width.
   * @param channels The number of channels.
   */
  inline CuDNNTensor(Scalar* data, std::size_t samples, std::size_t height, std::size_t width, std::size_t channels)
      : Base(data, samples * height * width * channels),
        _samples(samples),
        _height(height),
        _width(width),
        _channels(channels),
        _desc(),
        _filter_desc() {
    if (Base::size() > 0) {
      create_tensor_descriptor(_desc, samples, height, width, channels);
      create_filter_descriptor(_filter_desc, samples, height, width, channels);
    }
  }
  /**
   * @param samples The batch size.
   * @param height The height.
   * @param width The width.
   * @param channels The number of channels.
   */
  inline CuDNNTensor(std::size_t samples, std::size_t height, std::size_t width, std::size_t channels)
      : Base(samples * height * width * channels),
        _samples(samples),
        _height(height),
        _width(width),
        _channels(channels),
        _desc(),
        _filter_desc() {
    if (Base::size() > 0) {
      create_tensor_descriptor(_desc, samples, height, width, channels);
      create_filter_descriptor(_filter_desc, samples, height, width, channels);
    }
  }
  inline CuDNNTensor() : CuDNNTensor(0u, 0u, 0u, 0u) {}
  inline CuDNNTensor(const Tensor<Scalar, 4>& tensor)
      : CuDNNTensor(tensor.dimension(0), tensor.dimension(1), tensor.dimension(2), tensor.dimension(3)) {
    if (Base::size() > 0) {
      static std::array<std::size_t, 4> eigen_to_cudnn_layout({2u, 1u, 2u, 0u});
      Tensor<Scalar, 4> shuffled_tensor = tensor.shuffle(eigen_to_cudnn_layout);
      Base::copy_from_host(shuffled_tensor.data());
    }
  }
  inline CuDNNTensor(const Self& tensor)
      : Base(tensor),
        _samples(tensor._samples),
        _height(tensor._height),
        _width(tensor._width),
        _channels(tensor._channels),
        _desc(tensor._desc),
        _filter_desc(tensor._filter_desc) {}
  inline CuDNNTensor(Self&& tensor) : CuDNNTensor() { swap(*this, tensor); }
  inline ~CuDNNTensor() {
    if (Base::size() > 0) {
      destroy_tensor_descriptor(_desc);
      destroy_filter_descriptor(_filter_desc);
    }
  }
  inline Self& operator=(Self tensor) {
    swap(*this, tensor);
    return *this;
  }
  inline operator Tensor<Scalar, 4>() const {
    if (Base::size() == 0) return Tensor<Scalar, 4>();
    Tensor<Scalar, 4> out(_width, _height, _channels, _samples);
    Base::copy_to_host(out.data());
    static std::array<std::size_t, 4> cudnn_to_eigen_layout({3u, 1u, 0u, 2u});
    return out.shuffle(cudnn_to_eigen_layout);
  }
  /**
   * @return The batch size of the tensor.
   */
  inline std::size_t samples() const { return _samples; }
  /**
   * @return The height of the tensor.
   */
  inline std::size_t height() const { return _height; }
  /**
   * @return The width of the tensor.
   */
  inline std::size_t width() const { return _width; }
  /**
   * @return The number of channels of the tensor.
   */
  inline std::size_t channels() const { return _channels; }
  /**
   * @return A constant reference to the tensor descriptor.
   */
  inline const cudnnTensorDescriptor_t& desc() const { return _desc; }
  /**
   * @return A constant reference to the filter descriptor.
   */
  inline const cudnnFilterDescriptor_t& filter_desc() const { return _filter_desc; }
  /**
   * @param value The value to which all elements of the tensor are to be set.
   */
  inline void set_values(Scalar value) {
    cudnnAssert(cudnnSetTensor(CuDNNHandle::get_instance(), _desc, Base::data(), value));
  }
  /**
   * Performs a reduction along all ranks of the tensor.
   *
   * @param op_type The reduction operation type.
   * @return The result of the reduction.
   */
  inline Scalar reduce(cudnnReduceTensorOp_t op_type) const {
    Self reduced_tensor(1u, 1u, 1u, 1u);
    reduce_op(1, *this, op_type, 0, reduced_tensor);
    Scalar res;
    reduced_tensor.copy_to_host(&res);
    return res;
  }
  /**
   * Performs a reduction along the specified ranks of the tensor.
   *
   * @param op_type The reduction operation type.
   * @param ranks A size 4 boolean array denoting which ranks are to be reduced.
   * @return The result of the reduction.
   */
  inline Self reduce(cudnnReduceTensorOp_t op_type, const std::array<bool, 4>& ranks) const {
    Self reduced_tensor(ranks[0] ? 1u : _samples, ranks[1] ? 1u : _height, ranks[2] ? 1u : _width,
                        ranks[3] ? 1u : _channels);
    reduce_op(1, *this, op_type, 0, reduced_tensor);
    return reduced_tensor;
  }
  /**
   * @return The sum of all elements of the tensor.
   */
  inline Scalar sum() const { return reduce(CUDNN_REDUCE_TENSOR_ADD); }
  /**
   * @param ranks A size 4 boolean array denoting which ranks are to be reduced.
   * @return The reduced tensor.
   */
  inline Self sum(const std::array<bool, 4>& ranks) const { return reduce(CUDNN_REDUCE_TENSOR_ADD, ranks); }
  /**
   * @return The mean of all elements of the tensor.
   */
  inline Scalar avg() const { return reduce(CUDNN_REDUCE_TENSOR_AVG); }
  /**
   * @param ranks A size 4 boolean array denoting which ranks are to be reduced.
   * @return The reduced tensor.
   */
  inline Self avg(const std::array<bool, 4>& ranks) const { return reduce(CUDNN_REDUCE_TENSOR_AVG, ranks); }
  /**
   * @return The minimum of all elements of the tensor.
   */
  inline Scalar min() const { return reduce(CUDNN_REDUCE_TENSOR_MIN); }
  /**
   * @param ranks A size 4 boolean array denoting which ranks are to be reduced.
   * @return The reduced tensor.
   */
  inline Self min(const std::array<bool, 4>& ranks) const { return reduce(CUDNN_REDUCE_TENSOR_MIN, ranks); }
  /**
   * @return The maximum of all elements of the tensor.
   */
  inline Scalar max() const { return reduce(CUDNN_REDUCE_TENSOR_MAX); }
  /**
   * @param ranks A size 4 boolean array denoting which ranks are to be reduced.
   * @return The reduced tensor.
   */
  inline Self max(const std::array<bool, 4>& ranks) const { return reduce(CUDNN_REDUCE_TENSOR_MAX, ranks); }
  /**
   * @return The absolute maximum of all elements of the tensor.
   */
  inline Scalar abs_max() const { return reduce(CUDNN_REDUCE_TENSOR_AMAX); }
  /**
   * @param ranks A size 4 boolean array denoting which ranks are to be reduced.
   * @return The reduced tensor.
   */
  inline Self abs_max(const std::array<bool, 4>& ranks) const { return reduce(CUDNN_REDUCE_TENSOR_AMAX, ranks); }
  /**
   * @return The L1 norm of the tensor.
   */
  inline Scalar l1_norm() const { return reduce(CUDNN_REDUCE_TENSOR_NORM1); }
  /**
   * @param ranks A size 4 boolean array denoting which ranks are to be reduced.
   * @return The reduced tensor.
   */
  inline Self l1_norm(const std::array<bool, 4>& ranks) const { return reduce(CUDNN_REDUCE_TENSOR_NORM1, ranks); }
  /**
   * @return The L2 norm of the tensor.
   */
  inline Scalar l2_norm() const { return reduce(CUDNN_REDUCE_TENSOR_NORM2); }
  /**
   * @param ranks A size 4 boolean array denoting which ranks are to be reduced.
   * @return The reduced tensor.
   */
  inline Self l2_norm(const std::array<bool, 4>& ranks) const { return reduce(CUDNN_REDUCE_TENSOR_NORM2, ranks); }
  /**
   * @return A string representation of the tensor.
   */
  std::string to_string() const {
    std::stringstream strm;
    strm << "data type: " << DATA_TYPE << "; format: " << TENSOR_FORMAT << "; "
         << "[N:" << _samples << ", C:" << _channels << ", H:" << _height << ", W:" << _width << "]";
    return strm.str();
  }
  inline Self& operator+=(const Self& rhs) {
    add(1, rhs, 1, *this);
    return *this;
  }
  inline Self& operator-=(const Self& rhs) {
    add(-1, rhs, 1, *this);
    return *this;
  }
  inline Self& operator*=(const Self& rhs) {
    op(*this, 1, rhs, 1, CUDNN_OP_TENSOR_MUL, 1, *this);
    return *this;
  }
  inline Self& operator+=(Scalar rhs) {
    Self rhs_tensor(1u, 1u, 1u, 1u);
    rhs_tensor.copy_from_host(&rhs);
    op(*this, 1, rhs_tensor, 1, CUDNN_OP_TENSOR_ADD, 1, *this);
    return *this;
  }
  inline Self& operator-=(Scalar rhs) { return *this += -rhs; }
  inline Self& operator*=(Scalar rhs) {
    scale(rhs, *this);
    return *this;
  }
  inline Self& operator/=(Scalar rhs) { return *this *= (1 / rhs); }
  inline friend Self operator+(Self lhs, const Self& rhs) { return lhs += rhs; }
  inline friend Self operator-(Self lhs, const Self& rhs) { return lhs -= rhs; }
  inline friend Self operator*(Self lhs, const Self& rhs) { return lhs *= rhs; }
  inline friend Self operator+(Self lhs, Scalar rhs) { return lhs += rhs; }
  inline friend Self operator-(Self lhs, Scalar rhs) { return lhs -= rhs; }
  inline friend Self operator*(Self lhs, Scalar rhs) { return lhs *= rhs; }
  inline friend Self operator/(Self lhs, Scalar rhs) { return lhs /= rhs; }
  inline friend std::ostream& operator<<(std::ostream& os, const Self& tensor) {
    return os << tensor.to_string() << std::endl;
  }
  inline friend void swap(Self& tensor1, Self& tensor2) {
    using std::swap;
    swap(static_cast<Base&>(tensor1), static_cast<Base&>(tensor2));
    swap(tensor1._samples, tensor2._samples);
    swap(tensor1._height, tensor2._height);
    swap(tensor1._width, tensor2._width);
    swap(tensor1._channels, tensor2._channels);
    swap(tensor1._filter, tensor2._filter);
    swap(tensor1._desc, tensor2._desc);
    swap(tensor1._filter_desc, tensor2._filter_desc);
  }
  /**
   * @param desc A reference to the tensor descriptor object.
   * @param samples The batch size.
   * @param height The height.
   * @param width The width.
   * @param channels The number of channels.
   */
  inline static void create_tensor_descriptor(cudnnTensorDescriptor_t& desc, std::size_t samples, std::size_t height,
                                              std::size_t width, std::size_t channels) {
    cudnnAssert(cudnnCreateTensorDescriptor(&desc));
    cudnnAssert(cudnnSetTensor4dDescriptor(desc, TENSOR_FORMAT, DATA_TYPE, samples, channels, height, width));
  }
  /**
   * @param desc A constant reference to the tensor descriptor object.
   */
  inline static void destroy_tensor_descriptor(const cudnnTensorDescriptor_t& desc) {
    cudnnAssert(cudnnDestroyTensorDescriptor(desc));
  }
  /**
   * @param filter_desc A reference to the filter descriptor object.
   * @param samples The batch size.
   * @param height The height.
   * @param width The width.
   * @param channels The number of channels.
   */
  inline static void create_filter_descriptor(cudnnFilterDescriptor_t& filter_desc, std::size_t samples,
                                              std::size_t height, std::size_t width, std::size_t channels) {
    cudnnAssert(cudnnCreateFilterDescriptor(&filter_desc));
    cudnnAssert(cudnnSetFilter4dDescriptor(filter_desc, DATA_TYPE, TENSOR_FORMAT, samples, channels, height, width));
  }
  /**
   * @param filter_desc A constant reference to the filter descriptor object.
   */
  inline static void destroy_filter_descriptor(const cudnnFilterDescriptor_t& filter_desc) {
    cudnnAssert(cudnnDestroyFilterDescriptor(filter_desc));
  }
  /**
   * It scales the specified tensor by a certain factor.
   *
   * \f$A = \alpha * B\f$
   *
   * @param alpha The factor by which the tensor is to be scaled.
   * @param a The tensor to scale.
   */
  inline static void scale(Scalar alpha, /* in/out */ Self& a) {
    cudnnAssert(cudnnScaleTensor(CuDNNHandle::get_instance(), a.desc(), a.data(), &alpha));
  }
  /**
   * It adds tensor a to tensor b.
   *
   * \f$B = \alpha * A + \beta * B\f$
   *
   * @param alpha The scaling factor of the tensor to add to the other one.
   * @param a The tensor to add to the other tensor.
   * @param beta The scaling factor of the target tensor.
   * @param b The target tensor.
   */
  inline static void add(Scalar alpha, const Self& a, Scalar beta,
                         /* in/out */ Self& b) {
    cudnnAssert(cudnnAddTensor(CuDNNHandle::get_instance(), &alpha, a.desc(), a.data(), &beta, b.desc(), b.data()));
  }
  /**
   * It performs the specified operation on tensors a and b and saves the result
   * in c.
   *
   * \f$C = op(\alpha * A, \beta * B) + \gamma * C\f$
   *
   * @param a The first operand.
   * @param alpha The scaling factor of the first operand.
   * @param b The second operand.
   * @param beta The scaling factor of the second operand.
   * @param op_type The operation type.
   * @param gamma The scaling factor of the result tensor.
   * @param c The result tensor.
   */
  inline static void op(const Self& a, Scalar alpha, const Self& b, Scalar beta, cudnnOpTensorOp_t op_type,
                        Scalar gamma,
                        /* in/out */ Self& c) {
    cudnnOpTensorDescriptor_t desc;
    cudnnAssert(cudnnCreateOpTensorDescriptor(&desc));
    cudnnAssert(cudnnSetOpTensorDescriptor(desc, op_type, DATA_TYPE, NAN_PROP));
    cudnnAssert(cudnnOpTensor(desc, alpha, a.desc(), a.data(), beta, b.desc(), b.data(), gamma, c.desc(), c.data()));
    cudnnAssert(cudnnDestroyOpTensorDescriptor(desc));
  }
  /**
   * It performs the specified reduction operation on tensor a and adds it to
   * tensor b.
   *
   * \f$B = \alpha * reduce_op(A) + \beta * B\f$
   *
   * @param alpha The factor by which the result of the reduction is to be
   * scaled.
   * @param a The tensor to reduce.
   * @param op_type The reduction operation type.
   * @param beta The scaling factor of the target tensor.
   * @param b The target tensor.
   */
  inline static void reduce_op(Scalar alpha, const Self& a, cudnnReduceTensorOp_t op_type, Scalar beta,
                               /* in/out */ Self& b) {
    // Create the reduction operation descriptor.
    cudnnReduceTensorDescriptor_t desc;
    cudnnAssert(cudnnCreateReduceTensorDescriptor(&desc));
    cudnnAssert(cudnnSetReduceTensorDescriptor(desc, op_type, DATA_TYPE, NAN_PROP, CUDNN_REDUCE_TENSOR_NO_INDICES,
                                               CUDNN_32BIT_INDICES));
    // Calculate the array size needed for the indices (should be 0).
    std::size_t indices_size;
    cudnnAssert(cudnnGetReductionIndicesSize(CuDNNHandle::get_instance(), desc, a.desc(), b.desc(), &indices_size));
    Base indices(static_cast<std::size_t>(ceil(static_cast<Scalar>(indices_size) / sizeof(Scalar))));
    // Calculate the workspace size.
    std::size_t workspace_size;
    cudnnAssert(cudnnGetReductionWorkspaceSize(CuDNNHandle::get_instance(), desc, a.desc(), b.desc(), &workspace_size));
    Base workspace(static_cast<std::size_t>(ceil(static_cast<Scalar>(workspace_size) / sizeof(Scalar))));
    // Perform the reduction.
    cudnnAssert(cudnnReduceTensor(CuDNNHandle::get_instance(), indices.data(), indices_size, workspace.data(),
                                  workspace_size, &alpha, a.desc(), a.data(), &beta, b.desc(), b.data()));
    // Free resources.
    cudnnAssert(cudnnDestroyReduceTensorDescriptor(desc));
  }

 private:
  std::size_t _samples, _height, _width, _channels;
  cudnnTensorDescriptor_t _desc;
  cudnnFilterDescriptor_t _filter_desc;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CUDNN_CUDNNTENSOR_H_ */
