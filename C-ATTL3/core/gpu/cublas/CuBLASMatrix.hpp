/*
 * CuBLASMatrix.hpp
 *
 *  Created on: 12 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_GPU_CUBLAS_CUBLASMATRIX_HPP_
#define C_ATTL3_CORE_GPU_CUBLAS_CUBLASMATRIX_HPP_

#include <cublas_v2.h>

#include <cassert>
#include <type_traits>

#include "CuBLASHandle.hpp"
#include "core/EigenProxy.hpp"
#include "core/gpu/cuda/CUDAArray.hpp"

namespace cattle {
namespace gpu {

namespace {

template <typename Scalar>
using AsumRoutine = cublasStatus_t (*)(cublasHandle_t, int, const Scalar*, int, Scalar*);
template <typename Scalar>
__inline__ AsumRoutine<Scalar> asum_routine() {
  return &cublasDasum;
}
template <>
__inline__ AsumRoutine<float> asum_routine() {
  return &cublasSasum;
}

template <typename Scalar>
using Nrm2Routine = cublasStatus_t (*)(cublasHandle_t, int, const Scalar*, int, Scalar*);
template <typename Scalar>
__inline__ Nrm2Routine<Scalar> nrm2_routine() {
  return &cublasDnrm2;
}
template <>
__inline__ Nrm2Routine<float> nrm2_routine() {
  return &cublasSnrm2;
}

template <typename Scalar>
using ScalRoutine = cublasStatus_t (*)(cublasHandle_t, int, const Scalar*, Scalar*, int);
template <typename Scalar>
__inline__ ScalRoutine<Scalar> scal_routine() {
  return &cublasDscal;
}
template <>
__inline__ ScalRoutine<float> scal_routine() {
  return &cublasSscal;
}

template <typename Scalar>
using AxpyRoutine = cublasStatus_t (*)(cublasHandle_t, int, const Scalar*, const Scalar*, int, Scalar*, int);
template <typename Scalar>
__inline__ AxpyRoutine<Scalar> axpy_routine() {
  return &cublasDaxpy;
}
template <>
__inline__ AxpyRoutine<float> axpy_routine() {
  return &cublasSaxpy;
}

template <typename Scalar>
using GemmRoutine = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                       const Scalar*, const Scalar*, int, const Scalar*, int, const Scalar*, Scalar*,
                                       int);
template <typename Scalar>
__inline__ GemmRoutine<Scalar> gemm_routine() {
  return &cublasDgemm;
}
template <>
__inline__ GemmRoutine<float> gemm_routine() {
  return &cublasSgemm;
}

}  // namespace

/**
 * A template class for column-major cuBLAS device matrices of different data
 * types.
 */
template <typename Scalar>
class CuBLASMatrix : public CUDAArray<Scalar> {
  static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
  typedef CUDAArray<Scalar> Base;
  typedef CuBLASMatrix<Scalar> Self;

 public:
  /**
   * @param data The device array to wrap in a CuBLASMatrix. The ownership of
   * the pointer is not transfered to the matrix.
   * @param rows The number of rows of the matrix.
   * @param cols The number of columns of the matrix.
   */
  inline CuBLASMatrix(Scalar* data, std::size_t rows, std::size_t cols)
      : Base(data, rows * cols), _rows(rows), _cols(cols) {}
  /**
   * @param rows The number of rows of the matrix.
   * @param cols The number of columns of the matrix.
   */
  inline CuBLASMatrix(std::size_t rows, std::size_t cols) : Base(rows * cols), _rows(rows), _cols(cols) {}
  inline CuBLASMatrix() : CuBLASMatrix(0u, 0u) {}
  inline CuBLASMatrix(const Matrix<Scalar>& matrix) : CuBLASMatrix(matrix.rows(), matrix.cols()) {
    if (Base::size() > 0) Base::copy_from_host(matrix.data());
  }
  inline CuBLASMatrix(const Self& matrix) : Base(matrix), _rows(matrix._rows), _cols(matrix._cols) {}
  inline CuBLASMatrix(Self&& matrix) : CuBLASMatrix() { swap(*this, matrix); }
  virtual ~CuBLASMatrix() = default;
  inline Self& operator=(Self matrix) {
    swap(*this, matrix);
    return *this;
  }
  inline operator Matrix<Scalar>() const {
    if (Base::size() == 0) return Matrix<Scalar>();
    Matrix<Scalar> out(_rows, _cols);
    Base::copy_to_host(out.data());
    return out;
  }
  /**
   * @return The number of rows of the matrix.
   */
  inline std::size_t rows() const { return _rows; }
  /**
   * @return The number of columns of the matrix.
   */
  inline std::size_t cols() const { return _cols; }
  /**
   * @return The L1 norm of the matrix.
   */
  inline Scalar l1_norm() const {
    Scalar res;
    asum(Base::size(), 1, *this, res);
    return res;
  }
  /**
   * @return The L2 (Euclidian) norm of the matrix.
   */
  inline Scalar l2_norm() const {
    Scalar res;
    nrm2(Base::size(), 1, *this, res);
    return res;
  }
  inline Self& operator+=(const Self& rhs) {
    axpy(Base::size(), 1, rhs, 1, 1, *this);
    return *this;
  }
  inline Self& operator-=(const Self& rhs) {
    axpy(Base::size(), 1, rhs, -1, 1, *this);
    return *this;
  }
  inline Self& operator*=(const Self& rhs) {
    gemm(*this, false, rhs, false, 1, 1, *this);
    return *this;
  }
  inline Self& operator*=(Scalar rhs) {
    scal(Base::size(), rhs, 1, *this);
    return *this;
  }
  inline Self& operator/=(Scalar rhs) {
    scal(Base::size(), rhs, 1 / rhs, *this);
    return *this;
  }
  inline friend Self operator+(Self lhs, const Self& rhs) { return lhs += rhs; }
  inline friend Self operator-(Self lhs, const Self& rhs) { return lhs -= rhs; }
  inline friend Self operator*(Self lhs, const Self& rhs) { return lhs *= rhs; }
  inline friend Self operator*(Self lhs, Scalar rhs) { return lhs *= rhs; }
  inline friend Self operator/(Self lhs, Scalar rhs) { return lhs /= rhs; }
  inline friend void swap(Self& matrix1, Self& matrix2) {
    using std::swap;
    swap(static_cast<Base&>(matrix1), static_cast<Base&>(matrix2));
    swap(matrix1._rows, matrix2._rows);
    swap(matrix1._cols, matrix2._cols);
  }
  /**
   * It computes the sum of the absolute values of the matrix's coefficients.
   *
   * \f$R = \sum\limits_{i = 1}^n \left|A_i\right|\f$
   *
   * @param n The number of elements whose absolute value is to be summed up.
   * @param inc_a The stride between elements of the matrix.
   * @param a The matrix.
   * @param result The result of the computation.
   */
  inline static void asum(int n, int inc_a, const CuBLASMatrix<Scalar>& a,
                          /* out */ Scalar& result) {
    AsumRoutine<Scalar> asum = asum_routine<Scalar>();
    cublasAssert(asum(CuBLASHandle::get_instance(), n, a.data(), inc_a, &result));
  }
  /**
   * It computes the Euclidian norm of the matrix's coefficients.
   *
   * \f$R = \sqrt{\sum\limits_{i = 1}^n A_i^2}\f$
   *
   * @param n The number of elements whose second norm is to be calculated.
   * @param inc_a The stride between elements of the matrix.
   * @param a The matrix.
   * @param result The result of the computation.
   */
  inline static void nrm2(int n, int inc_a, const CuBLASMatrix<Scalar>& a,
                          /* out */ Scalar& result) {
    Nrm2Routine<Scalar> nrm2 = nrm2_routine<Scalar>();
    cublasAssert(nrm2(CuBLASHandle::get_instance(), n, a.data(), inc_a, &result));
  }
  /**
   * It scales the matrix by the specified factor.
   *
   * \f$A_i = \alpha * A_i, \forall i \in \{0,...,n\}\f$
   *
   * @param n The number of elements on which the operation is to be performed.
   * @param alpha The scaling factor.
   * @param inc_a The stride between elements of the matrix.
   * @param a The matrix to be scaled.
   */
  inline static void scal(int n, Scalar alpha, int inc_a,
                          /* in/out */ CuBLASMatrix<Scalar>& a) {
    ScalRoutine<Scalar> scal = scal_routine<Scalar>();
    cublasAssert(scal(CuBLASHandle::get_instance(), n, &alpha, a.data(), inc_a));
  }
  /**
   * It adds a scaled matrix to another matrix.
   *
   * \f$B_i = \alpha * A_i + B_i, \forall i \in \{0,...,n\}\f$
   *
   * @param n The number of elements on which the operation is to be performed.
   * @param inc_a The stride between elements of the scaled matrix.
   * @param a The matrix to scale and add to the other matrix.
   * @param alpha The scaling factor.
   * @param inc_b The stride between elements of the target matrix.
   * @param b The traget matrix.
   */
  inline static void axpy(int n, int inc_a, const CuBLASMatrix<Scalar>& a, Scalar alpha, int inc_b,
                          /* in/out */ CuBLASMatrix<Scalar>& b) {
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    AxpyRoutine<Scalar> axpy = axpy_routine<Scalar>();
    cublasAssert(axpy(CuBLASHandle::get_instance(), n, &alpha, a.data(), inc_a, b.data(), inc_b));
  }
  /**
   * It computes the product of the matrix multiplication.
   *
   * \f$C = \alpha * (A \textrm{ x } B) + \beta * C\f$
   *
   * @param a The multiplicand matrix.
   * @param transpose_a Whether the multiplicand is to be transposed for the
   * operation.
   * @param b The multiplier matrix.
   * @param transpose_b Whether the multiplier is to be transposed for the
   * operation.
   * @param alpha The factor by which the result of the multiplication is to be
   * scaled.
   * @param beta The factor by which c is to be scaled before the result of the
   * multiplication is added to it.
   * @param c The product of the matrix multiplication.
   */
  inline static void gemm(const CuBLASMatrix<Scalar>& a, bool transpose_a, const CuBLASMatrix<Scalar>& b,
                          bool transpose_b, Scalar alpha, Scalar beta,
                          /* out */ CuBLASMatrix<Scalar>& c) {
    std::size_t a_rows, a_cols, b_rows, b_cols;
    if (transpose_a) {
      a_rows = a.cols();
      a_cols = a.rows();
    } else {
      a_rows = a.rows();
      a_cols = a.cols();
    }
    if (transpose_b) {
      b_rows = b.cols();
      b_cols = b.rows();
    } else {
      b_rows = b.rows();
      b_cols = b.cols();
    }
    assert(a_cols == b_rows);
    assert(c.rows() == a_rows() && c.cols() == b_cols);
    cublasOperation_t a_op = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t b_op = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    // Resolve the GEMM precision based on the scalar type.
    GemmRoutine<Scalar> gemm = gemm_routine<Scalar>();
    // Perform the matrix multiplication.
    cublasAssert(gemm(CuBLASHandle::get_instance(), a_op, b_op, a_rows, b_cols, a_cols, &alpha, a.data(), a.rows(),
                      b.data(), b.rows(), &beta, c.data(), a_rows));
  }

 private:
  std::size_t _rows, _cols;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_CORE_GPU_CUBLAS_CUBLASMATRIX_HPP_ */
