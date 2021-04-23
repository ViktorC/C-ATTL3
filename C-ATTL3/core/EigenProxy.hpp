/*
 * EigenProxy.hpp
 *
 *  Created on: 12 Apr 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CORE_EIGENPROXY_H_
#define C_ATTL3_CORE_EIGENPROXY_H_

#define EIGEN_USE_THREADS

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <thread>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * The namespace containing all classes and typedefs of the C-ATTL3 library.
 */
namespace cattle {

/**
 * An alias for a single row matrix of an arbitrary scalar type.
 */
template <typename Scalar>
using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor, 1, Eigen::Dynamic>;

/**
 * An alias for a single column matrix of an arbitrary scalar type.
 */
template <typename Scalar>
using ColVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor, Eigen::Dynamic, 1>;

/**
 * An alias for a dynamically sized matrix of an arbitrary scalar type.
 */
template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * An alias for a class that can be used to map raw pointer data to a
 * dynamically sized Matrix of an arbitrary scalar type.
 */
template <typename Scalar>
using MatrixMap = Eigen::Map<Matrix<Scalar>>;

/**
 * An alias for a tensor of arbitrary rank and scalar type with dynamic
 * dimensionality.
 */
template <typename Scalar, std::size_t Rank>
using Tensor = Eigen::Tensor<Scalar, Rank, Eigen::ColMajor, std::size_t>;

/**
 * An alias for a class that can be used to map raw pointer data to a tensor of
 * arbitrary rank and scalar type with dynamic dimensionality.
 */
template <typename Scalar, std::size_t Rank>
using TensorMap = Eigen::TensorMap<Tensor<Scalar, Rank>>;

/**
 * An alias for permutation matrices.
 */
using PermMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

/**
 * An alias for self-adjoint eigen solvers.
 */
template <typename Scalar>
using EigenSolver = Eigen::SelfAdjointEigenSolver<Matrix<Scalar>>;

/**
 * An alias for Eigen's bi-diagonal divide and conquer singular-value
 * decomposition.
 */
template <typename Scalar>
using SVD = Eigen::BDCSVD<Matrix<Scalar>>;

/**
 * An alias for Eigen's singular-value decomposition options.
 */
using SVDOptions = Eigen::DecompositionOptions;

/**
 * @return The number of threads used by Eigen to accelerate operations
 * supporting multithreading.
 */
inline int num_of_eval_threads() { return Eigen::nbThreads(); }

/**
 * @param num_of_threads The number of threads Eigen should use to accelerate
 * operations supporting multithreading. The lower bound of the actual value
 * applied is 1 while the upper bound is the maximum of 1 and the level of
 * hardware concurrency detected.
 */
inline void set_num_of_eval_threads(int num_of_threads) {
  int max = std::max(1, (int)std::thread::hardware_concurrency());
  Eigen::setNbThreads(std::max(1, std::min(num_of_threads, max)));
}

/**
 * It serializes the matrix in a format such that the first two numbers denote
 * the matrix's number of rows and columns respectively and the remaining
 * numbers represent the coefficients of the matrix in column-major order.
 *
 * @param matrix The matrix to serialize.
 * @param out_stream The non-binary stream to serialize the matrix to.
 */
template <typename Scalar>
inline void serialize(const Matrix<Scalar>& matrix, std::ostream& out_stream) {
  out_stream << sizeof(Scalar);
  out_stream << " " << matrix.rows();
  out_stream << " " << matrix.cols();
  for (std::size_t i = 0; i < matrix.size(); ++i) out_stream << " " << *(matrix.data() + i);
  out_stream << std::flush;
}

/**
 * It serializes the matrix into a file at the specified file path.
 *
 * @param matrix The matrix to serialize.
 * @param file_path The path to the file to which the matrix is to be
 * serialized.
 */
template <typename Scalar>
inline void serialize(const Matrix<Scalar>& matrix, const std::string& file_path) {
  std::ofstream out_stream(file_path);
  assert(out_stream.is_open());
  serialize<Scalar>(matrix, out_stream);
}

/**
 * It serializes the matrix in a format such that the first 2 bytes denote the
 * size of a single coefficient of the matrix in bytes, the second and third 4
 * bytes denote the matrix's number of rows and columns respectively, and the
 * remaining bytes contain the coefficients of the matrix in column-major order.
 *
 * @param matrix The matrix to serialize.
 * @param out_stream The binary stream to serialize the matrix to.
 */
template <typename Scalar>
inline void serialize_binary(const Matrix<Scalar>& matrix, std::ostream& out_stream) {
  unsigned short scalar_size = static_cast<unsigned short>(sizeof(Scalar));
  out_stream.write(reinterpret_cast<const char*>(&scalar_size), std::streamsize(sizeof(unsigned short)));
  unsigned rows = static_cast<unsigned>(matrix.rows());
  unsigned cols = static_cast<unsigned>(matrix.cols());
  out_stream.write(reinterpret_cast<const char*>(&rows), std::streamsize(sizeof(unsigned)));
  out_stream.write(reinterpret_cast<const char*>(&cols), std::streamsize(sizeof(unsigned)));
  out_stream.write(reinterpret_cast<const char*>(matrix.data()), std::streamsize(matrix.size() * sizeof(Scalar)));
  out_stream << std::flush;
}

/**
 * It serializes the matrix into a binary file at the specified file path.
 *
 * @param matrix The matrix to serialize.
 * @param file_path The path to the binary file to which the matrix is to be
 * serialized.
 */
template <typename Scalar>
inline void serialize_binary(const Matrix<Scalar>& matrix, const std::string& file_path) {
  std::ofstream out_stream(file_path, std::ios::binary);
  assert(out_stream.is_open());
  serialize_binary<Scalar>(matrix, out_stream);
}

/**
 * It deserializes a matrix assuming the serialized format matches that used by
 * the serialize() method.
 *
 * @param in_stream The stream to the serialized matrix.
 * @return The unserialized matrix.
 */
template <typename Scalar>
inline Matrix<Scalar> deserialize(std::istream& in_stream) {
  unsigned rows, cols;
  in_stream >> rows;
  in_stream >> cols;
  Matrix<Scalar> matrix(rows, cols);
  for (std::size_t i = 0; i < matrix.size(); ++i) in_stream >> *(matrix.data() + i);
  return matrix;
}

/**
 * It deserializes a matrix from the file at the provided file path.
 *
 * @param file_path The path to the file containing the serialized matrix.
 * @return The deserialized matrix.
 */
template <typename Scalar>
inline Matrix<Scalar> deserialize(const std::string& file_path) {
  std::ifstream in_stream(file_path);
  assert(in_stream.is_open());
  return deserialize<Scalar>(in_stream);
}

/**
 * It deserializes a matrix assuming the serialized format matches that used by
 * the serialize_binary() method.
 *
 * @param in_stream The binary stream to the serialized matrix.
 * @return The unserialized matrix.
 */
template <typename Scalar>
inline Matrix<Scalar> deserialize_binary(std::istream& in_stream) {
  unsigned short scalar_size;
  in_stream.read(reinterpret_cast<char*>(&scalar_size), std::streamsize(sizeof(unsigned short)));
  assert(scalar_size == sizeof(Scalar));
  unsigned rows, cols;
  in_stream.read(reinterpret_cast<char*>(&rows), std::streamsize(sizeof(unsigned)));
  in_stream.read(reinterpret_cast<char*>(&cols), std::streamsize(sizeof(unsigned)));
  Matrix<Scalar> matrix(rows, cols);
  in_stream.read(reinterpret_cast<char*>(matrix.data()), std::streamsize(matrix.size() * sizeof(Scalar)));
  return matrix;
}

/**
 * It deserializes a matrix from the binary file at the provided file path.
 *
 * @param file_path The path to the binary file containing the serialized
 * matrix.
 * @return The deserialized matrix.
 */
template <typename Scalar>
inline Matrix<Scalar> deserialize_binary(const std::string& file_path) {
  std::ifstream in_stream(file_path, std::ios::binary);
  assert(in_stream.is_open());
  return deserialize_binary<Scalar>(in_stream);
}

}  // namespace cattle

#endif /* C_ATTL3_CORE_EIGENPROXY_H_ */
