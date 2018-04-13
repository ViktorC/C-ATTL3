/*
 * CUDAUtils.h
 *
 *  Created on: 12 Apr 2018
 *      Author: Viktor Csomor
 */

#ifndef UTILS_GPU_CUDAUTILS_H_
#define UTILS_GPU_CUDAUTILS_H_

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "utils/Eigen.h"

namespace cattle {
namespace internal {

template<typename Scalar>
class CUDAUtils {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	static Matrix<Scalar> mul(Matrix<Scalar>& a, Matrix<Scalar>& b) {
		assert(a.cols() == b.rows());
		Matrix<Scalar> c(a.rows(), b.cols());
		mat_mul(a.data(), b.data(), c.data(), a.rows(), a.cols(), b.rows(), b.cols());
		return c;
	}
private:
	CUDAUtils() { }
};

// CUDA kernels.
namespace {

template<typename Scalar>
__global__ bool mat_mul(const Scalar* a, const Scalar* b, Scalar* c, std::size_t a_rows, std::size_t a_cols,
		std::size_t b_rows, std::size_t b_cols) {
	cudaError_t cudaStat;
	cublasStatus_t cuBlasStat;
	cublasHandle_t handle;
	std::size_t a_size = a_rows * a_cols * sizeof(Scalar);
	std::size_t b_size = b_rows * b_cols * sizeof(Scalar);
	// Device arrays.
	Scalar* d_a;
	Scalar* d_b;
	Scalar* d_c;
	cudaStat = cudaMalloc(&d_a, a_size);
	if (cudaStat != cudaSuccess)
		return false;
	cudaStat = cudaMalloc(&d_b, b_size);
	if (cudaStat != cudaSuccess) {
		cudaFree(d_a);
		return false;
	}
	cudaStat = cudaMalloc(&d_c, a_rows * b_cols * sizeof(Scalar));
	if (cudaStat != cudaSuccess) {
		cudaFree(d_a);
		cudaFree(d_b);
		return false;
	}
	// Create the CUBLAS handle.
	cuBlasStat = cublasCreate(&handle);
	if (cuBlasStat != CUBLAS_STATUS_SUCCESS) {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return false;
	}
	// Copy the contents of the host arrays to the device arrays.
	cuBlasStat = cublasSetMatrix(a_rows, a_cols, sizeof(Scalar), a, a_rows, d_a, a_rows);
	if (cuBlasStat != CUBLAS_STATUS_SUCCESS) {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		cublasDestory(handle);
		return false;
	}
	cuBlasStat = cublasSetMatrix(b_rows, b_cols, sizeof(Scalar), b, b_rows, d_b, b_rows);
	if (cuBlasStat != CUBLAS_STATUS_SUCCESS) {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		cublasDestory(handle);
		return false;
	}
	// Perform the matrix multiplication.
	cuBlasStat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, a_rows, b_rows, a_cols,
			(Scalar) 1, d_a, a_rows, d_b, b_rows, (Scalar) 0, d_c, a_rows);
	if (cuBlasStat != CUBLAS_STATUS_SUCCESS) {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		cublasDestory(handle);
		return false;
	}
	/* Copy the contents of the device array holding the results of the matrix multiplication
	 * back to the host.
	 */
	cuBlasStat = cublasGetMatrix(a_rows, b_cols, sizeof(Scalar), d_c, a_rows, c, a_rows);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cublasDestory(handle);
	return cuBlasStat == CUBLAS_STATUS_SUCCESS;
}

}

}
} /* namespace cattle */

#endif /* UTILS_GPU_CUDAUTILS_H_ */
