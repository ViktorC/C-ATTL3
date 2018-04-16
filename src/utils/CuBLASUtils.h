/*
 * CuBLASUtils.h
 *
 *  Created on: 12 Apr 2018
 *      Author: Viktor Csomor
 */

#ifndef UTILS_GPU_CUBLASUTILS_H_
#define UTILS_GPU_CUBLASUTILS_H_

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <exception>
#include <type_traits>
#include "cublas_v2.h"
#include "Eigen.h"

namespace cattle {
namespace internal {

namespace {

template<typename Scalar>
using GemmRoutine = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
		int, int, int, const Scalar*, const Scalar*, int, const Scalar*, int, const Scalar*,
		Scalar*, int);

template<typename Scalar> __inline__ GemmRoutine<Scalar> get_gemm_routine() {
	return &cublasDgemm;
}

template<> __inline__ GemmRoutine<float> get_gemm_routine() {
	return &cublasSgemm;
}

}

/**
 * A singleton utility class providing methods for GPU accelerated linear algebra operations
 * using cuBLAS.
 */
template<typename Scalar>
class CuBLASUtils {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	CuBLASUtils(const CuBLASUtils&) = delete;
	~CuBLASUtils() {
		// Destroy the cuBLAS handle.
		cublasStatus_t cublas_stat = cublasDestroy(handle);
		assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
	}
	CuBLASUtils& operator=(const CuBLASUtils&) = delete;
	/**
	 * @return A reference to the only instance of the class.
	 */
	inline static CuBLASUtils& get_instance() {
		static CuBLASUtils instance;
		return instance;
	}
	/**
	 * It computes the product of the matrix multiplication.
	 *
	 * @param a The multiplicand matrix.
	 * @param b The multiplier matrix.
	 * @param transpose_a Whether the multiplicand is to be transposed for the operation.
	 * @param transpose_b Whether the multiplier is to be transposed for the operation.
	 * @return The product of the matrix multiplication.
	 */
	inline Matrix<Scalar> mul(Matrix<Scalar>& a, Matrix<Scalar>& b, bool transpose_a,
			bool transpose_b) {
		std::size_t a_orig_rows = a.rows();
		std::size_t a_orig_cols = a.cols();
		std::size_t b_orig_rows = b.rows();
		std::size_t b_orig_cols = b.cols();
		std::size_t a_rows, a_cols, b_rows, b_cols;
		if (transpose_a) {
			a_rows = a_orig_cols;
			a_cols = a_orig_rows;
		} else {
			a_rows = a_orig_rows;
			a_cols = a_orig_cols;
		}
		if (transpose_b) {
			b_rows = b_orig_cols;
			b_cols = b_orig_rows;
		} else {
			b_rows = b_orig_rows;
			b_cols = b_orig_cols;
		}
		assert(a_cols == b_rows);
		Matrix<Scalar> c(a_rows, b_cols);
		cudaError_t cuda_stat;
		cublasStatus_t cublas_stat;
		// Device arrays.
		Scalar* d_a;
		Scalar* d_b;
		Scalar* d_c;
		// Allocate the memory for the arrays on the device.
		cuda_stat = cudaMalloc(&d_a, a_rows * a_cols * sizeof(Scalar));
		if (cuda_stat != cudaSuccess)
			throw std::runtime_error("cuda malloc failure: " + cuda_stat);
		cuda_stat = cudaMalloc(&d_b, b_rows * b_cols * sizeof(Scalar));
		if (cuda_stat != cudaSuccess) {
			cudaFree(d_a);
			throw std::runtime_error("cuda malloc failure: " + cuda_stat);
		}
		cuda_stat = cudaMalloc(&d_c, a_rows * b_cols * sizeof(Scalar));
		if (cuda_stat != cudaSuccess) {
			cudaFree(d_a);
			cudaFree(d_b);
			throw std::runtime_error("cuda malloc failure: " + cuda_stat);
		}
		// Copy the contents of the host arrays to the device arrays.
		cublas_stat = cublasSetMatrix(a_orig_rows, a_orig_cols, sizeof(Scalar), a.data(),
				a_orig_rows, d_a, a_orig_rows);
		if (cublas_stat != CUBLAS_STATUS_SUCCESS) {
			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_c);
			throw std::runtime_error("cublas matrix mapping failure: " + cublas_stat);
		}
		cublas_stat = cublasSetMatrix(b_orig_rows, b_orig_cols, sizeof(Scalar), b.data(),
				b_orig_rows, d_b, b_orig_rows);
		if (cublas_stat != CUBLAS_STATUS_SUCCESS) {
			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_c);
			throw std::runtime_error("cublas matrix mapping failure: " + cublas_stat);
		}
		cublasOperation_t a_op = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t b_op = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
		Scalar alpha = 1;
		Scalar beta = 0;
		// Resolve the GEMM precision based on the scalar type.
		GemmRoutine<Scalar> gemm = get_gemm_routine<Scalar>();
		// Perform the matrix multiplication.
		cublas_stat = gemm(handle, a_op, b_op, a_rows, b_cols, a_cols, &alpha, d_a,
				a_orig_rows, d_b, b_orig_rows, &beta, d_c, a_rows);
		if (cublas_stat != CUBLAS_STATUS_SUCCESS) {
			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_c);
			throw std::runtime_error("cublas gemm failure: " + cublas_stat);
		}
		/* Copy the contents of the device array holding the results of the matrix
		 * multiplication back to the host. */
		cublas_stat = cublasGetMatrix(a_rows, b_cols, sizeof(Scalar), d_c, a_rows,
				c.data(), a_rows);
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		if (cublas_stat != CUBLAS_STATUS_SUCCESS)
			throw std::runtime_error("cublas matrix retrieval failure: " + cublas_stat);
		return c;
	}
private:
	CuBLASUtils() :
			handle() {
		// Create the cuBLAS handle.
		cublasStatus_t cublas_stat = cublasCreate(&handle);
		assert(cublas_stat == CUBLAS_STATUS_SUCCESS);
	}
	cublasHandle_t handle;
};

}
} /* namespace cattle */

#endif /* UTILS_GPU_CUBLASUTILS_H_ */
