/*
 * CuBLASHandle.hpp
 *
 *  Created on: 12 Apr 2018
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_UTILS_CUBLASHANDLE_H_
#define CATTL3_UTILS_CUBLASHANDLE_H_

#include <cassert>
#include <cstddef>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <exception>
#include <string>
#include <type_traits>

#include "CuBLASError.hpp"
#include "CUDAError.hpp"

namespace cattle {
namespace internal {

namespace {

template<typename Scalar>
using GemmRoutine = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
		int, int, int, const Scalar*, const Scalar*, int, const Scalar*, int, const Scalar*,
		Scalar*, int);

template<typename Scalar> __inline__ GemmRoutine<Scalar> resolve_gemm_routine() {
	return &cublasDgemm;
}

template<> __inline__ GemmRoutine<float> resolve_gemm_routine() {
	return &cublasSgemm;
}

}

/**
 * A singleton utility class providing methods for GPU accelerated linear algebra operations
 * using cuBLAS.
 */
template<typename Scalar>
class CuBLASHandle {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
public:
	CuBLASHandle(const CuBLASHandle&) = delete;
	~CuBLASHandle() {
		// Destroy the cuBLAS handle.
		cublasAssert(cublasDestroy(handle));
	}
	CuBLASHandle& operator=(const CuBLASHandle&) = delete;
	/**
	 * @return A reference to the only instance of the class.
	 */
	inline static CuBLASHandle& get_instance() {
		static CuBLASHandle instance;
		return instance;
	}
	/**
	 * It computes the product of the matrix multiplication.
	 *
	 * @param a The multiplicand matrix.
	 * @param a_orig_rows The number of rows in matrix a.
	 * @param a_orig_cols The number of columns in matrix b.
	 * @param transpose_a Whether the multiplicand is to be transposed for the operation.
	 * @param b The multiplier matrix.
	 * @param b_orig_rows The number of rows in matrix b.
	 * @param b_orig_cols The number of columns in matrix b.
	 * @param transpose_b Whether the multiplier is to be transposed for the operation.
	 * @param transpose_a
	 * @param transpose_b
	 * @param c The product of the matrix multiplication.
	 */
	inline void matrix_mul(Scalar* a, std::size_t a_orig_rows, std::size_t a_orig_cols, bool transpose_a,
			Scalar* b, std::size_t b_orig_rows, std::size_t b_orig_cols, bool transpose_b,
			/* out */ Scalar* c) const {
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
		// Device arrays.
		Scalar* d_a;
		Scalar* d_b;
		Scalar* d_c;
		// Allocate the memory for the arrays on the device.
		cudaAssert(cudaMalloc(&d_a, a_rows * a_cols * sizeof(Scalar)));
		cudaAssert(cudaMalloc(&d_b, b_rows * b_cols * sizeof(Scalar)));
		cudaAssert(cudaMalloc(&d_c, a_rows * b_cols * sizeof(Scalar)));
		// Copy the contents of the host arrays to the device arrays.
		cublasAssert(cublasSetMatrix(a_orig_rows, a_orig_cols, sizeof(Scalar), a, a_orig_rows,
				d_a, a_orig_rows));
		cublasAssert(cublasSetMatrix(b_orig_rows, b_orig_cols, sizeof(Scalar), b, b_orig_rows,
				d_b, b_orig_rows));
		cublasOperation_t a_op = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
		cublasOperation_t b_op = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
		Scalar alpha = 1;
		Scalar beta = 0;
		// Resolve the GEMM precision based on the scalar type.
		GemmRoutine<Scalar> gemm = resolve_gemm_routine<Scalar>();
		// Perform the matrix multiplication.
		cublasAssert(gemm(handle, a_op, b_op, a_rows, b_cols, a_cols, &alpha, d_a, a_orig_rows,
				d_b, b_orig_rows, &beta, d_c, a_rows));
		/* Copy the contents of the device array holding the results of the matrix
		 * multiplication back to the host. */
		cublasAssert(cublasGetMatrix(a_rows, b_cols, sizeof(Scalar), d_c, a_rows, c, a_rows));
		cudaAssert(cudaFree(d_a));
		cudaAssert(cudaFree(d_b));
		cudaAssert(cudaFree(d_c));
	}
private:
	cublasHandle_t handle;
	inline CuBLASHandle() :
			handle() {
		// Create the cuBLAS handle.
		cublasAssert(cublasCreate(&handle));
	}
};

}
} /* namespace cattle */

#endif /* CATTL3_UTILS_CUBLASHANDLE_H_ */
