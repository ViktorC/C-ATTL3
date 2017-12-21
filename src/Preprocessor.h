/*
 * Preprocessor.h
 *
 *  Created on: 12.12.2017
 *      Author: A6714
 */

#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include <Matrix.h>
#include <string>
#include <Vector.h>

namespace cppnn {

template<typename Scalar>
class Preprocessor {
public:
	virtual ~Preprocessor() = default;
	virtual void fit(const Matrix<Scalar>& data) = 0;
	virtual void transform(Matrix<Scalar>& data) const = 0;
};

template<typename Scalar>
class NormalizationPreprocessor : public Preprocessor<Scalar> {
public:
	NormalizationPreprocessor(bool standardize = false,
			Scalar epsilon = EPSILON) :
				standardize(standardize),
				epsilon(epsilon) {
		assert(epsilon > 0 && "epsilon must be greater than 0");
	};
	virtual ~NormalizationPreprocessor() = default;
	virtual void fit(const Matrix<Scalar>& data) {
		means = data.colwise().mean();
		if (standardize) {
			sd = RowVector<Scalar>(means.cols());
			for (int i = 0; i < sd.cols(); i++) {
				sd(i) = sqrt((data.col(i).array() - means(i)).square().mean() +
						epsilon);
			}
		}
	};
	virtual void transform(Matrix<Scalar>& data) const {
		assert(means.cols() == data.cols() && "mismatched fit and transform "
				"input matrix dimensions");
		data = data.rowwise() - means;
		if (standardize) {
			for (int i = 0; i < sd.cols(); i++) {
				data.col(i) /= sd(i);
			}
		}
	};
protected:
	static constexpr Scalar EPSILON = 1e-8;
	bool standardize;
	float epsilon;
	RowVector<Scalar> means;
	RowVector<Scalar> sd;
};

template<typename Scalar>
class PCAPreprocessor : public NormalizationPreprocessor<Scalar> {
public:
	PCAPreprocessor(bool standardize = false, bool whiten = false,
			float min_rel_var_to_retain = 1,
			Scalar epsilon = NormalizationPreprocessor<Scalar>::EPSILON) :
				NormalizationPreprocessor<Scalar>::NormalizationPreprocessor(standardize, epsilon),
				whiten(whiten),
				min_rel_var_to_retain(min_rel_var_to_retain) {
		assert(min_rel_var_to_retain > 0 && min_rel_var_to_retain <= 1 &&
				"the minimum relative variance to be retained must be greater "
				"then 0 and less than or equal to 1");
	};
	void fit(const Matrix<Scalar>& data) {
		NormalizationPreprocessor<Scalar>::fit(data);
		Matrix<Scalar> normalized_data = data.rowwise() -
				NormalizationPreprocessor<Scalar>::means;
		if (NormalizationPreprocessor<Scalar>::standardize) {
			for (int i = 0; i < NormalizationPreprocessor<Scalar>::sd.cols(); i++) {
				normalized_data.row(i) /= NormalizationPreprocessor<Scalar>::sd(i);
			}
		}
		// Compute the covariance matrix.
		Matrix<Scalar> cov = normalized_data.transpose() * normalized_data /
				normalized_data.rows();
		// Eigen decomposition.
		Eigen::SelfAdjointEigenSolver<Matrix<Scalar>> eigen_solver(cov);
		// Determine the number of components to retain.
		const ColVector<Scalar>& eigen_values = eigen_solver.eigenvalues();
		const Scalar min_var_to_retain = eigen_values.sum() * min_rel_var_to_retain;
		Scalar var = 0;
		int dims_to_retain = 0;
		for (; dims_to_retain < eigen_values.rows(); dims_to_retain++) {
			if (var >= min_var_to_retain)
				break;
			// The eigen values are sorted in an ascending order.
			var += eigen_values(eigen_values.rows() - (1 + dims_to_retain));
		}
		/* The eigen vectors are sorted based on the magnitude of their
		 * corresponding eigen values. */
		eigen_basis = eigen_solver.eigenvectors().rightCols(dims_to_retain);
		if (whiten) {
			// The eigen values are only needed if whitening is enabled.
			this->eigen_values = eigen_values.bottomRows(dims_to_retain).transpose();
		}
	};
	void transform(Matrix<Scalar>& data) const {
		NormalizationPreprocessor<Scalar>::transform(data);
		data *= eigen_basis;
		if (whiten) {
			for (int i = 0; i < data.cols(); i++) {
				// Add a small constant to avoid division by zero.
				data.col(i) *= 1 / sqrt(eigen_values(i) +
						NormalizationPreprocessor<Scalar>::epsilon);
			}
		}
	};
private:
	bool whiten;
	float min_rel_var_to_retain;
	Matrix<Scalar> eigen_basis;
	RowVector<Scalar> eigen_values;
};

} /* namespace cppnn */

#endif /* PREPROCESSOR_H_ */
