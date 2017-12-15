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

static const std::string* MISMTCHD_ROWS_ERR_MSG_PTR =
		new std::string("mismatched fit and transform input matrix dimensions");

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
	NormalizationPreprocessor(bool standardize) :
		standardize(standardize) { };
	virtual ~NormalizationPreprocessor() = default;
	virtual void fit(const Matrix<Scalar>& data) {
		means = data.rowwise().mean();
		if (standardize) {
			sd = (data.colwise() - means).square().rowwise().mean().sqrt();
		}
	};
	virtual void transform(Matrix<Scalar>& data) const {
		assert(means.cols() == data.rows() && MISMTCHD_ROWS_ERR_MSG_PTR);
		data = data.colwise() - means;
		if (standardize) {
			for (int i = 0; sd.cols(); i++) {
				data.row(i) = (data.row(i).array() - means(i)) / sd(i);
			}
		}
	};
protected:
	bool standardize;
	Vector<Scalar> means;
	Vector<Scalar> sd;
};

template<typename Scalar>
class PCAPreprocessor : public NormalizationPreprocessor<Scalar> {
public:
	PCAPreprocessor(bool standardize, bool whiten, unsigned max_dims_to_retain) :
			NormalizationPreprocessor(standardize),
			whiten(whiten),
			max_dims_to_retain(max_dims_to_retain) {
		assert(dims_to_retain > 0 && "minimum 1 dimension must be retained");
	};
	void fit(const Matrix<Scalar>& data) {
		NormalizationPreprocessor<Scalar>::fit(data);
		Matrix<Scalar> normalized_data = data.colwise() - mean;
		if (standardize) {
			for (int i = 0; sd.cols(); i++) {
				normalized_data.row(i) /= sd(i);
			}
		}
		// Compute the covariance matrix.
		Matrix<Scalar> cov = normalized_data.transpose() * normalized_data /
				normalized_data.rows();
		// Eigen decomposition.
		Eigen::SelfAdjointEigenSolver<Matrix<Scalar>> eig_solver(cov);
		unsigned _dims_to_retain = std::min(data.rows(), max_dims_to_retain);
		/* The eigen vectors are sorted based on the magnitude of their
		 * corresponding eigen values. */
		eigen_basis = eig_solver.eigenvectors().rightCols(_dims_to_retain);
		if (whiten) {

		}
	};
	void transform(Matrix<Scalar>& data) const {
		NormalizationPreprocessor<Scalar>::transform(data);
		data *= eigen_basis;
	};
private:
	bool whiten;
	int max_dims_to_retain;
	Matrix<Scalar> eigen_basis;

};

template<typename Scalar>
class WhiteningPreprocessor : public Preprocessor<Scalar> {
public:
	void fit(const Matrix<Scalar>& data) {

	};
	void transform(Matrix<Scalar>& data) const {

	};
};

} /* namespace cppnn */

#endif /* PREPROCESSOR_H_ */
