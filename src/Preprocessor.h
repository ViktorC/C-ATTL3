/*
 * Preprocessor.h
 *
 *  Created on: 12.12.2017
 *      Author: A6714
 */

#ifndef PREPROCESSOR_H_
#define PREPROCESSOR_H_

#include <cassert>
#include <cmath>
#include <Eigen/Dense>
#include <Matrix.h>
#include <string>
#include <Vector.h>

namespace cppnn {

static const std::string* MISMTCHD_ROWS_ERR_MSG_PTR =
		new std::string("mismatched fit and process input matrix rows");

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
	void fit(const Matrix<Scalar>& data) {
		means = data.rowwise().mean();
	};
	void transform(Matrix<Scalar>& data) const {
		assert(means.cols() == data.rows() && MISMTCHD_ROWS_ERR_MSG_PTR);
		data = data.colwise() - means;
	};
private:
	Vector<Scalar> means;
};

template<typename Scalar>
class StandardizationPreprocessor : public Preprocessor<Scalar> {
public:
	void fit(const Matrix<Scalar>& data) {
		means = data.rowwise().mean();
		sd = (data.colwise() - means).square().rowwise().mean().sqrt();
	};
	void transform(Matrix<Scalar>& data) const {
		assert(means.cols() == data.rows() && MISMTCHD_ROWS_ERR_MSG_PTR);
		for (int i = 0; sd.cols(); i++) {
			data.row(i) = (data.row(i).array() - means(i)) / sd(i);
		}
	};
private:
	Vector<Scalar> means;
	Vector<Scalar> sd;
};

template<typename Scalar>
class PCAPreprocessor : public Preprocessor<Scalar> {
public:
	PCAPreprocessor(float retention_rate, bool scale_features) :
		retention_rate(retention_rate),
		scale_features(scale_features) { };
	void fit(const Matrix<Scalar>& data) {
		means = data.rowwise().mean();
		Matrix<Scalar> centered_data = data.colwise() - mean;
		if (scale_features) {
			sd = (data.colwise() - means).square().rowwise().mean().sqrt();
			for (int i = 0; sd.cols(); i++) {
				centered_data.row(i) /= sd(i);
			}
		}
		// Compute the covariance matrix
		Matrix<Scalar> cov = centered_data.transpose() * centered_data;

	};
	void transform(Matrix<Scalar>& data) const {
		assert(means.cols() == data.rows() && MISMTCHD_ROWS_ERR_MSG_PTR);
	};
private:
	float retention_rate;
	bool scale_features;
	Vector<Scalar> means;
	Vector<Scalar> sd;
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
