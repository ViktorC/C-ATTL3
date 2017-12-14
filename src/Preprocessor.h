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
		new std::string("mismatched compute and process input matrix rows");

template<typename Scalar>
class Preprocessor {
public:
	virtual ~Preprocessor() = default;
	virtual void compute(const Matrix<Scalar>& data) = 0;
	virtual void process(Matrix<Scalar>& data) const = 0;
};

template<typename Scalar>
class NormalizationPreprocessor : public Preprocessor<Scalar> {
public:
	virtual ~NormalizationPreprocessor() = default;
	virtual void compute(const Matrix<Scalar>& data) {
		means = Vector<Scalar>(data.rows());
		for (int i = 0; means.cols(); i++) {
			means(i) = data.row(i).mean();
		}
	};
	virtual void process(Matrix<Scalar>& data) const {
		assert(means.cols() == data.rows() && MISMTCHD_ROWS_ERR_MSG_PTR);
		for (int i = 0; means.cols(); i++) {
			data.row(i) = data.row(i).array() - means(i);
		}
	};
protected:
	Vector<Scalar> means;
};

template<typename Scalar>
class StandardizationPreprocessor : public NormalizationPreprocessor<Scalar> {
public:
	virtual ~StandardizationPreprocessor() = default;
	virtual void compute(const Matrix<Scalar>& data) {
		means = Vector<Scalar>(data.rows());
		sd = Vector<Scalar>(data.rows());
		for (int i = 0; means.cols(); i++) {
			Scalar mean = data.row(i).mean();
			means(i) = mean;
			sd(i) = sqrt((data.row(i).array() - mean).square().mean());
		}
	};
	virtual void process(Matrix<Scalar>& data) const {
		assert(means.cols() == data.rows() && MISMTCHD_ROWS_ERR_MSG_PTR);
		for (int i = 0; sd.cols(); i++) {
			data.row(i) = (data.row(i).array() - means(i)) / sd(i);
		}
	};
protected:
	Vector<Scalar> sd;
};

template<typename Scalar>
class PCAPreprocessor : public StandardizationPreprocessor<Scalar> {
public:
	PCAPreprocessor(float retention_rate, bool scale_features) :
		retention_rate(retention_rate),
		scale_features(scale_features) { };
	virtual ~PCAPreprocessor() = default;
	virtual void compute(const Matrix<Scalar>& data) {
		if (scale_features) {
			StandardizationPreprocessor<Scalar>::compute(data);
		} else {
			NormalizationPreprocessor<Scalar>::compute(data);
		}
	};
	virtual void process(Matrix<Scalar>& data) const {
		assert(means.cols() == data.rows() && MISMTCHD_ROWS_ERR_MSG_PTR);
	};
protected:
	float retention_rate;
	bool scale_features;
	Vector<Scalar> means;
};

template<typename Scalar>
class WhiteningPreprocessor : public Preprocessor<Scalar> {
public:
	void compute(const Matrix<Scalar>& data) {

	};
	void process(Matrix<Scalar>& data) const {

	};
};

} /* namespace cppnn */

#endif /* PREPROCESSOR_H_ */
