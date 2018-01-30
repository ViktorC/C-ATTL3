/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

#include <cmath>
#include <DataProvider.h>
#include <Dimensions.h>
#include <iostream>
#include <Layer.h>
#include <Loss.h>
#include <memory>
#include <NeuralNetwork.h>
#include <Optimizer.h>
#include <Preprocessor.h>
#include <RegularizationPenalty.h>
#include <utility>
#include <vector>
#include <WeightInitialization.h>

typedef double Scalar;

using namespace cppnn;

int main() {
	std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;
	Tensor4Ptr<Scalar> training_obs_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(80, 32, 32, 3));
	Tensor4Ptr<Scalar> training_obj_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
	Tensor4Ptr<Scalar> test_obs_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(20, 32, 32, 3));
	Tensor4Ptr<Scalar> test_obj_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(20, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	PCAPreprocessor<Scalar> preproc(true, true);
//	preproc.fit(data);
//	preproc.transform(data);
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<NeuralNetPtr<Scalar>> nets;

	CompositeNeuralNetwork<Scalar> nn(std::move(nets));
	nn.init();
	LossSharedPtr<Scalar> loss(new QuadraticLoss<Scalar>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar> opt(loss, reg, 20);
//	std::cout << opt.verify_gradients(nn, test_prov) << std::endl;
	opt.optimize(nn, training_prov, test_prov, 500);
	return 0;
};
