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
	Tensor4Ptr<Scalar> test_obs_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(5, 32, 32, 3));
	Tensor4Ptr<Scalar> test_obj_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(5, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
//	PCAPreprocessor<Scalar> preproc(true, true);
//	preproc.fit(data);
//	preproc.transform(data);
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<std::pair<CompositeNeuralNetwork<Scalar>,bool>> res_modules;
	std::vector<NeuralNetPtr<Scalar>> nets;
	std::vector<SequentialNeuralNetwork<Scalar>> parallel_modules;
	parallel_modules.push_back(SequentialNeuralNetwork<Scalar>(LayerPtr<Scalar>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 1, init, 1, 0))));
	parallel_modules.push_back(SequentialNeuralNetwork<Scalar>(LayerPtr<Scalar>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 2, init, 5, 2))));
	nets.push_back(NeuralNetPtr<Scalar>(new ParallelNeuralNetwork<Scalar>(parallel_modules)));
//	nets.push_back(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(LayerPtr<Scalar>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 3, init, 1, 0)))));
	std::vector<LayerPtr<Scalar>> layers1(7);
	layers1[0] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(nets[0]->get_output_dims()));
	layers1[1] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers1[0]->get_output_dims()));
	layers1[2] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers1[1]->get_output_dims()));
	layers1[3] = LayerPtr<Scalar>(new ConvLayer<Scalar>(layers1[2]->get_output_dims(), 8, init));
	layers1[4] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(layers1[3]->get_output_dims()));
	layers1[5] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers1[4]->get_output_dims()));
	layers1[6] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers1[5]->get_output_dims()));
	nets.push_back(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(std::move(layers1))));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(std::move(nets)), false));
	std::vector<LayerPtr<Scalar>> layers2(3);
	layers2[0] = LayerPtr<Scalar>(new ConvLayer<Scalar>(res_modules[0].first.get_output_dims(), 8, init));
	layers2[1] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers2[0]->get_output_dims()));
	layers2[2] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers2[1]->get_output_dims()));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(std::move(layers2)))), false));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(
			LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(res_modules[1].first.get_output_dims()))))), false));
	std::vector<LayerPtr<Scalar>> layers3(3);
	layers3[0] = LayerPtr<Scalar>(new ConvLayer<Scalar>(res_modules[2].first.get_output_dims(), 8, init));
	layers3[1] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers3[0]->get_output_dims()));
	layers3[2] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers3[1]->get_output_dims()));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(std::move(layers3)))), false));
	std::vector<LayerPtr<Scalar>> layers4(5);
	layers4[0] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(res_modules[3].first.get_output_dims()));
	layers4[1] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers4[0]->get_output_dims()));
	layers4[2] = LayerPtr<Scalar>(new FCLayer<Scalar>(layers4[1]->get_output_dims(), 50, init));
	layers4[3] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers4[2]->get_output_dims()));
	layers4[4] = LayerPtr<Scalar>(new FCLayer<Scalar>(layers4[3]->get_output_dims(), 1, init));
	res_modules.push_back(std::make_pair(CompositeNeuralNetwork<Scalar>(NeuralNetPtr<Scalar>(new SequentialNeuralNetwork<Scalar>(std::move(layers4)))), false));
	ResidualNeuralNetwork<Scalar> nn(res_modules);
//	std::vector<LayerPtr<Scalar>> layers(20);
//	layers[0] = LayerPtr<Scalar>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 3, init, 1, 0));
//	layers[1] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(layers[0]->get_output_dims()));
//	layers[2] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers[1]->get_output_dims()));
//	layers[3] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers[2]->get_output_dims()));
//	layers[4] = LayerPtr<Scalar>(new ConvLayer<Scalar>(layers[3]->get_output_dims(), 8, init));
//	layers[5] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(layers[4]->get_output_dims()));
//	layers[6] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers[5]->get_output_dims()));
//	layers[7] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers[6]->get_output_dims()));
//	layers[8] = LayerPtr<Scalar>(new ConvLayer<Scalar>(layers[7]->get_output_dims(), 8, init));
//	layers[9] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers[8]->get_output_dims()));
//	layers[10] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers[9]->get_output_dims()));
//	layers[11] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(layers[10]->get_output_dims()));
//	layers[12] = LayerPtr<Scalar>(new ConvLayer<Scalar>(layers[11]->get_output_dims(), 8, init));
//	layers[13] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers[12]->get_output_dims()));
//	layers[14] = LayerPtr<Scalar>(new BatchNormLayer<Scalar>(layers[13]->get_output_dims()));
//	layers[15] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(layers[14]->get_output_dims()));
//	layers[16] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers[15]->get_output_dims()));
//	layers[17] = LayerPtr<Scalar>(new FCLayer<Scalar>(layers[16]->get_output_dims(), 50, init));
//	layers[18] = LayerPtr<Scalar>(new LeakyReLUActivationLayer<Scalar>(layers[17]->get_output_dims()));
//	layers[19] = LayerPtr<Scalar>(new FCLayer<Scalar>(layers[18]->get_output_dims(), 1, init));
//	SequentialNeuralNetwork<Scalar> nn(std::move(layers));
	nn.init();
	LossSharedPtr<Scalar> loss(new QuadraticLoss<Scalar>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar> opt(loss, reg, 20);
	std::cout << opt.verify_gradients(nn, test_prov) << std::endl;
//	opt.optimize(nn, training_prov, test_prov, 500);
	return 0;
};
