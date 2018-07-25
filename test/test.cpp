/*
 * test.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: Viktor Csomor
 */

//#include <cstring>
//#include <gtest/gtest.h>
//
//#include "gradient_test.hpp"
//#include "training_test.hpp"
//
//bool cattle::test::verbose;

#include "data_provider/CIFARDataProvider.hpp"
#include "data_provider/IMDBDataProvider.hpp"
#include "data_provider/JointFileDataProvider.hpp"
#include "data_provider/MNISTDataProvider.hpp"
#include "data_provider/MemoryDataProvider.hpp"
#include "data_provider/PartitionDataProvider.hpp"
#include "data_provider/SplitFileDataProvider.hpp"
#include "layer/activation/BinaryStepActivationLayer.hpp"
#include "layer/activation/ELUActivationLayer.hpp"
#include "layer/activation/IdentityActivationLayer.hpp"
#include "layer/activation/LeakyReLUActivationLayer.hpp"
#include "layer/activation/PReLUActivationLayer.hpp"
#include "layer/activation/PSwishActivationLayer.hpp"
#include "layer/activation/ReLUActivationLayer.hpp"
#include "layer/activation/ScaledActivationLayer.hpp"
#include "layer/activation/SigmoidActivationLayer.hpp"
#include "layer/activation/SoftmaxActivationLayer.hpp"
#include "layer/activation/SoftplusActivationLayer.hpp"
#include "layer/activation/SoftsignActivationLayer.hpp"
#include "layer/activation/SwishActivationLayer.hpp"
#include "layer/activation/TanhActivationLayer.hpp"
#include "layer/kernel/ConvKernelLayer.hpp"
#include "layer/kernel/DenseKernelLayer.hpp"
#include "layer/kernel/TransConvKernelLayer.hpp"
#include "layer/pool/MaxPoolLayer.hpp"
#include "layer/pool/MeanPoolLayer.hpp"
#include "layer/BatchNormLayer.hpp"
#include "layer/BroadcastLayer.hpp"
#include "layer/DropoutLayer.hpp"
#include "layer/ReshapeLayer.hpp"
#include "loss/AbsoluteLoss.hpp"
#include "loss/BinaryCrossEntropyLoss.hpp"
#include "loss/CrossEntropyLoss.hpp"
#include "loss/HingeLoss.hpp"
#include "loss/KullbackLeiblerLoss.hpp"
#include "loss/MultiLabelHingeLoss.hpp"
#include "loss/MultiLabelLogLoss.hpp"
#include "loss/NegatedLoss.hpp"
#include "loss/SoftmaxCrossEntropyLoss.hpp"
#include "loss/SquaredLoss.hpp"
#include "loss/UniversalLoss.hpp"
#include "neural_network/BidirectionalNeuralNetwork.hpp"
#include "neural_network/CompositeNeuralNetwork.hpp"
#include "neural_network/DenseNeuralNetwork.hpp"
#include "neural_network/FeedforwardNeuralNetwork.hpp"
#include "neural_network/LSTMNeuralNetwork.hpp"
#include "neural_network/ParallelNeuralNetwork.hpp"
#include "neural_network/RecurrentNeuralNetwork.hpp"
#include "neural_network/ResidualNeuralNetwork.hpp"
#include "neural_network/SequentialNeuralNetwork.hpp"
#include "neural_network/StackedNeuralNetwork.hpp"
#include "neural_network/UnidirectionalNeuralNetwork.hpp"
#include "parameter_initialization/ConstantParameterInitialization.hpp"
#include "parameter_initialization/GaussianParameterInitialization.hpp"
#include "parameter_initialization/GlorotParameterInitialization.hpp"
#include "parameter_initialization/HeParameterInitialization.hpp"
#include "parameter_initialization/IncrementalParameterInitialization.hpp"
#include "parameter_initialization/LeCunParameterInitialization.hpp"
#include "parameter_initialization/OneParameterInitialization.hpp"
#include "parameter_initialization/OrthogonalParameterInitialization.hpp"
#include "parameter_initialization/ZeroParameterInitialization.hpp"
#include "parameter_regularization/AbsoluteParameterRegularization.hpp"
#include "parameter_regularization/ElasticNetParameterRegularization.hpp"
#include "parameter_regularization/SquaredParameterRegularization.hpp"
#include "parameters/HostParameters.hpp"
#include "preprocessor/NormalizationPreprocessor.hpp"
#include "preprocessor/PCAPreprocessor.hpp"

int main(int argc, char** argv) {
//	using cattle::test::verbose;
//	static const char* verbose_flag = "-verbose";
//	for (int i = 1; i < argc; ++i) {
//		if (!strcmp(argv[i], verbose_flag)) {
//			verbose = true;
//			break;
//		}
//	}
//	::testing::InitGoogleTest(&argc, argv);
//	return RUN_ALL_TESTS();
}
