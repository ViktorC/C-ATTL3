///*
// * Optimizer.cpp
// *
// *  Created on: 6 Dec 2017
// *      Author: Viktor
// */
//
//#include <detail/matrix_def.hpp>
//#include <Layer.h>
//#include <matrix.hpp>
//#include <Optimizer.h>
//#include <tools/entry_proxy.hpp>
//#include <algorithm>
//#include <random>
//#include <vector.hpp>
//
//namespace cppnn {
//
//Optimizer::Optimizer(const Loss& loss, const Regularization& reg) :
//	loss(loss),
//	reg(reg) { };
//void Optimizer::init_weights(NeuralNetwork& net) {
//	std::default_random_engine gen;
//	double const abs_dist_Range = INIT_WEIGHT_ABS_MAX / net.get_input_size();
//	double const sd = abs_dist_Range * .34;
//	std::normal_distribution<> normal_distribution(0, sd);
//	for (unsigned i = 0; i < net.get_layers().size(); i++) {
//		viennacl::matrix<double> weights = net.get_layers()[i].get_weights();
//		unsigned rows = weights.size1();
//		unsigned cols = weights.size2();
//		for (unsigned j = 0; j < rows; j++) {
//			for (unsigned k = 0; k < cols; k++) {
//				if (j == rows - 1) {
//					// Set initial bias value to 0.
//					weights(j,k) = 0;
//				} else {
//					// Initialize weights using normal distribution centered around 0 with a small SD.
//					double const rand_weight = normal_distribution(gen);
//					double init_weight = rand_weight >= .0 ?
//							std::max(INIT_WEIGHT_ABS_MIN, rand_weight) :
//							std::min(-INIT_WEIGHT_ABS_MIN, rand_weight);
//					weights(j,k) = init_weight;
//				}
//			}
//		}
//	}
//};
//
//}
