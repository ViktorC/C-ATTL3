///*
// * NeuralNetwork.cpp
// *
// *  Created on: 04.12.2017
// *      Author: Viktor Csomor
// */
//
//#include <NeuralNetwork.h>
//
//#include <assert.h>
//#include <detail/matrix_def.hpp>
//#include <stddef.h>
//#include <tools/entry_proxy.hpp>
//#include <algorithm>
//#include <random>
//#include <vector>
//
//
//namespace cppnn {
//
//NeuralNetwork::NeuralNetwork(int input_size, std::vector<Layer>& layers) :
//		input_size(input_size),
//		layers(layers) {
//	if (input_size <= 0)
//		throw std::invalid_argument("input size must be greater than 0.");
//	if (layers.size() <= 0)
//		throw std::invalid_argument("layers must contain at least 1 element.");
//};
//int NeuralNetwork::get_input_size() const {
//	return input_size;
//}
//std::vector<Layer>& NeuralNetwork::get_layers() const {
//	return layers;
//}
//std::vector<double> NeuralNetwork::feed_forward(std::vector<double>& input) {
//	viennacl::matrix<double> input_mtrx(0, input.size());
//	for (unsigned i = 0; i < input.size(); i++) {
//		input_mtrx(0,i) = input(i);
//	}
//	viennacl::matrix<double>* input_mtrx_ptr = &input_mtrx;
//	for (unsigned i = 0; i < layers.size(); i++) {
//		viennacl::matrix<double>& input_mtrx_ref = *input_mtrx_ptr;
//		input_mtrx_ptr = &(layers[i].feed_forward(input_mtrx_ref));
//	}
//	return viennacl::row(*input_mtrx_ptr, 0);
//};
//void NeuralNetwork::feed_back(std::vector<double>& out_grads) {
//	viennacl::matrix<double> out_grads_mtrx(0, out_grads.size());
//	for (unsigned i = 0; i < out_grads.size(); i++) {
//		out_grads_mtrx(0,i) = out_grads(i);
//	}
//	viennacl::matrix<double>* out_grads_mtrx_ptr = &out_grads_mtrx;
//	for (int i = (int) layers.size() - 1; i >= 0; i--) {
//		viennacl::matrix<double>& out_grads_mtrx_ref = *out_grads_mtrx_ptr;
//		out_grads_mtrx_ptr = &(layers[i].feed_back(out_grads_mtrx_ref));
//	}
//};
//std::string NeuralNetwork::to_string() const {
//	std::string str = "<NN>\n";
//	for (unsigned i = 0; i < layers.size(); i++) {
//		str += "Layer " + std::to_string(i) + "--------------------\n";
//		str += layers[i].to_string();
//	}
//	str += "</NN>\n";
//	return str;
//};
//
//}
