///*
// * Layer.cpp
// *
// *  Created on: 04.12.2017
// *      Author: Viktor Csomor
// */
//
//#include <Activation.h>
//#include <Layer.h>
//#include <Matrix.h>
//#include <stdexcept>
//#include <string>
//#include <Vector.h>
//
//
//namespace cppnn {
//
//template<typename Scalar>
//Layer<Scalar>::Layer(int nodes, int prev_nodes, const Activation<Scalar>& act) :
//		prev_out(prev_nodes + 1),
//		prev_out_grads(prev_nodes + 1),
//		weights(prev_nodes + 1, nodes),
//		weight_grads(prev_nodes + 1, nodes),
//		in(nodes),
//		out(nodes),
//		act(act) {
//	// Bias trick.
//	prev_out(prev_nodes) = 1;
//};
//template<typename Scalar>
//Matrix<Scalar>& Layer<Scalar>::get_weights() const {
//	return weights;
//};
//template<typename Scalar>
//Matrix<Scalar>& Layer<Scalar>::get_weight_grads() const {
//	return weight_grads;
//};
//template<typename Scalar>
//Vector<Scalar>& Layer<Scalar>::feed_forward(Vector<Scalar>& prev_out) {
////	for (int i = 0; i < prev_nodes; i++) {
////		(*(this->prev_out))(0,i) = prev_out(0,i);
////	}
////	// Compute the neuron inputs by multiplying the output of the previous layer by the weights.
////	*in = viennacl::linalg::prod(*(this->prev_out), *weights);
////	// Activate the neurons.
////	*out = act.function(*in);
//	return *out;
//};
//template<typename Scalar>
//Vector<Scalar>& Layer<Scalar>::feed_back(Vector<Scalar>& out_grads) {
////	// Compute the gradients of the outputs with respect to the weighted inputs.
////	viennacl::matrix<double> in_grads(viennacl::linalg::element_prod(act.d_function(*in, *out), out_grads));
////	*weight_grads = viennacl::linalg::prod(trans(*prev_out), in_grads);
////	*prev_out_grads = viennacl::linalg::prod(in_grads, trans(*weights));
//	return *prev_out_grads;
//};
//template<typename Scalar>
//std::string Layer<Scalar>::to_string() const {
//	std::string str;
//	for (unsigned i = 0; i < weights->size1(); i++) {
//		for (unsigned j = 0; j < weights->size2(); j++) {
//			double w = (*weights)(i,j);
//			str += "Weight[" + std::to_string(i) + "," + std::to_string(j) +
//					"]: " + std::to_string(w) + "\n";
//		}
//	}
//	return str;
//};
//
//}
