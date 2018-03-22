/*
 * Layer.h
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include "Dimensions.h"
#include "Utils.h"
#include "WeightInitialization.h"

namespace cattle {

// TODO 1D and 2D convolution and pooling.
// TODO Optional GPU acceleration using cuBLAS and cuDNN.

// Forward declarations to NeuralNetwork and Optimizer so they can be friended.
template<typename Scalar, std::size_t Rank, bool Sequential> class NeuralNetwork;
template<typename Scalar, std::size_t Rank, bool Sequential> class Optimizer;

/**
 * An abstract class template representing layers in a neural network.
 */
template<typename Scalar, std::size_t Rank>
class Layer {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal rank");
	friend class NeuralNetwork<Scalar,Rank,true>;
	friend class NeuralNetwork<Scalar,Rank,false>;
	friend class Optimizer<Scalar,Rank,true>;
	friend class Optimizer<Scalar,Rank,false>;
	// Rank is increased by one to allow for batch training.
	typedef Tensor<Scalar,Rank + 1> Data;
public:
	virtual ~Layer() = default;
	/**
	 * A constant method implementing the clone pattern.
	 *
	 * @return A pointer to a copy of the instance. The instance does not take ownership of
	 * the returned pointer (i.e. the caller is responsible for deleting it).
	 */
	virtual Layer<Scalar,Rank>* clone() const = 0;
	/**
	 * A simple constant getter method for the input dimensionality of the layer.
	 *
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors accepted by the layer as its input (except for the first rank which denotes
	 * the variable sample size).
	 */
	virtual const Dimensions<int,Rank>& get_input_dims() const = 0;
	/**
	 * A simple constant getter method for the output dimensionality of the layer.
	 *
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors output by the layer along all ranks except the first one.
	 */
	virtual const Dimensions<int,Rank>& get_output_dims() const = 0;
	/**
	 * A method that returns whether the layer has parameters that can be learned.
	 *
	 * @return Whether the layer uses learnable parameters.
	 */
	inline bool is_parametric() {
		return get_params().rows() > 0 && get_params().cols() > 0;
	}
protected:
	/* Only expose methods that allow for the modification of the
	 * layer's state to friends and sub-classes. */
	/**
	 * It returns a clone of the layer instance using a reference to the original's parameters.
	 *
	 * @return A clone of the original layer instance sharing the same parameters with the
	 * original.
	 */
	virtual Layer<Scalar,Rank>* clone_with_shared_params() const = 0;
	/**
	 * A constant method that returns whether this layer functions as an input layer. An input
	 * layer does not need to propagate the gradients all the way during the backward pass as
	 * it is assumed that no other layer needs them derive the gradient on its parameters. It
	 * is therefore possible for an input layer to simply return a null tensor as the output of
	 * its backward pass.
	 *
	 * @return Whether this layer is the input layer of the neural network that contains it.
	 */
	virtual bool is_input_layer() const;
	/**
	 * Sets this instance's input layer status to the given value.
	 *
	 * @param input_layer Whether this layer is to be an input layer or not.
	 */
	virtual void set_input_layer(bool input_layer);
	/**
	 * It initializes the layer and its parameters.
	 */
	virtual void init() = 0;
	/**
	 * It empties the layer's caches such as those required for the derivation of the function
	 * represented by the layer.
	 */
	virtual void empty_cache() = 0;
	/**
	 * It returns a reference to the learnable parameters of the layer.
	 *
	 * @return A non-constant reference to the parameters of the layer that are to be learned.
	 */
	virtual Matrix<Scalar>& get_params() = 0;
	/**
	 * It returns a reference to the gradient of the learnable parameters of the layer.
	 *
	 * @return A non-constant reference to the gradient of the parameters of the layer.
	 */
	virtual Matrix<Scalar>& get_params_grad() = 0;
	/**
	 * It applies constraints such as max-norm to the parameters of the layer if applicable.
	 */
	virtual void enforce_constraints() = 0;
	/**
	 * It has the function represented by the layer applied to the input tensor.
	 *
	 * @param in A tensor representing a batch of observations. The observations are of
	 * the rank specified by the layer's template parameter and the input tensors rank is
	 * one greater.
	 * @param training Whether the input is to be processed in training or inference mode.
	 * If the forward pass is performed in inference mode, the backward pass is not
	 * guaranteed to work.
	 * @return The output of the function represented by the layer applied to the input
	 * tensor.
	 */
	virtual Data pass_forward(Data in, bool training) = 0;
	/**
	 * It back-propagates the derivative of the error function w.r.t. the output of the
	 * layer updating the gradient of its learnable parameters along the way.
	 *
	 * @param out_grads The derivative of the loss function w.r.t. the output of the
	 * layer
	 * @return The derivative of the loss function w.r.t. the output of the previous layer
	 * or a null tensor if the layer is an input layer.
	 */
	virtual Data pass_back(Data out_grads) = 0;
};

/**
 * An alias for a shared pointer to a WeightInitialization implementation instance of
 * an arbitrary scalar type.
 */
template<typename Scalar>
using WeightInitSharedPtr = std::shared_ptr<WeightInitialization<Scalar>>;

/**
 * An abstract base class template for layers representing linear kernel-based operations
 * such as matrix multiplication or convolution.
 */
template<typename Scalar, std::size_t Rank>
class KernelLayer : public Layer<Scalar,Rank> {
public:
	virtual ~KernelLayer() = default;
	const Dimensions<int,Rank>& get_input_dims() const {
		return input_dims;
	}
	const Dimensions<int,Rank>& get_output_dims() const {
		return output_dims;
	}
protected:
	inline KernelLayer(const Dimensions<int,Rank>& input_dims, Dimensions<int,Rank> output_dims, WeightInitSharedPtr<Scalar> weight_init,
			std::size_t weight_rows, std::size_t weight_cols, Scalar max_norm_constraint) :
				input_dims(input_dims),
				output_dims(output_dims),
				weight_init(weight_init),
				max_norm_constraint(max_norm_constraint),
				max_norm(Utils<Scalar>::decidedly_greater(max_norm_constraint, .0)),
				input_layer(false),
				weights(weight_rows, weight_cols),
				weights_grad(weight_rows, weight_cols),
				weights_ref(weights) {
		assert(weight_init != nullptr);
	}
	inline KernelLayer(const KernelLayer<Scalar,Rank>& layer, bool share_weights) :
		input_dims(layer.input_dims),
		output_dims(layer.output_dims),
		weight_init(layer.weight_init),
		max_norm_constraint(layer.max_norm_constraint),
		max_norm(layer.max_norm),
		input_layer(layer.input_layer),
		weights(share_weights ? Matrix<Scalar>(0, 0) : layer.weights),
		weights_grad(layer.weights_grad),
		weights_ref(share_weights ? layer.weights : weights) { }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() {
		weight_init->apply(weights_ref);
		weights_grad.setZero(weights_grad.rows(), weights_grad.cols());
	}
	inline Matrix<Scalar>& get_params() {
		return weights_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return weights_grad;
	}
	inline void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = weights.squaredNorm();
			if (l2_norm > max_norm_constraint)
				weights *= (max_norm_constraint / l2_norm);
		}
	}
	const Dimensions<int,Rank> input_dims;
	const Dimensions<int,Rank> output_dims;
	const WeightInitSharedPtr<Scalar> weight_init;
	const Scalar max_norm_constraint;
	const bool max_norm;
	bool input_layer;
	/* Eigen matrices are backed by arrays allocated on the heap, so these
	 * members do not burden the stack. */
	Matrix<Scalar> weights_grad;
	Matrix<Scalar>& weights_ref;
private:
	Matrix<Scalar> weights;
};

/**
 * A class template representing a fully connected layer.
 */
template<typename Scalar, std::size_t Rank>
class FCLayer : public KernelLayer<Scalar,Rank> {
	typedef KernelLayer<Scalar,Rank> Base;
	typedef Tensor<Scalar,Rank + 1> Data;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * @param output_size The length of the vector output for each sample.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the parametric kernel backing the layer.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline FCLayer(const Dimensions<int,Rank>& input_dims, std::size_t output_size, WeightInitSharedPtr<Scalar> weight_init,
			Scalar max_norm_constraint = 0) :
				Base::KernelLayer(input_dims, Dimensions<int,Rank>({ (int) output_size }), weight_init,
						input_dims.get_volume() + 1, output_size, max_norm_constraint) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new FCLayer(*this);
	}
protected:
	inline FCLayer(const FCLayer<Scalar,Rank>& layer, bool share_weights) :
			Base::KernelLayer(layer, share_weights),
			biased_in(layer.biased_in) { }
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return new FCLayer(*this, true);
	}
	inline void empty_cache() {
		biased_in = Matrix<Scalar>(0, 0);
	}
	inline Data pass_forward(Data in, bool training) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(in).template demote<>() == Base::input_dims);
		assert(in.dimension(0) > 0);
		unsigned input_size = Base::input_dims.get_volume();
		// Add a 1-column to the input for the bias trick.
		biased_in = Matrix<Scalar>(in.dimension(0), input_size + 1);
		biased_in.leftCols(input_size) = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(in));
		biased_in.col(input_size).setOnes();
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>((biased_in * Base::weights_ref).eval(), Base::output_dims);
	}
	inline Data pass_back(Data out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).template demote<>() == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && biased_in.rows() == out_grads.dimension(0));
		Matrix<Scalar> out_grads_mat = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads));
		// Compute the gradient of the outputs with respect to the weights.
		Base::weights_grad = biased_in.transpose() * out_grads_mat;
		if (Base::is_input_layer())
			return Data();
		/* Remove the bias row from the weight matrix, transpose it, and compute the derivative w.r.t. the
		 * previous layer's output. */
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>((out_grads_mat *
				Base::weights_ref.topRows(Base::input_dims.get_volume()).transpose()).eval(), Base::input_dims);
	}
private:
	// Staged computation caches
	Matrix<Scalar> biased_in;
};

/**
 * A class template representing a 3D convolutional layer.
 */
template<typename Scalar>
class ConvLayer : public KernelLayer<Scalar,3> {
	typedef KernelLayer<Scalar,3> Base;
	typedef Tensor<Scalar,4> Data;
	typedef std::array<int,4> RankwiseArray;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * The ranks of the input tensors denote the sample, height, width, and channel (N,H,W,C).
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the parametric kernel backing the layer.
	 * @param receptor_size The length of the sides of the base of the receptor cuboid.
	 * @param padding The length of padding to apply to the input tensor along its width and
	 * weight.
	 * @param stride The convolution stride i.e. the number of elements by which the receptor
	 * is to be shifted at each step of the convolution.
	 * @param dilation The size of the spatial (height- and width-wise) padding between voxels
	 * of the receptor.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline ConvLayer(const Dimensions<int,3>& input_dims, std::size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			std::size_t receptor_size = 3, std::size_t padding = 1, std::size_t stride = 1, std::size_t dilation = 0,
			Scalar max_norm_constraint = 0) :
				/* For every filter, there is a column in the weight matrix with the same number of
				 * elements as the size of the receptive field (F * F * D) + 1 for the bias row. */
				Base::KernelLayer(input_dims, Dimensions<int,3>({ calculate_output_dim(input_dims(0), receptor_size, padding, dilation, stride),
						calculate_output_dim(input_dims(1), receptor_size, padding, dilation, stride), (int) filters }), weight_init,
						receptor_size * receptor_size * input_dims(2) + 1, filters, max_norm_constraint),
				filters(filters),
				receptor_size(receptor_size),
				padding(padding),
				stride(stride),
				dilation(dilation),
				padded_height(input_dims(0) + 2 * padding),
				padded_width(input_dims(1) + 2 * padding),
				dil_receptor_size(receptor_size + (receptor_size - 1) * dilation) {
		assert(filters > 0);
		assert(receptor_size > 0);
		assert(stride > 0);
		assert(input_dims(1) + 2 * padding >= receptor_size + (receptor_size - 1) * dilation &&
				input_dims(2) + 2 * padding >= receptor_size + (receptor_size - 1) * dilation);
	}
	inline Layer<Scalar,3>* clone() const {
		return new ConvLayer(*this);
	}
protected:
	inline ConvLayer(const ConvLayer<Scalar>& layer, bool share_weights) :
			Base::KernelLayer(layer, share_weights),
			filters(layer.filters),
			receptor_size(layer.receptor_size),
			padding(layer.padding),
			stride(layer.stride),
			dilation(layer.dilation),
			padded_height(layer.padded_height),
			padded_width(layer.padded_width),
			dil_receptor_size(layer.dil_receptor_size),
			biased_in_vec(layer.biased_in_vec) { }
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return new ConvLayer(*this, true);
	}
	inline void empty_cache() {
		biased_in_vec = std::vector<Matrix<Scalar>>(0);
	}
	inline Data pass_forward(Data in, bool training) {
		assert(Utils<Scalar>::template get_dims<4>(in).template demote<>() == Base::input_dims);
		assert(in.dimension(0) > 0);
		// Spatial padding.
		std::array<std::pair<int,int>,4> paddings;
		paddings[0] = std::make_pair(0, 0);
		paddings[1] = std::make_pair(padding, padding);
		paddings[2] = std::make_pair(padding, padding);
		paddings[3] = std::make_pair(0, 0);
		Data padded_in = in.pad(paddings);
		// Free the memory occupied by the now-expendable input tensor.
		in = Data();
		int rows = padded_in.dimension(0);
		int patches = Base::output_dims(0) * Base::output_dims(1);
		int receptor_vol = receptor_size * receptor_size * Base::input_dims(2);
		int height_rem = padded_height - dil_receptor_size;
		int width_rem = padded_width - dil_receptor_size;
		// Prepare the base offsets and extents for slicing and dilation.
		RankwiseArray row_offsets({ 0, 0, 0, 0 });
		RankwiseArray row_extents({ 1, Base::output_dims(0), Base::output_dims(1), Base::output_dims(2) });
		RankwiseArray patch_extents({ 1, dil_receptor_size, dil_receptor_size, padded_in.dimension(3) });
		RankwiseArray patch_offsets({ 0, 0, 0, 0 });
		RankwiseArray dil_strides({ 1, (int) dilation + 1, (int) dilation + 1, 1 });
		Data out(padded_in.dimension(0), Base::output_dims(0), Base::output_dims(1), Base::output_dims(2));
		biased_in_vec = std::vector<Matrix<Scalar>>(padded_in.dimension(0));
		/* 'Tensor-row' by 'tensor-row', stretch the receptor locations into row vectors, form a matrix out of
		 * them, and multiply it by the weight matrix. */
		for (int i = 0; i < rows; ++i) {
			row_offsets[0] = i;
			patch_offsets[0] = i;
			int patch_ind = 0;
			{
				Matrix<Scalar> in_mat_i(patches, receptor_vol);
				for (int j = 0; j <= height_rem; j += stride) {
					patch_offsets[1] = j;
					for (int k = 0; k <= width_rem; k += stride) {
						patch_offsets[2] = k;
						Data patch;
						// If the patch is dilated, skip the 'internal padding' when stretching it into a row.
						if (dilation > 0)
							patch = padded_in.slice(patch_offsets, patch_extents).stride(dil_strides);
						else
							patch = padded_in.slice(patch_offsets, patch_extents);
						in_mat_i.row(patch_ind++) = Utils<Scalar>::template map_tensor_to_mat<4>(std::move(patch));
					}
				}
				assert(patch_ind == patches);
				// Set the appended column's elements to 1.
				biased_in_vec[i] = Matrix<Scalar>(patches, receptor_vol + 1);
				Matrix<Scalar>& biased_in = biased_in_vec[i];
				biased_in.col(receptor_vol).setOnes();
				biased_in.leftCols(receptor_vol) = in_mat_i;
			}
			/* Flatten the matrix product into a row vector, reshape it into a 'single-row' sub-tensor, and
			 * assign it to the output tensor's corresponding 'row'. */
			Matrix<Scalar> out_i = biased_in_vec[i] * Base::weights_ref;
			out.slice(row_offsets, row_extents) = Utils<Scalar>::template map_mat_to_tensor<4>(MatrixMap<Scalar>(out_i.data(),
					1, out_i.rows() * out_i.cols()).matrix(), Base::output_dims);
		}
		return out;
	}
	inline Data pass_back(Data out_grads) {
		assert(Utils<Scalar>::template get_dims<4>(out_grads).template demote<>() == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && biased_in_vec.size() == (unsigned) out_grads.dimension(0));
		int rows = out_grads.dimension(0);
		int depth = Base::input_dims(2);
		int height_rem = padded_height - dil_receptor_size;
		int width_rem = padded_width - dil_receptor_size;
		Dimensions<int,3> comp_patch_dims({ (int) receptor_size, (int) receptor_size, Base::input_dims(2) });
		RankwiseArray out_grads_row_offsets({ 0, 0, 0, 0 });
		RankwiseArray out_grads_row_extents({ 1, Base::output_dims(0), Base::output_dims(1), Base::output_dims(2) });
		RankwiseArray patch_extents({ 1, dil_receptor_size, dil_receptor_size, depth });
		RankwiseArray patch_offsets({ 0, 0, 0, 0 });
		RankwiseArray strides({ 1, (int) dilation + 1, (int) dilation + 1, 1 });
		Data prev_out_grads(rows, padded_height, padded_width, depth);
		prev_out_grads.setZero();
		Base::weights_grad.setZero(Base::weights_grad.rows(), Base::weights_grad.cols());
		for (int i = 0; i < rows; ++i) {
			out_grads_row_offsets[0] = i;
			Matrix<Scalar> prev_out_grads_mat_i;
			{
				Data slice_i = out_grads.slice(out_grads_row_offsets, out_grads_row_extents).eval();
				MatrixMap<Scalar> out_grads_mat_map_i = MatrixMap<Scalar>(slice_i.data(), Base::output_dims(0) *
						Base::output_dims(1), filters);
				// Accumulate the gradients across the observations.
				Base::weights_grad += biased_in_vec[i].transpose() * out_grads_mat_map_i;
				if (Base::is_input_layer())
					continue;
				/* Remove the bias row from the weight matrix, transpose it, and compute the gradient of the
				 * previous layer's output. */
				prev_out_grads_mat_i = out_grads_mat_map_i * Base::weights_ref.topRows(Base::weights_ref.rows() - 1).transpose();
			}
			/* Given the gradient of the stretched out receptor patches, perform a 'backwards' convolution
			 * to get the derivative w.r.t. the individual input nodes. */
			int patch_ind = 0;
			patch_offsets[0] = i;
			for (int j = 0; j <= height_rem; j += stride) {
				patch_offsets[1] = j;
				for (int k = 0; k <= width_rem; k += stride) {
					patch_offsets[2] = k;
					// Accumulate the gradients where the receptor-patch-tensors overlap.
					if (dilation > 0) {
						prev_out_grads.slice(patch_offsets, patch_extents).stride(strides) +=
								Utils<Scalar>::template map_mat_to_tensor<4>(prev_out_grads_mat_i.row(patch_ind++), comp_patch_dims);
					} else {
						prev_out_grads.slice(patch_offsets, patch_extents) += Utils<Scalar>::template map_mat_to_tensor<4>(
								prev_out_grads_mat_i.row(patch_ind++), comp_patch_dims);
					}
				}
			}
			assert(patch_ind == prev_out_grads_mat_i.rows());
		}
		// Cut off the padding.
		RankwiseArray no_padding_offsets({ 0, (int) padding, (int) padding, 0 });
		RankwiseArray no_padding_extents({ rows, Base::input_dims(0), Base::input_dims(1), Base::input_dims(2) });
		return prev_out_grads.slice(no_padding_offsets, no_padding_extents);
	}
	/**
	 * It computes the dimension of the channel-rank of the output tensor of the layer.
	 *
	 * @param input_dim The dimensionality of the input tensor.
	 * @param receptor_size The spatial extent of the receptor.
	 * @param padding The spatial padding.
	 * @param dilation The dilation of the receptor.
	 * @param stride The convolution stride.
	 * @return The depth of the output tensors produced by the layer.
	 */
	static int calculate_output_dim(int input_dim, int receptor_size, int padding, int dilation, int stride) {
		return (input_dim - receptor_size - (receptor_size - 1) * dilation + 2 * padding) / stride + 1;
	}
private:
	const std::size_t filters;
	const std::size_t receptor_size;
	const std::size_t padding;
	const std::size_t stride;
	const std::size_t dilation;
	const int padded_height;
	const int padded_width;
	const int dil_receptor_size;
	// Staged computation caches
	std::vector<Matrix<Scalar>> biased_in_vec;
};

/**
 * An abstract class template that represents a non-linear activation function layer.
 */
template<typename Scalar, std::size_t Rank>
class ActivationLayer : public Layer<Scalar,Rank> {
	typedef Tensor<Scalar,Rank + 1> Data;
public:
	virtual ~ActivationLayer() = default;
	inline const Dimensions<int,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<int,Rank>& get_output_dims() const {
		return dims;
	}
protected:
	inline ActivationLayer(const Dimensions<int,Rank>& dims) :
			dims(dims),
			input_layer(false),
			params(share_weights ? Matrix<Scalar>(0, 0) : layer.weights),
			params_grad(0, 0),
			params_ref(params) { }
	inline ActivationLayer(const ActivationLayer<Scalar,Rank>& layer, bool share_weights) :
			dims(layer.dims),
			input_layer(layer.input_layer),
			params(share_weights ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_weights ? layer.params : params),
			in(layer.in),
			out(layer.out) { }
	/**
	 * Applies the non-linearity to the specified input matrix.
	 *
	 * @param in The input tensor mapped to a matrix.
	 * @return The activated input matrix.
	 */
	virtual Matrix<Scalar> activate(const Matrix<Scalar>& in) = 0;
	/**
	 * Differentiates the activation function and returns the derivative of the loss
	 * function w.r.t. the output of the previous layer in matrix form.
	 *
	 * @param in The input tensor mapped to a matrix.
	 * @param out The output produced by layer when applied to the input matrix.
	 * @param out_grads The derivative of the loss function w.r.t. to the output of
	 * this layer.
	 * @return The derivative of the loss function w.r.t. to the output of the
	 * previous layer.
	 */
	virtual Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) = 0;
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() { }
	inline void empty_cache() {
		in = Matrix<Scalar>(0, 0);
		out = Matrix<Scalar>(0, 0);
	}
	inline Matrix<Scalar>& get_params() {
		return params_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void enforce_constraints() { }
	inline Data pass_forward(Data in, bool training) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(in).template demote<>() == dims);
		assert(in.dimension(0) > 0);
		this->in = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(in));
		out = activate(this->in);
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(out, dims);
	}
	inline Data pass_back(Data out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).template demote<>() == dims);
		assert(out_grads.dimension(0) > 0 && out.rows() == out_grads.dimension(0));
		if (input_layer)
			return Data();
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(d_activate(in, out,
				Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads))), dims);
	}
	const Dimensions<int,Rank> dims;
	bool input_layer;
	Matrix<Scalar> params_grad;
	Matrix<Scalar>& params_ref;
	// Staged computation caches
	Matrix<Scalar> in;
	Matrix<Scalar> out;
private:
	Matrix<Scalar> params;
};

/**
 * A class template representing an identity activation layer that merely outputs
 * its input.
 */
template<typename Scalar, std::size_t Rank>
class IdentityActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline IdentityActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new IdentityActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in;
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return out_grads;
	}
};

/**
 * A class template that represents a linearly scaling activation layer.
 */
template<typename Scalar, std::size_t Rank>
class ScalingActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param scale The factor by which the input is to be scaled.
	 */
	inline ScalingActivationLayer(const Dimensions<int,Rank>& dims, Scalar scale) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			scale(scale) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ScalingActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in * scale;
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return out_grads * scale;
	}
private:
	const Scalar scale;
};

/**
 * A class template that represents a binary step activation function that outputs either
 * 1 or 0 based on the signum of its input. This function is not theoretically differentiable.
 */
template<typename Scalar, std::size_t Rank>
class BinaryStepActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline BinaryStepActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new BinaryStepActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : .0); });
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return Matrix<Scalar>::Zero(in.rows(), in.cols());
	}
};

/**
 * A class template representing a sigmoid activation function layer.
 */
template<typename Scalar, std::size_t Rank>
class SigmoidActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SigmoidActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SigmoidActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return ((-in).array().exp() + 1).inverse();
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return (out.array() *  (-out.array() + 1)) * out_grads.array();
	}
};

/**
 * A class template representing a hyperbolic tangent activation function layer.
 */
template<typename Scalar, std::size_t Rank>
class TanhActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline TanhActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new TanhActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.array().tanh();
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return (-out.array() * out.array() + 1) * out_grads.array();
	}
};

/**
 * A class template for a softmax activation function layer. Unlike most other activation
 * layers which represent element-wise functions, the softmax layer represents a multivariate
 * function.
 */
template<typename Scalar, std::size_t Rank>
class SoftmaxActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param epsilon A small constant to maintain numerical stability.
	 */
	inline SoftmaxActivationLayer(const Dimensions<int,Rank>& dims, Scalar epsilon = Utils<Scalar>::EPSILON2) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			epsilon(epsilon) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SoftmaxActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		/* First subtract the value of the greatest coefficient from each element row-wise
		 * to avoid an overflow due to raising e to great powers. */
		Matrix<Scalar> out = (in.array().colwise() - in.array().rowwise().maxCoeff()).exp();
		return out.array().colwise() / (out.array().rowwise().sum() + epsilon);
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		Matrix<Scalar> d_in(in.rows(), in.cols());
		for (int i = 0; i < d_in.rows(); ++i) {
			RowVector<Scalar> row_i = out.row(i);
			auto jacobian = (row_i.asDiagonal() - (row_i.transpose() * row_i));
			d_in.row(i) = out_grads.row(i) * jacobian;
		}
		return d_in;
	}
private:
	const Scalar epsilon;
};

/**
 * A class template representing a rectified linear unit (ReLU) activation function. ReLU
 * layers set all negative elements of the input to 0. This function is not theoretically
 * differentiable.
 */
template<typename Scalar, std::size_t Rank>
class ReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline ReLUActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ReLUActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(.0);
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : .0); })
				.cwiseProduct(out_grads);
	}
};

/**
 * A class template representing a leaky rectified linear unit activation function. Unlike
 * traditional ReLU layers leaky ReLU layers do not set negative elements of the input to
 * 0 but scale them by a small constant alpha. This function is not theoretically
 * differentiable.
 */
template<typename Scalar, std::size_t Rank>
class LeakyReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param alpha The factor by which negative inputs are to be scaled.
	 */
	inline LeakyReLUActivationLayer(const Dimensions<int,Rank>& dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new LeakyReLUActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(in * alpha);
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : alpha); })
				.cwiseProduct(out_grads);
	}
private:
	const Scalar alpha;
};

/**
 * A class template representing an exponential linear unit (ELU) activation function. ELUs
 * apply an exponential (e based) function scaled by alpha to the negative elements of the input.
 * ELU layers are not theoretically differentiable.
 */
template<typename Scalar, std::size_t Rank>
class ELUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param alpha The factor by which negative inputs are to be scaled.
	 */
	inline ELUActivationLayer(const Dimensions<int,Rank>& dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ELUActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i > .0 ? i : (alpha * (exp(i) - 1))); });
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		Matrix<Scalar> d_in(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); ++i) {
			for (int j = 0; j < in.rows(); ++j)
				d_in(j,i) = (in(j,i) > .0 ? 1.0 : (out(j,i) + alpha)) * out_grads(j,i);
		}
		return d_in;
	}
private:
	const Scalar alpha;
};

/**
 * A class template representing a parametric rectified linear unit (PReLU) activation function.
 * PReLU layers are Leaky ReLU activation functions with element-wise, learnable alphas. PReLU
 * activation functions are not theoretically differentiable.
 */
template<typename Scalar, std::size_t Rank>
class PReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param init_alpha The initial factor by which negative inputs are to be scaled.
	 */
	inline PReLUActivationLayer(const Dimensions<int,Rank>& dims, Scalar init_alpha = 1e-1) :
			Base::ActivationLayer(dims),
			init_alpha(init_alpha) {
		Base::params.resize(1, dims.get_volume());
		Base::params_grad.resize(1, dims.get_volume());
	}
	inline Layer<Scalar,Rank>* clone() const {
		return new PReLUActivationLayer(*this);
	}
protected:
	inline void init() {
		Base::params.setConstant(init_alpha);
		Base::params_grad.setZero(1, Base::dims.get_volume());
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(in * Base::params.row(0).asDiagonal());
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		Base::params_grad.row(0).setZero();
		Matrix<Scalar> d_in = Matrix<Scalar>(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); ++i) {
			for (int j = 0; j < in.rows(); ++j) {
				Scalar in_ji = in(j,i);
				if (in_ji >= 0)
					d_in(j,i) = out_grads(j,i);
				else {
					Scalar out_ji = out_grads(j,i);
					d_in(j,i) = Base::params(0,i) * out_ji;
					Base::params_grad(0,i) += in_ji * out_ji;
				}
			}
		}
		return d_in;
	}
private:
	const Scalar init_alpha;
};

/**
 * An abstract class template representing a pooling layer for batches of rank 3 data.
 */
template<typename Scalar>
class PoolingLayer : public Layer<Scalar,3> {
	typedef Tensor<Scalar,4> Data;
	typedef std::array<int,4> RankwiseArray;
public:
	inline PoolingLayer(const Dimensions<int,3>& input_dims, std::size_t receptor_size, std::size_t stride) :
			input_dims(input_dims),
			output_dims({ calculate_output_dim(input_dims(0), receptor_size, stride),
					calculate_output_dim(input_dims(1), receptor_size, stride), input_dims(2) }),
			receptor_size(receptor_size),
			stride(stride),
			receptor_area(receptor_size * receptor_size),
			height_rem(input_dims(0) - receptor_size),
			width_rem(input_dims(1) - receptor_size),
			input_layer(false),
			params(0, 0),
			params_grad(0, 0) {
		assert(input_dims(0) >= (int) receptor_size && input_dims(1) >= (int) receptor_size);
		assert(receptor_size > 0);
		assert(stride > 0);
	}
	inline const Dimensions<int,3>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<int,3>& get_output_dims() const {
		return output_dims;
	}
protected:
	/**
	 * Initializes the cache required for back-propagation.
	 */
	virtual void init_cache() = 0;
	/**
	 * Reduces the input vector into a single coefficient.
	 *
	 * @param patch A spatial patch of the input tensor (of size receptor*receptor) stretched
	 * into a row vector.
	 * @param patch_ind The index of the patch.
	 * @return The single numeral representing the result of the reduction.
	 */
	virtual Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) = 0;
	/**
	 * Differentiates the reduction function and returns the derivative of the loss function
	 * w.r.t. the output of the previous layer.
	 *
	 * @param grad The derivative of the loss function w.r.t. the output of this layer.
	 * @param patch_ind The index of the patch.
	 * @return The derivative of the loss function w.r.t. the output of the previous layer.
	 */
	virtual RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) = 0;
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() { }
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void enforce_constraints() { };
	inline Data pass_forward(Data in, bool training) {
		assert(Utils<Scalar>::template get_dims<4>(in).template demote<>() == input_dims);
		assert(in.dimension(0) > 0);
		rows = in.dimension(0);
		int depth = input_dims(2);
		RankwiseArray patch_offsets({ 0, 0, 0, 0 });
		RankwiseArray patch_extents({ 1, (int) receptor_size, (int) receptor_size, 1 });
		Data out(rows, output_dims(0), output_dims(1), output_dims(2));
		init_cache();
		int patch_ind = 0;
		for (int i = 0; i < rows; ++i) {
			patch_offsets[0] = i;
			int out_j = 0;
			for (int j = 0; j <= height_rem; j += stride, ++out_j) {
				patch_offsets[1] = j;
				int out_k = 0;
				for (int k = 0; k <= width_rem; k += stride, ++out_k) {
					patch_offsets[2] = k;
					for (int l = 0; l < depth; ++l) {
						patch_offsets[3] = l;
						Data patch = in.slice(patch_offsets, patch_extents);
						out(i,out_j,out_k,l) = reduce(Utils<Scalar>::template map_tensor_to_mat<4>(std::move(patch)), patch_ind++);
					}
				}
			}
		}
		return out;
	}
	inline Data pass_back(Data out_grads) {
		assert(Utils<Scalar>::template get_dims<4>(out_grads).template demote<>() == output_dims);
		assert(out_grads.dimension(0) > 0 && rows == out_grads.dimension(0));
		if (input_layer)
			return Data();
		int depth = input_dims(2);
		Dimensions<int,3> patch_dims({ (int) receptor_size, (int) receptor_size, 1 });
		RankwiseArray patch_offsets({ 0, 0, 0, 0});
		RankwiseArray patch_extents({ 1, patch_dims(0), patch_dims(1), patch_dims(2) });
		Data prev_out_grads(rows, input_dims(0), input_dims(1), depth);
		prev_out_grads.setZero();
		int patch_ind = 0;
		for (int i = 0; i < rows; ++i) {
			patch_offsets[0] = i;
			int out_grads_j = 0;
			for (int j = 0; j <= height_rem; j += stride, ++out_grads_j) {
				patch_offsets[1] = j;
				int out_grads_k = 0;
				for (int k = 0; k <= width_rem; k += stride, ++out_grads_k) {
					patch_offsets[2] = k;
					for (int l = 0; l < depth; ++l) {
						patch_offsets[3] = l;
						prev_out_grads.slice(patch_offsets, patch_extents) +=
								Utils<Scalar>::template map_mat_to_tensor<4>(d_reduce(out_grads(i,out_grads_j,out_grads_k,l),
										patch_ind++), patch_dims);
					}
				}
			}
		}
		return prev_out_grads;
	}
	/**
	 * It calculates the depth of the tensor output by the layer for all of its ranks except
	 * the first one (which denotes the samples in the batch).
	 *
	 * @param input_dim The dimensionality of the input tensor.
	 * @param receptor_size The spatial extent of the receptor.
	 * @param stride The stride at which the receptor is applied to the input tensor.
	 * @return The depth of the output tensor.
	 */
	static int calculate_output_dim(int input_dim, int receptor_size, int stride) {
		return (input_dim - receptor_size) / stride + 1;
	}
	const Dimensions<int,3> input_dims;
	const Dimensions<int,3> output_dims;
	const std::size_t receptor_size;
	const std::size_t stride;
	const int receptor_area;
	const int height_rem;
	const int width_rem;
	bool input_layer;
	// No actual parameters.
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
	// Keep track of the input rows.
	int rows;
};

/**
 * A class template representing a pooling layer that reduces patches of the input by taking their
 * sums.
 */
template<typename Scalar>
class SumPoolingLayer : public PoolingLayer<Scalar> {
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_size The spatial extent of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of elements
	 * by which the receptor is to be shifted after every step of the pooling process).
	 */
	inline SumPoolingLayer(const Dimensions<int,3>& input_dims, std::size_t receptor_size = 2, std::size_t stride = 2) :
			PoolingLayer<Scalar>::PoolingLayer(input_dims, receptor_size, stride) { }
	inline Layer<Scalar,3>* clone() const {
		return new SumPoolingLayer(*this);
	}
protected:
	inline void init_cache() { };
	inline Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) {
		return patch.sum();
	}
	inline RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) {
		return RowVector<Scalar>::Constant(PoolingLayer<Scalar>::receptor_area, grad);
	}
	inline void empty_cache() { }
};

/**
 * A class template representing a pooling layer that reduces patches of the input by taking their
 * means.
 */
template<typename Scalar>
class MeanPoolingLayer : public PoolingLayer<Scalar> {
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_size The spatial extent of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of elements
	 * by which the receptor is to be shifted after every step of the pooling process).
	 */
	inline MeanPoolingLayer(const Dimensions<int,3>& input_dims, std::size_t receptor_size = 2, std::size_t stride = 2) :
			PoolingLayer<Scalar>::PoolingLayer(input_dims, receptor_size, stride) { }
	inline Layer<Scalar,3>* clone() const {
		return new MeanPoolingLayer(*this);
	}
protected:
	inline void init_cache() { }
	inline Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) {
		return patch.mean();
	}
	inline RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) {
		return RowVector<Scalar>::Constant(PoolingLayer<Scalar>::receptor_area,
				grad / (Scalar) PoolingLayer<Scalar>::receptor_area);
	}
	inline void empty_cache() { }
};

/**
 * A class template representing a pooling layer that reduces patches of the input by taking their
 * maximums.
 */
template<typename Scalar>
class MaxPoolingLayer : public PoolingLayer<Scalar> {
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_size The spatial extent of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of elements
	 * by which the receptor is to be shifted after every step of the pooling process).
	 */
	inline MaxPoolingLayer(const Dimensions<int,3>& input_dims, unsigned receptor_size = 2, unsigned stride = 2) :
			PoolingLayer<Scalar>::PoolingLayer(input_dims, receptor_size, stride) { }
	inline Layer<Scalar,3>* clone() const {
		return new MaxPoolingLayer(*this);
	}
protected:
	inline void init_cache() {
		max_inds = std::vector<unsigned>(PoolingLayer<Scalar>::rows *
				PoolingLayer<Scalar>::output_dims.get_volume());
	}
	inline Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) {
		int max_ind = 0;
		Scalar max = Utils<Scalar>::MIN;
		for (int i = 0; i < patch.cols(); ++i) {
			Scalar val_i = patch(i);
			if (val_i > max) {
				max = val_i;
				max_ind = i;
			}
		}
		max_inds[patch_ind] = max_ind;
		return max;
	}
	inline RowVector<Scalar> d_reduce(Scalar grad, unsigned patch_ind) {
		RowVector<Scalar> d_patch = RowVector<Scalar>::Zero(PoolingLayer<Scalar>::receptor_area);
		d_patch(max_inds[patch_ind]) = grad;
		return d_patch;
	}
	inline void empty_cache() {
		max_inds = std::vector<unsigned>(0);
	}
private:
	// Cache
	std::vector<unsigned> max_inds;
};

/**
 * An abstract base class template for a batch normalization layer.
 */
template<typename Scalar, std::size_t Rank>
class BatchNormLayerBase : public Layer<Scalar,Rank> {
public:
	virtual ~BatchNormLayerBase() = default;
	inline const Dimensions<int,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<int,Rank>& get_output_dims() const {
		return dims;
	}
protected:
	typedef Tensor<Scalar,Rank + 1> Data;
	inline BatchNormLayerBase(const Dimensions<int,Rank>& dims, int depth, Scalar norm_avg_decay, Scalar epsilon) :
			dims(dims),
			depth(depth),
			norm_avg_decay(norm_avg_decay),
			epsilon(epsilon),
			input_layer(false),
			avg_means(depth, dims.get_volume() / depth),
			avg_inv_sds(depth, dims.get_volume() / depth),
			avgs_init(false),
			params(2 * depth, dims.get_volume() / depth),
			params_grad(2 * depth, dims.get_volume() / depth),
			cache_vec(depth) {
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() {
		for (int i = 0; i < params.rows(); i += 2) {
			// Gamma
			params.row(i).setOnes();
			// Beta
			params.row(i + 1).setZero();
		}
		params_grad.setZero(params.rows(), params.cols());
		avg_means.setZero(avg_means.rows(), avg_means.cols());
		avg_inv_sds.setZero(avg_means.rows(), avg_inv_sds.cols());
		avgs_init = false;
	}
	inline void empty_cache() {
		for (unsigned i = 0; i < cache_vec.size(); ++i) {
			Cache& cache = cache_vec[i];
			cache.inv_in_sd = RowVector<Scalar>(0);
			cache.std_in = Matrix<Scalar>(0, 0);
		}
	}
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void enforce_constraints() { }
	inline Data _pass_forward(Data in, const Dimensions<int,Rank>& output_dims, bool training, int i) {
		Matrix<Scalar> in_ch_i = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(in));
		if (training) {
			Cache& cache = cache_vec[i];
			RowVector<Scalar> means = in_ch_i.colwise().mean();
			Matrix<Scalar> norm_in = in_ch_i.rowwise() - means;
			cache.inv_in_sd = (norm_in.array().square().colwise().mean() + epsilon).sqrt().inverse();
			cache.std_in = norm_in * cache.inv_in_sd.asDiagonal();
			in_ch_i = cache.std_in;
			// Maintain a moving average of means and variances for testing.
			if (avgs_init) {
				avg_means.row(i) = (1.0 - norm_avg_decay) * avg_means.row(i) + norm_avg_decay * means;
				avg_inv_sds.row(i) = (1.0 - norm_avg_decay) * avg_inv_sds.row(i) + norm_avg_decay *
						cache.inv_in_sd;
			} else {
				avg_means.row(i) = means;
				avg_inv_sds.row(i) = cache.inv_in_sd;
				avgs_init = true;
			}
		} else // For testing, use the moving averages.
			in_ch_i = (in_ch_i.rowwise() - avg_means.row(i)) * avg_inv_sds.row(i).asDiagonal();
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(((in_ch_i * params.row(2 * i).asDiagonal())
				.rowwise() + params.row(2 * i + 1)).eval(), output_dims);
	}
	inline Data _pass_back(Data out_grads, const Dimensions<int,Rank>& output_dims, int i) {
		int rows = out_grads.dimension(0);
		Cache& cache = cache_vec[i];
		Matrix<Scalar> std_in_grads_i;
		/* Back-propagate the gradient through the batch normalization 'function' and also calculate the
		 * gradients of the betas and gammas. */
		{ // Manage memory by scope restriction.
			Matrix<Scalar> out_grads_ch_map_i = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads));
			params_grad.row(2 * i) = out_grads_ch_map_i.cwiseProduct(cache.std_in).colwise().sum();
			params_grad.row(2 * i + 1) = out_grads_ch_map_i.colwise().sum();
			if (input_layer)
				return Data();
			std_in_grads_i = out_grads_ch_map_i * params.row(2 * i).asDiagonal();
		}
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(((((rows * std_in_grads_i).rowwise() -
				std_in_grads_i.colwise().sum()) - cache.std_in *
				(cache.std_in.cwiseProduct(std_in_grads_i).colwise().sum().asDiagonal())) *
				((1.0 / rows) * cache.inv_in_sd).asDiagonal()).eval(), output_dims);
	}
	const Dimensions<int,Rank> dims;
	const int depth;
	const Scalar norm_avg_decay;
	const Scalar epsilon;
	bool input_layer;
	// Dynamic batch normalization parameters.
	Matrix<Scalar> avg_means;
	Matrix<Scalar> avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
	// Staged computation cache_vec
	struct Cache {
		RowVector<Scalar> inv_in_sd;
		Matrix<Scalar> std_in;
	};
	std::vector<Cache> cache_vec;
};

/**
 * A class template for a batch normalization layer.
 */
template<typename Scalar, std::size_t Rank>
class BatchNormLayer : public BatchNormLayerBase<Scalar,Rank> {
	typedef BatchNormLayerBase<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(const Dimensions<int,Rank>& dims, Scalar norm_avg_decay = .1, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			Base::template BatchNormLayerBase(dims, 1, norm_avg_decay, epsilon) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new BatchNormLayer(*this);
	}
protected:
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(in).template demote<>() == Base::dims);
		assert(in.dimension(0) > 0);
		return Base::_pass_forward(std::move(in), Base::dims, training, 0);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).template demote<>() == Base::dims);
		assert(out_grads.dimension(0) > 0 && Base::cache_vec[0].std_in.rows() == out_grads.dimension(0));
		return Base::_pass_back(std::move(out_grads), Base::dims, 0);
	}
};

/**
 * A partial template specialization for multi-channel input tensors.
 */
template<typename Scalar>
class BatchNormLayer<Scalar,3> : public BatchNormLayerBase<Scalar,3> {
	typedef BatchNormLayerBase<Scalar,3> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(Dimensions<int,3> dims, Scalar norm_avg_decay = .1, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			Base::template BatchNormLayerBase(dims, dims(2), norm_avg_decay, epsilon) { }
	inline Layer<Scalar,3>* clone() const {
		return new BatchNormLayer(*this);
	}
protected:
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert(Utils<Scalar>::template get_dims<4>(in).template demote<>() == Base::dims);
		assert(in.dimension(0) > 0);
		if (Base::depth == 1)
			return Base::_pass_forward(std::move(in), Base::dims, training, 0);
		else { // Multi-channel image data; depth-wise normalization.
			int rows = in.dimension(0);
			typename Base::Data out(rows, Base::dims(0), Base::dims(1), Base::dims(2));
			Dimensions<int,3> slice_dims({ Base::dims(0), Base::dims(1), 1 });
			std::array<int,4> offsets({ 0, 0, 0, 0 });
			std::array<int,4> extents({ rows, slice_dims(0), slice_dims(1), slice_dims(2) });
			for (int i = 0; i < Base::depth; ++i) {
				offsets[3] = i;
				typename Base::Data in_slice_i = in.slice(offsets, extents);
				out.slice(offsets, extents) = Base::_pass_forward(std::move(in_slice_i), slice_dims, training, i);
			}
			return out;
		}
	}
	inline typename Base::Data pass_back(typename Base::Data out_grads) {
		assert(Utils<Scalar>::template get_dims<4>(out_grads).template demote<>() == Base::dims);
		assert(out_grads.dimension(0) > 0 && Base::cache_vec[0].std_in.rows() == out_grads.dimension(0));
		if (Base::depth == 1)
			return Base::_pass_back(std::move(out_grads), Base::dims, 0);
		else {
			int rows = out_grads.dimension(0);
			typename Base::Data prev_out_grads;
			if (!Base::is_input_layer())
				prev_out_grads = typename Base::Data(rows, Base::dims(0), Base::dims(1), Base::dims(2));
			Dimensions<int,3> slice_dims({ Base::dims(0), Base::dims(1), 1 });
			std::array<int,4> offsets({ 0, 0, 0, 0 });
			std::array<int,4> extents({ rows, slice_dims(0), slice_dims(1), slice_dims(2) });
			for (int i = 0; i < Base::depth; ++i) {
				offsets[3] = i;
				typename Base::Data out_grads_slice_i = out_grads.slice(offsets, extents);
				if (Base::is_input_layer())
					Base::_pass_back(std::move(out_grads_slice_i), slice_dims, i);
				else
					prev_out_grads.slice(offsets, extents) = Base::_pass_back(std::move(out_grads_slice_i), slice_dims, i);
			}
			return prev_out_grads;
		}
	}
};

/**
 * A class template representing a drop-out layer.
 */
template<typename Scalar, std::size_t Rank>
class DropoutLayer : public Layer<Scalar,Rank> {
	typedef Tensor<Scalar,Rank + 1> Data;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param dropout_prob The probability of an element of the input tensor being set to 0.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline DropoutLayer(const Dimensions<int,Rank>& dims, Scalar dropout_prob, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			dims(dims),
			dropout_prob(dropout_prob),
			epsilon(epsilon),
			dropout(Utils<Scalar>::decidedly_greater(dropout_prob, .0)),
			input_layer(false),
			params(0, 0),
			params_grad(0, 0) {
		assert(dropout_prob <= 1 && "dropout prob must not be greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline Layer<Scalar,Rank>* clone() const {
		return new DropoutLayer(*this);
	}
	inline const Dimensions<int,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<int,Rank>& get_output_dims() const {
		return dims;
	}
protected:
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() { }
	inline void empty_cache() {
		dropout_mask = Matrix<Scalar>(0, 0);
	}
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void enforce_constraints() { }
	inline Data pass_forward(Data in, bool training) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(in).template demote<>() == dims);
		assert(in.dimension(0) > 0);
		if (training && dropout) {
			Matrix<Scalar> in_mat = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(in));
			dropout_mask = Matrix<Scalar>(in_mat.rows(), in_mat.cols());
			dropout_mask.setRandom(in_mat.rows(), in_mat.cols());
			// Inverted dropout.
			Scalar scaling_factor = 1 / (1 - dropout_prob + epsilon);
			dropout_mask = ((dropout_mask.array() + 1) / 2).unaryExpr([this,scaling_factor](Scalar i) {
				return (Scalar) (i <= dropout_prob ? .0 : scaling_factor);
			});
			return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(in_mat.cwiseProduct(dropout_mask).eval(), dims);
		}
		return in;
	}
	inline Data pass_back(Data out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).template demote<>() == dims);
		assert(out_grads.dimension(0) > 0 && dropout_mask.rows() == out_grads.dimension(0));
		if (input_layer)
			return Data();
		// The derivative of the dropout 'function'.
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads))
				.cwiseProduct(dropout_mask).eval(), dims);
	}
private:
	const Dimensions<int,Rank> dims;
	const Scalar dropout_prob;
	const Scalar epsilon;
	const bool dropout;
	bool input_layer;
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
	// Staged computation cache_vec
	Matrix<Scalar> dropout_mask;
};

} /* namespace cattle */

#endif /* LAYER_H_ */
