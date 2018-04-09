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
#include "RegularizationPenalty.h"
#include "WeightInitialization.h"

namespace cattle {

// TODO CPU convolution performance improvement.
// TODO Optional GPU acceleration using cuBLAS and cuDNN.

/**
 * An alias for a shared pointer to a WeightInitialization implementation instance of
 * an arbitrary scalar type.
 */
template<typename Scalar>
using WeightInitSharedPtr = std::shared_ptr<WeightInitialization<Scalar>>;

/**
 * An alias for a shared pointer to a regularization penalty of an arbitrary scalar type.
 */
template<typename Scalar>
using RegPenSharedPtr = std::shared_ptr<RegularizationPenalty<Scalar>>;

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
public:
	static const RegPenSharedPtr<Scalar> DEFAULT_REG_PEN;
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
	virtual const Dimensions<std::size_t,Rank>& get_input_dims() const = 0;
	/**
	 * A simple constant getter method for the output dimensionality of the layer.
	 *
	 * @return A constant reference to the member variable denoting the dimensions of the
	 * tensors output by the layer along all ranks except the first one.
	 */
	virtual const Dimensions<std::size_t,Rank>& get_output_dims() const = 0;
	/**
	 * A method that returns whether the layer has parameters that can be learned.
	 *
	 * @return Whether the layer uses learnable parameters.
	 */
	inline bool is_parametric() {
		return get_params().rows() > 0 && get_params().cols() > 0;
	}
protected:
	// Rank is increased by one to allow for batch training.
	static constexpr std::size_t DATA_RANK = Rank + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
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
	 * It calculates the regularization penalty of the layer's parameters. If the layer is not
	 * parametric, 0 is returned.
	 *
	 * @return A scalar representing the penalty on the magnitude of the layer's parameters.
	 */
	virtual Scalar get_regularization_penalty() = 0;
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
	 * layer updating the gradient of its learnable parameters along the way if there are
	 * any. If there are, it also calculates the derivative of the regularization penalty
	 * w.r.t. to the layer's parameters and adds it to their gradient.
	 *
	 * @param out_grads The derivative of the loss function w.r.t. the output of the
	 * layer
	 * @return The derivative of the loss function w.r.t. the output of the previous layer
	 * or a null tensor if the layer is an input layer.
	 */
	virtual Data pass_back(Data out_grads) = 0;
};

// Initialize the static default regularization penalty.
template<typename Scalar, std::size_t Rank>
const RegPenSharedPtr<Scalar> Layer<Scalar,Rank>::DEFAULT_REG_PEN(new NoRegularizationPenalty<Scalar>());

/**
 * An abstract base class template for layers representing linear kernel-based operations
 * such as matrix multiplication or convolution.
 */
template<typename Scalar, std::size_t Rank>
class KernelLayer : public Layer<Scalar,Rank> {
public:
	virtual ~KernelLayer() = default;
	const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return input_dims;
	}
	const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return output_dims;
	}
protected:
	inline KernelLayer(const Dimensions<std::size_t,Rank>& input_dims, Dimensions<std::size_t,Rank> output_dims,
			WeightInitSharedPtr<Scalar> weight_init, RegPenSharedPtr<Scalar> weight_reg, std::size_t weight_rows,
			std::size_t weight_cols, Scalar max_norm_constraint) :
				input_dims(input_dims),
				output_dims(output_dims),
				weight_init(weight_init),
				weight_reg(weight_reg),
				max_norm_constraint(max_norm_constraint),
				max_norm(internal::Utils<Scalar>::decidedly_greater(max_norm_constraint, (Scalar) 0)),
				input_layer(false),
				weights_shared(false),
				weights(weight_rows, weight_cols),
				weights_grad(weight_rows, weight_cols),
				weights_ref(weights) {
		assert(weight_init != nullptr);
		assert(weight_reg != nullptr);
	}
	inline KernelLayer(const KernelLayer<Scalar,Rank>& layer, bool share_params = false) :
			input_dims(layer.input_dims),
			output_dims(layer.output_dims),
			weight_init(layer.weight_init),
			weight_reg(layer.weight_reg),
			max_norm_constraint(layer.max_norm_constraint),
			max_norm(layer.max_norm),
			input_layer(layer.input_layer),
			weights_shared(share_params),
			weights(share_params ? Matrix<Scalar>(0, 0) : layer.weights),
			weights_grad(layer.weights_grad),
			weights_ref(share_params ? layer.weights : weights) { }
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
	inline Scalar get_regularization_penalty() {
		return weights_shared ? 0 : weight_reg->function(weights_ref);
	}
	const Dimensions<std::size_t,Rank> input_dims;
	const Dimensions<std::size_t,Rank> output_dims;
	const WeightInitSharedPtr<Scalar> weight_init;
	const RegPenSharedPtr<Scalar> weight_reg;
	const Scalar max_norm_constraint;
	const bool max_norm;
	const bool weights_shared;
	/* Eigen matrices are backed by arrays allocated on the heap, so these
	 * members do not burden the stack. */
	Matrix<Scalar> weights_grad;
	Matrix<Scalar>& weights_ref;
private:
	bool input_layer;
	mutable Matrix<Scalar> weights;
};

/**
 * A class template representing a fully connected layer.
 */
template<typename Scalar, std::size_t Rank>
class FCLayer : public KernelLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef KernelLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * @param output_size The length of the vector output for each sample.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the parametric kernel backing the layer.
	 * @param weight_reg The regularization function to apply to the layer's parameters.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline FCLayer(const Dimensions<std::size_t,Rank>& input_dims, std::size_t output_size, WeightInitSharedPtr<Scalar> weight_init,
			RegPenSharedPtr<Scalar> weight_reg = Root::DEFAULT_REG_PEN, Scalar max_norm_constraint = 0) :
				Base::KernelLayer(input_dims, Dimensions<std::size_t,Rank>({ output_size }), weight_init, weight_reg,
						input_dims.get_volume() + 1, output_size, max_norm_constraint),
				out_conversion_dims(Base::output_dims.template promote<>()),
				prev_out_conversion_dims(Base::input_dims.template promote<>()) { }
	inline Root* clone() const {
		return new FCLayer(*this);
	}
protected:
	inline FCLayer(const FCLayer<Scalar,Rank>& layer, bool share_params = false) :
			Base::KernelLayer(layer, share_params),
			out_conversion_dims(layer.out_conversion_dims),
			prev_out_conversion_dims(layer.prev_out_conversion_dims),
			biased_in(layer.biased_in) { }
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return new FCLayer(*this, true);
	}
	inline void empty_cache() {
		biased_in = Matrix<Scalar>(0, 0);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(in.dimensions()).template demote<>()) == Base::input_dims);
		assert(in.dimension(0) > 0);
		unsigned input_size = Base::input_dims.get_volume();
		// Add a 1-column to the input for the bias trick.
		biased_in = Matrix<Scalar>(in.dimension(0), input_size + 1);
		biased_in.leftCols(input_size) = MatrixMap<Scalar>(in.data(), in.dimension(0), input_size);
		biased_in.col(input_size).setOnes();
		Matrix<Scalar> out = biased_in * Base::weights_ref;
		out_conversion_dims[0] = out.rows();
		return TensorMap<Scalar,Root::DATA_RANK>(out.data(), out_conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && biased_in.rows() == out_grads.dimension(0));
		MatrixMap<Scalar> out_grads_mat(out_grads.data(), out_grads.dimension(0), Base::output_dims.get_volume());
		// Compute the gradient of the outputs with respect to the weights.
		if (Base::weights_shared)
			Base::weights_grad = biased_in.transpose() * out_grads_mat;
		else
			Base::weights_grad = biased_in.transpose() * out_grads_mat + Base::weight_reg->d_function(Base::weights_ref);
		if (Base::is_input_layer())
			return typename Root::Data();
		/* Remove the bias row from the weight matrix, transpose it, and compute the derivative w.r.t. the
		 * previous layer's output. */
		Matrix<Scalar> prev_out_grads = out_grads_mat * Base::weights_ref.topRows(Base::input_dims.get_volume()).transpose();
		prev_out_conversion_dims[0] = prev_out_grads.rows();
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grads.data(), prev_out_conversion_dims);
	}
private:
	RankwiseArray out_conversion_dims;
	RankwiseArray prev_out_conversion_dims;
	// Staged computation caches
	Matrix<Scalar> biased_in;
};

/**
 * A class template for convolutional layers.
 */
template<typename Scalar>
class ConvLayer : public KernelLayer<Scalar,3> {
	typedef Layer<Scalar,3> Root;
	typedef KernelLayer<Scalar,3> Base;
	typedef std::array<std::size_t,4> RankwiseArray;
	typedef std::array<std::pair<std::size_t,std::size_t>,4> PaddingsArray;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * The ranks of the input tensors denote the sample, height, width, and channel (N,H,W,C).
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the parametric kernel backing the layer.
	 * @param weight_reg The regularization function to apply to the layer's parameters.
	 * @param receptor_height The height of the base of the receptor cuboid.
	 * @param receptor_width The width of the base of the receptor cuboid.
	 * @param padding The length of padding to apply to the input tensor along its width and
	 * weight.
	 * @param stride The convolution stride i.e. the number of elements by which the receptor
	 * is to be shifted at each step of the convolution.
	 * @param dilation The size of the spatial (height- and width-wise) padding between voxels
	 * of the receptor.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline ConvLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			RegPenSharedPtr<Scalar> weight_reg = Root::DEFAULT_REG_PEN, std::size_t receptor_height = 3, std::size_t receptor_width = 3,
			std::size_t padding = 1, std::size_t stride = 1, std::size_t dilation = 0, Scalar max_norm_constraint = 0) :
				/* For every filter, there is a column in the weight matrix with the same number of
				 * elements as the area of the receptive field (F_H * F_W * D) + 1 for the bias row. */
				Base::KernelLayer(input_dims, Dimensions<std::size_t,3>({
						calculate_output_dim(input_dims(0), receptor_height, padding, dilation, stride),
						calculate_output_dim(input_dims(1), receptor_width, padding, dilation, stride),
						filters }), weight_init, weight_reg, receptor_height * receptor_width * input_dims(2) + 1,
						filters, max_norm_constraint),
				filters(filters),
				receptor_height(receptor_height),
				receptor_width(receptor_width),
				padding(padding),
				stride(stride),
				dilation(dilation),
				padded_height(input_dims(0) + 2 * padding),
				padded_width(input_dims(1) + 2 * padding),
				dil_receptor_height(receptor_height + (receptor_height - 1) * dilation),
				dil_receptor_width(receptor_width + (receptor_width - 1) * dilation),
				out_conversion_dims({ 0u, Base::output_dims(0), Base::output_dims(1), Base::output_dims(2) }),
				patch_offsets({ 0u, 0u, 0u, 0u }),
				patch_extents({ 0u, dil_receptor_height, dil_receptor_width, input_dims(2) }),
				dil_strides({ 1u, dilation + 1u, dilation + 1u, 1u }),
				paddings({ std::make_pair(0, 0), std::make_pair(padding, padding), std::make_pair(padding, padding),
						std::make_pair(0, 0) }) {
		assert(filters > 0);
		assert(receptor_height > 0);
		assert(receptor_width > 0);
		assert(stride > 0);
		assert(input_dims(1) + 2 * padding >= dil_receptor_height &&
				input_dims(2) + 2 * padding >= dil_receptor_width);
	}
	inline Root* clone() const {
		return new ConvLayer(*this);
	}
protected:
	inline ConvLayer(const ConvLayer<Scalar>& layer, bool share_params = false) :
			Base::KernelLayer(layer, share_params),
			filters(layer.filters),
			receptor_height(layer.receptor_height),
			receptor_width(layer.receptor_width),
			padding(layer.padding),
			stride(layer.stride),
			dilation(layer.dilation),
			padded_height(layer.padded_height),
			padded_width(layer.padded_width),
			dil_receptor_height(layer.dil_receptor_height),
			dil_receptor_width(layer.dil_receptor_width),
			out_conversion_dims(layer.out_conversion_dims),
			patch_offsets(layer.patch_offsets),
			patch_extents(layer.patch_extents),
			dil_strides(layer.dil_strides),
			paddings(layer.paddings) { }
	inline Root* clone_with_shared_params() const {
		return new ConvLayer(*this, true);
	}
	inline void empty_cache() {
		biased_in = Matrix<Scalar>(0, 0);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == Base::input_dims);
		assert(in.dimension(0) > 0);
		// Spatial padding.
		if (padding > 0)
			in = typename Root::Data(in.pad(paddings));
		std::size_t rows = in.dimension(0);
		std::size_t total_patches = rows * Base::output_dims(0) * Base::output_dims(1);
		std::size_t receptor_vol = Base::weights_ref.rows() - 1;
		/* Flatten the receptor cuboids into row vectors and concatenate them. Each row stands for one stretched
		 * out receptor of one sample. The same receptor location along all samples of the batch is represented
		 * by a contiguous block of these rows. There is one block for each receptor location. */
		std::size_t patch_ind = 0;
		patch_extents[0] = rows;
		biased_in = Matrix<Scalar>(total_patches, receptor_vol + 1);
		for (std::size_t j = 0; j <= padded_height - dil_receptor_height; j += stride) {
			patch_offsets[1] = j;
			for (std::size_t k = 0; k <= padded_width - dil_receptor_width; k += stride) {
				patch_offsets[2] = k;
				typename Root::Data patch;
				// If the patch is dilated, skip the 'internal padding' when flattening it into a matrix.
				if (dilation > 0)
					patch = in.slice(patch_offsets, patch_extents).stride(dil_strides);
				else
					patch = in.slice(patch_offsets, patch_extents);
				biased_in.block(patch_ind, 0, rows, receptor_vol) = MatrixMap<Scalar>(patch.data(), rows, receptor_vol);
				patch_ind += rows;
			}
		}
		assert(patch_ind == total_patches);
		// Bias trick.
		biased_in.col(receptor_vol).setOnes();
		Matrix<Scalar> out = biased_in * Base::weights_ref;
		out_conversion_dims[0] = rows;
		return TensorMap<Scalar,4>(out.data(), out_conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,4>(out_grads.dimensions()).template demote<>()) == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && biased_in.rows() / (Base::output_dims(0) *
				Base::output_dims(1)) == out_grads.dimension(0));
		std::size_t rows = out_grads.dimension(0);
		std::size_t patches = Base::output_dims(0) * Base::output_dims(1);
		std::size_t receptor_vol = Base::weights_ref.rows() - 1;
		MatrixMap<Scalar> out_grads_mat_map = MatrixMap<Scalar>(out_grads.data(),
				rows * Base::output_dims(0) * Base::output_dims(1), filters);
		if (Base::weights_shared)
			Base::weights_grad = biased_in.transpose() * out_grads_mat_map;
		else
			Base::weights_grad = biased_in.transpose() * out_grads_mat_map + Base::weight_reg->d_function(Base::weights_ref);
		if (Base::is_input_layer())
			return typename Root::Data();
		/* Remove the bias row from the weight matrix, transpose it, and compute the gradient of the
		 * previous layer's output. */
		Matrix<Scalar> prev_out_grads_mat = out_grads_mat_map * Base::weights_ref.topRows(receptor_vol).transpose();
		/* Given the gradient of the stretched out receptor patches, perform a 'backwards' convolution
		 * to get the derivative w.r.t. the individual input nodes. */
		typename Root::Data prev_out_grads(rows, padded_height, padded_width, Base::input_dims(2));
		prev_out_grads.setZero();
		std::size_t patch_ind = 0;
		patch_extents[0] = rows;
		for (std::size_t j = 0; j <= padded_height - dil_receptor_height; j += stride) {
			patch_offsets[1] = j;
			for (std::size_t k = 0; k <= padded_width - dil_receptor_width; k += stride) {
				patch_offsets[2] = k;
				// Accumulate the gradients where the receptor-patch-tensors overlap.
				Matrix<Scalar> prev_out_grads_block = prev_out_grads_mat.block(patch_ind, 0, rows, receptor_vol);
				TensorMap<Scalar,4> prev_out_grads_block_map(prev_out_grads_block.data(), patch_extents);
				if (dilation > 0)
					prev_out_grads.slice(patch_offsets, patch_extents).stride(dil_strides) += prev_out_grads_block_map;
				else
					prev_out_grads.slice(patch_offsets, patch_extents) += prev_out_grads_block_map;
				patch_ind += rows;
			}
		}
		assert(patch_ind == prev_out_grads_mat.rows());
		if (padding > 0) {
			// Cut off the padding.
			RankwiseArray no_padding_offsets({ 0, padding, padding, 0 });
			RankwiseArray no_padding_extents({ rows, Base::input_dims(0), Base::input_dims(1), Base::input_dims(2) });
			return prev_out_grads.slice(no_padding_offsets, no_padding_extents);
		} else
			return prev_out_grads;
	}
	/**
	 * It computes the dimension of the output tensor along a rank.
	 *
	 * @param input_dim The dimensionality of the input tensor.
	 * @param receptor_size The spatial extent of the receptor.
	 * @param padding The spatial padding.
	 * @param dilation The dilation of the receptor.
	 * @param stride The convolution stride.
	 * @return The dimension of the output tensor along the specified rank.
	 */
	static std::size_t calculate_output_dim(std::size_t input_dim, std::size_t receptor_size, std::size_t padding,
			std::size_t dilation, std::size_t stride) {
		return (input_dim - receptor_size - (receptor_size - 1) * dilation + 2 * padding) / stride + 1;
	}
private:
	// The defining attributes of the convolutional layer.
	const std::size_t filters;
	const std::size_t receptor_height;
	const std::size_t receptor_width;
	const std::size_t padding;
	const std::size_t stride;
	const std::size_t dilation;
	// Pre-computed values to improve propagation-time performance.
	const std::size_t padded_height;
	const std::size_t padded_width;
	const std::size_t dil_receptor_height;
	const std::size_t dil_receptor_width;
	RankwiseArray out_conversion_dims;
	RankwiseArray patch_offsets;
	RankwiseArray patch_extents;
	RankwiseArray dil_strides;
	PaddingsArray paddings;
	// Staged computation caches
	Matrix<Scalar> biased_in;
};

/**
 * An abstract class template that represents an activation function layer.
 */
template<typename Scalar, std::size_t Rank>
class ActivationLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	virtual ~ActivationLayer() = default;
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return dims;
	}
protected:
	inline ActivationLayer(const Dimensions<std::size_t,Rank>& dims, std::size_t param_rows = 0,
			std::size_t params_cols = 0) :
				dims(dims),
				input_layer(false),
				params(param_rows, params_cols),
				params_grad(param_rows, params_cols),
				params_ref(params) { }
	inline ActivationLayer(const ActivationLayer<Scalar,Rank>& layer, bool share_params = false) :
			dims(layer.dims),
			input_layer(layer.input_layer),
			params(share_params ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_params ? layer.params : params) { }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() { }
	inline Matrix<Scalar>& get_params() {
		return params_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void enforce_constraints() { }
	inline Scalar get_regularization_penalty() {
		return 0;
	}
	inline void empty_cache() { }
	const Dimensions<std::size_t,Rank> dims;
	Matrix<Scalar> params_grad;
	Matrix<Scalar>& params_ref;
private:
	bool input_layer;
	mutable Matrix<Scalar> params;
};

/**
 * A class template representing an identity activation layer that merely outputs
 * its input.
 */
template<typename Scalar, std::size_t Rank>
class IdentityActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline IdentityActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new IdentityActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return in;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		return out_grads;
	}
private:
	std::size_t batch_size;
};

/**
 * A class template that represents a linearly scaling activation layer.
 */
template<typename Scalar, std::size_t Rank>
class ScalingActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param scale The factor by which the input is to be scaled.
	 */
	inline ScalingActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar scale) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			scale(scale) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ScalingActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return in * scale;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		return out_grads * scale;
	}
private:
	const Scalar scale;
	std::size_t batch_size;
};

/**
 * A class template that represents a binary step activation function that outputs either
 * 1 or 0 based on the signum of its input. This function is not theoretically differentiable.
 */
template<typename Scalar, std::size_t Rank>
class BinaryStepActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline BinaryStepActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new BinaryStepActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : .0); });
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		return out_grads.constant((Scalar) 0);
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a sigmoid activation function layer.
 */
template<typename Scalar, std::size_t Rank>
class SigmoidActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SigmoidActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SigmoidActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline void empty_cache() {
		out = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		auto act = ((-in).exp() + in.constant((Scalar) 1)).inverse();
		if (training) {
			out = act;
			return out;
		}
		return act;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && out.dimension(0) == out_grads.dimension(0));
		return (out * (-out + out.constant((Scalar) 1))) * out_grads;
	}
private:
	// Staged computation cache.
	typename Root::Data out;
};

/**
 * A class template representing a hyperbolic tangent activation function layer.
 */
template<typename Scalar, std::size_t Rank>
class TanhActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline TanhActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new TanhActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline void empty_cache() {
		out = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		auto act = in.tanh();
		if (training) {
			out = act;
			return out;
		}
		return act;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && out.dimension(0) == out_grads.dimension(0));
		return (-out * out + out.constant((Scalar) 1)) * out_grads;
	}
private:
	typename Root::Data out;
};

/**
 * A class template for a softmax activation function layer. Unlike most other activation
 * layers which represent element-wise functions, the softmax layer represents a multivariate
 * function.
 */
template<typename Scalar, std::size_t Rank>
class SoftmaxActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param epsilon A small constant to maintain numerical stability.
	 */
	inline SoftmaxActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar epsilon = internal::Utils<Scalar>::EPSILON2) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			epsilon(epsilon),
			conversion_dims(dims.template promote<>()) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SoftmaxActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline void empty_cache() {
		out = Matrix<Scalar>(0, 0);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
		/* First subtract the value of the greatest coefficient from each element row-wise
		 * to avoid an overflow due to raising e to great powers. */
		Matrix<Scalar> act = (in_mat.array().colwise() - in_mat.array().rowwise().maxCoeff()).exp();
		act = act.array().colwise() / (act.array().rowwise().sum() + epsilon);
		conversion_dims[0] = rows;
		if (training) {
			out = std::move(act);
			return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
		}
		return TensorMap<Scalar,Root::DATA_RANK>(act.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && out.rows() == out_grads.dimension(0));
		std::size_t rows = out_grads.dimension(0);
		MatrixMap<Scalar> out_grads_mat(out_grads.data(), rows, out_grads.size() / rows);
		Matrix<Scalar> prev_out_grads(out.rows(), out.cols());
		for (int i = 0; i < prev_out_grads.rows(); ++i) {
			RowVector<Scalar> row_i = out.row(i);
			// FIXME Do not evaluate the expressions into a temporary variable.
			Matrix<Scalar> jacobian = row_i.asDiagonal();
			jacobian -= row_i.transpose() * row_i;
			prev_out_grads.row(i) = out_grads_mat.row(i) * jacobian;
		}
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grads.data(), conversion_dims);
	}
private:
	const Scalar epsilon;
	RankwiseArray conversion_dims;
	// Staged computation cache matrix.
	Matrix<Scalar> out;
};

/**
 * A class template representing a rectified linear unit (ReLU) activation function. ReLU
 * layers set all negative elements of the input to 0. This function is not theoretically
 * differentiable.
 */
template<typename Scalar, std::size_t Rank>
class ReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline ReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ReLUActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline void empty_cache() {
		in = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		if (training)
			this->in = in;
		return in.cwiseMax((Scalar) 0);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && in.dimension(0) == out_grads.dimension(0));
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= 0); }) * out_grads;
	}
private:
	typename Root::Data in;
};

/**
 * A class template representing a leaky rectified linear unit activation function. Unlike
 * traditional ReLU layers leaky ReLU layers do not set negative elements of the input to
 * 0 but scale them by a small constant alpha. This function is not theoretically
 * differentiable.
 */
template<typename Scalar, std::size_t Rank>
class LeakyReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param alpha The factor by which negative inputs are to be scaled.
	 */
	inline LeakyReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new LeakyReLUActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline void empty_cache() {
		in = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		if (training)
			this->in = in;
		return in.cwiseMax(in * alpha);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && in.dimension(0) == out_grads.dimension(0));
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i >= 0 ? 1 : alpha); }) * out_grads;
	}
private:
	const Scalar alpha;
	typename Root::Data in;
};

/**
 * A class template representing an exponential linear unit (ELU) activation function. ELUs
 * apply an exponential (e based) function scaled by alpha to the negative elements of the input.
 * ELU layers are not theoretically differentiable.
 */
template<typename Scalar, std::size_t Rank>
class ELUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param alpha The factor by which negative inputs are to be scaled.
	 */
	inline ELUActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha),
			conversion_dims(dims.template promote<>()) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ELUActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline void empty_cache() {
		in = Matrix<Scalar>(0, 0);
		out = Matrix<Scalar>(0, 0);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		if (training) {
			this->in = MatrixMap<Scalar>(in.data(), in.dimension(0), Base::dims.get_volume());
			out = this->in.unaryExpr([this](Scalar i) {
				return (Scalar) (i > .0 ? i : (alpha * (exp(i) - 1)));
			});
			conversion_dims[0] = out.rows();
			return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
		}
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i > 0 ? i : (alpha * (exp(i) - 1))); });
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && conversion_dims[0] == out_grads.dimension(0));
		MatrixMap<Scalar> out_grads_mat(out_grads.data(), conversion_dims[0], Base::dims.get_volume());
		Matrix<Scalar> prev_out_grads(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); ++i) {
			for (int j = 0; j < in.rows(); ++j)
				prev_out_grads(j,i) = (Scalar) ((in(j,i) > 0 ? 1 : (out(j,i) + alpha)) * out_grads_mat(j,i));
		}
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grads.data(), conversion_dims);
	}
private:
	const Scalar alpha;
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in;
	Matrix<Scalar> out;
};

/**
 * A class template representing a parametric rectified linear unit (PReLU) activation function.
 * PReLU layers are Leaky ReLU activation functions with element-wise, learnable alphas. PReLU
 * activation functions are not theoretically differentiable.
 */
template<typename Scalar, std::size_t Rank>
class PReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param init_alpha The initial factor by which negative inputs are to be scaled.
	 */
	inline PReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims, RegPenSharedPtr<Scalar> param_reg = Root::DEFAULT_REG_PEN,
			Scalar init_alpha = 1e-1, Scalar max_norm_constraint = 0) :
				Base::ActivationLayer(dims, 1, dims.get_volume()),
				param_reg(param_reg),
				init_alpha(init_alpha),
				max_norm_constraint(max_norm_constraint),
				max_norm(internal::Utils<Scalar>::decidedly_greater(max_norm_constraint, (Scalar) 0)),
				params_shared(false),
				conversion_dims(dims.template promote<>()) {
		assert(param_reg != nullptr);
	}
	inline Layer<Scalar,Rank>* clone() const {
		return new PReLUActivationLayer(*this);
	}
protected:
	inline PReLUActivationLayer(const PReLUActivationLayer<Scalar,Rank>& layer, bool share_params = false) :
			Base::ActivationLayer(layer, share_params),
			param_reg(layer.param_reg),
			init_alpha(layer.init_alpha),
			max_norm_constraint(layer.max_norm_constraint),
			max_norm(layer.max_norm),
			params_shared(share_params),
			conversion_dims(layer.conversion_dims) { }
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return new PReLUActivationLayer(*this, true);
	}
	inline void init() {
		Base::params_ref.setConstant(init_alpha);
		Base::params_grad.setZero(1, Base::dims.get_volume());
	}
	inline void empty_cache() {
		in = Matrix<Scalar>(0, 0);
	}
	inline void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = Base::params_ref.squaredNorm();
			if (l2_norm > max_norm_constraint)
				Base::params_ref *= (max_norm_constraint / l2_norm);
		}
	}
	inline Scalar get_regularization_penalty() {
		return params_shared ? (Scalar) 0 : param_reg->function(Base::params_ref);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		this->in = MatrixMap<Scalar>(in.data(), rows, Base::dims.get_volume());
		Matrix<Scalar> out = this->in.cwiseMax(this->in * Base::params_ref.row(0).asDiagonal());
		conversion_dims[0] = rows;
		return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && conversion_dims[0] == out_grads.dimension(0));
		if (params_shared)
			Base::params_grad.row(0).setZero();
		else
			Base::params_grad = param_reg->d_function(Base::params_ref);
		MatrixMap<Scalar> out_grads_map(out_grads.data(), conversion_dims[0], Base::dims.get_volume());
		Matrix<Scalar> prev_out_grads = Matrix<Scalar>(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); ++i) {
			for (int j = 0; j < in.rows(); ++j) {
				Scalar in_ji = in(j,i);
				if (in_ji >= 0)
					prev_out_grads(j,i) = out_grads_map(j,i);
				else {
					Scalar out_ji = out_grads_map(j,i);
					prev_out_grads(j,i) = Base::params_ref(0,i) * out_ji;
					Base::params_grad(0,i) += in_ji * out_ji;
				}
			}
		}

		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grads.data(), conversion_dims);
	}
private:
	const RegPenSharedPtr<Scalar> param_reg;
	const Scalar init_alpha;
	const Scalar max_norm_constraint;
	const bool max_norm;
	const bool params_shared;
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in;
};

/**
 * An abstract class template representing a pooling layer for batches of rank 3 data.
 */
template<typename Scalar>
class PoolingLayer : public Layer<Scalar,3> {
	typedef Layer<Scalar,3> Base;
public:
	inline const Dimensions<std::size_t,3>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<std::size_t,3>& get_output_dims() const {
		return output_dims;
	}
protected:
	typedef std::array<std::size_t,4> RankwiseArray;
	typedef std::array<std::size_t,2> ReductionRanksArray;
	typedef Tensor<Scalar,2> ReducedData;
	inline PoolingLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height, std::size_t receptor_width,
			std::size_t stride, std::size_t dilation) :
				input_dims(input_dims),
				output_dims({ calculate_output_dim(input_dims(0), receptor_height, dilation, stride),
						calculate_output_dim(input_dims(1), receptor_width, dilation, stride), input_dims(2) }),
				receptor_height(receptor_height),
				receptor_width(receptor_width),
				stride(stride),
				dilation(dilation),
				dil_receptor_height(receptor_height + (receptor_height - 1) * dilation),
				dil_receptor_width(receptor_width + (receptor_width - 1) * dilation),
				height_rem(input_dims(0) - dil_receptor_height),
				width_rem(input_dims(1) - dil_receptor_width),
				input_layer(false),
				reduction_ranks({ 1u, 2u }),
				broadcast({ 1u, receptor_height, receptor_width, 1u }),
				patch_offsets({ 0u, 0u, 0u, 0u }),
				patch_extents({ 0u, dil_receptor_height, dil_receptor_width, input_dims(2) }),
				reduced_patch_offsets({ 0u, 0u, 0u, 0u }),
				reduced_patch_extents({ 0u, 1u, 1u, input_dims(2) }),
				dil_strides({ 1u, dilation + 1u, dilation + 1u, 1u }),
				params(0, 0),
				params_grad(0, 0) {
		assert(input_dims(0) >= dil_receptor_height && input_dims(1) >= dil_receptor_width);
		assert(receptor_height > 0 && receptor_width > 0);
		assert(stride > 0);
	}
	/**
	 * Initializes the cache required for back-propagation.
	 */
	virtual void init_cache() = 0;
	/**
	 * Reduces the input tensor patch along the specified ranks.
	 *
	 * @param patch A tensor representing a spatial patch of the input tensor.
	 * @param patch_ind The index of the patch.
	 * @return The reduced tensor.
	 */
	virtual typename Base::Data reduce(const typename Base::Data& patch, std::size_t patch_ind) = 0;
	/**
	 * Differentiates the reduction function and returns the derivative of the loss function
	 * w.r.t. the non-reduced patch.
	 *
	 * @param grad The derivative of the loss function w.r.t. the reduced patch.
	 * @param patch_ind The index of the patch.
	 * @return The derivative of the loss function w.r.t. the non-reduced patch.
	 */
	virtual typename Base::Data d_reduce(const typename Base::Data& grad, std::size_t patch_ind) = 0;
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
	inline void enforce_constraints() { }
	inline Scalar get_regularization_penalty() {
		return 0;
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == input_dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		patch_extents[0] = rows;
		reduced_patch_extents[0] = rows;
		typename Base::Data out(rows, output_dims(0), output_dims(1), output_dims(2));
		init_cache();
		std::size_t patch_ind = 0;
		std::size_t out_i = 0;
		for (std::size_t i = 0; i <= height_rem; i += stride, ++out_i) {
			patch_offsets[1] = i;
			reduced_patch_offsets[1] = out_i;
			std::size_t out_j = 0;
			for (std::size_t j = 0; j <= width_rem; j += stride, ++out_j) {
				patch_offsets[2] = j;
				reduced_patch_offsets[2] = out_j;
				typename Base::Data patch;
				// Dilated receptor support.
				if (dilation > 0)
					patch = in.slice(patch_offsets, patch_extents).stride(dil_strides);
				else
					patch = in.slice(patch_offsets, patch_extents);
				out.slice(reduced_patch_offsets, reduced_patch_extents) = reduce(patch, patch_ind++);
			}
		}
		return out;
	}
	inline typename Base::Data pass_back(typename Base::Data out_grads) {
		assert((Dimensions<std::size_t,4>(out_grads.dimensions()).template demote<>()) == output_dims);
		assert(out_grads.dimension(0) > 0 && patch_extents[0] == out_grads.dimension(0));
		if (input_layer)
			return typename Base::Data();
		typename Base::Data prev_out_grads(patch_extents[0], input_dims(0), input_dims(1),  input_dims(2));
		prev_out_grads.setZero();
		std::size_t patch_ind = 0;
		std::size_t out_grads_i = 0;
		for (std::size_t i = 0; i <= height_rem; i += stride, ++out_grads_i) {
			patch_offsets[1] = i;
			reduced_patch_offsets[1] = out_grads_i;
			std::size_t out_grads_j = 0;
			for (std::size_t j = 0; j <= width_rem; j += stride, ++out_grads_j) {
				patch_offsets[2] = j;
				reduced_patch_offsets[2] = out_grads_j;
				typename Base::Data reduced_patch_grads = out_grads.slice(reduced_patch_offsets, reduced_patch_extents);
				// Accumulate the gradients where the patches overlap.
				if (dilation > 0)
					prev_out_grads.slice(patch_offsets, patch_extents).stride(dil_strides) +=
							d_reduce(reduced_patch_grads, patch_ind++);
				else
					prev_out_grads.slice(patch_offsets, patch_extents) += d_reduce(reduced_patch_grads, patch_ind++);
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
	 * @param dilation The dilation of the receptor.
	 * @param stride The stride at which the receptor is applied to the input tensor.
	 * @return The depth of the output tensor.
	 */
	static std::size_t calculate_output_dim(std::size_t input_dim, std::size_t receptor_size, std::size_t dilation,
			std::size_t stride) {
		return (input_dim - receptor_size - (receptor_size - 1) * dilation) / stride + 1;
	}
	const Dimensions<std::size_t,3> input_dims;
	const Dimensions<std::size_t,3> output_dims;
	const std::size_t receptor_height;
	const std::size_t receptor_width;
	const std::size_t stride;
	const std::size_t dilation;
	const std::size_t dil_receptor_height;
	const std::size_t dil_receptor_width;
	const std::size_t height_rem;
	const std::size_t width_rem;
	// Arrays for tensor manipulation.
	ReductionRanksArray reduction_ranks;
	RankwiseArray broadcast;
	RankwiseArray patch_offsets;
	RankwiseArray patch_extents;
	RankwiseArray reduced_patch_offsets;
	RankwiseArray reduced_patch_extents;
	RankwiseArray dil_strides;
private:
	bool input_layer;
	// No actual parameters.
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
};

/**
 * A class template representing a pooling layer that reduces patches of the input by taking their
 * sums.
 */
template<typename Scalar>
class SumPoolingLayer : public PoolingLayer<Scalar> {
	typedef Layer<Scalar,3> Root;
	typedef PoolingLayer<Scalar> Base;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of elements
	 * by which the receptor is to be shifted after every step of the pooling process).
	 * @param dilation The extent of the dilation to apply to the receptor field.
	 */
	inline SumPoolingLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t stride = 2, std::size_t dilation = 0) :
				Base::PoolingLayer(input_dims, receptor_height, receptor_width, stride, dilation) { }
	inline Root* clone() const {
		return new SumPoolingLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline void init_cache() { };
	inline typename Root::Data reduce(const typename Root::Data& patch, std::size_t patch_ind) {
		typename Base::ReducedData reduced_patch = patch.sum(Base::reduction_ranks);
		return TensorMap<Scalar,4>(reduced_patch.data(), Base::reduced_patch_extents);
	}
	inline typename Root::Data d_reduce(const typename Root::Data& grad, std::size_t patch_ind) {
		return grad.broadcast(Base::broadcast);
	}
	inline void empty_cache() { }
};

/**
 * A class template representing a pooling layer that reduces patches of the input by taking their
 * means.
 */
template<typename Scalar>
class MeanPoolingLayer : public PoolingLayer<Scalar> {
	typedef Layer<Scalar,3> Root;
	typedef PoolingLayer<Scalar> Base;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of elements
	 * by which the receptor is to be shifted after every step of the pooling process).
	 * @param dilation The extent of the dilation to apply to the receptor field.
	 */
	inline MeanPoolingLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t stride = 2, std::size_t dilation = 0) :
				Base::PoolingLayer(input_dims, receptor_height, receptor_width, stride, dilation),
				receptor_area(receptor_height * receptor_width) { }
	inline Root* clone() const {
		return new MeanPoolingLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline void init_cache() { }
	inline typename Root::Data reduce(const typename Root::Data& patch, std::size_t patch_ind) {
		typename Base::ReducedData reduced_patch = patch.mean(Base::reduction_ranks);
		return TensorMap<Scalar,4>(reduced_patch.data(), Base::reduced_patch_extents);
	}
	inline typename Root::Data d_reduce(const typename Root::Data& grad, std::size_t patch_ind) {
		return (grad / (Scalar) receptor_area).broadcast(Base::broadcast);
	}
	inline void empty_cache() { }
private:
	std::size_t receptor_area;
};

/**
 * A class template representing a pooling layer that reduces patches of the input by taking their
 * maximums.
 */
template<typename Scalar>
class MaxPoolingLayer : public PoolingLayer<Scalar> {
	typedef Layer<Scalar,3> Root;
	typedef PoolingLayer<Scalar> Base;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of elements
	 * by which the receptor is to be shifted after every step of the pooling process).
	 * @param dilation The extent of the dilation to apply to the receptor field.
	 */
	inline MaxPoolingLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t stride = 2, std::size_t dilation = 0) :
				Base::PoolingLayer(input_dims, receptor_height, receptor_width, stride, dilation) { }
	inline Root* clone() const {
		return new MaxPoolingLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline void init_cache() {
		max_inds = std::vector<std::vector<unsigned>>(Base::output_dims(0) * Base::output_dims(1));
	}
	inline typename Root::Data reduce(const typename Root::Data& patch, std::size_t patch_ind) {
		std::size_t rows = patch.dimension(0);
		std::size_t depth = patch.dimension(3);
		std::vector<unsigned> inds(rows * depth);
		typename Root::Data reduced_patch(rows, 1u, 1u, depth);
		for (std::size_t i = 0; i < rows; ++i) {
			for (std::size_t j = 0; j < depth; ++j) {
				Scalar max = internal::Utils<Scalar>::MIN;
				unsigned max_height = 0;
				unsigned max_width = 0;
				for (std::size_t k = 0; k < Base::receptor_height; ++k) {
					for (std::size_t l = 0; l < Base::receptor_width; ++l) {
						Scalar val = patch(i,k,l,j);
						if (val > max) {
							max = val;
							max_height = k;
							max_width = l;
						}
					}
				}
				inds[i * depth + j] = max_height * Base::receptor_width + max_width;
				reduced_patch(i,0u,0u,j) = max;
			}
		}
		max_inds[patch_ind] = inds;
		return reduced_patch;
	}
	inline typename Root::Data d_reduce(const typename Root::Data& grad, std::size_t patch_ind) {
		std::size_t rows = grad.dimension(0);
		std::size_t depth = grad.dimension(3);
		typename Root::Data patch(rows, Base::receptor_height, Base::receptor_width, depth);
		patch.setZero();
		std::vector<unsigned>& inds = max_inds[patch_ind];
		for (std::size_t i = 0; i < rows; ++i) {
			for (std::size_t j = 0; j < depth; ++j) {
				unsigned max_ind = inds[i * depth + j];
				unsigned height = max_ind / Base::receptor_width;
				unsigned width = max_ind % Base::receptor_width;
				patch(i,height,width,j) = grad(i,0u,0u,j);
			}
		}
		return patch;
	}
	inline void empty_cache() {
		max_inds = std::vector<std::vector<unsigned>>(0);
	}
private:
	// Cache
	std::vector<std::vector<unsigned>> max_inds;
};

/**
 * An abstract base class template for a batch normalization layer.
 */
template<typename Scalar, std::size_t Rank>
class BatchNormLayerBase : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	virtual ~BatchNormLayerBase() = default;
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return dims;
	}
protected:
	typedef std::array<std::size_t,Base::DATA_RANK> RankwiseArray;
	inline BatchNormLayerBase(const Dimensions<std::size_t,Rank>& dims, std::size_t depth, RegPenSharedPtr<Scalar> gamma_reg,
			RegPenSharedPtr<Scalar> beta_reg, Scalar gamma_max_norm_constraint, Scalar beta_max_norm_constraint,
			Scalar norm_avg_decay, Scalar epsilon) :
				dims(dims),
				depth(depth),
				gamma_reg(gamma_reg),
				beta_reg(beta_reg),
				gamma_max_norm_constraint(gamma_max_norm_constraint),
				beta_max_norm_constraint(beta_max_norm_constraint),
				gamma_max_norm(internal::Utils<Scalar>::decidedly_greater(gamma_max_norm_constraint, (Scalar) 0)),
				beta_max_norm(internal::Utils<Scalar>::decidedly_greater(beta_max_norm_constraint, (Scalar) 0)),
				norm_avg_decay(norm_avg_decay),
				epsilon(epsilon),
				input_layer(false),
				params_shared(false),
				avg_means(depth, dims.get_volume() / depth),
				avg_inv_sds(depth, dims.get_volume() / depth),
				avgs_init(false),
				params(2 * depth, dims.get_volume() / depth),
				params_grad(2 * depth, dims.get_volume() / depth),
				params_ref(params),
				cache_vec(depth) {
		assert(gamma_reg != nullptr);
		assert(beta_reg != nullptr);
		assert(norm_avg_decay >= 0 && norm_avg_decay <= 1 &&
				"norm avg decay must not be less than 0 or greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline BatchNormLayerBase(const BatchNormLayerBase<Scalar,Rank>& layer, bool share_params = false) :
			dims(layer.dims),
			depth(layer.depth),
			gamma_reg(layer.gamma_reg),
			beta_reg(layer.beta_reg),
			gamma_max_norm_constraint(layer.gamma_max_norm_constraint),
			beta_max_norm_constraint(layer.beta_max_norm_constraint),
			gamma_max_norm(layer.gamma_max_norm),
			beta_max_norm(layer.beta_max_norm),
			norm_avg_decay(layer.norm_avg_decay),
			epsilon(layer.epsilon),
			input_layer(layer.input_layer),
			params_shared(share_params),
			avg_means(layer.avg_means),
			avg_inv_sds(layer.avg_inv_sds),
			avgs_init(layer.avgs_init),
			params(share_params ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_params ? layer.params : params),
			cache_vec(layer.cache_vec) { }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() {
		// Gamma.
		for (int i = 0; i < depth; ++i)
			params_ref.row(i).setOnes();
		// Beta.
		for (int i = depth; i < 2 * depth; ++i)
			params_ref.row(i).setZero();
		params_grad.setZero(params_ref.rows(), params_ref.cols());
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
		return params_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void enforce_constraints() {
		Scalar l2_norm;
		if (gamma_max_norm) {
			l2_norm = params_ref.topRows(depth).squaredNorm();
			if (l2_norm > gamma_max_norm_constraint)
				params_ref.topRows(depth) *= (gamma_max_norm_constraint / l2_norm);
		}
		if (beta_max_norm) {
			l2_norm = params_ref.bottomRows(depth).squaredNorm();
			if (l2_norm > beta_max_norm_constraint)
				params_ref.bottomRows(depth) *= (beta_max_norm_constraint / l2_norm);
		}
	}
	inline Scalar get_regularization_penalty() {
		return params_shared ? (Scalar) 0 : (gamma_reg->function(params_ref.topRows(depth)) +
				beta_reg->function(params_ref.bottomRows(depth)));
	}
	inline typename Base::Data _pass_forward(typename Base::Data in, const RankwiseArray& output_dims,
			bool training, int i) {
		std::size_t rows = in.dimension(0);
		MatrixMap<Scalar> in_mat(in.data(), rows, in.size() / rows);
		if (training) {
			Cache& cache = cache_vec[i];
			RowVector<Scalar> means = in_mat.colwise().mean();
			Matrix<Scalar> norm_in = in_mat.rowwise() - means;
			// FIXME If the squared mean is close to 0, the gradients are inaccurate. If epsilon is small, they explode, too.
			cache.inv_in_sd = (norm_in.array().square().colwise().mean() + epsilon).sqrt().inverse();
			cache.std_in = norm_in * cache.inv_in_sd.asDiagonal();
			in_mat = cache.std_in;
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
			in_mat = (in_mat.rowwise() - avg_means.row(i)) * avg_inv_sds.row(i).asDiagonal();
		Matrix<Scalar> out = (in_mat * params_ref.row(i).asDiagonal()).rowwise() + params_ref.row(depth + i);
		return TensorMap<Scalar,Base::DATA_RANK>(out.data(), output_dims);
	}
	inline typename Base::Data _pass_back(typename Base::Data out_grads, const RankwiseArray& prev_out_dims, int i) {
		std::size_t rows = out_grads.dimension(0);
		Cache& cache = cache_vec[i];
		Matrix<Scalar> std_in_grads;
		/* Back-propagate the gradient through the batch normalization 'function' and also calculate the
		 * gradients of the betas and gammas. */
		{ // Manage memory by scope restriction.
			MatrixMap<Scalar> out_grads_mat(out_grads.data(), rows, out_grads.size() / rows);
			params_grad.row(i) = out_grads_mat.cwiseProduct(cache.std_in).colwise().sum();
			params_grad.row(depth + i) = out_grads_mat.colwise().sum();
			if (input_layer)
				return typename Base::Data();
			std_in_grads = out_grads_mat * params_ref.row(i).asDiagonal();
		}
		Matrix<Scalar> prev_out_grads = (((rows * std_in_grads).rowwise() - std_in_grads.colwise().sum()) -
				cache.std_in * (cache.std_in.cwiseProduct(std_in_grads).colwise().sum().asDiagonal())) *
				((1.0 / rows) * cache.inv_in_sd).asDiagonal();
		return TensorMap<Scalar,Base::DATA_RANK>(prev_out_grads.data(), prev_out_dims);
	}
	const Dimensions<std::size_t,Rank> dims;
	const std::size_t depth;
	const RegPenSharedPtr<Scalar> gamma_reg;
	const RegPenSharedPtr<Scalar> beta_reg;
	const Scalar gamma_max_norm_constraint;
	const Scalar beta_max_norm_constraint;
	const bool gamma_max_norm;
	const bool beta_max_norm;
	const Scalar norm_avg_decay;
	const Scalar epsilon;
	const bool params_shared;
	Matrix<Scalar> params_grad;
	Matrix<Scalar>& params_ref;
private:
	bool input_layer;
	// Dynamic batch normalization parameters.
	Matrix<Scalar> avg_means;
	Matrix<Scalar> avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	mutable Matrix<Scalar> params;
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
	typedef Layer<Scalar,Rank> Root;
	typedef BatchNormLayerBase<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(const Dimensions<std::size_t,Rank>& dims, RegPenSharedPtr<Scalar> gamma_reg = Root::DEFAULT_REG_PEN,
			RegPenSharedPtr<Scalar> beta_reg = Root::DEFAULT_REG_PEN, Scalar gamma_max_norm_constraint = 0,
			Scalar beta_max_norm_constraint = 0, Scalar norm_avg_decay = .1, Scalar epsilon = internal::Utils<Scalar>::EPSILON3) :
				Base::BatchNormLayerBase(dims, dims(2), gamma_reg, beta_reg, gamma_max_norm_constraint,
						beta_max_norm_constraint, norm_avg_decay, epsilon),
				conversion_dims(Base::dims.template promote<>()) { }
	inline Root* clone() const {
		return new BatchNormLayer(*this);
	}
protected:
	inline BatchNormLayer(const BatchNormLayer<Scalar,Rank>& layer, bool share_params = false) :
			Base::BatchNormLayerBase(layer, share_params) { }
	inline Root* clone_with_shared_params() const {
		return new BatchNormLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		conversion_dims[0] = in.dimension();
		return Base::_pass_forward(std::move(in), conversion_dims, training, 0);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && conversion_dims[0] == out_grads.dimension(0));
		typename Root::Data prev_out_grads = Base::_pass_back(std::move(out_grads), conversion_dims, 0);
		if (!Base::params_shared) {
			Base::params_grad.topRows(Base::depth) +=
					Base::gamma_reg->d_function(Base::params_ref.topRows(Base::depth));
			Base::params_grad.bottomRows(Base::depth) +=
					Base::beta_reg->d_function(Base::params_ref.bottomRows(Base::depth));
		}
		return prev_out_grads;
	}
private:
	typename Base::RankwiseArray conversion_dims;
};

/**
 * A partial template specialization for multi-channel input tensors.
 */
template<typename Scalar>
class BatchNormLayer<Scalar,3> : public BatchNormLayerBase<Scalar,3> {
	typedef Layer<Scalar,3> Root;
	typedef BatchNormLayerBase<Scalar,3> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(Dimensions<std::size_t,3> dims, RegPenSharedPtr<Scalar> gamma_reg = Root::DEFAULT_REG_PEN,
			RegPenSharedPtr<Scalar> beta_reg = Root::DEFAULT_REG_PEN, Scalar gamma_max_norm_constraint = 0,
			Scalar beta_max_norm_constraint = 0, Scalar norm_avg_decay = .1, Scalar epsilon = internal::Utils<Scalar>::EPSILON3) :
				Base::BatchNormLayerBase(dims, dims(2), gamma_reg, beta_reg, gamma_max_norm_constraint,
						beta_max_norm_constraint, norm_avg_decay, epsilon),
				offsets({ 0u, 0u, 0u, 0u }),
				extents({ 0u, dims(0), dims(1), 1u }) { }
	inline Root* clone() const {
		return new BatchNormLayer(*this);
	}
protected:
	inline BatchNormLayer(const BatchNormLayer<Scalar,3>& layer, bool share_params = false) :
			Base::BatchNormLayerBase(layer, share_params),
			offsets(layer.offsets),
			extents(layer.extents) { }
	inline Root* clone_with_shared_params() const {
		return new BatchNormLayer(*this, true);
	}
	inline typename Base::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		std::size_t rows = in.dimension(0);
		if (Base::dims(2) == 1) {
			std::array<std::size_t,Root::DATA_RANK> out_dims = Base::dims.template promote<>();
			out_dims[0] = rows;
			return Base::_pass_forward(std::move(in), out_dims, training, 0);
		} else { // Multi-channel image data; depth-wise normalization.
			typename Root::Data out(rows, Base::dims(0), Base::dims(1), Base::dims(2));
			extents[0] = rows;
			for (int i = 0; i < Base::dims(2); ++i) {
				offsets[3] = i;
				typename Root::Data in_slice_i = in.slice(offsets, extents);
				out.slice(offsets, extents) = Base::_pass_forward(std::move(in_slice_i), extents, training, i);
			}
			return out;
		}
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,4>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && extents[0] == out_grads.dimension(0));
		std::size_t rows = out_grads.dimension(0);
		typename Root::Data prev_out_grads;
		if (Base::dims(2) == 1) {
			std::array<std::size_t,Root::DATA_RANK> prev_out_dims = Base::dims.template promote<>();
			prev_out_dims[0] = out_grads.dimension(0);
			prev_out_grads = Base::_pass_back(std::move(out_grads), prev_out_dims, 0);
		} else {
			if (!Base::is_input_layer())
				prev_out_grads = typename Base::Data(rows, Base::dims(0), Base::dims(1), Base::dims(2));
			for (int i = 0; i < Base::dims(2); ++i) {
				offsets[3] = i;
				typename Root::Data out_grads_slice = out_grads.slice(offsets, extents);
				if (Base::is_input_layer())
					Base::_pass_back(std::move(out_grads_slice), extents, i);
				else
					prev_out_grads.slice(offsets, extents) = Base::_pass_back(std::move(out_grads_slice), extents, i);
			}
		}
		if (!Base::params_shared) {
			Base::params_grad.topRows(Base::depth) +=
					Base::gamma_reg->d_function(Base::params_ref.topRows(Base::depth));
			Base::params_grad.bottomRows(Base::depth) +=
					Base::beta_reg->d_function(Base::params_ref.bottomRows(Base::depth));
		}
		return prev_out_grads;
	}
private:
	typename Base::RankwiseArray offsets;
	typename Base::RankwiseArray extents;
};

/**
 * A class template representing a drop-out layer.
 */
template<typename Scalar, std::size_t Rank>
class DropoutLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param dropout_prob The probability of an element of the input tensor being set to 0.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline DropoutLayer(const Dimensions<std::size_t,Rank>& dims, Scalar dropout_prob,
			Scalar epsilon = internal::Utils<Scalar>::EPSILON3) :
				dims(dims),
				dropout_prob(dropout_prob),
				epsilon(epsilon),
				dropout(internal::Utils<Scalar>::decidedly_greater(dropout_prob, .0)),
				input_layer(false),
				params(0, 0),
				params_grad(0, 0) {
		assert(dropout_prob <= 1 && "dropout prob must not be greater than 1");
		assert(epsilon > 0 && "epsilon must be greater than 0");
	}
	inline Base* clone() const {
		return new DropoutLayer(*this);
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return dims;
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() { }
	inline void empty_cache() {
		dropout_mask = typename Base::Data();
	}
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void enforce_constraints() { }
	inline Scalar get_regularization_penalty() {
		return 0;
	}
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		if (training && dropout) {
			typename Base::Data random_tensor(in.dimensions());
			random_tensor.setRandom();
			// Inverted dropout.
			Scalar scaling_factor = 1 / (1 - dropout_prob + epsilon);
			dropout_mask = ((random_tensor + random_tensor.constant((Scalar) 1)) / 2)
					.unaryExpr([this,scaling_factor](Scalar i) { return (Scalar) (i <= dropout_prob ? 0 : scaling_factor); });
			return in * dropout_mask;
		}
		return in;
	}
	inline typename Base::Data pass_back(typename Base::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == dims);
		assert(out_grads.dimension(0) > 0 && dropout_mask.rows() == out_grads.dimension(0));
		if (input_layer)
			return typename Base::Data();
		// The derivative of the dropout 'function'.
		return out_grads * dropout_mask;
	}
private:
	const Dimensions<std::size_t,Rank> dims;
	const Scalar dropout_prob;
	const Scalar epsilon;
	const bool dropout;
	bool input_layer;
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
	// Staged computation cache.
	typename Base::Data dropout_mask;
};

} /* namespace cattle */

#endif /* LAYER_H_ */
