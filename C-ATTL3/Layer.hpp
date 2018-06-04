/*
 * Layer.hpp
 *
 *  Created on: 04.12.2017
 *      Author: Viktor Csomor
 */

#ifndef CATTL3_LAYER_H_
#define CATTL3_LAYER_H_

#ifdef CATTL3_USE_CUDA
#define CATTL3_USE_CUBLAS
#define CATTL3_USE_CUDNN
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>

#include "Dimensions.hpp"
#include "ParameterRegularization.hpp"
#include "utils/EigenProxy.hpp"
#include "utils/NumericUtils.hpp"
#include "WeightInitialization.hpp"

#ifdef CATTL3_USE_CUBLAS
#include "utils/CuBLASHandle.hpp"
#endif

#ifdef CATTL3_USE_CUDNN
#include "utils/CuDNNHandle.hpp"
#endif

namespace cattle {

// TODO Convolution and pooling for 1st and 2nd degree tensors.
// TODO FFT and/or Winograd filtering for convolution.
// TODO More comprehensive GPU acceleration.

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
using ParamRegSharedPtr = std::shared_ptr<ParamaterRegularization<Scalar>>;

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
	static const ParamRegSharedPtr<Scalar> NO_PARAM_REG;
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
	 * A constant method for determining whether the parameters of the layer,
	 * if there are any, are to be updated during optimization.
	 *
	 * @return Whether the parameters should not be updated during optimization.
	 */
	virtual bool is_frozen() const = 0;
	/**
	 * A method for setting whether the parameters of the layer should not be updated
	 * during optimization. Frozen layers are not regularized either.
	 *
	 * @param frozen Whether the parameters of the layer are to be frozen, i.e. not
	 * updatable via optimization.
	 */
	virtual void set_frozen(bool frozen) = 0;
	/**
	 * It initializes the layer and its parameters.
	 */
	virtual void init() = 0;
	/**
	 * A method that returns whether the layer has parameters that can be learned.
	 *
	 * @return Whether the layer uses learnable parameters.
	 */
	inline bool is_parametric() {
		return get_params().rows() > 0 && get_params().cols() > 0;
	}
	/**
	 * @return A string representation of the layer.
	 */
	inline virtual std::string to_string() {
		static const int header_length = 128;
		std::stringstream strm;
		std::stringstream id_num_strm;
		id_num_strm << this;
		std::string id = "<" + std::string(typeid(*this).name()) + id_num_strm.str() + ">";
		strm << id << std::string(std::max(0, (int) (header_length - id.length())), '-') << std::endl;
		strm << "\tinput dims: " << get_input_dims().to_string() << std::endl;
		strm << "\toutput dims: " << get_output_dims().to_string() << std::endl;
		if (is_parametric()) {
			strm << "\tparams:" << std::endl;
			Matrix<Scalar>& params = get_params();
			for (int j = 0; j < params.rows(); ++j) {
				strm << "\t[ ";
				for (int k = 0; k < params.cols(); ++k) {
					strm << std::setw(11) << std::setprecision(4) << params(j,k);
					if (k != params.cols() - 1)
						strm << ", ";
				}
				strm << " ]" << std::endl;
			}
		}
		return strm.str();
	}
	friend std::ostream& operator<<(std::ostream& os, Layer<Scalar,Rank>& layer) {
		return os << layer.to_string() << std::flush;
	}
protected:
	// Rank is increased by one to allow for batch training.
	static constexpr std::size_t DATA_RANK = Rank + 1;
	typedef Tensor<Scalar,DATA_RANK> Data;
	/* Only expose methods that allow for the modification of the layer's state to friends and
	 * sub-classes (except the initialization method). */
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
	virtual bool is_input_layer() const = 0;
	/**
	 * Sets this instance's input layer status to the given value.
	 *
	 * @param input_layer Whether this layer is to be an input layer or not.
	 */
	virtual void set_input_layer(bool input_layer) = 0;
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
	 * It computes the derivative of the regularization function w.r.t. the parameters of the
	 * layer and adds it to their gradient. If the layer is not parametric, calling this
	 * method has no effect.
	 */
	virtual void regularize() = 0;
	/**
	 * It calculates the regularization penalty of the layer's parameters. If the layer is not
	 * parametric, 0 is returned.
	 *
	 * @return A scalar representing the penalty on the magnitude of the layer's parameters.
	 */
	virtual Scalar get_regularization_penalty() = 0;
	/**
	 * It applies constraints such as max-norm to the parameters of the layer (if applicable).
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
const ParamRegSharedPtr<Scalar> Layer<Scalar,Rank>::NO_PARAM_REG = std::make_shared<NoParameterRegularization<Scalar>>();

/**
 * An abstract base class template for layers representing linear kernel-based operations
 * such as matrix multiplication or convolution.
 */
template<typename Scalar, std::size_t Rank>
class KernelLayer : public Layer<Scalar,Rank> {
public:
	virtual ~KernelLayer() = default;
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return output_dims;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() {
		weight_init->apply(weights_ref);
		weights_grad.setZero(weights_grad.rows(), weights_grad.cols());
	}
protected:
	inline KernelLayer(const Dimensions<std::size_t,Rank>& input_dims, Dimensions<std::size_t,Rank> output_dims,
			WeightInitSharedPtr<Scalar> weight_init, ParamRegSharedPtr<Scalar> weight_reg, std::size_t weight_rows,
			std::size_t weight_cols, Scalar max_norm_constraint) :
				input_dims(input_dims),
				output_dims(output_dims),
				weight_init(weight_init),
				weight_reg(weight_reg),
				max_norm_constraint(max_norm_constraint),
				max_norm(internal::NumericUtils<Scalar>::decidedly_greater(max_norm_constraint, (Scalar) 0)),
				input_layer(false),
				frozen(false),
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
			frozen(layer.frozen),
			weights(share_params ? Matrix<Scalar>(0, 0) : layer.weights),
			weights_grad(layer.weights_grad),
			weights_ref(share_params ? layer.weights : weights) { }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline Matrix<Scalar>& get_params() {
		return weights_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return weights_grad;
	}
	inline void regularize() {
		weights_grad += weight_reg->d_function(weights_ref);
	}
	inline Scalar get_regularization_penalty() {
		return weight_reg->function(weights_ref);
	}
	inline void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = weights.squaredNorm();
			if (l2_norm > max_norm_constraint)
				weights *= (max_norm_constraint / l2_norm);
		}
	}
	const Dimensions<std::size_t,Rank> input_dims;
	const Dimensions<std::size_t,Rank> output_dims;
	const WeightInitSharedPtr<Scalar> weight_init;
	const ParamRegSharedPtr<Scalar> weight_reg;
	const Scalar max_norm_constraint;
	const bool max_norm;
	/* Eigen matrices are backed by arrays allocated on the heap, so these
	 * members do not burden the stack. */
	Matrix<Scalar> weights_grad;
	Matrix<Scalar>& weights_ref;
private:
	bool input_layer;
	bool frozen;
	mutable Matrix<Scalar> weights;
};

/**
 * A class template representing a fully connected layer.
 */
template<typename Scalar, std::size_t Rank>
class DenseLayer : public KernelLayer<Scalar,Rank> {
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
	inline DenseLayer(const Dimensions<std::size_t,Rank>& input_dims, std::size_t output_size, WeightInitSharedPtr<Scalar> weight_init,
			ParamRegSharedPtr<Scalar> weight_reg = Root::NO_PARAM_REG, Scalar max_norm_constraint = 0) :
				Base::KernelLayer(input_dims, Dimensions<std::size_t,Rank>({ output_size }), weight_init, weight_reg,
						input_dims.get_volume() + 1, output_size, max_norm_constraint),
				out_conversion_dims(Base::output_dims.template promote<>()),
				prev_out_conversion_dims(Base::input_dims.template promote<>()) { }
	inline Root* clone() const {
		return new DenseLayer(*this);
	}
protected:
	inline DenseLayer(const DenseLayer<Scalar,Rank>& layer, bool share_params = false) :
			Base::KernelLayer(layer, share_params),
			out_conversion_dims(layer.out_conversion_dims),
			prev_out_conversion_dims(layer.prev_out_conversion_dims),
			biased_in_mat(layer.biased_in_mat) { }
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return new DenseLayer(*this, true);
	}
	inline void empty_cache() {
		biased_in_mat = Matrix<Scalar>(0, 0);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(in.dimensions()).template demote<>()) == Base::input_dims);
		assert(in.dimension(0) > 0);
		unsigned input_size = Base::input_dims.get_volume();
		// Add a 1-column to the input for the bias trick.
		biased_in_mat = Matrix<Scalar>(in.dimension(0), input_size + 1);
		biased_in_mat.leftCols(input_size) = MatrixMap<Scalar>(in.data(), in.dimension(0), input_size);
		biased_in_mat.col(input_size).setOnes();
#ifndef CATTL3_USE_CUBLAS
		Matrix<Scalar> out = biased_in_mat * Base::weights_ref;
#else
		Matrix<Scalar> out = internal::CuBLASHandle<Scalar>::get_instance().mul(biased_in_mat, Base::weights_ref, false, false);
#endif
		out_conversion_dims[0] = out.rows();
		return TensorMap<Scalar,Root::DATA_RANK>(out.data(), out_conversion_dims);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && biased_in_mat.rows() == out_grads.dimension(0));
		// Compute the gradient of the outputs with respect to the weights.
#ifndef CATTL3_USE_CUBLAS
		MatrixMap<Scalar> out_grads_mat(out_grads.data(), out_grads.dimension(0), Base::output_dims.get_volume());
		Base::weights_grad = biased_in_mat.transpose() * out_grads_mat;
		if (Base::is_input_layer())
			return typename Root::Data();
		/* Remove the bias row from the weight matrix, transpose it, and compute the derivative w.r.t. the
		 * previous layer's output. */
		Matrix<Scalar> prev_out_grads = out_grads_mat * Base::weights_ref.topRows(Base::input_dims.get_volume()).transpose();
#else
		Matrix<Scalar> out_grads_mat = MatrixMap<Scalar>(out_grads.data(), out_grads.dimension(0),
				Base::output_dims.get_volume());
		Base::weights_grad = internal::CuBLASHandle<Scalar>::get_instance().mul(biased_in_mat, out_grads_mat, true, false);
		if (Base::is_input_layer())
			return typename Root::Data();
		Matrix<Scalar> weights_without_bias = Base::weights_ref.topRows(Base::input_dims.get_volume());
		Matrix<Scalar> prev_out_grads = internal::CuBLASHandle<Scalar>::get_instance().mul(out_grads_mat,
				weights_without_bias, false, true);
#endif
		prev_out_conversion_dims[0] = prev_out_grads.rows();
		return TensorMap<Scalar,Root::DATA_RANK>(prev_out_grads.data(), prev_out_conversion_dims);
	}
private:
	RankwiseArray out_conversion_dims;
	RankwiseArray prev_out_conversion_dims;
	// Staged computation caches
	Matrix<Scalar> biased_in_mat;
};

/**
 * An abstract base class template for a 2D convolutional layer.
 */
template<typename Scalar, std::size_t Rank>
class ConvolutionLayerBase : public KernelLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef KernelLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,4> Array4D;
	typedef std::array<std::pair<std::size_t,std::size_t>,4> PaddingsArray4D;
protected:
	inline ConvolutionLayerBase(const Dimensions<std::size_t,Rank>& input_dims, const Dimensions<std::size_t,Rank>& output_dims,
			std::size_t filters, WeightInitSharedPtr<Scalar> weight_init, ParamRegSharedPtr<Scalar> weight_reg,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_padding, std::size_t horizontal_padding,
			std::size_t vertical_stride, std::size_t horizontal_stride, std::size_t vertical_dilation, std::size_t horizontal_dilation,
			Scalar max_norm_constraint) :
				/* For every filter, there is a column in the weight matrix with the same number of
				 * elements as the area of the receptive field (F * F * D) + 1 for the bias row. */
				Base::KernelLayer(input_dims, output_dims, weight_init, weight_reg,
						receptor_height * receptor_width * input_dims.template extend<3 - Rank>()(2) + 1,
						filters, max_norm_constraint),
				ext_input_dims(input_dims.template extend<3 - Rank>()),
				ext_output_dims(calculate_3d_output_dims(output_dims, filters)),
				filters(filters),
				receptor_height(receptor_height),
				receptor_width(receptor_width),
				vertical_padding(vertical_padding),
				horizontal_padding(horizontal_padding),
				vertical_stride(vertical_stride),
				horizontal_stride(horizontal_stride),
				vertical_dilation(vertical_dilation),
				horizontal_dilation(horizontal_dilation),
				padded_height(ext_input_dims(0) + 2 * vertical_padding),
				padded_width(ext_input_dims(1) + 2 * horizontal_padding),
				dil_receptor_height(receptor_height + (receptor_height - 1) * vertical_dilation),
				dil_receptor_width(receptor_width + (receptor_width - 1) * horizontal_dilation),
				patches_per_sample(ext_output_dims(0) * ext_output_dims(1)),
				out_conversion_dims({ 0u, ext_output_dims(0), ext_output_dims(1), ext_output_dims(2) }),
				patch_offsets({ 0u, 0u, 0u, 0u }),
				patch_extents({ 0u, dil_receptor_height, dil_receptor_width, ext_input_dims(2) }),
				dil_strides({ 1u, vertical_dilation + 1u, horizontal_dilation + 1u, 1u }),
				no_padding_offsets({ 0u, vertical_padding, horizontal_padding, 0u }),
				no_padding_extents({ 0u, ext_input_dims(0), ext_input_dims(1), ext_input_dims(2) }),
				paddings({ std::make_pair(0, 0), std::make_pair(vertical_padding, vertical_padding),
						std::make_pair(horizontal_padding, horizontal_padding), std::make_pair(0, 0) }) {
		assert(filters > 0);
		assert(receptor_height > 0);
		assert(receptor_width > 0);
		assert(vertical_stride > 0 && horizontal_stride > 0);
		assert(ext_input_dims(0) + 2 * vertical_padding >= dil_receptor_height &&
				ext_input_dims(1) + 2 * horizontal_padding >= dil_receptor_width);
	}
	inline ConvolutionLayerBase(const ConvolutionLayerBase<Scalar,Rank>& layer, bool share_params = false) :
			Base::KernelLayer(layer, share_params),
			ext_input_dims(layer.ext_input_dims),
			ext_output_dims(layer.ext_output_dims),
			filters(layer.filters),
			receptor_height(layer.receptor_height),
			receptor_width(layer.receptor_width),
			vertical_padding(layer.vertical_padding),
			horizontal_padding(layer.horizontal_padding),
			vertical_stride(layer.vertical_stride),
			horizontal_stride(layer.horizontal_stride),
			vertical_dilation(layer.vertical_dilation),
			horizontal_dilation(layer.horizontal_dilation),
			padded_height(layer.padded_height),
			padded_width(layer.padded_width),
			dil_receptor_height(layer.dil_receptor_height),
			dil_receptor_width(layer.dil_receptor_width),
			patches_per_sample(layer.patches_per_sample),
			out_conversion_dims(layer.out_conversion_dims),
			patch_offsets(layer.patch_offsets),
			patch_extents(layer.patch_extents),
			dil_strides(layer.dil_strides),
			no_padding_offsets(layer.no_padding_offsets),
			no_padding_extents(layer.no_padding_extents),
			paddings(layer.paddings),
			biased_in_conv_mat(layer.biased_in_conv_mat) { }
	inline void empty_cache() {
		biased_in_conv_mat = Matrix<Scalar>(0, 0);
	}
	inline Tensor<Scalar,4> _pass_forward(Tensor<Scalar,4> in, bool training) {
		// Spatial padding.
		if (vertical_padding > 0 || horizontal_padding > 0)
			in = Tensor<Scalar,4>(in.pad(paddings));
		std::size_t rows = in.dimension(0);
		std::size_t total_patches = rows * patches_per_sample;
		std::size_t receptor_vol = Base::weights_ref.rows() - 1;
		/* Flatten the receptor cuboids into row vectors and concatenate them. Each row stands for one stretched
		 * out receptor of one sample. The same receptor location along all samples of the batch is represented
		 * by a contiguous block of these rows. */
		std::size_t patch_ind = 0;
		patch_extents[0] = rows;
		biased_in_conv_mat = Matrix<Scalar>(total_patches, receptor_vol + 1);
		for (std::size_t i = 0; i <= padded_width - dil_receptor_width; i += horizontal_stride) {
			patch_offsets[2] = i;
			for (std::size_t j = 0; j <= padded_height - dil_receptor_height; j += vertical_stride) {
				patch_offsets[1] = j;
				Tensor<Scalar,4> patch;
				// If the patch is dilated, skip the 'internal padding' when flattening it into a matrix.
				if (vertical_dilation > 0 || horizontal_dilation > 0)
					patch = in.slice(patch_offsets, patch_extents).stride(dil_strides);
				else
					patch = in.slice(patch_offsets, patch_extents);
				biased_in_conv_mat.block(patch_ind, 0, rows, receptor_vol) = MatrixMap<Scalar>(patch.data(), rows, receptor_vol);
				patch_ind += rows;
			}
		}
		assert(patch_ind == total_patches);
		// Bias trick.
		biased_in_conv_mat.col(receptor_vol).setOnes();
#ifndef CATTL3_USE_CUBLAS
		Matrix<Scalar> out = biased_in_conv_mat * Base::weights_ref;
#else
		Matrix<Scalar> out = internal::CuBLASHandle<Scalar>::get_instance().mul(biased_in_conv_mat,
				Base::weights_ref, false, false);
#endif
		out_conversion_dims[0] = rows;
		return TensorMap<Scalar,4>(out.data(), out_conversion_dims);
	}
	inline Tensor<Scalar,4> _pass_back(Tensor<Scalar,4> out_grads) {
		std::size_t rows = out_grads.dimension(0);
		std::size_t total_patches = rows * patches_per_sample;
		std::size_t receptor_vol = Base::weights_ref.rows() - 1;
#ifndef CATTL3_USE_CUBLAS
		MatrixMap<Scalar> out_grads_mat(out_grads.data(), total_patches, filters);
		Base::weights_grad = biased_in_conv_mat.transpose() * out_grads_mat;
		if (Base::is_input_layer())
			return Tensor<Scalar,4>();
		/* Remove the bias row from the weight matrix, transpose it, and compute the gradient of the
		 * previous layer's output. */
		Matrix<Scalar> prev_out_grads_conv_mat = out_grads_mat * Base::weights_ref.topRows(receptor_vol).transpose();
#else
		Matrix<Scalar> out_grads_mat = MatrixMap<Scalar>(out_grads.data(), total_patches, filters);
		Base::weights_grad = internal::CuBLASHandle<Scalar>::get_instance().mul(biased_in_conv_mat, out_grads_mat, true, false);
		if (Base::is_input_layer())
			return Tensor<Scalar,4>();
		Matrix<Scalar> weights_without_bias = Base::weights_ref.topRows(receptor_vol);
		Matrix<Scalar> prev_out_grads_conv_mat = internal::CuBLASHandle<Scalar>::get_instance().mul(out_grads_mat,
				weights_without_bias, false, true);
#endif
		/* Given the gradient of the stretched out receptor patches, perform a 'backwards' convolution
		 * to get the derivative w.r.t. the individual input nodes. */
		Tensor<Scalar,4> prev_out_grads(rows, padded_height, padded_width, ext_input_dims(2));
		prev_out_grads.setZero();
		std::size_t patch_ind = 0;
		patch_extents[0] = rows;
		for (std::size_t i = 0; i <= padded_width - dil_receptor_width; i += horizontal_stride) {
			patch_offsets[2] = i;
			for (std::size_t j = 0; j <= padded_height - dil_receptor_height; j += vertical_stride) {
				patch_offsets[1] = j;
				// Accumulate the gradients where the receptor-patch-tensors overlap.
				Matrix<Scalar> prev_out_grads_conv_mat_block = prev_out_grads_conv_mat.block(patch_ind, 0,
						rows, receptor_vol);
				TensorMap<Scalar,4> prev_out_grads_patch(prev_out_grads_conv_mat_block.data(), rows,
						receptor_height, receptor_width, ext_input_dims(2));
				if (vertical_dilation > 0 || horizontal_dilation > 0)
					prev_out_grads.slice(patch_offsets, patch_extents).stride(dil_strides) += prev_out_grads_patch;
				else
					prev_out_grads.slice(patch_offsets, patch_extents) += prev_out_grads_patch;
				patch_ind += rows;
			}
		}
		assert(patch_ind == prev_out_grads_conv_mat.rows());
		if (vertical_padding > 0 || horizontal_padding > 0) {
			// Cut off the padding.
			no_padding_extents[0] = rows;
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
	inline static std::size_t calculate_output_dim(std::size_t input_dim, std::size_t receptor_size, std::size_t padding,
			std::size_t dilation, std::size_t stride) {
		return (input_dim - receptor_size - (receptor_size - 1) * dilation + 2 * padding) / stride + 1;
	}
	// The defining attributes of the convolutional layer.
	const Dimensions<std::size_t,3> ext_input_dims;
	const Dimensions<std::size_t,3> ext_output_dims;
	const std::size_t filters;
	const std::size_t receptor_height;
	const std::size_t receptor_width;
	const std::size_t vertical_padding;
	const std::size_t horizontal_padding;
	const std::size_t vertical_stride;
	const std::size_t horizontal_stride;
	const std::size_t vertical_dilation;
	const std::size_t horizontal_dilation;
private:
	/**
	 * Calculates the dimensions of the 2D convolution performed on the rank-3 data.
	 *
	 * @param dims The reshaped output dimensions.
	 * @param filters The number of convolution filters.
	 * @return The original output dimensions of the convolution.
	 */
	inline static Dimensions<std::size_t,3> calculate_3d_output_dims(const Dimensions<std::size_t,Rank>& dims,
			std::size_t filters) {
		Dimensions<std::size_t,3> ext_dims = dims.template extend<3 - Rank>();
		ext_dims(2) *= filters;
		ext_dims(Rank - 1) /= filters;
		return ext_dims;
	}
	// Pre-computed values to improve propagation-time performance.
	const std::size_t padded_height;
	const std::size_t padded_width;
	const std::size_t dil_receptor_height;
	const std::size_t dil_receptor_width;
	const std::size_t patches_per_sample;
	Array4D out_conversion_dims;
	Array4D patch_offsets;
	Array4D patch_extents;
	Array4D dil_strides;
	Array4D no_padding_offsets;
	Array4D no_padding_extents;
	PaddingsArray4D paddings;
	// Staged computation caches
	Matrix<Scalar> biased_in_conv_mat;
};

/**
 * A class template for a 2D convolutional layer operating on rank-3 data batches (rank-4 tensors).  The results
 * of the convolutions of the filters and the input tensor are concatenated along the highest (4th) rank of the
 * output tensor.
 */
template<typename Scalar, std::size_t Rank = 3>
class ConvolutionLayer : public ConvolutionLayerBase<Scalar,Rank> {
	typedef Layer<Scalar,3> Root;
	typedef KernelLayer<Scalar,3> KernelBase;
	typedef ConvolutionLayerBase<Scalar,3> ConvBase;
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
	 * @param vertical_padding The extent of padding to apply to the input tensor along its height (both
	 * at the top and at the bottom).
	 * @param horizontal_padding The extent of padding to apply to the input tensor along its width (both
	 * at the left and at the right).
	 * @param vertical_stride The vertical convolution stride i.e. the number of elements by which the
	 * receptor is to be shifted along the height of the input tensor.
	 * @param horizontal_stride The horizonzal convolution stride i.e. the number of elements by which the
	 * receptor is to be shifted along the width of the input tensor.
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline ConvolutionLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			ParamRegSharedPtr<Scalar> weight_reg = Root::NO_PARAM_REG, std::size_t receptor_height = 3, std::size_t receptor_width = 3,
			std::size_t vertical_padding = 1, std::size_t horizontal_padding = 1, std::size_t vertical_stride = 1,
			std::size_t horizontal_stride = 1, std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0,
			Scalar max_norm_constraint = 0) :
				ConvBase::ConvolutionLayerBase(input_dims, {
						ConvBase::calculate_output_dim(input_dims(0), receptor_height, vertical_padding, vertical_dilation, vertical_stride),
						ConvBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_padding, horizontal_dilation, horizontal_stride),
						filters }, filters, weight_init, weight_reg, receptor_height, receptor_width, vertical_padding, horizontal_padding,
						vertical_stride, horizontal_stride, vertical_dilation, horizontal_dilation, max_norm_constraint) { }
	inline Root* clone() const {
		return new ConvolutionLayer(*this);
	}
protected:
	inline ConvolutionLayer(const ConvolutionLayer<Scalar,Rank>& layer, bool share_params = false) :
			ConvBase::ConvolutionLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone_with_shared_params() const {
		return new ConvolutionLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return ConvBase::_pass_forward(std::move(in), training);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,4>(out_grads.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		return ConvBase::_pass_back(std::move(out_grads));
	}
private:
	std::size_t batch_size;
};

/**
 * A class template for a 2D convolutional layer operating on rank-2 data batches (rank-3 tensors).  The results
 * of the convolutions of the filters and the input tensor are concatenated along the highest (3rd) rank of the
 * output tensor.
 */
template<typename Scalar>
class ConvolutionLayer<Scalar,2> : public ConvolutionLayerBase<Scalar,2> {
	typedef Layer<Scalar,2> Root;
	typedef KernelLayer<Scalar,2> KernelBase;
	typedef ConvolutionLayerBase<Scalar,2> ConvBase;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * The ranks of the input tensors denote the sample, height, and width (N,H,W).
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the parametric kernel backing the layer.
	 * @param weight_reg The regularization function to apply to the layer's parameters.
	 * @param receptor_height The height of the receptor field.
	 * @param receptor_width The width of the receptor field.
	 * @param vertical_padding The extent of padding to apply to the input tensor along its height (both
	 * at the top and at the bottom).
	 * @param horizontal_padding The extent of padding to apply to the input tensor along its width (both
	 * at the left and at the right).
	 * @param vertical_stride The vertical convolution stride i.e. the number of elements by which the
	 * receptor is to be shifted along the height of the input tensor.
	 * @param horizontal_stride The horizonzal convolution stride i.e. the number of elements by which the
	 * receptor is to be shifted along the width of the input tensor.
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline ConvolutionLayer(const Dimensions<std::size_t,2>& input_dims, std::size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			ParamRegSharedPtr<Scalar> weight_reg = Root::NO_PARAM_REG, std::size_t receptor_height = 3, std::size_t receptor_width = 3,
			std::size_t vertical_padding = 1, std::size_t horizontal_padding = 1, std::size_t vertical_stride = 1,
			std::size_t horizontal_stride = 1, std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0,
			Scalar max_norm_constraint = 0) :
				ConvBase::ConvolutionLayerBase(input_dims, {
						ConvBase::calculate_output_dim(input_dims(0), receptor_height, vertical_padding, vertical_dilation, vertical_stride),
						ConvBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_padding, horizontal_dilation, horizontal_stride) *
						filters  }, filters, weight_init, weight_reg, receptor_height, receptor_width, vertical_padding, horizontal_padding,
						vertical_stride, horizontal_stride, vertical_dilation, horizontal_dilation, max_norm_constraint) { }
	inline Root* clone() const {
		return new ConvolutionLayer(*this);
	}
protected:
	inline ConvolutionLayer(const ConvolutionLayer<Scalar,2>& layer, bool share_params = false) :
			ConvBase::ConvolutionLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone_with_shared_params() const {
		return new ConvolutionLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,3>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return ConvBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), in.dimension(2), 1u }), training)
				.reshape(std::array<std::size_t,3>({ batch_size, KernelBase::output_dims(0), KernelBase::output_dims(1) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,3>(out_grads.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = ConvBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, ConvBase::ext_output_dims(0), ConvBase::ext_output_dims(1), ConvBase::ext_output_dims(2) }));
		if (KernelBase::is_input_layer())
			return Tensor<Scalar,3>();
		return TensorMap<Scalar,3>(prev_out_grads.data(), { batch_size, KernelBase::input_dims(0), KernelBase::input_dims(1) });
	}
private:
	std::size_t batch_size;
};


/**
 * A class template for a 1D convolutional layer operating on rank-1 data batches (rank-2 tensors).  The results
 * of the convolutions of the filters and the input tensor are concatenated along the highest (2nd) rank of the
 * output tensor.
 */
template<typename Scalar>
class ConvolutionLayer<Scalar,1> : public ConvolutionLayerBase<Scalar,1> {
	typedef Layer<Scalar,1> Root;
	typedef KernelLayer<Scalar,1> KernelBase;
	typedef ConvolutionLayerBase<Scalar,1> ConvBase;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * The ranks of the input tensors denote the sample and the length (N,L).
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the parametric kernel backing the layer.
	 * @param weight_reg The regularization function to apply to the layer's parameters.
	 * @param receptor_length The length of the receptor.
	 * @param padding The extent of padding to apply to the input tensor along its length on both ends.
	 * @param stride The convolution stride i.e. the number of elements by which the receptor is to be
	 * shifted along the length of the input tensor.
	 * @param dilation The extent of dilation to apply to the receptor.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	ConvolutionLayer(const Dimensions<std::size_t,1>& input_dims, std::size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			ParamRegSharedPtr<Scalar> weight_reg = Root::NO_PARAM_REG, std::size_t receptor_length = 3, std::size_t padding = 1,
			std::size_t stride = 1, std::size_t dilation = 0, Scalar max_norm_constraint = 0) :
				ConvBase::ConvolutionLayerBase(input_dims, {
						ConvBase::calculate_output_dim(input_dims(0), receptor_length, padding, dilation, stride) * filters  },
						filters, weight_init, weight_reg, receptor_length, 1, padding, 0, stride, 1, dilation, 0,
						max_norm_constraint) { }
	inline Root* clone() const {
		return new ConvolutionLayer(*this);
	}
protected:
	inline ConvolutionLayer(const ConvolutionLayer<Scalar,1>& layer, bool share_params = false) :
			ConvBase::ConvolutionLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone_with_shared_params() const {
		return new ConvolutionLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,2>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return ConvBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), 1u, 1u }), training)
				.reshape(std::array<std::size_t,2>({ batch_size, KernelBase::output_dims(0) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,2>(out_grads.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = ConvBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, ConvBase::ext_output_dims(0), ConvBase::ext_output_dims(1), ConvBase::ext_output_dims(2) }));
		if (KernelBase::is_input_layer())
			return Tensor<Scalar,2>();
		return TensorMap<Scalar,2>(prev_out_grads.data(), { batch_size, KernelBase::input_dims(0) });
	}
private:
	std::size_t batch_size;
};

/**
 * An abstract base class template for a transposed 2D convolutional layer.
 */
template<typename Scalar, std::size_t Rank>
class DeconvolutionLayerBase : public KernelLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef KernelLayer<Scalar,Rank> Base;
	typedef std::array<std::size_t,4> Array4D;
	typedef std::array<std::pair<std::size_t,std::size_t>,4> PaddingsArray4D;
public:
	inline DeconvolutionLayerBase(const Dimensions<std::size_t,Rank>& input_dims, const Dimensions<std::size_t,Rank>& output_dims,
			std::size_t filters, WeightInitSharedPtr<Scalar> weight_init, ParamRegSharedPtr<Scalar> weight_reg,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_padding, std::size_t horizontal_padding,
			std::size_t vertical_stride, std::size_t horizontal_stride, std::size_t vertical_dilation, std::size_t horizontal_dilation,
			Scalar max_norm_constraint) :
				Base::KernelLayer(input_dims, output_dims, weight_init, weight_reg, input_dims.template extend<3 - Rank>()(2) + 1,
						receptor_height * receptor_width * filters, max_norm_constraint),
				ext_input_dims(input_dims.template extend<3 - Rank>()),
				ext_output_dims(calculate_3d_output_dims(output_dims, filters)),
				filters(filters),
				receptor_height(receptor_height),
				receptor_width(receptor_width),
				vertical_padding(vertical_padding),
				horizontal_padding(horizontal_padding),
				vertical_stride(vertical_stride),
				horizontal_stride(horizontal_stride),
				vertical_dilation(vertical_dilation),
				horizontal_dilation(horizontal_dilation),
				padded_height(ext_output_dims(0) + 2 * vertical_padding),
				padded_width(ext_output_dims(1) + 2 * horizontal_padding),
				dil_receptor_height(receptor_height + (receptor_height - 1) * vertical_dilation),
				dil_receptor_width(receptor_width + (receptor_width - 1) * horizontal_dilation),
				patches_per_sample(ext_input_dims(0) * ext_input_dims(1)),
				prev_out_conversion_dims({ 0u, ext_input_dims(0), ext_input_dims(1), ext_input_dims(2) }),
				patch_offsets({ 0u, 0u, 0u, 0u }),
				patch_extents({ 0u, dil_receptor_height, dil_receptor_width, filters }),
				dil_strides({ 1u, vertical_dilation + 1u, horizontal_dilation + 1u, 1u }),
				no_padding_offsets({ 0u, vertical_padding, horizontal_padding, 0u }),
				no_padding_extents({ 0u, ext_output_dims(0), ext_output_dims(1), ext_output_dims(2) }),
				paddings({ std::make_pair(0, 0), std::make_pair(vertical_padding, vertical_padding),
						std::make_pair(horizontal_padding, horizontal_padding), std::make_pair(0, 0) }) {
		assert(filters > 0);
		assert(receptor_height > 0);
		assert(receptor_width > 0);
		assert(vertical_stride > 0 && horizontal_stride > 0);
		assert(ext_output_dims(0) + 2 * vertical_padding >= dil_receptor_height &&
				ext_output_dims(1) + 2 * horizontal_padding >= dil_receptor_width);
	}
protected:
	inline DeconvolutionLayerBase(const DeconvolutionLayerBase<Scalar,Rank>& layer, bool share_params = false) :
			Base::KernelLayer(layer, share_params),
			filters(layer.filters),
			receptor_height(layer.receptor_height),
			receptor_width(layer.receptor_width),
			vertical_padding(layer.vertical_padding),
			horizontal_padding(layer.horizontal_padding),
			vertical_stride(layer.vertical_stride),
			horizontal_stride(layer.horizontal_stride),
			vertical_dilation(layer.vertical_dilation),
			horizontal_dilation(layer.horizontal_dilation),
			padded_height(layer.padded_height),
			padded_width(layer.padded_width),
			dil_receptor_height(layer.dil_receptor_height),
			dil_receptor_width(layer.dil_receptor_width),
			patches_per_sample(layer.patches_per_sample),
			prev_out_conversion_dims(layer.prev_out_conversion_dims),
			patch_offsets(layer.patch_offsets),
			patch_extents(layer.patch_extents),
			dil_strides(layer.dil_strides),
			no_padding_offsets(layer.no_padding_offsets),
			no_padding_extents(layer.no_padding_extents),
			paddings(layer.paddings),
			biased_in_mat(layer.biased_in_mat) { }
	inline void empty_cache() {
		biased_in_mat = Matrix<Scalar>(0, 0);
	}
	inline Tensor<Scalar,4> _pass_forward(Tensor<Scalar,4> in, bool training) {
		std::size_t rows = in.dimension(0);
		std::size_t depth = ext_input_dims(2);
		std::size_t receptor_vol = Base::weights_ref.cols();
		std::size_t total_patches = rows * patches_per_sample;
		biased_in_mat = Matrix<Scalar>(total_patches, depth + 1);
		biased_in_mat.block(0, 0, total_patches, depth) = MatrixMap<Scalar>(in.data(), total_patches, depth);
		biased_in_mat.col(depth).setOnes();
#ifndef CATTL3_USE_CUBLAS
		Matrix<Scalar> out_conv_mat = biased_in_mat * Base::weights_ref;
#else
		Matrix<Scalar> out_conv_mat = internal::CuBLASHandle<Scalar>::get_instance().mul(biased_in_mat,
				Base::weights_ref, false, false);
#endif
		/* Given the values of the stretched out receptor patches, accumulate them in the output tensor. */
		Tensor<Scalar,4> out(rows, padded_height, padded_width, ext_output_dims(2));
		out.setZero();
		std::size_t patch_ind = 0;
		patch_extents[0] = rows;
		for (std::size_t i = 0; i <= padded_width - dil_receptor_width; i += horizontal_stride) {
			patch_offsets[2] = i;
			for (std::size_t j = 0; j <= padded_height - dil_receptor_height; j += vertical_stride) {
				patch_offsets[1] = j;
				// Accumulate the gradients where the receptor-patch-tensors overlap.
				Matrix<Scalar> out_conv_mat_block = out_conv_mat.block(patch_ind, 0, rows, receptor_vol);
				TensorMap<Scalar,4> out_patch(out_conv_mat_block.data(), rows, receptor_height,
						receptor_width, ext_output_dims(2));
				if (vertical_dilation > 0 || horizontal_dilation > 0)
					out.slice(patch_offsets, patch_extents).stride(dil_strides) += out_patch;
				else
					out.slice(patch_offsets, patch_extents) += out_patch;
				patch_ind += rows;
			}
		}
		assert(patch_ind == total_patches);
		if (vertical_padding > 0 || horizontal_padding > 0) {
			// Cut off the padding.
			no_padding_extents[0] = rows;
			return out.slice(no_padding_offsets, no_padding_extents);
		} else
			return out;
	}
	inline Tensor<Scalar,4> _pass_back(Tensor<Scalar,4> out_grads) {
		// Spatial padding.
		if (vertical_padding > 0 || horizontal_padding > 0)
			out_grads = Tensor<Scalar,4>(out_grads.pad(paddings));
		std::size_t rows = out_grads.dimension(0);
		std::size_t depth = ext_input_dims(2);
		std::size_t receptor_vol = Base::weights_ref.cols();
		std::size_t total_patches = rows * patches_per_sample;
		std::size_t patch_ind = 0;
		patch_extents[0] = rows;
		Matrix<Scalar> out_grads_conv_mat(total_patches, receptor_vol);
		for (std::size_t i = 0; i <= padded_width - dil_receptor_width; i += horizontal_stride) {
			patch_offsets[2] = i;
			for (std::size_t j = 0; j <= padded_height - dil_receptor_height; j += vertical_stride) {
				patch_offsets[1] = j;
				Tensor<Scalar,4> patch;
				// If the patch is dilated, skip the 'internal padding' when flattening it into a matrix.
				if (vertical_dilation > 0 || horizontal_dilation > 0)
					patch = out_grads.slice(patch_offsets, patch_extents).stride(dil_strides);
				else
					patch = out_grads.slice(patch_offsets, patch_extents);
				out_grads_conv_mat.block(patch_ind, 0, rows, receptor_vol) = MatrixMap<Scalar>(patch.data(),
						rows, receptor_vol);
				patch_ind += rows;
			}
		}
		assert(patch_ind == total_patches);
#ifndef CATTL3_USE_CUBLAS
		Base::weights_grad = biased_in_mat.transpose() * out_grads_conv_mat;
		if (Base::is_input_layer())
			return Tensor<Scalar,4>();
		Matrix<Scalar> prev_out_grads = out_grads_conv_mat * Base::weights_ref.topRows(depth).transpose();
#else
		Base::weights_grad = internal::CuBLASHandle<Scalar>::get_instance().mul(biased_in_mat,
				out_grads_conv_mat, true, false);
		if (Base::is_input_layer())
			return Tensor<Scalar,4>();
		Matrix<Scalar> weights_without_bias = Base::weights_ref.topRows(depth);
		Matrix<Scalar> prev_out_grads = internal::CuBLASHandle<Scalar>::get_instance().mul(out_grads_conv_mat,
				weights_without_bias, false, true);
#endif
		prev_out_conversion_dims[0] = rows;
		return TensorMap<Scalar,4>(prev_out_grads.data(), prev_out_conversion_dims);
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
	inline static std::size_t calculate_output_dim(std::size_t input_dim, std::size_t receptor_size, std::size_t padding,
			std::size_t dilation, std::size_t stride) {
		return (input_dim - 1) * stride + receptor_size + (receptor_size - 1) * dilation - 2 * padding;
	}
	// The defining attributes of the deconvolutional layer.
	const Dimensions<std::size_t,3> ext_input_dims;
	const Dimensions<std::size_t,3> ext_output_dims;
	const std::size_t filters;
	const std::size_t receptor_height;
	const std::size_t receptor_width;
	const std::size_t vertical_padding;
	const std::size_t horizontal_padding;
	const std::size_t vertical_stride;
	const std::size_t horizontal_stride;
	const std::size_t vertical_dilation;
	const std::size_t horizontal_dilation;
private:
	/**
	 * Calculates the dimensions of the transposed 2D convolution performed on the rank-3 data.
	 *
	 * @param dims The reshaped output dimensions.
	 * @param filters The number of convolution filters.
	 * @return The original output dimensions of the transposed convolution.
	 */
	inline static Dimensions<std::size_t,3> calculate_3d_output_dims(const Dimensions<std::size_t,Rank>& dims,
			std::size_t filters) {
		Dimensions<std::size_t,3> ext_dims = dims.template extend<3 - Rank>();
		ext_dims(2) *= filters;
		ext_dims(Rank - 1) /= filters;
		return ext_dims;
	}
	// Pre-computed values to improve propagation-time performance.
	const std::size_t padded_height;
	const std::size_t padded_width;
	const std::size_t dil_receptor_height;
	const std::size_t dil_receptor_width;
	const std::size_t patches_per_sample;
	Array4D prev_out_conversion_dims;
	Array4D patch_offsets;
	Array4D patch_extents;
	Array4D dil_strides;
	Array4D no_padding_offsets;
	Array4D no_padding_extents;
	PaddingsArray4D paddings;
	// Staged computation caches
	Matrix<Scalar> biased_in_mat;
};

/**
 * A class template for a transposed 2D convolutional layer operating on rank-3 data batches (rank-4 tensors).
 * The results of the convolutions of the filters and the input tensor are concatenated along the highest (4th)
 * rank of the output tensor.
 */
template<typename Scalar, std::size_t Rank = 3>
class DeconvolutionLayer : public DeconvolutionLayerBase<Scalar,Rank> {
	typedef Layer<Scalar,3> Root;
	typedef KernelLayer<Scalar,3> KernelBase;
	typedef DeconvolutionLayerBase<Scalar,3> DeconvBase;
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
	 * @param vertical_padding The extent of vertical padding to use for the transposed convolution.
	 * @param horizontal_padding The extent of horizontal padding to use for the transposed convolution.
	 * @param vertical_stride The vertical convolution stride i.e. the number of elements by which the
	 * receptor is to be shifted along the height of the output tensor.
	 * @param horizontal_stride The horizonzal convolution stride i.e. the number of elements by which the
	 * receptor is to be shifted along the width of the output tensor.
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline DeconvolutionLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			ParamRegSharedPtr<Scalar> weight_reg = Root::NO_PARAM_REG, std::size_t receptor_height = 3, std::size_t receptor_width = 3,
			std::size_t vertical_padding = 1, std::size_t horizontal_padding = 1, std::size_t vertical_stride = 1,
			std::size_t horizontal_stride = 1, std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0,
			Scalar max_norm_constraint = 0) :
				DeconvBase::DeconvolutionLayerBase(input_dims,{
						DeconvBase::calculate_output_dim(input_dims(0), receptor_height, vertical_padding, vertical_dilation, vertical_stride),
						DeconvBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_padding, horizontal_dilation, horizontal_stride),
						filters }, filters, weight_init, weight_reg, receptor_height, receptor_width, vertical_padding, horizontal_padding,
						vertical_stride, horizontal_stride, vertical_dilation, horizontal_dilation, max_norm_constraint) { }
	inline Root* clone() const {
		return new DeconvolutionLayer(*this);
	}
protected:
	inline DeconvolutionLayer(const DeconvolutionLayer<Scalar,3>& layer, bool share_params = false) :
			DeconvBase::DeconvolutionLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone_with_shared_params() const {
		return new DeconvolutionLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return DeconvBase::_pass_forward(std::move(in), training);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,4>(out_grads.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		return DeconvBase::_pass_back(std::move(out_grads));
	}
private:
	std::size_t batch_size;
};

/**
 * A class template for a transposed 2D convolutional layer operating on rank-2 data batches (rank-3 tensors).
 * The results of the convolutions of the filters and the input tensor are concatenated along the highest (3rd)
 * rank of the output tensor.
 */
template<typename Scalar>
class DeconvolutionLayer<Scalar,2> : public DeconvolutionLayerBase<Scalar,2> {
	typedef Layer<Scalar,2> Root;
	typedef KernelLayer<Scalar,2> KernelBase;
	typedef DeconvolutionLayerBase<Scalar,2> DeconvBase;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * The ranks of the input tensors denote the sample, height, and width (N,H,W).
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the parametric kernel backing the layer.
	 * @param weight_reg The regularization function to apply to the layer's parameters.
	 * @param receptor_height The height of the receptor field.
	 * @param receptor_width The width of the receptor field.
	 * @param vertical_padding The extent of padding to apply to the input tensor along its height (both
	 * at the top and at the bottom).
	 * @param horizontal_padding The extent of padding to apply to the input tensor along its width (both
	 * at the left and at the right).
	 * @param vertical_stride The vertical convolution stride i.e. the number of elements by which the
	 * receptor is to be shifted along the height of the input tensor.
	 * @param horizontal_stride The horizonzal convolution stride i.e. the number of elements by which the
	 * receptor is to be shifted along the width of the input tensor.
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	DeconvolutionLayer(const Dimensions<std::size_t,2>& input_dims, std::size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			ParamRegSharedPtr<Scalar> weight_reg = Root::NO_PARAM_REG, std::size_t receptor_height = 3, std::size_t receptor_width = 3,
			std::size_t vertical_padding = 1, std::size_t horizontal_padding = 1, std::size_t vertical_stride = 1,
			std::size_t horizontal_stride = 1, std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0,
			Scalar max_norm_constraint = 0) :
				DeconvBase::DeconvolutionLayerBase(input_dims,{
						DeconvBase::calculate_output_dim(input_dims(0), receptor_height, vertical_padding, vertical_dilation, vertical_stride),
						DeconvBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_padding, horizontal_dilation, horizontal_stride) *
						filters }, filters, weight_init, weight_reg, receptor_height, receptor_width, vertical_padding, horizontal_padding,
						vertical_stride, horizontal_stride, vertical_dilation, horizontal_dilation, max_norm_constraint) { }
	inline Root* clone() const {
		return new DeconvolutionLayer(*this);
	}
protected:
	inline DeconvolutionLayer(const DeconvolutionLayer<Scalar,2>& layer, bool share_params = false) :
			DeconvBase::DeconvolutionLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone_with_shared_params() const {
		return new DeconvolutionLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,3>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return DeconvBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), in.dimension(2), 1u }), training)
				.reshape(std::array<std::size_t,3>({ batch_size, KernelBase::output_dims(0), KernelBase::output_dims(1) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,3>(out_grads.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = DeconvBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, DeconvBase::ext_output_dims(0), DeconvBase::ext_output_dims(1), DeconvBase::ext_output_dims(2) }));
		if (KernelBase::is_input_layer())
			return Tensor<Scalar,3>();
		return TensorMap<Scalar,3>(prev_out_grads.data(), { batch_size, KernelBase::input_dims(0), KernelBase::input_dims(1) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template for a transposed 2D convolutional layer operating on rank-1 data batches (rank-2 tensors).
 * The results of the convolutions of the filters and the input tensor are concatenated along the highest (2nd)
 * rank of the output tensor.
 */
template<typename Scalar>
class DeconvolutionLayer<Scalar,1> : public DeconvolutionLayerBase<Scalar,1> {
	typedef Layer<Scalar,1> Root;
	typedef KernelLayer<Scalar,1> KernelBase;
	typedef DeconvolutionLayerBase<Scalar,1> DeconvBase;
public:
	/**
	 * @param input_dims The dimensionality of the observations to be processed by the layer.
	 * The ranks of the input tensors denote the sample and the length (N,L).
	 * @param filters The number of filters to use.
	 * @param weight_init A shared pointer to a weight initialization used to initialize the
	 * values of the parametric kernel backing the layer.
	 * @param weight_reg The regularization function to apply to the layer's parameters.
	 * @param receptor_length The length of the receptor.
	 * @param padding The extent of padding to apply to the input tensor along its length on both ends.
	 * @param stride The convolution stride i.e. the number of elements by which the receptor is to be
	 * shifted along the length of the input tensor.
	 * @param dilation The extent of dilation to apply to the receptor.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	DeconvolutionLayer(const Dimensions<std::size_t,1>& input_dims, std::size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			ParamRegSharedPtr<Scalar> weight_reg = Root::NO_PARAM_REG, std::size_t receptor_length = 3, std::size_t padding = 1,
			std::size_t stride = 1, std::size_t dilation = 0, Scalar max_norm_constraint = 0) :
				DeconvBase::DeconvolutionLayerBase(input_dims,{
						DeconvBase::calculate_output_dim(input_dims(0), receptor_length, padding, dilation, stride) * filters  },
						filters, weight_init, weight_reg, receptor_length, 1, padding, 0, stride, 1, dilation, 0,
						max_norm_constraint) { }
	inline Root* clone() const {
		return new DeconvolutionLayer(*this);
	}
protected:
	inline DeconvolutionLayer(const DeconvolutionLayer<Scalar,1>& layer, bool share_params = false) :
			DeconvBase::DeconvolutionLayerBase(layer, share_params),
			batch_size(layer.batch_size) { }
	inline Root* clone_with_shared_params() const {
		return new DeconvolutionLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,2>(in.dimensions()).template demote<>()) == KernelBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return DeconvBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), 1u, 1u }), training)
				.reshape(std::array<std::size_t,2>({ batch_size, KernelBase::output_dims(0) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,2>(out_grads.dimensions()).template demote<>()) == KernelBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = DeconvBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, DeconvBase::ext_output_dims(0), DeconvBase::ext_output_dims(1), DeconvBase::ext_output_dims(2) }));
		if (KernelBase::is_input_layer())
			return Tensor<Scalar,2>();
		return TensorMap<Scalar,2>(prev_out_grads.data(), { batch_size, KernelBase::input_dims(0) });
	}
private:
	std::size_t batch_size;
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
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
protected:
	inline ActivationLayer(const Dimensions<std::size_t,Rank>& dims, std::size_t param_rows = 0,
			std::size_t params_cols = 0) :
				dims(dims),
				input_layer(false),
				frozen(false),
				params(param_rows, params_cols),
				params_grad(param_rows, params_cols),
				params_ref(params) { }
	inline ActivationLayer(const ActivationLayer<Scalar,Rank>& layer, bool share_params = false) :
			dims(layer.dims),
			input_layer(layer.input_layer),
			frozen(layer.frozen),
			params(share_params ? Matrix<Scalar>(0, 0) : layer.params),
			params_grad(layer.params_grad),
			params_ref(share_params ? layer.params : params) { }
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline Matrix<Scalar>& get_params() {
		return params_ref;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() {
		return 0;
	}
	inline void enforce_constraints() { }
	inline void empty_cache() { }
	const Dimensions<std::size_t,Rank> dims;
	Matrix<Scalar> params_grad;
	Matrix<Scalar>& params_ref;
private:
	bool input_layer;
	bool frozen;
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
class ScaledActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param scale The factor by which the input is to be scaled.
	 */
	inline ScaledActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar scale) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			scale(scale) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ScaledActivationLayer(*this);
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
		return out_grads.constant(0);
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
		auto out = ((-in).exp() + in.constant(1)).inverse();
		if (training) {
			this->out = out;
			return this->out;
		}
		return out;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && out.dimension(0) == out_grads.dimension(0));
		return (out * (-out + out.constant(1))) * out_grads;
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
		auto out = in.tanh();
		if (training) {
			this->out = out;
			return this->out;
		}
		return out;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && out.dimension(0) == out_grads.dimension(0));
		return (-out * out + out.constant(1)) * out_grads;
	}
private:
	typename Root::Data out;
};

/**
 * A class template representing a softsign activation function layer, an alternative to the
 * tanh layer.
 */
template<typename Scalar, std::size_t Rank>
class SoftsignActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SoftsignActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SoftsignActivationLayer(*this);
	}
protected:
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return clone();
	}
	inline void empty_cache() {
		denominator = typename Root::Data();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		auto denominator = in.constant(1) + in.abs();
		if (training) {
			this->denominator = denominator;
			return in / this->denominator;
		}
		return in / denominator;
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && in.dimension(0) == out_grads.dimension(0));
		return denominator.square().inverse() * out_grads;
	}
private:
	// Staged computation cache.
	typename Root::Data denominator;
};

/**
 * A class template representing a softplus activation function layer. The softplus activation function
 * is a differentiable function that approximates the rectified linear unit function.
 */
template<typename Scalar, std::size_t Rank>
class SoftplusActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 */
	inline SoftplusActivationLayer(const Dimensions<std::size_t,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SoftplusActivationLayer(*this);
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
		if (training) {
			this->in = std::move(in);
			return (this->in.exp() + this->in.constant(1)).log();
		}
		return (in.exp() + in.constant(1)).log();
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && in.dimension(0) == out_grads.dimension(0));
		return ((-in).exp() + in.constant(1)).inverse() * out_grads;
	}
private:
	// Staged computation cache.
	typename Root::Data in;
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
	inline SoftmaxActivationLayer(const Dimensions<std::size_t,Rank>& dims, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
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
				return (Scalar) (i >= 0 ? i : (alpha * (exp(i) - 1)));
			});
			conversion_dims[0] = out.rows();
			return TensorMap<Scalar,Root::DATA_RANK>(out.data(), conversion_dims);
		}
		return in.unaryExpr([this](Scalar i) { return (Scalar) (i >= 0 ? i : (alpha * (exp(i) - 1))); });
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && conversion_dims[0] == out_grads.dimension(0));
		MatrixMap<Scalar> out_grads_mat(out_grads.data(), conversion_dims[0], Base::dims.get_volume());
		Matrix<Scalar> prev_out_grads(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); ++i) {
			for (int j = 0; j < in.rows(); ++j)
				prev_out_grads(j,i) = (Scalar) ((in(j,i) >= 0 ? 1 : (out(j,i) + alpha)) * out_grads_mat(j,i));
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
	 * @param param_reg The regularization function to apply to the layer's parameters.
	 * @param init_alpha The initial factor by which negative inputs are to be scaled.
	 * @param max_norm_constraint An optional max-norm constraint. If it is 0 or less, no
	 * constraint is applied.
	 */
	inline PReLUActivationLayer(const Dimensions<std::size_t,Rank>& dims, ParamRegSharedPtr<Scalar> param_reg = Root::NO_PARAM_REG,
			Scalar init_alpha = 1e-1, Scalar max_norm_constraint = 0) :
				Base::ActivationLayer(dims, 1, dims.get_volume()),
				param_reg(param_reg),
				init_alpha(init_alpha),
				max_norm_constraint(max_norm_constraint),
				max_norm(internal::NumericUtils<Scalar>::decidedly_greater(max_norm_constraint, (Scalar) 0)),
				conversion_dims(dims.template promote<>()) {
		assert(param_reg != nullptr);
	}
	inline Layer<Scalar,Rank>* clone() const {
		return new PReLUActivationLayer(*this);
	}
	inline void init() {
		Base::params_ref.setConstant(init_alpha);
		Base::params_grad.setZero(1, Base::dims.get_volume());
	}
protected:
	inline PReLUActivationLayer(const PReLUActivationLayer<Scalar,Rank>& layer, bool share_params = false) :
			Base::ActivationLayer(layer, share_params),
			param_reg(layer.param_reg),
			init_alpha(layer.init_alpha),
			max_norm_constraint(layer.max_norm_constraint),
			max_norm(layer.max_norm),
			conversion_dims(layer.conversion_dims) { }
	inline Layer<Scalar,Rank>* clone_with_shared_params() const {
		return new PReLUActivationLayer(*this, true);
	}
	inline void empty_cache() {
		in = Matrix<Scalar>(0, 0);
	}
	inline void regularize() {
		Base::params_grad += param_reg->d_function(Base::params_ref);
	}
	inline Scalar get_regularization_penalty() {
		return param_reg->function(Base::params_ref);
	}
	inline void enforce_constraints() {
		if (max_norm) {
			Scalar l2_norm = Base::params_ref.squaredNorm();
			if (l2_norm > max_norm_constraint)
				Base::params_ref *= (max_norm_constraint / l2_norm);
		}
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
		Base::params_grad.row(0).setZero();
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
	const ParamRegSharedPtr<Scalar> param_reg;
	const Scalar init_alpha;
	const Scalar max_norm_constraint;
	const bool max_norm;
	RankwiseArray conversion_dims;
	// Staged computation caches.
	Matrix<Scalar> in;
};

/**
 * An abstract base class template representing a pooling layer.
 */
template<typename Scalar, std::size_t Rank>
class PoolLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
public:
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return output_dims;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
protected:
	typedef std::array<std::size_t,4> Array4D;
	typedef std::array<std::size_t,2> ReductionRanksArray2D;
	inline PoolLayer(const Dimensions<std::size_t,Rank>& input_dims, const Dimensions<std::size_t,Rank>& output_dims,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride,
			std::size_t vertical_dilation, std::size_t horizontal_dilation) :
				input_dims(input_dims),
				output_dims(output_dims),
				ext_input_dims(input_dims.template extend<3 - Rank>()),
				ext_output_dims(output_dims.template extend<3 - Rank>()),
				receptor_height(receptor_height),
				receptor_width(receptor_width),
				vertical_stride(vertical_stride),
				horizontal_stride(horizontal_stride),
				vertical_dilation(vertical_dilation),
				horizontal_dilation(horizontal_dilation),
				dil_receptor_height(receptor_height + (receptor_height - 1) * vertical_dilation),
				dil_receptor_width(receptor_width + (receptor_width - 1) * horizontal_dilation),
				height_rem(ext_input_dims(0) - dil_receptor_height),
				width_rem(ext_input_dims(1) - dil_receptor_width),
				input_layer(false),
				frozen(false),
				reduction_ranks({ 1u, 2u }),
				broadcast({ 1u, receptor_height, receptor_width, 1u }),
				patch_offsets({ 0u, 0u, 0u, 0u }),
				patch_extents({ 0u, dil_receptor_height, dil_receptor_width, ext_input_dims(2) }),
				reduced_patch_offsets({ 0u, 0u, 0u, 0u }),
				reduced_patch_extents({ 0u, 1u, 1u, ext_input_dims(2) }),
				dil_strides({ 1u, vertical_dilation + 1u, horizontal_dilation + 1u, 1u }),
				params(0, 0),
				params_grad(0, 0) {
		assert(ext_input_dims(0) >= dil_receptor_height && ext_input_dims(1) >= dil_receptor_width);
		assert(receptor_height > 0 && receptor_width > 0);
		assert(vertical_stride > 0 && horizontal_stride > 0);
	}
	/**
	 * Initializes the cache required for back-propagation.
	 */
	virtual void _init_cache() = 0;
	/**
	 * Reduces the input tensor patch along the specified ranks.
	 *
	 * @param patch A tensor representing a spatial patch of the input tensor.
	 * @param patch_ind The index of the patch.
	 * @return The reduced tensor.
	 */
	virtual Tensor<Scalar,4> _reduce(const Tensor<Scalar,4>& patch, std::size_t patch_ind) = 0;
	/**
	 * Differentiates the reduction function and returns the derivative of the loss function
	 * w.r.t. the non-reduced patch.
	 *
	 * @param grad The derivative of the loss function w.r.t. the reduced patch.
	 * @param patch_ind The index of the patch.
	 * @return The derivative of the loss function w.r.t. the non-reduced patch.
	 */
	virtual Tensor<Scalar,4> _d_reduce(const Tensor<Scalar,4>& grad, std::size_t patch_ind) = 0;
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() {
		return 0;
	}
	inline void enforce_constraints() { }
	inline Tensor<Scalar,4> _pass_forward(Tensor<Scalar,4> in, bool training) {
		std::size_t rows = in.dimension(0);
		patch_extents[0] = rows;
		reduced_patch_extents[0] = rows;
		Tensor<Scalar,4> out(rows, ext_output_dims(0), ext_output_dims(1), ext_output_dims(2));
		_init_cache();
		std::size_t patch_ind = 0;
		std::size_t out_i = 0;
		for (std::size_t i = 0; i <= width_rem; i += horizontal_stride, ++out_i) {
			patch_offsets[2] = i;
			reduced_patch_offsets[2] = out_i;
			std::size_t out_j = 0;
			for (std::size_t j = 0; j <= height_rem; j += vertical_stride, ++out_j) {
				patch_offsets[1] = j;
				reduced_patch_offsets[1] = out_j;
				Tensor<Scalar,4> patch;
				// Dilated receptor support.
				if (vertical_dilation > 0 || horizontal_dilation > 0)
					patch = in.slice(patch_offsets, patch_extents).stride(dil_strides);
				else
					patch = in.slice(patch_offsets, patch_extents);
				out.slice(reduced_patch_offsets, reduced_patch_extents) = _reduce(patch, patch_ind++);
			}
		}
		return out;
	}
	inline Tensor<Scalar,4> _pass_back(Tensor<Scalar,4> out_grads) {
		if (input_layer)
			return Tensor<Scalar,4>();
		Tensor<Scalar,4> prev_out_grads(patch_extents[0], ext_input_dims(0), ext_input_dims(1),  ext_input_dims(2));
		prev_out_grads.setZero();
		std::size_t patch_ind = 0;
		std::size_t out_grads_i = 0;
		for (std::size_t i = 0; i <= width_rem; i += horizontal_stride, ++out_grads_i) {
			patch_offsets[2] = i;
			reduced_patch_offsets[2] = out_grads_i;
			std::size_t out_grads_j = 0;
			for (std::size_t j = 0; j <= height_rem; j += vertical_stride, ++out_grads_j) {
				patch_offsets[1] = j;
				reduced_patch_offsets[1] = out_grads_j;
				Tensor<Scalar,4> reduced_patch_grads = out_grads.slice(reduced_patch_offsets, reduced_patch_extents);
				// Accumulate the gradients where the patches overlap.
				if (vertical_dilation > 0 || horizontal_dilation > 0)
					prev_out_grads.slice(patch_offsets, patch_extents).stride(dil_strides) +=
							_d_reduce(reduced_patch_grads, patch_ind++);
				else
					prev_out_grads.slice(patch_offsets, patch_extents) += _d_reduce(reduced_patch_grads, patch_ind++);
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
	inline static std::size_t calculate_output_dim(std::size_t input_dim, std::size_t receptor_size, std::size_t dilation,
			std::size_t stride) {
		return (input_dim - receptor_size - (receptor_size - 1) * dilation) / stride + 1;
	}
	const Dimensions<std::size_t,Rank> input_dims;
	const Dimensions<std::size_t,Rank> output_dims;
	const Dimensions<std::size_t,3> ext_input_dims;
	const Dimensions<std::size_t,3> ext_output_dims;
	const std::size_t receptor_height;
	const std::size_t receptor_width;
	const std::size_t vertical_stride;
	const std::size_t horizontal_stride;
	const std::size_t vertical_dilation;
	const std::size_t horizontal_dilation;
	const std::size_t dil_receptor_height;
	const std::size_t dil_receptor_width;
	const std::size_t height_rem;
	const std::size_t width_rem;
	// Arrays for tensor manipulation.
	ReductionRanksArray2D reduction_ranks;
	Array4D broadcast;
	Array4D patch_offsets;
	Array4D patch_extents;
	Array4D reduced_patch_offsets;
	Array4D reduced_patch_extents;
	Array4D dil_strides;
private:
	bool input_layer;
	bool frozen;
	// No actual parameters.
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
};

/**
 * An abstract class template representing a sum pooling layer that reduces patches of the input by taking
 * their sums.
 */
template<typename Scalar, std::size_t Rank>
class SumPoolLayerBase : public PoolLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef PoolLayer<Scalar,Rank> Base;
public:
	inline SumPoolLayerBase(const Dimensions<std::size_t,Rank>& input_dims, const Dimensions<std::size_t,Rank>& output_dims,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride,
			std::size_t vertical_dilation, std::size_t horizontal_dilation) :
				Base::PoolLayer(input_dims, output_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride, vertical_dilation, horizontal_dilation) { }
protected:
	inline void empty_cache() { }
	inline void _init_cache() { }
	inline Tensor<Scalar,4> _reduce(const Tensor<Scalar,4>& patch, std::size_t patch_ind) {
		Tensor<Scalar,2> reduced_patch = patch.sum(Base::reduction_ranks);
		return TensorMap<Scalar,4>(reduced_patch.data(), Base::reduced_patch_extents);
	}
	inline Tensor<Scalar,4> _d_reduce(const Tensor<Scalar,4>& grad, std::size_t patch_ind) {
		return grad.broadcast(Base::broadcast);
	}
};

/**
 * A class template representing a 2D sum pooling layer operating on rank-3 data.
 */
template<typename Scalar, std::size_t Rank = 3>
class SumPoolLayer : public SumPoolLayerBase<Scalar,Rank> {
	typedef Layer<Scalar,3> Root;
	typedef PoolLayer<Scalar,3> PoolBase;
	typedef SumPoolLayerBase<Scalar,3> SumPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor field.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor field.
	 */
	inline SumPoolLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2,
			std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0) :
				SumPoolBase::SumPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_height, vertical_dilation, vertical_stride),
						PoolBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_dilation, horizontal_stride),
						input_dims(2) }, receptor_height, receptor_width, vertical_stride, horizontal_stride,
						vertical_dilation, horizontal_dilation) { }
	inline Root* clone() const {
		return new SumPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(std::move(in), training);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,4>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		return PoolBase::_pass_back(std::move(out_grads));
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 2D sum pooling layer operating on rank-2 data.
 */
template<typename Scalar>
class SumPoolLayer<Scalar,2> : public SumPoolLayerBase<Scalar,2> {
	typedef Layer<Scalar,2> Root;
	typedef PoolLayer<Scalar,2> PoolBase;
	typedef SumPoolLayerBase<Scalar,2> SumPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor field.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor field.
	 */
	inline SumPoolLayer(const Dimensions<std::size_t,2>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2,
			std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0) :
				SumPoolBase::SumPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_height, vertical_dilation, vertical_stride),
						PoolBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_dilation, horizontal_stride) },
						receptor_height, receptor_width, vertical_stride, horizontal_stride, vertical_dilation,
						horizontal_dilation) { }
	inline Root* clone() const {
		return new SumPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,3>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), in.dimension(2), 1u }), training)
				.reshape(std::array<std::size_t,3>({ batch_size, PoolBase::output_dims(0), PoolBase::output_dims(1) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,3>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,3>();
		return TensorMap<Scalar,3>(prev_out_grads.data(), { batch_size, PoolBase::input_dims(0), PoolBase::input_dims(1) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 1D sum pooling layer.
 */
template<typename Scalar>
class SumPoolLayer<Scalar,1> : public SumPoolLayerBase<Scalar,1> {
	typedef Layer<Scalar,1> Root;
	typedef PoolLayer<Scalar,1> PoolBase;
	typedef SumPoolLayerBase<Scalar,1> SumPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_length The length of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the length of the input tensor).
	 * @param dilation The extent of dilation to apply to the receptor field.
	 */
	inline SumPoolLayer(const Dimensions<std::size_t,1>& input_dims, std::size_t receptor_length = 2,
			std::size_t stride = 2, std::size_t dilation = 0) :
				SumPoolBase::SumPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_length, dilation, stride) },
						receptor_length, 1, stride, 1, dilation, 0) { }
	inline Root* clone() const {
		return new SumPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,2>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), 1u, 1u }), training)
				.reshape(std::array<std::size_t,2>({ batch_size, PoolBase::output_dims(0) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,2>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,2>();
		return TensorMap<Scalar,2>(prev_out_grads.data(), { batch_size, PoolBase::input_dims(0) });
	}
private:
	std::size_t batch_size;
};

/**
 * An abstract class template representing a pooling layer that reduces patches of the input by taking their
 * means.
 */
template<typename Scalar, std::size_t Rank>
class MeanPoolLayerBase : public PoolLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef PoolLayer<Scalar,Rank> Base;
public:
	inline MeanPoolLayerBase(const Dimensions<std::size_t,Rank>& input_dims, const Dimensions<std::size_t,Rank>& output_dims,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride,
			std::size_t vertical_dilation, std::size_t horizontal_dilation) :
				Base::PoolLayer(input_dims, output_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride, vertical_dilation, horizontal_dilation),
				receptor_area(receptor_height * receptor_width) { }
protected:
	inline void empty_cache() { }
	inline void _init_cache() { }
	inline Tensor<Scalar,4> _reduce(const Tensor<Scalar,4>& patch, std::size_t patch_ind) {
		Tensor<Scalar,2> reduced_patch = patch.mean(Base::reduction_ranks);
		return TensorMap<Scalar,4>(reduced_patch.data(), Base::reduced_patch_extents);
	}
	inline Tensor<Scalar,4> _d_reduce(const Tensor<Scalar,4>& grad, std::size_t patch_ind) {
		return (grad / (Scalar) receptor_area).broadcast(Base::broadcast);
	}
private:
	std::size_t receptor_area;
};

/**
 * A class template representing a 2D mean pooling layer operating on rank-3 data.
 */
template<typename Scalar, std::size_t Rank = 3>
class MeanPoolLayer : public MeanPoolLayerBase<Scalar,Rank> {
	typedef Layer<Scalar,3> Root;
	typedef PoolLayer<Scalar,3> PoolBase;
	typedef MeanPoolLayerBase<Scalar,3> MeanPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor field.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor field.
	 */
	inline MeanPoolLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2,
			std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0) :
				MeanPoolBase::MeanPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_height, vertical_dilation, vertical_stride),
						PoolBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_dilation, horizontal_stride),
						input_dims(2) }, receptor_height, receptor_width, vertical_stride, horizontal_stride,
						vertical_dilation, horizontal_dilation) { }
	inline Root* clone() const {
		return new MeanPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(std::move(in), training);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,4>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		return PoolBase::_pass_back(std::move(out_grads));
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 2D mean pooling layer operating on rank-2 data.
 */
template<typename Scalar>
class MeanPoolLayer<Scalar,2> : public MeanPoolLayerBase<Scalar,2> {
	typedef Layer<Scalar,2> Root;
	typedef PoolLayer<Scalar,2> PoolBase;
	typedef MeanPoolLayerBase<Scalar,2> MeanPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor field.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor field.
	 */
	inline MeanPoolLayer(const Dimensions<std::size_t,2>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2,
			std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0) :
				MeanPoolBase::MeanPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_height, vertical_dilation, vertical_stride),
						PoolBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_dilation, horizontal_stride) },
						receptor_height, receptor_width, vertical_stride, horizontal_stride, vertical_dilation,
						horizontal_dilation) { }
	inline Root* clone() const {
		return new MeanPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,3>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), in.dimension(2), 1u }), training)
				.reshape(std::array<std::size_t,3>({ batch_size, PoolBase::output_dims(0), PoolBase::output_dims(1) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,3>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,3>();
		return TensorMap<Scalar,3>(prev_out_grads.data(), { batch_size, PoolBase::input_dims(0), PoolBase::input_dims(1) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 1D mean pooling layer.
 */
template<typename Scalar>
class MeanPoolLayer<Scalar,1> : public MeanPoolLayerBase<Scalar,1> {
	typedef Layer<Scalar,1> Root;
	typedef PoolLayer<Scalar,1> PoolBase;
	typedef MeanPoolLayerBase<Scalar,1> MeanPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_length The length of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the length of the input tensor).
	 * @param dilation The extent of dilation to apply to the receptor field.
	 */
	inline MeanPoolLayer(const Dimensions<std::size_t,1>& input_dims, std::size_t receptor_length = 2,
			std::size_t stride = 2, std::size_t dilation = 0) :
				MeanPoolBase::MeanPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_length, dilation, stride) },
						receptor_length, 1, stride, 1, dilation, 0) { }
	inline Root* clone() const {
		return new MeanPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,2>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), 1u, 1u }), training)
				.reshape(std::array<std::size_t,2>({ batch_size, PoolBase::output_dims(0) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,2>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,2>();
		return TensorMap<Scalar,2>(prev_out_grads.data(), { batch_size, PoolBase::input_dims(0) });
	}
private:
	std::size_t batch_size;
};

/**
 * An abstract class template representing a pooling layer that reduces patches of the input by taking their
 * maxima.
 */
template<typename Scalar, std::size_t Rank>
class MaxPoolLayerBase : public PoolLayer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Root;
	typedef PoolLayer<Scalar,Rank> Base;
public:
	inline MaxPoolLayerBase(const Dimensions<std::size_t,Rank>& input_dims, const Dimensions<std::size_t,Rank>& output_dims,
			std::size_t receptor_height, std::size_t receptor_width, std::size_t vertical_stride, std::size_t horizontal_stride,
			std::size_t vertical_dilation, std::size_t horizontal_dilation) :
				Base::PoolLayer(input_dims, output_dims, receptor_height, receptor_width, vertical_stride,
						horizontal_stride, vertical_dilation, horizontal_dilation) { }
protected:
	inline void empty_cache() {
		max_inds = std::vector<std::vector<unsigned>>(0);
	}
	inline void _init_cache() {
		max_inds = std::vector<std::vector<unsigned>>(Base::ext_output_dims(0) * Base::ext_output_dims(1));
	}
	inline Tensor<Scalar,4> _reduce(const Tensor<Scalar,4>& patch, std::size_t patch_ind) {
		std::size_t rows = patch.dimension(0);
		std::size_t depth = patch.dimension(3);
		std::vector<unsigned> inds(rows * depth);
		Tensor<Scalar,4> reduced_patch(rows, 1u, 1u, depth);
		for (std::size_t i = 0; i < depth; ++i) {
			for (std::size_t j = 0; j < rows; ++j) {
				Scalar max = internal::NumericUtils<Scalar>::MIN;
				unsigned max_height = 0;
				unsigned max_width = 0;
				for (std::size_t k = 0; k < Base::receptor_width; ++k) {
					for (std::size_t l = 0; l < Base::receptor_height; ++l) {
						Scalar val = patch(j,l,k,i);
						if (val > max) {
							max = val;
							max_height = l;
							max_width = k;
						}
					}
				}
				inds[i * rows + j] = max_width * Base::receptor_height + max_height;
				reduced_patch(j,0u,0u,i) = max;
			}
		}
		max_inds[patch_ind] = inds;
		return reduced_patch;
	}
	inline Tensor<Scalar,4> _d_reduce(const Tensor<Scalar,4>& grad, std::size_t patch_ind) {
		std::size_t rows = grad.dimension(0);
		std::size_t depth = grad.dimension(3);
		Tensor<Scalar,4> patch(rows, Base::receptor_height, Base::receptor_width, depth);
		patch.setZero();
		std::vector<unsigned>& inds = max_inds[patch_ind];
		for (std::size_t i = 0; i < depth; ++i) {
			for (std::size_t j = 0; j < rows; ++j) {
				unsigned max_ind = inds[i * rows + j];
				unsigned max_height = max_ind % Base::receptor_height;
				unsigned max_width = max_ind / Base::receptor_height;
				patch(j,max_height,max_width,i) = grad(j,0u,0u,i);
			}
		}
		return patch;
	}
private:
	// Cache
	std::vector<std::vector<unsigned>> max_inds;
};

/**
 * A class template representing a 2D max pooling layer operating on rank-3 data.
 */
template<typename Scalar, std::size_t Rank = 3>
class MaxPoolLayer : public MaxPoolLayerBase<Scalar,Rank> {
	typedef Layer<Scalar,3> Root;
	typedef PoolLayer<Scalar,3> PoolBase;
	typedef MaxPoolLayerBase<Scalar,3> MaxPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor field.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor field.
	 */
	inline MaxPoolLayer(const Dimensions<std::size_t,3>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2,
			std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0) :
				MaxPoolBase::MaxPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_height, vertical_dilation, vertical_stride),
						PoolBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_dilation, horizontal_stride),
						input_dims(2) }, receptor_height, receptor_width, vertical_stride, horizontal_stride,
						vertical_dilation, horizontal_dilation) { }
	inline Root* clone() const {
		return new MaxPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,4>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(std::move(in), training);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,4>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		return PoolBase::_pass_back(std::move(out_grads));
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 2D max pooling layer operating on rank-2 data.
 */
template<typename Scalar>
class MaxPoolLayer<Scalar,2> : public MaxPoolLayerBase<Scalar,2> {
	typedef Layer<Scalar,2> Root;
	typedef PoolLayer<Scalar,2> PoolBase;
	typedef MaxPoolLayerBase<Scalar,2> MaxPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_height The height of the pooling receptor.
	 * @param receptor_width The width of the pooling receptor.
	 * @param vertical_stride The vertical stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the height of the input tensor).
	 * @param horizontal_stride The horizontal stride at which the input is to be pooled (i.e. the number
	 * of elements by which the receptor is to be shifted along the width of the input tensor).
	 * @param vertical_dilation The extent of vertical dilation to apply to the receptor field.
	 * @param horizontal_dilation The extent of horizontal dilation to apply to the receptor field.
	 */
	inline MaxPoolLayer(const Dimensions<std::size_t,2>& input_dims, std::size_t receptor_height = 2,
			std::size_t receptor_width = 2, std::size_t vertical_stride = 2, std::size_t horizontal_stride = 2,
			std::size_t vertical_dilation = 0, std::size_t horizontal_dilation = 0) :
				MaxPoolBase::MaxPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_height, vertical_dilation, vertical_stride),
						PoolBase::calculate_output_dim(input_dims(1), receptor_width, horizontal_dilation, horizontal_stride) },
						receptor_height, receptor_width, vertical_stride, horizontal_stride, vertical_dilation,
						horizontal_dilation) { }
	inline Root* clone() const {
		return new MaxPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,3>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), in.dimension(2), 1u }), training)
				.reshape(std::array<std::size_t,3>({ batch_size, PoolBase::output_dims(0), PoolBase::output_dims(1) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,3>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,3>();
		return TensorMap<Scalar,3>(prev_out_grads.data(), { batch_size, PoolBase::input_dims(0), PoolBase::input_dims(1) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a 1D max pooling layer.
 */
template<typename Scalar>
class MaxPoolLayer<Scalar,1> : public MaxPoolLayerBase<Scalar,1> {
	typedef Layer<Scalar,1> Root;
	typedef PoolLayer<Scalar,1> PoolBase;
	typedef MaxPoolLayerBase<Scalar,1> MaxPoolBase;
public:
	/**
	 * @param input_dims The dimensionality of the input tensor.
	 * @param receptor_length The length of the pooling receptor.
	 * @param stride The stride at which the input is to be pooled (i.e. the number of
	 * elements by which the receptor is to be shifted along the length of the input tensor).
	 * @param dilation The extent of dilation to apply to the receptor field.
	 */
	inline MaxPoolLayer(const Dimensions<std::size_t,1>& input_dims, std::size_t receptor_length = 2,
			std::size_t stride = 2, std::size_t dilation = 0) :
				MaxPoolBase::MaxPoolLayerBase(input_dims, {
						PoolBase::calculate_output_dim(input_dims(0), receptor_length, dilation, stride) },
						receptor_length, 1, stride, 1, dilation, 0) { }
	inline Root* clone() const {
		return new MaxPoolLayer(*this);
	}
protected:
	inline Root* clone_with_shared_params() const {
		return clone();
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,2>(in.dimensions()).template demote<>()) == PoolBase::input_dims);
		assert(in.dimension(0) > 0);
		batch_size = in.dimension(0);
		return PoolBase::_pass_forward(TensorMap<Scalar,4>(in.data(), { batch_size, in.dimension(1), 1u, 1u }), training)
				.reshape(std::array<std::size_t,2>({ batch_size, PoolBase::output_dims(0) }));
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,2>(out_grads.dimensions()).template demote<>()) == PoolBase::output_dims);
		assert(out_grads.dimension(0) > 0 && batch_size == out_grads.dimension(0));
		Tensor<Scalar,4> prev_out_grads = PoolBase::_pass_back(TensorMap<Scalar,4>(out_grads.data(),
				{ batch_size, PoolBase::ext_output_dims(0), PoolBase::ext_output_dims(1), PoolBase::ext_output_dims(2) }));
		if (PoolBase::is_input_layer())
			return Tensor<Scalar,2>();
		return TensorMap<Scalar,2>(prev_out_grads.data(), { batch_size, PoolBase::input_dims(0) });
	}
private:
	std::size_t batch_size;
};

/**
 * A class template representing a broadcasting layer that repeats the contents of its input tensors
 * along its ranks.
 */
template<typename Scalar, std::size_t Rank>
class BroadcastLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Rank + 1> RankwiseArray;
public:
	/**
	 * @param input_dims The nominal input dimensions of the layer.
	 * @param broadcast The number of times the input tensor's contents are
	 * repeated along each rank. All elements should be greater than 0.
	 */
	inline BroadcastLayer(const Dimensions<std::size_t,Rank>& input_dims,
			const Dimensions<std::size_t,Rank>& broadcast) :
				input_dims(input_dims),
				output_dims(input_dims * broadcast),
				input_layer(false),
				frozen(false),
				broadcast(broadcast.template promote<>()),
				params(0, 0),
				params_grad(0, 0) {
		slice_offsets.fill(0);
		for (std::size_t i = 0; i < Rank; ++i)
			assert(broadcast(i) > 0);
	}
	inline Base* clone() const {
		return new BroadcastLayer(*this);
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return output_dims;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
protected:
	inline Base* clone_with_shared_params() const {
		return clone();
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void empty_cache() { }
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() {
		return 0;
	}
	inline void enforce_constraints() { }
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == input_dims);
		assert(in.dimension(0) > 0);
		rows = in.dimension(0);
		return in.broadcast(broadcast);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == output_dims);
		assert(out_grads.dimension(0) > 0 && rows == out_grads.dimension(0));
		typename Base::Data prev_out_grads = std::move(out_grads);
		slice_offsets.fill(0);
		slice_extents = output_dims.template promote<>();
		slice_extents[0] = rows;
		for (std::size_t i = 0; i < Rank; ++i) {
			if (broadcast[i + 1] <= 1)
				continue;
			slice_extents[i + 1] = input_dims(i);
			typename Base::Data work_tensor(slice_extents);
			work_tensor.setZero();
			for (std::size_t j = 0; j < broadcast[i + 1]; ++j) {
				work_tensor += prev_out_grads.slice(slice_offsets, slice_extents);
				slice_offsets[i + 1] += input_dims(i);
			}
			slice_offsets[i + 1] = 0;
			prev_out_grads = std::move(work_tensor);
		}
		return prev_out_grads;
	}
private:
	const Dimensions<std::size_t,Rank> input_dims;
	const Dimensions<std::size_t,Rank> output_dims;
	RankwiseArray broadcast;
	RankwiseArray slice_offsets;
	RankwiseArray slice_extents;
	std::size_t rows;
	bool input_layer;
	bool frozen;
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
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
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
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
protected:
	typedef std::array<std::size_t,Base::DATA_RANK> RankwiseArray;
	inline BatchNormLayerBase(const Dimensions<std::size_t,Rank>& dims, ParamRegSharedPtr<Scalar> gamma_reg,
			ParamRegSharedPtr<Scalar> beta_reg, Scalar gamma_max_norm_constraint, Scalar beta_max_norm_constraint,
			Scalar norm_avg_decay, Scalar epsilon) :
				dims(dims),
				depth(dims.template extend<3 - Rank>()(2)),
				gamma_reg(gamma_reg),
				beta_reg(beta_reg),
				gamma_max_norm_constraint(gamma_max_norm_constraint),
				beta_max_norm_constraint(beta_max_norm_constraint),
				gamma_max_norm(internal::NumericUtils<Scalar>::decidedly_greater(gamma_max_norm_constraint, (Scalar) 0)),
				beta_max_norm(internal::NumericUtils<Scalar>::decidedly_greater(beta_max_norm_constraint, (Scalar) 0)),
				norm_avg_decay(norm_avg_decay),
				epsilon(epsilon),
				input_layer(false),
				frozen(false),
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
			frozen(layer.frozen),
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
	inline void regularize() {
		params_grad.topRows(depth) += gamma_reg->d_function(params_ref.topRows(depth));
		params_grad.bottomRows(depth) += beta_reg->d_function(params_ref.bottomRows(depth));
	}
	inline Scalar get_regularization_penalty() {
		return gamma_reg->function(params_ref.topRows(depth)) + beta_reg->function(params_ref.bottomRows(depth));
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
	const ParamRegSharedPtr<Scalar> gamma_reg;
	const ParamRegSharedPtr<Scalar> beta_reg;
	const Scalar gamma_max_norm_constraint;
	const Scalar beta_max_norm_constraint;
	const bool gamma_max_norm;
	const bool beta_max_norm;
	const Scalar norm_avg_decay;
	const Scalar epsilon;
	Matrix<Scalar>& params_ref;
private:
	bool input_layer;
	bool frozen;
	// Dynamic batch normalization parameters.
	Matrix<Scalar> avg_means;
	Matrix<Scalar> avg_inv_sds;
	bool avgs_init;
	// Betas and gammas
	mutable Matrix<Scalar> params;
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
	typedef Layer<Scalar,Rank> Root;
	typedef BatchNormLayerBase<Scalar,Rank> Base;
public:
	/**
	 * @param dims The dimensionality of the input tensor.
	 * @param gamma_reg The regularization function to apply to the layer's gamma parameters.
	 * @param beta_reg The regularization function to apply to the layer's beta parameters.
	 * @param gamma_max_norm_constraint An optional max-norm constraint to enforce on the
	 * gamma parameters. If it is 0 or less, no constraint is applied.
	 * @param beta_max_norm_constraint An optional max-norm constraint to enforce on the
	 * beta parameters. If it is 0 or less, no constraint is applied.
	 * @param norm_avg_decay The decay rate of the maintained means and variances.
	 * @param epsilon A small constant used to maintain numerical stability.
	 */
	inline BatchNormLayer(const Dimensions<std::size_t,Rank>& dims, ParamRegSharedPtr<Scalar> gamma_reg = Root::NO_PARAM_REG,
			ParamRegSharedPtr<Scalar> beta_reg = Root::NO_PARAM_REG, Scalar gamma_max_norm_constraint = 0,
			Scalar beta_max_norm_constraint = 0, Scalar norm_avg_decay = .1, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				Base::BatchNormLayerBase(dims, gamma_reg, beta_reg, gamma_max_norm_constraint, beta_max_norm_constraint,
						norm_avg_decay, epsilon),
				conversion_dims(dims.template promote<>()) { }
	inline Root* clone() const {
		return new BatchNormLayer(*this);
	}
protected:
	inline BatchNormLayer(const BatchNormLayer<Scalar,Rank>& layer, bool share_params = false) :
			Base::BatchNormLayerBase(layer, share_params),
			conversion_dims(layer.conversion_dims) { }
	inline Root* clone_with_shared_params() const {
		return new BatchNormLayer(*this, true);
	}
	inline typename Root::Data pass_forward(typename Root::Data in, bool training) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(in.dimensions()).template demote<>()) == Base::dims);
		assert(in.dimension(0) > 0);
		conversion_dims[0] = in.dimension(0);
		return Base::_pass_forward(std::move(in), conversion_dims, training, 0);
	}
	inline typename Root::Data pass_back(typename Root::Data out_grads) {
		assert((Dimensions<std::size_t,Root::DATA_RANK>(out_grads.dimensions()).template demote<>()) == Base::dims);
		assert(out_grads.dimension(0) > 0 && conversion_dims[0] == out_grads.dimension(0));
		typename Root::Data prev_out_grads = Base::_pass_back(std::move(out_grads), conversion_dims, 0);
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
		 * @param gamma_reg The regularization function to apply to the layer's gamma parameters.
		 * @param beta_reg The regularization function to apply to the layer's beta parameters.
		 * @param gamma_max_norm_constraint An optional max-norm constraint to enforce on the
		 * gamma parameters. If it is 0 or less, no constraint is applied.
		 * @param beta_max_norm_constraint An optional max-norm constraint to enforce on the
		 * beta parameters. If it is 0 or less, no constraint is applied.
		 * @param norm_avg_decay The decay rate of the maintained means and variances.
		 * @param epsilon A small constant used to maintain numerical stability.
		 */
	inline BatchNormLayer(Dimensions<std::size_t,3> dims, ParamRegSharedPtr<Scalar> gamma_reg = Root::NO_PARAM_REG,
			ParamRegSharedPtr<Scalar> beta_reg = Root::NO_PARAM_REG, Scalar gamma_max_norm_constraint = 0,
			Scalar beta_max_norm_constraint = 0, Scalar norm_avg_decay = .1, Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				Base::BatchNormLayerBase(dims, gamma_reg, beta_reg, gamma_max_norm_constraint, beta_max_norm_constraint,
						norm_avg_decay, epsilon),
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
			extents[0] = rows;
			return Base::_pass_forward(std::move(in), extents, training, 0);
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
		if (Base::dims(2) == 1)
			prev_out_grads = Base::_pass_back(std::move(out_grads), extents, 0);
		else {
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
			Scalar epsilon = internal::NumericUtils<Scalar>::EPSILON2) :
				dims(dims),
				dropout_prob(dropout_prob),
				epsilon(epsilon),
				dropout(internal::NumericUtils<Scalar>::decidedly_greater(dropout_prob, .0)),
				input_layer(false),
				frozen(false),
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
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
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
	inline void empty_cache() {
		dropout_mask = typename Base::Data();
	}
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() {
		return 0;
	}
	inline void enforce_constraints() { }
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == dims);
		assert(in.dimension(0) > 0);
		if (training && dropout) {
			// Inverted dropout.
			Scalar scaling_factor = (Scalar) 1 / (1 - dropout_prob + epsilon);
			dropout_mask = ((in.random() + in.constant(1)) / (Scalar) 2).unaryExpr([this,scaling_factor](Scalar i) {
				return (Scalar) (i <= dropout_prob ? 0 : scaling_factor);
			});
			return in * dropout_mask;
		}
		return in;
	}
	inline typename Base::Data pass_back(typename Base::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == dims);
		assert(out_grads.dimension(0) > 0 && dropout_mask.dimension(0) == out_grads.dimension(0));
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
	bool frozen;
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
	// Staged computation cache.
	typename Base::Data dropout_mask;
};

/**
 * A class template representing a reshaping layer that outputs a reshaped copy of the input
 * tensor with the same volume. The data backing the tensor is not shifted in any way.
 */
template<typename Scalar, std::size_t Rank>
class ReshapeLayer : public Layer<Scalar,Rank> {
	typedef Layer<Scalar,Rank> Base;
	typedef std::array<std::size_t,Rank + 1> RankwiseArray;
public:
	/**
	 * @param input_dims The nominal input dimensions of the layer.
	 * @param output_dims The dimensions of the reshaped tensor. The output tensor must have
	 * the same volume as the input tensor.
	 */
	inline ReshapeLayer(const Dimensions<std::size_t,Rank>& input_dims,
			const Dimensions<std::size_t,Rank>& output_dims) :
				input_dims(input_dims),
				output_dims(output_dims),
				input_layer(false),
				frozen(false),
				input_conversion_dims(output_dims.template promote<>()),
				output_conversion_dims(input_dims.template promote<>()),
				params(0, 0),
				params_grad(0, 0) {
		assert(input_dims.get_volume() == output_dims.get_volume());
	}
	inline Base* clone() const {
		return new ReshapeLayer(*this);
	}
	inline const Dimensions<std::size_t,Rank>& get_input_dims() const {
		return input_dims;
	}
	inline const Dimensions<std::size_t,Rank>& get_output_dims() const {
		return output_dims;
	}
	inline bool is_frozen() const {
		return frozen;
	}
	inline void set_frozen(bool frozen) {
		this->frozen = frozen;
	}
	inline void init() { }
protected:
	inline Base* clone_with_shared_params() const {
		return clone();
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void empty_cache() { }
	inline Matrix<Scalar>& get_params() {
		return params;
	}
	inline Matrix<Scalar>& get_params_grad() {
		return params_grad;
	}
	inline void regularize() { }
	inline Scalar get_regularization_penalty() {
		return 0;
	}
	inline void enforce_constraints() { }
	inline typename Base::Data pass_forward(typename Base::Data in, bool training) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(in.dimensions()).template demote<>()) == input_dims);
		assert(in.dimension(0) > 0);
		input_conversion_dims[0] = in.dimension(0);
		return in.reshape(input_conversion_dims);
	}
	inline typename Base::Data pass_back(typename Base::Data out_grads) {
		assert((Dimensions<std::size_t,Base::DATA_RANK>(out_grads.dimensions()).template demote<>()) == output_dims);
		assert(out_grads.dimension(0) > 0 && input_conversion_dims[0] == out_grads.dimension(0));
		output_conversion_dims[0] = input_conversion_dims[0];
		return out_grads.reshape(output_conversion_dims);
	}
private:
	const Dimensions<std::size_t,Rank> input_dims;
	const Dimensions<std::size_t,Rank> output_dims;
	RankwiseArray input_conversion_dims;
	RankwiseArray output_conversion_dims;
	bool input_layer;
	bool frozen;
	Matrix<Scalar> params;
	Matrix<Scalar> params_grad;
};

} /* namespace cattle */

#endif /* CATTL3_LAYER_H_ */
