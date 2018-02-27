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
#include <Dimensions.h>
#include <memory>
#include <type_traits>
#include <Utils.h>
#include <utility>
#include <WeightInitialization.h>

namespace cattle {

// TODO Optional GPU acceleration using cuBLAS and cuDNN.

// Forward declarations to NeuralNetwork and Optimizer so they can be friended.
template<typename Scalar, size_t Rank, bool Sequential> class NeuralNetwork;
template<typename Scalar, size_t Rank, bool Sequential> class Optimizer;

template<typename Scalar, size_t Rank>
class Layer {
	static_assert(std::is_floating_point<Scalar>::value, "non floating-point scalar type");
	static_assert(Rank > 0 && Rank < 4, "illegal rank");
	friend class NeuralNetwork<Scalar,Rank,true>;
	friend class NeuralNetwork<Scalar,Rank,false>;
	friend class Optimizer<Scalar,Rank,true>;
	friend class Optimizer<Scalar,Rank,false>;
	typedef Tensor<Scalar,Rank + 1> Data;
public:
	virtual ~Layer() = default;
	// Clone pattern.
	virtual Layer<Scalar,Rank>* clone() const = 0;
	virtual const Dimensions<int,Rank>& get_input_dims() const = 0;
	virtual const Dimensions<int,Rank>& get_output_dims() const = 0;
	inline bool is_parametric() {
		return get_params().rows() > 0 && get_params().cols() > 0;
	}
protected:
	/* Only expose methods that allow for the modification of the
	 * layer's state to friends and sub-classes. */
	virtual bool is_input_layer() const;
	virtual void set_input_layer(bool input_layer);
	virtual void init() = 0;
	virtual void empty_cache() = 0;
	virtual Matrix<Scalar>& get_params() = 0;
	virtual Matrix<Scalar>& get_param_grads() = 0;
	virtual void enforce_constraints() = 0;
	// Rank is increased by one to allow for batch training.
	virtual Data pass_forward(Data in, bool training) = 0;
	virtual Data pass_back(Data out_grads) = 0;
};

template<typename Scalar>
using WeightInitSharedPtr = std::shared_ptr<WeightInitialization<Scalar>>;

template<typename Scalar, size_t Rank>
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
			size_t weight_rows, size_t weight_cols, Scalar max_norm_constraint) :
				input_dims(input_dims),
				output_dims(output_dims),
				weight_init(weight_init),
				max_norm_constraint(max_norm_constraint),
				max_norm(Utils<Scalar>::decidedly_greater(max_norm_constraint, .0)),
				input_layer(input_layer),
				weights(weight_rows, weight_cols),
				weight_grads(weight_rows, weight_cols) {
		assert(weight_init != nullptr);
	}
	inline bool is_input_layer() const {
		return input_layer;
	}
	inline void set_input_layer(bool input_layer) {
		this->input_layer = input_layer;
	}
	inline void init() {
		weight_init->apply(weights);
		weight_grads.setZero(weight_grads.rows(), weight_grads.cols());
	}
	inline Matrix<Scalar>& get_params() {
		return weights;
	}
	inline Matrix<Scalar>& get_param_grads() {
		return weight_grads;
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
	Matrix<Scalar> weights;
	Matrix<Scalar> weight_grads;
};

template<typename Scalar, size_t Rank>
class FCLayer : public KernelLayer<Scalar,Rank> {
	typedef KernelLayer<Scalar,Rank> Base;
	typedef Tensor<Scalar,Rank + 1> Data;
public:
	inline FCLayer(const Dimensions<int,Rank>& input_dims, size_t output_size, WeightInitSharedPtr<Scalar> weight_init,
			Scalar max_norm_constraint = 0) :
				Base::KernelLayer(input_dims, Dimensions<int,Rank>({ (int) output_size }), weight_init,
						input_dims.get_volume() + 1, output_size, max_norm_constraint) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new FCLayer(*this);
	}
protected:
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
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>((biased_in * Base::weights).eval(), Base::output_dims);
	}
	inline Data pass_back(Data out_grads) {
		assert(Utils<Scalar>::template get_dims<Rank + 1>(out_grads).template demote<>() == Base::output_dims);
		assert(out_grads.dimension(0) > 0 && biased_in.rows() == out_grads.dimension(0));
		Matrix<Scalar> out_grads_mat = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads));
		// Compute the gradients of the outputs with respect to the weights.
		Base::weight_grads = biased_in.transpose() * out_grads_mat;
		if (Base::is_input_layer())
			return Utils<Scalar>::template get_null_tensor<Rank + 1>();
		/* Remove the bias row from the weight matrix, transpose it, and compute gradients w.r.t. the
		 * previous layer's output. */
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>((out_grads_mat *
				Base::weights.topRows(Base::input_dims.get_volume()).transpose()).eval(), Base::input_dims);
	}
private:
	// Staged computation caches
	Matrix<Scalar> biased_in;
};

// 3D convolutional layer.
template<typename Scalar>
class ConvLayer : public KernelLayer<Scalar,3> {
	typedef KernelLayer<Scalar,3> Base;
	typedef Tensor<Scalar,4> Data;
	typedef std::array<int,4> RankwiseArray;
public:
	inline ConvLayer(const Dimensions<int,3>& input_dims, size_t filters, WeightInitSharedPtr<Scalar> weight_init,
			size_t receptor_size = 3, size_t padding = 1, size_t stride = 1, size_t dilation = 0,
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
		in = Utils<Scalar>::template get_null_tensor<4>();
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
			Matrix<Scalar> out_i = biased_in_vec[i] * Base::weights;
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
		Base::weight_grads.setZero(Base::weight_grads.rows(), Base::weight_grads.cols());
		for (int i = 0; i < rows; ++i) {
			out_grads_row_offsets[0] = i;
			Matrix<Scalar> prev_out_grads_mat_i;
			{
				Data slice_i = out_grads.slice(out_grads_row_offsets, out_grads_row_extents);
				MatrixMap<Scalar> out_grads_mat_map_i = MatrixMap<Scalar>(slice_i.data(), Base::output_dims(0) *
						Base::output_dims(1), filters);
				// Accumulate the gradients of the outputs w.r.t. the weights for each observation a.k.a 'tensor-row'.
				Base::weight_grads += biased_in_vec[i].transpose() * out_grads_mat_map_i;
				if (Base::is_input_layer())
					continue;
				/* Remove the bias row from the weight matrix, transpose it, and compute the gradients w.r.t. the
				 * previous layer's output. */
				prev_out_grads_mat_i = out_grads_mat_map_i * Base::weights.topRows(Base::weights.rows() - 1).transpose();
			}
			/* Given the gradients w.r.t. the stretched out receptor patches, perform a 'backwards' convolution
			 * to get the gradients w.r.t. the individual input nodes. */
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
	static int calculate_output_dim(int input_dim, int receptor_size, int padding, int dilation, int stride) {
		return (input_dim - receptor_size - (receptor_size - 1) * dilation + 2 * padding) / stride + 1;
	}
private:
	const size_t filters;
	const size_t receptor_size;
	const size_t padding;
	const size_t stride;
	const size_t dilation;
	const int padded_height;
	const int padded_width;
	const int dil_receptor_size;
	// Staged computation caches
	std::vector<Matrix<Scalar>> biased_in_vec;
};

template<typename Scalar, size_t Rank>
class ActivationLayer : public Layer<Scalar,Rank> {
	typedef Tensor<Scalar,Rank + 1> Data;
public:
	inline ActivationLayer(const Dimensions<int,Rank>& dims) :
			dims(dims),
			input_layer(false),
			params(0, 0),
			param_grads(0, 0) { }
	virtual ~ActivationLayer() = default;
	inline const Dimensions<int,Rank>& get_input_dims() const {
		return dims;
	}
	inline const Dimensions<int,Rank>& get_output_dims() const {
		return dims;
	}
protected:
	virtual Matrix<Scalar> activate(const Matrix<Scalar>& in) = 0;
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
		return params;
	}
	inline Matrix<Scalar>& get_param_grads() {
		return param_grads;
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
			return Utils<Scalar>::template get_null_tensor<Rank + 1>();
		return Utils<Scalar>::template map_mat_to_tensor<Rank + 1>(d_activate(in, out,
				Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads))), dims);
	}
	const Dimensions<int,Rank> dims;
	bool input_layer;
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Staged computation caches
	Matrix<Scalar> in;
	Matrix<Scalar> out;
};

template<typename Scalar, size_t Rank>
class IdentityActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline IdentityActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new IdentityActivationLayer(*this);
	}
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in;
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return out_grads;
	}
};

template<typename Scalar, size_t Rank>
class ScalingActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline ScalingActivationLayer(const Dimensions<int,Rank>& dims, Scalar scale) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			scale(scale) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ScalingActivationLayer(*this);
	}
protected:
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

template<typename Scalar, size_t Rank>
class BinaryStepActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline BinaryStepActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new BinaryStepActivationLayer(*this);
	}
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : .0); });
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return Matrix<Scalar>::Zero(in.rows(), in.cols());
	}
};

template<typename Scalar, size_t Rank>
class SigmoidActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline SigmoidActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SigmoidActivationLayer(*this);
	}
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return ((-in).array().exp() + 1).inverse();
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return (out.array() *  (-out.array() + 1)) * out_grads.array();
	}
};

template<typename Scalar, size_t Rank>
class TanhActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline TanhActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new TanhActivationLayer(*this);
	}
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.array().tanh();
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return (-out.array() * out.array() + 1) * out_grads.array();
	}
};

template<typename Scalar, size_t Rank>
class SoftmaxActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline SoftmaxActivationLayer(const Dimensions<int,Rank>& dims, Scalar epsilon = Utils<Scalar>::EPSILON2) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			epsilon(epsilon) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new SoftmaxActivationLayer(*this);
	}
protected:
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

template<typename Scalar, size_t Rank>
class ReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline ReLUActivationLayer(const Dimensions<int,Rank>& dims) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ReLUActivationLayer(*this);
	}
protected:
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(.0);
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		return in.unaryExpr([](Scalar i) { return (Scalar) (i >= .0 ? 1.0 : .0); })
				.cwiseProduct(out_grads);
	}
};

template<typename Scalar, size_t Rank>
class LeakyReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline LeakyReLUActivationLayer(const Dimensions<int,Rank>& dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new LeakyReLUActivationLayer(*this);
	}
protected:
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

template<typename Scalar, size_t Rank>
class ELUActivationLayer : public ActivationLayer<Scalar,Rank> {
public:
	inline ELUActivationLayer(const Dimensions<int,Rank>& dims, Scalar alpha = 1e-1) :
			ActivationLayer<Scalar,Rank>::ActivationLayer(dims),
			alpha(alpha) { }
	inline Layer<Scalar,Rank>* clone() const {
		return new ELUActivationLayer(*this);
	}
protected:
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

template<typename Scalar, size_t Rank>
class PReLUActivationLayer : public ActivationLayer<Scalar,Rank> {
	typedef ActivationLayer<Scalar,Rank> Base;
public:
	inline PReLUActivationLayer(const Dimensions<int,Rank>& dims, Scalar init_alpha = 1e-1) :
			Base::ActivationLayer(dims),
			init_alpha(init_alpha) {
		Base::params.resize(1, dims.get_volume());
		Base::param_grads.resize(1, dims.get_volume());
	}
	inline Layer<Scalar,Rank>* clone() const {
		return new PReLUActivationLayer(*this);
	}
protected:
	inline void init() {
		Base::params.setConstant(init_alpha);
		Base::param_grads.setZero(1, Base::dims.get_volume());
	}
	inline Matrix<Scalar> activate(const Matrix<Scalar>& in) {
		return in.cwiseMax(in * Base::params.row(0).asDiagonal());
	}
	inline Matrix<Scalar> d_activate(const Matrix<Scalar>& in, const Matrix<Scalar>& out,
			const Matrix<Scalar>& out_grads) {
		Base::param_grads.row(0).setZero();
		Matrix<Scalar> d_in = Matrix<Scalar>(in.rows(), in.cols());
		for (int i = 0; i < in.cols(); ++i) {
			for (int j = 0; j < in.rows(); ++j) {
				Scalar in_ji = in(j,i);
				if (in_ji >= 0)
					d_in(j,i) = out_grads(j,i);
				else {
					Scalar out_ji = out_grads(j,i);
					d_in(j,i) = Base::params(0,i) * out_ji;
					Base::param_grads(0,i) += in_ji * out_ji;
				}
			}
		}
		return d_in;
	}
private:
	const Scalar init_alpha;
};

template<typename Scalar>
class PoolingLayer : public Layer<Scalar,3> {
	typedef Tensor<Scalar,4> Data;
	typedef std::array<int,4> RankwiseArray;
public:
	inline PoolingLayer(const Dimensions<int,3>& input_dims, size_t receptor_size, size_t stride) :
			input_dims(input_dims),
			output_dims({ calculate_output_dim(input_dims(0), receptor_size, stride),
					calculate_output_dim(input_dims(1), receptor_size, stride), input_dims(2) }),
			receptor_size(receptor_size),
			stride(stride),
			receptor_area(receptor_size * receptor_size),
			height_rem(input_dims(0) - receptor_size),
			width_rem(input_dims(1) - receptor_size),
			input_layer(input_layer),
			params(0, 0),
			param_grads(0, 0) {
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
	virtual void init_cache() = 0;
	virtual Scalar reduce(const RowVector<Scalar>& patch, unsigned patch_ind) = 0;
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
	inline Matrix<Scalar>& get_param_grads() {
		return param_grads;
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
			return Utils<Scalar>::template get_null_tensor<4>();
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
	static int calculate_output_dim(int input_dim, int receptor_size, int stride) {
		return (input_dim - receptor_size) / stride + 1;
	}
	const Dimensions<int,3> input_dims;
	const Dimensions<int,3> output_dims;
	const size_t receptor_size;
	const size_t stride;
	const int receptor_area;
	const int height_rem;
	const int width_rem;
	bool input_layer;
	// No actual parameters.
	Matrix<Scalar> params;
	Matrix<Scalar> param_grads;
	// Keep track of the input rows.
	int rows;
};

template<typename Scalar>
class SumPoolingLayer : public PoolingLayer<Scalar> {
public:
	inline SumPoolingLayer(const Dimensions<int,3>& input_dims, size_t receptor_size = 2, size_t stride = 2) :
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

template<typename Scalar>
class MeanPoolingLayer : public PoolingLayer<Scalar> {
public:
	inline MeanPoolingLayer(const Dimensions<int,3>& input_dims, size_t receptor_size = 2, size_t stride = 2) :
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

template<typename Scalar>
class MaxPoolingLayer : public PoolingLayer<Scalar> {
public:
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

template<typename Scalar, size_t Rank>
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
			param_grads(2 * depth, dims.get_volume() / depth),
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
		param_grads.setZero(params.rows(), params.cols());
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
	inline Matrix<Scalar>& get_param_grads() {
		return param_grads;
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
		 * gradients on the betas and gammas. */
		{ // Manage memory by scope restriction.
			Matrix<Scalar> out_grads_ch_map_i = Utils<Scalar>::template map_tensor_to_mat<Rank + 1>(std::move(out_grads));
			param_grads.row(2 * i) = out_grads_ch_map_i.cwiseProduct(cache.std_in).colwise().sum();
			param_grads.row(2 * i + 1) = out_grads_ch_map_i.colwise().sum();
			if (input_layer)
				return Utils<Scalar>::template get_null_tensor<Rank + 1>();
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
	Matrix<Scalar> param_grads;
	// Staged computation cache_vec
	struct Cache {
		RowVector<Scalar> inv_in_sd;
		Matrix<Scalar> std_in;
	};
	std::vector<Cache> cache_vec;
};

// Batch norm for all but multi-channel input tensors.
template<typename Scalar, size_t Rank>
class BatchNormLayer : public BatchNormLayerBase<Scalar,Rank> {
	typedef BatchNormLayerBase<Scalar,Rank> Base;
public:
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

// Partial template specialization for multi-channel input tensors.
template<typename Scalar>
class BatchNormLayer<Scalar,3> : public BatchNormLayerBase<Scalar,3> {
	typedef BatchNormLayerBase<Scalar,3> Base;
public:
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
			if (!Layer<Scalar,3>::is_input_layer())
				prev_out_grads = typename Base::Data(rows, Base::dims(0), Base::dims(1), Base::dims(2));
			Dimensions<int,3> slice_dims({ Base::dims(0), Base::dims(1), 1 });
			std::array<int,4> offsets({ 0, 0, 0, 0 });
			std::array<int,4> extents({ rows, slice_dims(0), slice_dims(1), slice_dims(2) });
			for (int i = 0; i < Base::depth; ++i) {
				offsets[3] = i;
				typename Base::Data out_grads_slice_i = out_grads.slice(offsets, extents);
				if (Layer<Scalar,3>::is_input_layer())
					Base::_pass_back(std::move(out_grads_slice_i), slice_dims, i);
				else
					prev_out_grads.slice(offsets, extents) = Base::_pass_back(std::move(out_grads_slice_i), slice_dims, i);
			}
			return prev_out_grads;
		}
	}
};

template<typename Scalar, size_t Rank>
class DropoutLayer : public Layer<Scalar,Rank> {
	typedef Tensor<Scalar,Rank + 1> Data;
public:
	inline DropoutLayer(const Dimensions<int,Rank>& dims, Scalar dropout_prob, Scalar epsilon = Utils<Scalar>::EPSILON3) :
			dims(dims),
			dropout_prob(dropout_prob),
			epsilon(epsilon),
			dropout(Utils<Scalar>::decidedly_greater(dropout_prob, .0)),
			input_layer(false),
			params(0, 0),
			param_grads(0, 0) {
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
	inline Matrix<Scalar>& get_param_grads() {
		return param_grads;
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
			return Utils<Scalar>::template get_null_tensor<Rank + 1>();
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
	Matrix<Scalar> param_grads;
	// Staged computation cache_vec
	Matrix<Scalar> dropout_mask;
};

} /* namespace cattle */

#endif /* LAYER_H_ */
