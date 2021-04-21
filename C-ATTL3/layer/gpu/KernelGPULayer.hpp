/*
 * KernelGPULayer.hpp
 *
 *  Created on: 18 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_KERNELGPULAYER_H_
#define C_ATTL3_LAYER_GPU_KERNELGPULAYER_H_

#include "core/gpu/GPULayer.hpp"
#include "layer/KernelLayer.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar, std::size_t Rank>
class KernelGPULayer : public GPULayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef GPULayer<Scalar, Rank> Base;
  typedef KernelGPULayer<Scalar, Rank> Self;

 public:
  inline virtual ~KernelGPULayer() = default;
  inline const Base& get_params_owner() const { return owner; }
  inline const typename Root::Dims& get_input_dims() const { return input_dims; }
  inline const typename Root::Dims& get_output_dims() const { return output_dims; }
  inline bool is_input_layer() const { return input_layer; }
  inline void set_input_layer(bool input_layer) { this->input_layer = input_layer; }
  inline std::vector<const Parameters<Scalar>*> get_params() const {
    return std::vector<const Parameters<Scalar>*>({weights.get(), bias.get()});
  }
  inline std::vector<Parameters<Scalar>*> get_params() {
    return std::vector<Parameters<Scalar>*>({weights.get(), bias.get()});
  }
  inline const typename Base::GPUDims& get_gpu_input_dims() const { return gpu_input_dims; }
  inline const typename Base::GPUDims& get_gpu_output_dims() const { return gpu_output_dims; }
  inline std::vector<const GPUParameters<Scalar>*> get_gpu_params() const {
    return std::vector<const GPUParameters<Scalar>*>({weights.get(), bias.get()});
  }
  inline std::vector<GPUParameters<Scalar>*> get_gpu_params() {
    return std::vector<GPUParameters<Scalar>*>({weights.get(), bias.get()});
  }

 protected:
  inline KernelGPULayer(const typename Root::Dims& input_dims, const typename Root::Dims& output_dims,
                        const typename Base::GPUDims& gpu_input_dims, const typename Base::GPUDims& gpu_output_dims,
                        GPUParamsSharedPtr<Scalar> weights, GPUParamsSharedPtr<Scalar> bias)
      : owner(*this),
        input_dims(input_dims),
        output_dims(output_dims),
        gpu_input_dims(gpu_input_dims),
        gpu_output_dims(gpu_output_dims),
        weights(weights),
        bias(bias),
        input_layer(false) {
    assert(weights && bias);
  }
  inline KernelGPULayer(const Self& layer, bool share_params = false)
      : owner(share_params ? layer.owner : *this),
        input_dims(layer.input_dims),
        output_dims(layer.output_dims),
        gpu_input_dims(layer.gpu_input_dims),
        gpu_output_dims(layer.gpu_output_dims),
        weights(share_params ? layer.weights : GPUParamsSharedPtr<Scalar>(layer.weights->gpu_clone())),
        bias(share_params ? layer.bias : GPUParamsSharedPtr<Scalar>(layer.bias->gpu_clone())),
        input_layer(layer.input_layer) {}
  const Self& owner;
  const typename Root::Dims input_dims, output_dims;
  const typename Base::GPUDims gpu_input_dims, gpu_output_dims;
  GPUParamsSharedPtr<Scalar> weights, bias;
  bool input_layer;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_KERNELGPULAYER_H_ */
