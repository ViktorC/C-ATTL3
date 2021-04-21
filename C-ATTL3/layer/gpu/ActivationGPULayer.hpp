/*
 * ActivationGPULayer.hpp
 *
 *  Created on: 18 Aug 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_GPU_ACTIVATIONGPULAYER_H_
#define C_ATTL3_LAYER_GPU_ACTIVATIONGPULAYER_H_

#include "core/gpu/GPULayer.hpp"
#include "layer/ActivationLayer.hpp"

namespace cattle {
namespace gpu {

template <typename Scalar, std::size_t Rank>
class ActivationGPULayer : public virtual GPULayer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Root;
  typedef GPULayer<Scalar, Rank> Base;
  typedef ActivationGPULayer<Scalar, Rank> Self;

 public:
  inline virtual ~ActivationGPULayer() = default;
  inline const Root &get_params_owner() const { return owner; }
  inline const typename Root::Dims &get_input_dims() const { return dims; }
  inline const typename Root::Dims &get_output_dims() const { return dims; }
  inline bool is_input_layer() const { return input_layer; }
  inline void set_input_layer(bool input_layer) { this->input_layer = input_layer; }
  inline std::vector<const Parameters<Scalar> *> get_params() const {
    return params ? std::vector<const Parameters<Scalar> *>({params.get()})
                  : std::vector<const Parameters<Scalar> *>(0);
  }
  inline std::vector<Parameters<Scalar> *> get_params() {
    return params ? std::vector<Parameters<Scalar> *>({params.get()}) : std::vector<Parameters<Scalar> *>(0);
  }
  inline virtual Base *gpu_clone_with_shared_params() { return Base::gpu_clone(); }
  inline const typename Base::GPUDims &get_gpu_input_dims() const { return gpu_dims; }
  inline const typename Base::GPUDims &get_gpu_output_dims() const { return gpu_dims; }
  inline std::vector<const GPUParameters<Scalar> *> get_gpu_params() const {
    return params ? std::vector<const GPUParameters<Scalar> *>({params.get()})
                  : std::vector<const GPUParameters<Scalar> *>(0);
  }
  inline std::vector<GPUParameters<Scalar> *> get_gpu_params() {
    return params ? std::vector<GPUParameters<Scalar> *>({params.get()}) : std::vector<GPUParameters<Scalar> *>(0);
  }

 protected:
  inline ActivationGPULayer(const typename Root::Dims &dims, GPUParamsSharedPtr<Scalar> params = nullptr)
      : owner(*this), dims(dims), gpu_dims(dims.template extend<3 - Rank>()), params(params), input_layer(false) {}
  inline ActivationGPULayer(const Self &layer, bool share_params = false)
      : owner(share_params && layer.params ? layer.owner : *this),
        dims(layer.dims),
        gpu_dims(layer.gpu_dims),
        params(share_params ? layer.params
                            : (!layer.params ? nullptr : GPUParamsSharedPtr<Scalar>(layer.params->gpu_clone()))),
        input_layer(layer.input_layer) {}

 private:
  const Self &owner;
  const typename Base::Dims dims;
  const typename Base::GPUDims gpu_dims;
  GPUParamsSharedPtr<Scalar> params;
  bool input_layer;
};

} /* namespace gpu */
} /* namespace cattle */

#endif /* C_ATTL3_LAYER_GPU_ACTIVATIONGPULAYER_H_ */
