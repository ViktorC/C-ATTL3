/*
 * ActivationLayer.hpp
 *
 *  Created on: 22 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_LAYER_ACTIVATIONLAYER_H_
#define C_ATTL3_LAYER_ACTIVATIONLAYER_H_

#include <memory>

#include "core/Layer.hpp"

namespace cattle {

/**
 * An alias for a shared pointer to a Parameters instance.
 */
template <typename Scalar>
using ParamsSharedPtr = std::shared_ptr<Parameters<Scalar>>;

/**
 * An abstract class template that represents an activation function layer.
 */
template <typename Scalar, std::size_t Rank>
class ActivationLayer : public Layer<Scalar, Rank> {
  typedef Layer<Scalar, Rank> Base;
  typedef ActivationLayer<Scalar, Rank> Self;

 public:
  virtual ~ActivationLayer() = default;
  virtual Base* clone() const = 0;
  inline Base* clone_with_shared_params() { return clone(); }
  inline const Base& get_params_owner() const { return owner; }
  inline const typename Base::Dims& get_input_dims() const { return dims; }
  inline const typename Base::Dims& get_output_dims() const { return dims; }
  inline bool is_input_layer() const { return input_layer; }
  inline void set_input_layer(bool input_layer) { this->input_layer = input_layer; }
  inline std::vector<const Parameters<Scalar>*> get_params() const {
    return params ? std::vector<const Parameters<Scalar>*>({params.get()}) : std::vector<const Parameters<Scalar>*>(0);
  }
  inline std::vector<Parameters<Scalar>*> get_params() {
    return params ? std::vector<Parameters<Scalar>*>({params.get()}) : std::vector<Parameters<Scalar>*>(0);
  }

 protected:
  /**
   * @param dims The input and output dimensions of the layer.
   * @param params The parameters of the layer; it can be null if the layer is
   * not parametric.
   */
  inline ActivationLayer(const typename Base::Dims& dims, ParamsSharedPtr<Scalar> params = nullptr)
      : owner(*this), dims(dims), params(params), input_layer(false) {}
  inline ActivationLayer(const Self& layer, bool share_params = false)
      : owner(share_params && layer.params ? layer.owner : *this),
        dims(layer.dims),
        params(share_params ? layer.params
                            : (!layer.params ? nullptr : ParamsSharedPtr<Scalar>(layer.params->clone()))),
        input_layer(layer.input_layer) {}
  const Self& owner;
  const typename Base::Dims dims;
  ParamsSharedPtr<Scalar> params;
  bool input_layer;
};

} /* namespace cattle */

#endif /* C_ATTL3_LAYER_ACTIVATIONLAYER_H_ */
