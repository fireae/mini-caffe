#ifndef REVERSE_LAYER_HPP
#define REVERSE_LAYER_HPP

#include <vector>

#include "./neuron_layer.hpp"

namespace caffe {

/*
 * @brief Reverses the data of the input Blob into the output blob.
 *
 * Note: This is a useful layer if you want to reverse the time of
 * a recurrent layer.
 */

class ReverseLayer : public NeuronLayer {
 public:
  explicit ReverseLayer(const LayerParameter& param);

  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Reverse"; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
      
  int axis_;
};

}  // namespace caffe

#endif  // REVERSE_LAYER_HPP
