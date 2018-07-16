#ifndef REVERSE_TIME_LAYER_HPP
#define REVERSE_TIME_LAYER_HPP

#include <vector>
#include "../layer.hpp"

namespace caffe {

/*
 * @brief Reverses the data of the input Blob into the output blob.
 *
 * Note: This is a useful layer if you want to reverse the time of
 * a recurrent layer.
 */

class ReverseTimeLayer : public Layer {
 public:
  explicit ReverseTimeLayer(const LayerParameter& param);

  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "ReverseTime"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  bool copy_remaining_;
};

}  // namespace caffe

#endif  // REVERSE_TIME_LAYER_HPP
