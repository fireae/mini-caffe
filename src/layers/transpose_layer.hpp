#ifndef CAFFE_TRANSPOSE_LAYER_HPP_
#define CAFFE_TRANSPOSE_LAYER_HPP_

#include <vector>

#include "../layer.hpp"

namespace caffe {

class TransposeLayer : public Layer{
 public:
  explicit TransposeLayer(const LayerParameter& param)
      : Layer(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Transpose"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
 
 private:
  TransposeParameter transpose_param_;
  vector<int> permute(const vector<int>& vec);
  BlobInt bottom_counts_;
  BlobInt top_counts_;
  BlobInt forward_map_;
  BlobInt backward_map_;
  BlobInt buf_;
};

}  // namespace caffe

#endif  // CAFFE_TRANSPOSE_LAYER_HPP_
