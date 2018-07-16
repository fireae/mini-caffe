
#include <vector>
#include "./reverse_layer.hpp"

#include "../util/math_functions.hpp"
namespace caffe {


ReverseLayer::ReverseLayer(const LayerParameter& param)
  : NeuronLayer(param)
  , axis_(param.reverse_param().axis()) {
  CHECK_GE(axis_, 0);
}


void ReverseLayer::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  NeuronLayer::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";

  CHECK_LT(axis_, bottom[0]->num_axes())
        << "Axis must be less than the number of axis for reversing";
}


void ReverseLayer::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const real_t* src = bottom[0]->cpu_data();

  const int count = top[0]->count();
  const int axis_count = top[0]->count(axis_);
  const int copy_amount
        = (axis_ + 1 == top[0]->num_axes()) ? 1 : top[0]->count(axis_ + 1);
  const int num_fix = (axis_ > 0) ? count / axis_count : 1;
  const int sub_iter_max = top[0]->shape(axis_);

  for (int fix = 0; fix < num_fix; ++fix) {
    real_t* target = top[0]->mutable_cpu_data()
                    + (fix + 1) * copy_amount * sub_iter_max - copy_amount;
    for (int i = 0; i < sub_iter_max; ++i) {
      caffe_copy(copy_amount, src, target);
      src += copy_amount;     // normal order
      target -= copy_amount;
    }
  }
}

#ifndef USE_CUDA
STUB_GPU(ReverseLayer);
#endif

REGISTER_LAYER_CLASS(Reverse);

}  // namespace caffe
