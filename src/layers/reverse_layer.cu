#include <vector>

#include "./reverse_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {


void ReverseLayer::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const real_t* src = bottom[0]->gpu_data();

  const int count = top[0]->count();
  const int axis_count = top[0]->count(axis_);
  const int copy_amount
        = (axis_ + 1 == top[0]->num_axes()) ? 1 : top[0]->count(axis_ + 1);
  const int num_fix = (axis_ > 0) ? count / axis_count : 1;
  const int sub_iter_max = top[0]->shape(axis_);

  for (int fix = 0; fix < num_fix; ++fix) {
    real_t* target = top[0]->mutable_gpu_data()
            + (fix + 1) * copy_amount * sub_iter_max - copy_amount;
    for (int i = 0; i < sub_iter_max; ++i) {
      caffe_copy(copy_amount, src, target);
      src += copy_amount;     // normal order
      target -= copy_amount;  // reverse order
    }
  }
}



}  // namespace caffe
