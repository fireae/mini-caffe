#include <vector>
#include <algorithm>
#include <cmath>

#include "../util/math_functions.hpp"
#include "./lstm_layer_Junhyuk.hpp"

namespace caffe {


__device__ real_t sigmoid(const real_t x) {
  return real_t(1) / (real_t(1) + exp(-x));
}


__device__ real_t tanh(const real_t x) {
  return real_t(2) * sigmoid(real_t(2) * x) - real_t(1);
}


__global__ void ClipAdd(const int nthreads, const int dim, int t,
    const real_t* clip, const real_t* add_vec, real_t* data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const real_t clip_t = clip ? clip[n] : real_t(t > 0);
    data[index] += clip_t * add_vec[index];
  }
}


__global__ void ActivationForward(const int nthreads, const int H,
                                const real_t* pre_gate, real_t* gate) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % (4*H);
    gate[index] = d < 3*H ? sigmoid(pre_gate[index]) : tanh(pre_gate[index]);
  }
}


__global__ void LSTMForward(const int nthreads, const int H, const int t,
    const real_t* c_prev, const real_t* gate, const real_t* clip,
    real_t* c_t, real_t* h_t) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / H;
    const int d = index % H;
    const real_t* offset = gate + 4*H*n;
    const real_t i_t = offset[d];
    const real_t f_t = offset[H + d];
    const real_t o_t = offset[2*H + d];
    const real_t g_t = offset[3*H + d];
    const real_t c_t_1 = c_prev[index];
    const real_t clip_t = clip ? clip[n] : real_t(t > 0);
    c_t[index] = clip_t * f_t * c_t_1 + i_t * g_t;
    h_t[index] = o_t * tanh(c_t[index]);
  }
}


__global__ void LSTMBackward(const int nthreads, const int H, const int t, 
    const real_t* c_prev, const real_t* gate, const real_t* c_t, 
    const real_t* clip, real_t* dc_t, const real_t* dh_t, 
    real_t* dc_prev, real_t* gate_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / H;
    const int d = index % H;
    const real_t* gate_t = gate + 4*H*n;
    const real_t i_t = gate_t[d];
    const real_t f_t = gate_t[H + d];
    const real_t o_t = gate_t[2*H + d];
    const real_t g_t = gate_t[3*H + d];
    const real_t c_t_1 = c_prev[index];
    const real_t c = c_t[index];
    const real_t tanh_c = tanh(c);
    const real_t clip_t = clip ? clip[n] : real_t(t > 0);
    real_t* dc_t_1 = dc_prev + index;
    real_t* gate_diff_t = gate_diff + 4*H*n;
    real_t* di_t = gate_diff_t + d;
    real_t* df_t = gate_diff_t + H + d;
    real_t* do_t = gate_diff_t + 2*H + d;
    real_t* dg_t = gate_diff_t + 3*H + d;
    
    // Output gate : tanh(c(t)) * h_diff(t)
    *do_t = dh_t[index] * tanh_c;
    // Cell state : o(t) * tanh'(c(t)) * h_diff(t) + f(t+1) * c_diff(t+1)
    dc_t[index] += dh_t[index] * o_t * (real_t(1) - tanh_c * tanh_c);
    // c_diff(t-1) += f(t) * c_diff(t)
    *dc_t_1 = clip_t * dc_t[index] * f_t;
    // Forget gate : c(t-1) * c_diff(t)
    *df_t = clip_t * dc_t[index] * c_t_1;
    // Input gate : g(t) * c_diff(t)
    *di_t = dc_t[index] * g_t;
    // Input modulation gate : i(t) * c_diff(t)
    *dg_t = dc_t[index] * i_t;
  }
}


__global__ void ActivationBackward(const int nthreads, const int H, 
    const real_t clip_threshold, const real_t* gate, const real_t* gate_diff, 
    real_t* pre_gate_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % (4 * H);
    const real_t gate_val = gate[index];
    if (d < 3 * H) {
      pre_gate_diff[index] = gate_diff[index] * gate_val * (real_t(1) - gate_val);
    } else {
      pre_gate_diff[index] = gate_diff[index] * (real_t(1) - gate_val * gate_val);
    }
    if (clip_threshold > real_t(0)) {
      if (pre_gate_diff[index] < -clip_threshold) {
        pre_gate_diff[index] = -clip_threshold;
      }
      else if (pre_gate_diff[index] > clip_threshold) {
        pre_gate_diff[index] = clip_threshold;
      }
    }
  }
}


void LstmLayer::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  CHECK_EQ(top[0]->gpu_data(), top_.gpu_data());
  real_t* top_data = top_.mutable_gpu_data();
  const real_t* bottom_data = bottom[0]->gpu_data();
  const real_t* clip = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->gpu_data();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  const real_t* weight_i = this->blobs_[0]->gpu_data();
  const real_t* weight_h = this->blobs_[1]->gpu_data();
  const real_t* bias = this->blobs_[2]->gpu_data();
  real_t* pre_gate_data = pre_gate_.mutable_gpu_data();
  real_t* gate_data = gate_.mutable_gpu_data();
  real_t* cell_data = cell_.mutable_gpu_data();

  // Initialize previous state
  if (clip) {
    caffe_copy(c_0_.count(), c_T_.gpu_data(), c_0_.mutable_gpu_data());
    caffe_copy(h_0_.count(), h_T_.gpu_data(), h_0_.mutable_gpu_data());
  }
  else {
    caffe_gpu_set(c_0_.count(), real_t(0.), c_0_.mutable_gpu_data());
    caffe_gpu_set(h_0_.count(), real_t(0.), h_0_.mutable_gpu_data());
  }

  // Compute input to hidden forward propagation
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 4*H_, I_, real_t(1.),
      bottom_data, weight_i, real_t(0.), pre_gate_data);
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 4*H_, 1, real_t(1.),
      bias_multiplier_.gpu_data(), bias, real_t(1.), pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    real_t* h_t = top_data + top_.offset(t);
    real_t* c_t = cell_data + cell_.offset(t);
    real_t* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
    real_t* gate_t = gate_data + gate_.offset(t);
    const real_t* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
    const real_t* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.gpu_data();
    const real_t* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.gpu_data();

    caffe_gpu_gemm(CblasNoTrans, CblasTrans, N_, 4*H_, H_, real_t(1.), 
        h_t_1, weight_h, real_t(0.), h_to_gate_.mutable_gpu_data());
    ClipAdd<<<CAFFE_GET_BLOCKS(4*N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        4*N_*H_, 4*H_, t, clip_t, h_to_gate_.gpu_data(), pre_gate_t);
    CUDA_POST_KERNEL_CHECK;
    ActivationForward<<<CAFFE_GET_BLOCKS(4*N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        4*N_*H_, H_, pre_gate_t, gate_t);
    CUDA_POST_KERNEL_CHECK;
    LSTMForward<<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, H_, t, c_t_1, gate_t, clip_t, c_t, h_t);
    CUDA_POST_KERNEL_CHECK;
  }

  // Preserve cell state and output value for truncated BPTT
  caffe_copy(N_*H_, cell_data + cell_.offset(T_-1), c_T_.mutable_gpu_data());
  caffe_copy(N_*H_, top_data + top_.offset(T_-1), h_T_.mutable_gpu_data());
}


}  // namespace caffe
