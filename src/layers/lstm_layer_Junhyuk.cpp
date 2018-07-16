#include <vector>

#include "../util/math_functions.hpp"
#include "./lstm_layer_Junhyuk.hpp"
#include "../filler.hpp"

namespace caffe {


inline real_t sigmoid(real_t x) {
  return 1. / (1. + exp(-x));
}


void LstmLayer::LayerSetUp(const vector<Blob*>& bottom,//bottom[0]: [T]x[N]x[Channels]
      const vector<Blob*>& top) {
  clipping_threshold_ = this->layer_param_.lstm_param().clipping_threshold();
  N_ = bottom[0]->shape(1);// this->layer_param_.lstm_param().batch_size(); // batch_size
  H_ = this->layer_param_.lstm_param().num_output(); // number of hidden units
  I_ = bottom[0]->shape(2);// bottom[0]->count() / bottom[0]->num(); // input dimension

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    shared_ptr<Filler > weight_filler(GetFiller(
        this->layer_param_.lstm_param().weight_filler()));
 
    // input-to-hidden weights
    // Intialize the weight
    vector<int> weight_shape;
    weight_shape.push_back(4*H_);
    weight_shape.push_back(I_);
    this->blobs_[0].reset(new Blob(weight_shape));
    weight_filler->Fill(this->blobs_[0].get());

    // hidden-to-hidden weights
    // Intialize the weight
    weight_shape.clear();
    weight_shape.push_back(4*H_);
    weight_shape.push_back(H_);
    this->blobs_[1].reset(new Blob(weight_shape));
    weight_filler->Fill(this->blobs_[1].get());

    // If necessary, intiialize and fill the bias term
    vector<int> bias_shape(1, 4*H_);
    this->blobs_[2].reset(new Blob(bias_shape));
    shared_ptr<Filler > bias_filler(GetFiller(
        this->layer_param_.lstm_param().bias_filler()));
    bias_filler->Fill(this->blobs_[2].get());
  }  // parameter initialization
  vector<int> cell_shape;
  cell_shape.push_back(N_);
  cell_shape.push_back(H_);
  c_0_.Reshape(cell_shape);
  h_0_.Reshape(cell_shape);
  c_T_.Reshape(cell_shape);
  h_T_.Reshape(cell_shape);
  h_to_h_.Reshape(cell_shape);

  vector<int> gate_shape;
  gate_shape.push_back(N_);
  gate_shape.push_back(4);
  gate_shape.push_back(H_);
  h_to_gate_.Reshape(gate_shape);
}


void LstmLayer::Reshape(const vector<Blob*>& bottom,//bottom[0]: [T]x[N]x[Channels]
      const vector<Blob*>& top) {//top[0] [T*N]x[H]
  // Figure out the dimensions
	T_ = bottom[0]->shape(0);// bottom[0]->num() / N_; // length of sequence
	N_ = bottom[0]->shape(1);
//   CHECK_EQ(bottom[0]->num() % N_, 0) << "Input size "
//     "should be multiple of batch size";
//   CHECK_EQ(bottom[0]->count() / T_ / N_, I_) << "Input size "
//     "incompatible with inner product parameters.";
  vector<int> original_top_shape;
  original_top_shape.push_back(T_);
  original_top_shape.push_back(N_);
  original_top_shape.push_back(H_);
  top[0]->Reshape(original_top_shape);

  // Gate initialization
  vector<int> gate_shape;
  gate_shape.push_back(T_);
  gate_shape.push_back(N_);
  gate_shape.push_back(4);
  gate_shape.push_back(H_);
  pre_gate_.Reshape(gate_shape);
  gate_.Reshape(gate_shape);
  
  vector<int> top_shape;
  top_shape.push_back(T_);
  top_shape.push_back(N_);
  top_shape.push_back(H_);
  cell_.Reshape(top_shape);
  top_.Reshape(top_shape);
  top_.ShareData(*top[0]);

  vector<int> cell_shape;
  cell_shape.push_back(N_);
  cell_shape.push_back(H_);
  c_0_.Reshape(cell_shape);
  h_0_.Reshape(cell_shape);
  c_T_.Reshape(cell_shape);
  h_T_.Reshape(cell_shape);
  h_to_h_.Reshape(cell_shape);

  vector<int> gate_shape1;
  gate_shape1.push_back(N_);
  gate_shape1.push_back(4);
  gate_shape1.push_back(H_);
  h_to_gate_.Reshape(gate_shape1);
  
  // Set up the bias multiplier
  vector<int> multiplier_shape(1, N_*T_);
  bias_multiplier_.Reshape(multiplier_shape);
  caffe_set(bias_multiplier_.count(), real_t(1), 
    bias_multiplier_.mutable_cpu_data());
}


void LstmLayer::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  CHECK_EQ(top[0]->cpu_data(), top_.cpu_data());
  real_t* top_data = top_.mutable_cpu_data();
  const real_t* bottom_data = bottom[0]->cpu_data();
  const real_t* clip = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->cpu_data();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  const real_t* weight_i = this->blobs_[0]->cpu_data();
  const real_t* weight_h = this->blobs_[1]->cpu_data();
  const real_t* bias = this->blobs_[2]->cpu_data();
  real_t* pre_gate_data = pre_gate_.mutable_cpu_data();
  real_t* gate_data = gate_.mutable_cpu_data();
  real_t* cell_data = cell_.mutable_cpu_data();
  real_t* h_to_gate = h_to_gate_.mutable_cpu_data();

  // Initialize previous state
  if (clip) {
    caffe_copy(c_0_.count(), c_T_.cpu_data(), c_0_.mutable_cpu_data());
    caffe_copy(h_0_.count(), h_T_.cpu_data(), h_0_.mutable_cpu_data());
  }
  else {
    caffe_set(c_0_.count(), real_t(0.), c_0_.mutable_cpu_data());
    caffe_set(h_0_.count(), real_t(0.), h_0_.mutable_cpu_data());
  }

  // Compute input to hidden forward propagation
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 4*H_, I_, real_t(1.),
      bottom_data, weight_i, real_t(0.), pre_gate_data);
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 4*H_, 1, real_t(1.),
      bias_multiplier_.cpu_data(), bias, real_t(1.), pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    real_t* h_t = top_data + top_.offset(t);//[T]x[N]x[H]
    real_t* c_t = cell_data + cell_.offset(t);//[T]x[N]x[H]
    real_t* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
    real_t* gate_t = gate_data + gate_.offset(t);
    real_t* h_to_gate_t = h_to_gate;
    const real_t* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
    const real_t* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.cpu_data();
    const real_t* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.cpu_data();

    // Hidden-to-hidden propagation
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_, 4*H_, H_, real_t(1.), 
        h_t_1, weight_h, real_t(0.), h_to_gate);

    for (int n = 0; n < N_; ++n) {
      const bool cont = clip_t ? clip_t[n] : t > 0;
      if (cont) {
        caffe_add(4*H_, pre_gate_t, h_to_gate, pre_gate_t);
      }
      for (int d = 0; d < H_; ++d) {
        // Apply nonlinearity
        gate_t[d] = sigmoid(pre_gate_t[d]);
        gate_t[H_ + d] = cont ? sigmoid(pre_gate_t[H_ + d]) : real_t(0.);
        gate_t[2*H_ + d] = sigmoid(pre_gate_t[2*H_ + d]);
        gate_t[3*H_ + d] = tanh(pre_gate_t[3*H_ + d]);

        // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
        c_t[d] = gate_t[H_ + d] * c_t_1[d] + gate_t[d] * gate_t[3*H_ + d];
        h_t[d] = gate_t[2*H_ + d] * tanh(c_t[d]);
      }
      
      h_t += H_;
      c_t += H_;
      c_t_1 += H_;
      pre_gate_t += 4*H_;
      gate_t += 4*H_;
      h_to_gate_t += 4*H_;
    }
  }
  // Preserve cell state and output value for truncated BPTT
  caffe_copy(N_*H_, cell_data + cell_.offset(T_-1), c_T_.mutable_cpu_data());
  caffe_copy(N_*H_, top_data + top_.offset(T_-1), h_T_.mutable_cpu_data());
}



#ifndef USE_CUDA
STUB_GPU(LstmLayer);
#endif

REGISTER_LAYER_CLASS(Lstm);

}  // namespace caffe
