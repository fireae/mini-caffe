#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>


#include "../filler.hpp"
#include "./DenseBlock_layer.hpp"

namespace caffe {

void DenseBlockLayer::LayerSetUp(const vector<Blob *> &bottom,
                                 const vector<Blob *> &top) {
  this->cpuInited = false;
  this->N = bottom[0]->shape()[0];
  this->H = bottom[0]->shape()[2];
  this->W = bottom[0]->shape()[3];

  DenseBlockParameter dbParam = this->layer_param_.denseblock_param();
  this->numTransition = dbParam.numtransition();
  // this->initChannel = dbParam.initchannel();
  this->initChannel = bottom[0]->channels(); // modified by jxs
  this->growthRate = dbParam.growthrate();
  this->trainCycleIdx = 0; // initially, trainCycleIdx = 0
  this->EMA_decay = dbParam.moving_average_fraction();
#ifdef USE_CUDA
  this->workspace_size_bytes = dbParam.workspace_mb() * 1024 * 1024;
  this->gpu_idx_ = dbParam.gpuidx();
#endif
  this->useDropout = dbParam.use_dropout();
  this->dropoutAmount = dbParam.dropout_amount();
  this->DB_randomSeed = 124816;
  this->useBC = dbParam.use_bc();
  this->BC_ultra_spaceEfficient = dbParam.bc_ultra_space_efficient();
  // Parameter Blobs
  // for transition i,
  // blobs_[i] is its filter blob
  // blobs_[numTransition + i] is its scaler blob
  // blobs_[2*numTransition + i] is its bias blob
  // blobs_[3*numTransition + i] is its globalMean
  // blobs_[4*numTransition + i] is its globalVar
  if (useBC) {
    this->blobs_.resize(10 * this->numTransition + 1);
  } else {
    this->blobs_.resize(5 * this->numTransition + 1);
  }
  for (int transitionIdx = 0; transitionIdx < this->numTransition;
       ++transitionIdx) {
    // filter
    // No BC case
    if (!useBC) {
      int inChannels = initChannel + transitionIdx * growthRate;
      int filterShape_Arr[] = {growthRate, inChannels, 3, 3};
      vector<int> filterShape(filterShape_Arr, filterShape_Arr + 4);
      this->blobs_[transitionIdx].reset(new Blob(filterShape));
      shared_ptr<Filler> filter_Filler(GetFiller(dbParam.filter_filler()));
      filter_Filler->Fill(this->blobs_[transitionIdx].get());
    } else {
      // 3*3 kernel
      int filter_33_shapeArr[] = {growthRate, 4 * growthRate, 3, 3};
      vector<int> filter33Shape(filter_33_shapeArr, filter_33_shapeArr + 4);
      this->blobs_[transitionIdx].reset(new Blob(filter33Shape));
      shared_ptr<Filler> filter_Filler3(GetFiller(dbParam.filter_filler()));
      filter_Filler3->Fill(this->blobs_[transitionIdx].get());

      // 1*1 kernel
      int inChannels = initChannel + transitionIdx * growthRate;
      int filter_11_shapeArr[] = {4 * growthRate, inChannels, 1, 1};
      vector<int> filter11Shape(filter_11_shapeArr, filter_11_shapeArr + 4);
      this->blobs_[5 * numTransition + transitionIdx].reset(
          new Blob(filter11Shape));
      shared_ptr<Filler> filter_Filler1(GetFiller(dbParam.filter_filler()));
      filter_Filler1->Fill(
          this->blobs_[5 * numTransition + transitionIdx].get());
    }
    // scaler & bias
    int inChannels = initChannel + transitionIdx * growthRate;
    int BNparamShape_Arr[] = {1, inChannels, 1, 1};
    vector<int> BNparamShape(BNparamShape_Arr, BNparamShape_Arr + 4);
    // scaler
    this->blobs_[numTransition + transitionIdx].reset(new Blob(BNparamShape));
    shared_ptr<Filler> weight_filler0(GetFiller(dbParam.bn_scaler_filler()));
    weight_filler0->Fill(this->blobs_[numTransition + transitionIdx].get());

    int BN_4G_Shape[] = {1, 4 * growthRate, 1, 1};
    vector<int> BN_4Gparam_ShapeVec(BN_4G_Shape, BN_4G_Shape + 4);
    // scaler BC
    if (useBC) {
      this->blobs_[6 * numTransition + transitionIdx].reset(
          new Blob(BN_4Gparam_ShapeVec));
      shared_ptr<Filler> weight_filler0_4G(
          GetFiller(dbParam.bn_scaler_filler()));
      weight_filler0_4G->Fill(
          this->blobs_[6 * numTransition + transitionIdx].get());
    }
    // bias
    this->blobs_[2 * numTransition + transitionIdx].reset(
        new Blob(BNparamShape));
    shared_ptr<Filler> weight_filler1(GetFiller(dbParam.bn_bias_filler()));
    weight_filler1->Fill(this->blobs_[2 * numTransition + transitionIdx].get());
    // bias BC
    if (useBC) {
      this->blobs_[7 * numTransition + transitionIdx].reset(
          new Blob(BN_4Gparam_ShapeVec));
      shared_ptr<Filler> weight_filler1_4G(GetFiller(dbParam.bn_bias_filler()));
      weight_filler1_4G->Fill(
          this->blobs_[7 * numTransition + transitionIdx].get());
    }
    // globalMean
    this->blobs_[3 * numTransition + transitionIdx].reset(
        new Blob(BNparamShape));
    for (int blobIdx = 0; blobIdx < inChannels; ++blobIdx) {
      shared_ptr<Blob> localB = this->blobs_[3 * numTransition + transitionIdx];
      localB->mutable_cpu_data()[localB->offset(0, blobIdx, 0, 0)] = 0;
    }
    // globalMean BC
    if (useBC) {
      this->blobs_[8 * numTransition + transitionIdx].reset(
          new Blob(BN_4Gparam_ShapeVec));
      shared_ptr<Blob> localB = this->blobs_[8 * numTransition + transitionIdx];
      for (int blobIdx = 0; blobIdx < 4 * growthRate; ++blobIdx) {
        localB->mutable_cpu_data()[localB->offset(0, blobIdx, 0, 0)] = 0;
      }
    }
    // globalVar
    this->blobs_[4 * numTransition + transitionIdx].reset(
        new Blob(BNparamShape));
    for (int blobIdx = 0; blobIdx < inChannels; ++blobIdx) {
      shared_ptr<Blob> localB = this->blobs_[4 * numTransition + transitionIdx];
      localB->mutable_cpu_data()[localB->offset(0, blobIdx, 0, 0)] = 1;
    }
    // globalVar BC
    if (useBC) {
      this->blobs_[9 * numTransition + transitionIdx].reset(
          new Blob(BN_4Gparam_ShapeVec));
      shared_ptr<Blob> localB = this->blobs_[9 * numTransition + transitionIdx];
      for (int blobIdx = 0; blobIdx < 4 * growthRate; ++blobIdx) {
        localB->mutable_cpu_data()[localB->offset(0, blobIdx, 0, 0)] = 1;
      }
    }
  }
  // final parameter for the equivalent of blobs_[2] in Caffe-BN
  vector<int> singletonShapeVec;
  singletonShapeVec.push_back(1);
  int singletonIdx = useBC ? 10 * numTransition : 5 * numTransition;
  this->blobs_[singletonIdx].reset(new Blob(singletonShapeVec));
  this->blobs_[singletonIdx]->mutable_cpu_data()[0] = real_t(0);
  // parameter specification: globalMean/Var weight decay and lr is 0
  if (!useBC) {
    for (int i = 0; i < this->blobs_.size(); ++i) {
      if (this->layer_param_.param_size() != i) {
        CHECK_EQ(0, 1) << "Nope";
      }
      ParamSpec *fixed_param_spec = this->layer_param_.add_param();
      // global Mean/Var
      if (i >= 3 * this->numTransition) {
        fixed_param_spec->set_lr_mult(0.f);
        fixed_param_spec->set_decay_mult(0.f);
      }
      // BN Scaler and Bias
      else if (i >= this->numTransition) {
        fixed_param_spec->set_lr_mult(1.f);
        fixed_param_spec->set_decay_mult(1.f);
      } else {
        fixed_param_spec->set_lr_mult(1.f);
        fixed_param_spec->set_decay_mult(1.f);
      }
    }
  } else {
    for (int i = 0; i < this->blobs_.size(); ++i) {
      if (this->layer_param_.param_size() != i) {
        CHECK_EQ(0, 1) << "Nope";
      }
      ParamSpec *fixed_param_spec = this->layer_param_.add_param();
      if ((i >= 3 * numTransition) && (i < 5 * numTransition)) {
        fixed_param_spec->set_lr_mult(0.f);
        fixed_param_spec->set_decay_mult(0.f);
      } else if (i >= 8 * numTransition) {
        fixed_param_spec->set_lr_mult(0.f);
        fixed_param_spec->set_decay_mult(0.f);
      } else {
        fixed_param_spec->set_lr_mult(1.f);
        fixed_param_spec->set_decay_mult(1.f);
      }
    }
  }

#ifdef USE_CUDA
  GPU_Initialization();
#endif
}

void DenseBlockLayer::Reshape(const vector<Blob *> &bottom,
                              const vector<Blob *> &top) {
  int batch_size = bottom[0]->shape()[0];
  int h = bottom[0]->shape()[2];
  int w = bottom[0]->shape()[3];

#ifdef USE_CUDA
  reshape_gpu_data(this->H, this->W, this->N, h, w, batch_size);
#endif
  this->N = batch_size;
  this->H = h;
  this->W = w;
  int topShapeArr[] = {
      this->N, this->initChannel + this->numTransition * this->growthRate,
      this->H, this->W};
  vector<int> topShape(topShapeArr, topShapeArr + 4);
  top[0]->Reshape(topShape);
}

void DenseBlockLayer::syncBlobs(DenseBlockLayer *originLayer) {
  vector<shared_ptr<Blob>> &originBlobs = originLayer->blobs();
  for (int blobIdx = 0; blobIdx < originBlobs.size(); ++blobIdx) {
    shared_ptr<Blob> localBlob = originBlobs[blobIdx];
    Blob *newBlob = new Blob(localBlob->shape());
    newBlob->CopyFrom(*(localBlob.get()), false);
    shared_ptr<Blob> sharedPtrBlob(newBlob);
    this->blobs_[blobIdx] = sharedPtrBlob;
  }
}

real_t getZeroPaddedValue(bool isDiff, Blob *inputData, int n, int c, int h,
                          int w) {
  int n_blob = inputData->shape(0);
  int c_blob = inputData->shape(1);
  int h_blob = inputData->shape(2);
  int w_blob = inputData->shape(3);
  if ((n < 0) || (n >= n_blob))
    return 0;
  if ((c < 0) || (c >= c_blob))
    return 0;
  if ((h < 0) || (h >= h_blob))
    return 0;
  if ((w < 0) || (w >= w_blob))
    return 0;
    return inputData->data_at(n, c, h, w);
}

// Assumption, h_filter and w_filter must be 3 for now
// naivest possible implementation of convolution, CPU forward and backward
// should not be used in production. CPU version of convolution assume img H,W
// does not change after convolution, which corresponds to denseBlock without BC
// input of size N*c_input*h_img*w_img

void convolution_Fwd(Blob *input, Blob *output, Blob *filter, int N,
                     int c_output, int c_input, int h_img, int w_img,
                     int h_filter, int w_filter) {
  int outputShape[] = {N, c_output, h_img, w_img};
  vector<int> outputShapeVec(outputShape, outputShape + 4);
  output->Reshape(outputShapeVec);
  real_t *outputPtr = output->mutable_cpu_data();
  for (int n = 0; n < N; ++n) {
    for (int c_outIdx = 0; c_outIdx < c_output; ++c_outIdx) {
      for (int hIdx = 0; hIdx < h_img; ++hIdx) {
        for (int wIdx = 0; wIdx < w_img; ++wIdx) {
          outputPtr[output->offset(n, c_outIdx, hIdx, wIdx)] = 0;
          for (int c_inIdx = 0; c_inIdx < c_input; ++c_inIdx) {
            for (int filter_x = 0; filter_x < h_filter; ++filter_x) {
              for (int filter_y = 0; filter_y < w_filter; ++filter_y) {
                int localX = hIdx + (h_filter / 2) - filter_x;
                int localY = wIdx + (w_filter / 2) - filter_y;
                outputPtr[output->offset(n, c_outIdx, hIdx, wIdx)] +=
                    (filter->data_at(c_outIdx, c_inIdx, filter_x, filter_y) *
                     getZeroPaddedValue(false, input, n, c_inIdx, localX,
                                        localY));
              }
            }
          }
        }
      }
    }
  }
}

void ReLU_Fwd(Blob *bottom, Blob *top, int N, int C, int h_img, int w_img) {
  // Reshape top
  int topShapeArr[] = {N, C, h_img, w_img};
  vector<int> topShapeVec(topShapeArr, topShapeArr + 4);
  top->Reshape(topShapeVec);
  // ReLU Fwd
  real_t *topPtr = top->mutable_cpu_data();
  for (int n = 0; n < N; ++n) {
    for (int cIdx = 0; cIdx < C; ++cIdx) {
      for (int hIdx = 0; hIdx < h_img; ++hIdx) {
        for (int wIdx = 0; wIdx < w_img; ++wIdx) {
          real_t bottomData = bottom->data_at(n, cIdx, hIdx, wIdx);
          topPtr[top->offset(n, cIdx, hIdx, wIdx)] =
              bottomData >= 0 ? bottomData : 0;
        }
      }
    }
  }
}

real_t getMean(Blob *A, int channelIdx) {
  int N = A->shape(0);
  int H = A->shape(2);
  int W = A->shape(3);
  int totalCount = N * H * W;

  real_t sum = 0;
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        sum += A->data_at(n, channelIdx, h, w);
      }
    }
  }
  return sum / totalCount;
}

real_t getVar(Blob *A, int channelIdx) {
  int N = A->shape(0);
  int H = A->shape(2);
  int W = A->shape(3);
  int totalCount = N * H * W;
  real_t mean = getMean(A, channelIdx);

  real_t sum = 0;
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        sum += (A->data_at(n, channelIdx, h, w) - mean) *
               (A->data_at(n, channelIdx, h, w) - mean);
      }
    }
  }
  return sum / totalCount;
}

void BN_inf_Fwd(Blob *input, Blob *output, int N, int C, int h_img, int w_img,
                Blob *globalMean, Blob *globalVar, Blob *scaler, Blob *bias,
                Blob *factor_b) {
  int channelShape[] = {1, C, 1, 1};
  vector<int> channelShapeVec(channelShape, channelShape + 4);
  Blob *localInf_Mean = new Blob(channelShapeVec);
  Blob *localInf_Var = new Blob(channelShapeVec);
  real_t scale_factor =
      factor_b->cpu_data()[0] == 0 ? 0 : (1 / factor_b->cpu_data()[0]);
  caffe_cpu_scale(localInf_Mean->count(), scale_factor, globalMean->cpu_data(),
                  localInf_Mean->mutable_cpu_data());
  caffe_cpu_scale(localInf_Var->count(), scale_factor, globalVar->cpu_data(),
                  localInf_Var->mutable_cpu_data());
  // Reshape output
  int outputShape[] = {N, C, h_img, w_img};
  vector<int> outputShapeVec(outputShape, outputShape + 4);
  output->Reshape(outputShapeVec);
  // BN Fwd inf
  double epsilon = 1e-5;
  real_t *outputPtr = output->mutable_cpu_data();

  for (int n = 0; n < N; ++n) {
    for (int cIdx = 0; cIdx < C; ++cIdx) {
      real_t denom = 1.0 / sqrt(localInf_Var->data_at(0, cIdx, 0, 0) + epsilon);
      for (int hIdx = 0; hIdx < h_img; ++hIdx) {
        for (int wIdx = 0; wIdx < w_img; ++wIdx) {
          outputPtr[output->offset(n, cIdx, hIdx, wIdx)] =
              scaler->data_at(0, cIdx, 0, 0) *
                  (denom * (input->data_at(n, cIdx, hIdx, wIdx) -
                            localInf_Mean->data_at(0, cIdx, 0, 0))) +
              bias->data_at(0, cIdx, 0, 0);
        }
      }
    }
  }
}

void BN_train_Fwd(Blob *bottom, Blob *top, Blob *output_xhat, Blob *globalMean,
                  Blob *globalVar, Blob *batchMean, Blob *batchVar,
                  Blob *scaler, Blob *bias, int N, int C, int h_img, int w_img,
                  real_t EMA_decay) {
  // reshape output
  int outputShape[] = {N, C, h_img, w_img};
  vector<int> outputShapeVec(outputShape, outputShape + 4);
  top->Reshape(outputShapeVec);
  output_xhat->Reshape(outputShapeVec);
  // BN Fwd train
  double epsilon = 1e-5;
  // get batch/global Mean/Var
  for (int channelIdx = 0; channelIdx < C; ++channelIdx) {
    int variance_adjust_m = N * h_img * w_img;
    // batch
    real_t *batchMean_mutable = batchMean->mutable_cpu_data();
    real_t *batchVar_mutable = batchVar->mutable_cpu_data();
    batchMean_mutable[channelIdx] = getMean(bottom, channelIdx);
    batchVar_mutable[channelIdx] =
        (variance_adjust_m / (variance_adjust_m - 1.0)) *
        getVar(bottom, channelIdx);
    // global
    real_t *globalMean_mutable = globalMean->mutable_cpu_data();
    real_t *globalVar_mutable = globalVar->mutable_cpu_data();
    globalMean_mutable[channelIdx] =
        EMA_decay * globalMean->data_at(0, channelIdx, 0, 0) +
        batchMean->data_at(0, channelIdx, 0, 0);
    globalVar_mutable[channelIdx] =
        EMA_decay * globalVar->data_at(0, channelIdx, 0, 0) +
        batchVar->data_at(0, channelIdx, 0, 0);
  }
  // process data
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < h_img; ++h) {
        for (int w = 0; w < w_img; ++w) {
          real_t *xhat_mutable = output_xhat->mutable_cpu_data();
          xhat_mutable[output_xhat->offset(n, c, h, w)] =
              (bottom->data_at(n, c, h, w) - batchMean->data_at(0, c, 0, 0)) /
              sqrt(batchVar->data_at(0, c, 0, 0) + epsilon);
          real_t *output_mutable = top->mutable_cpu_data();
          output_mutable[top->offset(n, c, h, w)] =
              (scaler->data_at(0, c, 0, 0)) *
                  (output_xhat->data_at(n, c, h, w)) +
              bias->data_at(0, c, 0, 0);
        }
      }
    }
  }
}

void DenseBlockLayer::CPU_Initialization() {
  this->batch_Mean.resize(this->numTransition);
  this->batch_Var.resize(this->numTransition);

  this->merged_conv.resize(this->numTransition + 1);
  this->BN_XhatVec.resize(this->numTransition);
  this->postBN_blobVec.resize(this->numTransition);
  this->postReLU_blobVec.resize(this->numTransition);
  this->postConv_blobVec.resize(this->numTransition);
  if (useBC) {
    BC_BN_XhatVec.resize(this->numTransition);
    postBN_BCVec.resize(this->numTransition);
    postReLU_BCVec.resize(this->numTransition);
    postConv_BCVec.resize(this->numTransition);
    batch_Mean4G.resize(numTransition);
    batch_Var4G.resize(numTransition);
  }
  for (int transitionIdx = 0; transitionIdx < this->numTransition;
       ++transitionIdx) {
    int conv_y_Channels = this->growthRate;
    int mergeChannels = this->initChannel + this->growthRate * transitionIdx;
    int channelShapeArr[] = {1, mergeChannels, 1, 1};
    int conv_y_ShapeArr[] = {this->N, conv_y_Channels, this->H, this->W};
    int mergeShapeArr[] = {this->N, mergeChannels, this->H, this->W};
    vector<int> channelShape(channelShapeArr, channelShapeArr + 4);
    vector<int> conv_y_Shape(conv_y_ShapeArr, conv_y_ShapeArr + 4);
    vector<int> mergeShape(mergeShapeArr, mergeShapeArr + 4);

    this->batch_Mean[transitionIdx] = new Blob(channelShape);
    this->batch_Var[transitionIdx] = new Blob(channelShape);

    this->merged_conv[transitionIdx] = new Blob(mergeShape);
    this->BN_XhatVec[transitionIdx] = new Blob(mergeShape);
    this->postBN_blobVec[transitionIdx] = new Blob(mergeShape);
    this->postReLU_blobVec[transitionIdx] = new Blob(mergeShape);
    this->postConv_blobVec[transitionIdx] = new Blob(conv_y_Shape);
    if (useBC) {
      int quadGShapeArr[] = {N, 4 * growthRate, H, W};
      int quadChannelArr[] = {1, 4 * growthRate, 1, 1};
      vector<int> quadGShape(quadGShapeArr, quadGShapeArr + 4);
      vector<int> quadChannelShape(quadChannelArr, quadChannelArr + 4);
      this->BC_BN_XhatVec[transitionIdx] = new Blob(quadGShape);
      this->postBN_BCVec[transitionIdx] = new Blob(quadGShape);
      this->postReLU_BCVec[transitionIdx] = new Blob(quadGShape);
      this->postConv_BCVec[transitionIdx] = new Blob(quadGShape);
      batch_Mean4G[transitionIdx] = new Blob(quadChannelShape);
      batch_Var4G[transitionIdx] = new Blob(quadChannelShape);
    }
  }
  // the last element of merged_conv serve as output of forward
  int extraMergeOutputShapeArr[] = {
      this->N, this->initChannel + this->growthRate * this->numTransition,
      this->H, this->W};
  vector<int> extraMergeOutputShapeVector(extraMergeOutputShapeArr,
                                          extraMergeOutputShapeArr + 4);
  this->merged_conv[this->numTransition] =
      new Blob(extraMergeOutputShapeVector);
}

void mergeChannelData(Blob *outputBlob, Blob *blobA, Blob *blobB) {
  int N = blobA->shape(0);
  int frontC = blobA->shape(1);
  int backC = blobB->shape(1);
  int H = blobA->shape(2);
  int W = blobA->shape(3);

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < frontC + backC; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          real_t inData;
          if (c < frontC) {
            inData = blobA->cpu_data()[blobA->offset(n, c, h, w)];
          } else {
            int readC = c - frontC;
            inData = blobB->cpu_data()[blobB->offset(n, readC, h, w)];
          }
          outputBlob->mutable_cpu_data()[outputBlob->offset(n, c, h, w)] =
              inData;
        }
      }
    }
  }
}

void BlobSetZero(Blob *B, int count) {
  real_t *B_mutable_data = B->mutable_cpu_data();
  for (int i = 0; i < count; ++i) {
    B_mutable_data[i] = 0;
  }
}

void DenseBlockLayer::LoopEndCleanup_cpu() {
  for (int transitionIdx = 0; transitionIdx < this->numTransition;
       ++transitionIdx) {
    int tensorCount = this->N * growthRate * this->H * this->W;
    int tensorMergeCount =
        this->N * (this->initChannel + this->growthRate * transitionIdx) *
        this->H * this->W;
    BlobSetZero(this->merged_conv[transitionIdx], tensorMergeCount);
    BlobSetZero(this->BN_XhatVec[transitionIdx], tensorMergeCount);
    BlobSetZero(this->postBN_blobVec[transitionIdx], tensorMergeCount);
    BlobSetZero(this->postReLU_blobVec[transitionIdx], tensorMergeCount);
    BlobSetZero(this->postConv_blobVec[transitionIdx], tensorCount);
  }
}

void DenseBlockLayer::Forward_cpu(const vector<Blob *> &bottom,
                                  const vector<Blob *> &top) {
  // init CPU
  if (!this->cpuInited) {
    // std::cout<<"fwd cpu init"<<std::endl;
    this->CPU_Initialization();
    this->cpuInited = true;
    // std::cout<<"fwd cpu init done"<<std::endl;
  }
  int bnTimerIdx = useBC ? 10 * numTransition : 5 * numTransition;
  // deploy init data
  this->merged_conv[0]->CopyFrom(*(bottom[0]));
  // init CPU finish
  for (int transitionIdx = 0; transitionIdx < this->numTransition;
       ++transitionIdx) {
    // BN
    Blob *BN_bottom = this->merged_conv[transitionIdx];
    Blob *BN_top = this->postBN_blobVec[transitionIdx];
    Blob *Scaler = this->blobs_[numTransition + transitionIdx].get();
    Blob *Bias = this->blobs_[2 * numTransition + transitionIdx].get();
    int localChannels = this->initChannel + transitionIdx * this->growthRate;
    {
      // std::cout<<"cpu BN test forward"<<std::endl;
      BN_inf_Fwd(BN_bottom, BN_top, this->N, localChannels, this->H, this->W,
                 this->blobs_[3 * this->numTransition + transitionIdx].get(),
                 this->blobs_[4 * this->numTransition + transitionIdx].get(),
                 Scaler, Bias, this->blobs_[bnTimerIdx].get());
    } 
    // ReLU
    Blob *ReLU_top = this->postReLU_blobVec[transitionIdx];
    ReLU_Fwd(BN_top, ReLU_top, this->N, localChannels, this->H, this->W);
    // if useBC, Conv1*1-BN(BC)-ReLU(BC)
    if (useBC) {
      // BC Conv 1*1
      Blob *BC_filterBlob =
          this->blobs_[5 * numTransition + transitionIdx].get();
      Blob *BC_conv_x = postReLU_blobVec[transitionIdx];
      Blob *BC_conv_y = postConv_BCVec[transitionIdx];
      int BC_conv_inChannel = initChannel + growthRate * transitionIdx;
      int BC_conv_outChannel = 4 * growthRate;
      convolution_Fwd(BC_conv_x, BC_conv_y, BC_filterBlob, N,
                      BC_conv_outChannel, BC_conv_inChannel, H, W, 1, 1);
      // BC BN
      Blob *BC_BN_x = postConv_BCVec[transitionIdx];
      Blob *BC_BN_y = postBN_BCVec[transitionIdx];
      Blob *BC_Scaler = this->blobs_[6 * numTransition + transitionIdx].get();
      Blob *BC_Bias = this->blobs_[7 * numTransition + transitionIdx].get();
      Blob *BC_Mean = this->blobs_[8 * numTransition + transitionIdx].get();
      Blob *BC_Var = this->blobs_[9 * numTransition + transitionIdx].get();
      {
        BN_inf_Fwd(BC_BN_x, BC_BN_y, N, 4 * growthRate, H, W, BC_Mean, BC_Var,
                   BC_Scaler, BC_Bias, this->blobs_[bnTimerIdx].get());
      } 
      // BC ReLU
      Blob *ReLU_x = postBN_BCVec[transitionIdx];
      Blob *ReLU_y = postReLU_BCVec[transitionIdx];
      ReLU_Fwd(ReLU_x, ReLU_y, N, 4 * growthRate, H, W);
    }
    // Conv
    Blob *filterBlob = this->blobs_[transitionIdx].get();
    Blob *conv_x =
        useBC ? postReLU_BCVec[transitionIdx] : postReLU_blobVec[transitionIdx];
    Blob *conv_y = this->postConv_blobVec[transitionIdx];
    int inConvChannel =
        useBC ? 4 * growthRate : initChannel + growthRate * transitionIdx;
    convolution_Fwd(conv_x, conv_y, filterBlob, N, growthRate, inConvChannel, H,
                    W, 3, 3);
    // post Conv merge
    Blob *mergeOutput = merged_conv[transitionIdx + 1];
    Blob *mergeInputA = merged_conv[transitionIdx];
    Blob *mergeInputB = postConv_blobVec[transitionIdx];
    mergeChannelData(mergeOutput, mergeInputA, mergeInputB);
  }
  // deploy output data
  top[0]->CopyFrom(*(this->merged_conv[this->numTransition]));
}

void DenseBlockLayer::Forward_cpu_public(const vector<Blob *> &bottom,
                                         const vector<Blob *> &top) {
  this->Forward_cpu(bottom, top);
}

#ifndef USE_CUDA
STUB_GPU(DenseBlockLayer);
#endif

REGISTER_LAYER_CLASS(DenseBlock);

} // namespace caffe
