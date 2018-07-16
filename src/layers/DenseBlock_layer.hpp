#ifndef CAFFE_DENSEBLOCK_LAYER_HPP_
#define CAFFE_DENSEBLOCK_LAYER_HPP_

#include <vector>
#include <string>

#include "../layer.hpp"

namespace caffe {

class DenseBlockLayer : public Layer {
 public:
  explicit DenseBlockLayer(const LayerParameter& param)
      : Layer(param) {}

  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top); 
  
  virtual inline const char* type() const { return "DenseBlock"; } 

  virtual void Forward_cpu_public(const vector<Blob*>& bottom, const vector<Blob*>& top);

  void Forward_gpu_public(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void syncBlobs(DenseBlockLayer* originLayer);

  int initChannel, growthRate, numTransition; 
  int N,H,W; //N,H,W of the input tensor, inited in reshape phase
  
  bool useDropout;
  float dropoutAmount;
  unsigned long long DB_randomSeed;
  bool useBC;
  bool BC_ultra_spaceEfficient;
  
 protected:
  
  virtual void CPU_Initialization();

  void GPU_Initialization();
  void reshape_gpu_data(int oldh, int oldw, int oldn, int h, int w, int newn);

  virtual void LoopEndCleanup_cpu();

  void LoopEndCleanup_gpu();

  void resetDropoutDesc(); 

  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  //performance related parameter
  int gpu_idx_;

  //common Blobs for both CPU & GPU mode
  //in this->blobs_, containing all filters for Convolution, scalers and bias for BN
  
  //start CPU specific data section
  bool cpuInited;
  //at T has shape (1,initC+T*growth,1,1)
  vector<Blob*> batch_Mean; 
  vector<Blob*> batch_Var;
  vector<Blob*> batch_Mean4G;
  vector<Blob*> batch_Var4G;

  vector<Blob*> merged_conv;//at T has shape (N,initC+T*growth,H,W), but this vector has T+1 elements

  vector<Blob*> BN_XhatVec;//at T has shape (N,initC+T*growth,H,W)
  vector<Blob*> postBN_blobVec;
  vector<Blob*> postReLU_blobVec;
  vector<Blob*> postConv_blobVec;//at T has shape(N,growth,H,W)
  //BC related CPU 
  vector<Blob*> BC_BN_XhatVec;//at T has shape(N,4*growthRate,H,W)
  vector<Blob*> postBN_BCVec;
  vector<Blob*> postReLU_BCVec;
  vector<Blob*> postConv_BCVec; 
  //end CPU specific data section

  int trainCycleIdx; //used in BN train phase for EMA Mean/Var estimation
					 //convolution Related
  int pad_h, pad_w, conv_verticalStride, conv_horizentalStride;
  int filter_H, filter_W;
  //Decay value used in EMA of BN
  real_t EMA_decay;

#ifdef USE_CUDA
  //start GPU specific data section
  //GPU ptr for efficient space usage only, these pointers not allocated when CPU_ONLY, these are not Blobs because Descriptor is not traditional 
  //bool gpuInited;
  real_t* postConv_data_gpu;
  real_t* postConv_grad_gpu;
  real_t* postDropout_data_gpu;
  real_t* postDropout_grad_gpu;
  real_t* postBN_data_gpu;
  real_t* postBN_grad_gpu;
  real_t* postReLU_data_gpu;
  real_t* postReLU_grad_gpu;
  real_t* workspace;
  real_t* workspace2;
  //gpu workspace size
  int workspace_size_bytes;

  vector<real_t*> ResultSaveMean_gpu;
  vector<real_t*> ResultSaveInvVariance_gpu;
  vector<void*> dropout_state_gpu;
  vector<size_t> dropout_stateSize;
  vector<void*> dropout_reserve_gpu;
  vector<size_t> dropout_reserveSize;
  real_t* Mean_tmp;//used in BN inf
  real_t* Var_tmp;//used in BN inf
  
  //BC related parameters 
  vector<real_t*> postConv_4GVec; //used if not ultra space efficient mode
  real_t* postConv_4G; //used if ultra space efficient mode
  real_t* postBN_4G;
  real_t* postReLU_4G;  
  real_t* postConv_4G_grad;
  real_t* postBN_4G_grad;
  real_t* postReLU_4G_grad;
  cudnnTensorDescriptor_t * quadG_tensorDesc;
  cudnnTensorDescriptor_t * quadG_paramDesc;
  cudnnConvolutionDescriptor_t* convBC_Descriptor;
   vector<real_t*> BC_MeanInfVec;
  vector<real_t*> BC_VarInfVec;
  vector<real_t*> ResultSaveMean_BC;
  vector<real_t*> ResultSaveInvVariance_BC;
   vector<cudnnFilterDescriptor_t *> BC_filterDescriptorVec;
  //chosen Fwd,BwdFilter,BwdData algos for BC-Conv/Normal-Conv
  vector<cudnnConvolutionFwdAlgo_t *> conv_FwdAlgoVec;
  vector<cudnnConvolutionFwdAlgo_t *> BC_FwdAlgoVec;
  vector<cudnnConvolutionBwdFilterAlgo_t *> conv_BwdFilterAlgoVec;
  vector<cudnnConvolutionBwdFilterAlgo_t *> BC_BwdFilterAlgoVec;
  vector<cudnnConvolutionBwdDataAlgo_t *> conv_BwdDataAlgoVec;
  vector<cudnnConvolutionBwdDataAlgo_t *> BC_BwdDataAlgoVec; 
   //BC_dropout
  //vector<void*> BC_dropout_state;
  //vector<void*> BC_dropout_reserve;
  //vector<size_t> BC_dropout_stateSize;
  //vector<size_t> BC_dropout_reserveSize;
  //real_t* postDropout_4G;
  //real_t* postDropout_4G_grad;
  
 
   //gpu handles and descriptors
  cudnnHandle_t* cudnnHandlePtr;
  cudaStream_t* cudaPrimalStream;
  vector<cudnnHandle_t*> extraHandles;
  vector<cudaStream_t*> extraStreams;

  vector<cudnnTensorDescriptor_t *> tensorDescriptorVec_conv_x;//local Conv X
  cudnnTensorDescriptor_t * tensorDescriptor_conv_y;//local Conv Y
  vector<cudnnTensorDescriptor_t *> tensorDescriptor_BN;//<channelwise>
  //Dropout descriptor 
  vector<cudnnDropoutDescriptor_t *> dropoutDescriptorVec;
  //filter descriptor for conv
  vector<cudnnFilterDescriptor_t *> filterDescriptorVec;
  //ReLU Activation Descriptor  
  cudnnActivationDescriptor_t* ReLUDesc;
  //conv descriptor for conv
  cudnnConvolutionDescriptor_t* conv_Descriptor;
#endif
  //end GPU specific data setion
};

}  // namespace caffe

#endif  // CAFFE_DENSEBLOCK_LAYER_HPP_

