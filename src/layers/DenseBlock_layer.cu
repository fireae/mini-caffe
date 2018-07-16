#include <time.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
//#include <dirent.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include "caffe/layers/DenseBlock_layer.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/cudnn.hpp"

namespace caffe {

	bool dirExists_cu(string dirStr) {
		/* const char* dirCStr = dirStr.c_str();
		 DIR* dir = opendir(dirCStr);
		 if (ENOENT == errno){
		   return false;
		 }
		 closedir(dir);*/
		return true;
	}

	void tryCreateDirectory_cu(string fileName) {
		/* vector<string> strVec;
		 boost::split(strVec,fileName,boost::is_any_of("/"));
		 string newStr="";
		 for (int i=0;i<strVec.size()-1;++i){
		   newStr += strVec[i] + (i==strVec.size()-2?"":"/");
		 }
		 boost::filesystem::path dirToCreate(newStr);
		 if (!dirExists_cu(newStr)){
		   boost::filesystem::create_directories(dirToCreate);
		 }*/
	}


	string itos_cu(int i) {
		string output = boost::lexical_cast<string>(i);
		return output;
	}

	
	void gpu_copy_one_to_many(const real_t* inPtr_gpu, real_t* outPtr_gpu, int numChunks, int chunkSize_input, int chunkStride_output) {
		for (int chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
			const real_t* inPtr_local = inPtr_gpu + chunkIdx*chunkSize_input;
			real_t* outPtr_local = outPtr_gpu + chunkIdx*chunkStride_output;

			//printf("inpointer %p\n",inPtr_gpu);
			//printf("outpointer %p\n",outPtr_gpu);
			CUDA_CHECK(cudaMemcpy(outPtr_local, inPtr_local, chunkSize_input * sizeof(real_t), cudaMemcpyDeviceToDevice));
		}
	}

	
	void gpu_copy_many_to_one(real_t* inPtr_gpu, real_t* outPtr_gpu, int numChunks, int chunkSize_output, int chunkStride_input) {
		for (int chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
			real_t* inPtr_local = inPtr_gpu + chunkIdx*chunkStride_input;
			real_t* outPtr_local = outPtr_gpu + chunkIdx*chunkSize_output;
			CUDA_CHECK(cudaMemcpy(outPtr_local, inPtr_local, chunkSize_output * sizeof(real_t), cudaMemcpyDeviceToDevice));
		}
	}

	
	void print_gpuPtr(real_t* gpuPtr, int numValues) {
		real_t* cpuPtr = new real_t[numValues];
		cudaMemcpy(cpuPtr, gpuPtr, numValues * sizeof(real_t), cudaMemcpyDeviceToHost);
		for (int i = 0; i < numValues; ++i) {
			std::cout << cpuPtr[i] << ",";
		}
		std::cout << std::endl;
	}

	
	void log_gpuPtr(real_t* gpuPtr, int numValues, string fileName) {
		real_t* cpuPtr = new real_t[numValues];
		cudaMemcpy(cpuPtr, gpuPtr, numValues * sizeof(real_t), cudaMemcpyDeviceToHost);
		const char* fileName_cstr = fileName.c_str();
		tryCreateDirectory_cu(fileName_cstr);
		std::ofstream outWriter(fileName_cstr, std::ofstream::out);
		for (int i = 0; i < numValues; ++i) {
			outWriter << cpuPtr[i] << ",";
		}
		outWriter << std::endl;
	}

	
	void DenseBlockLayer::logInternal_gpu(string dir, int TIdx, bool logDynamic, bool logDiff) {
		string localDir = dir + "/gpu_" + itos_cu(this->logId) + "/";
		if (logDynamic) {
			int postBufferSize = this->N * (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W;
			int quadGBufferSize = N * 4 * growthRate*H*W;
			if (logDiff) {
				//postConv_grad_gpu
				log_gpuPtr(this->postConv_grad_gpu, postBufferSize, localDir + "postConv_grad_gpu_transition" + itos_cu(TIdx));
				//postBN_grad_gpu
				log_gpuPtr(this->postBN_grad_gpu, postBufferSize, localDir + "postBN_grad_gpu_transition" + itos_cu(TIdx));
				//postReLU_grad_gpu
				log_gpuPtr(this->postReLU_grad_gpu, postBufferSize, localDir + "postReLU_grad_gpu_transition" + itos_cu(TIdx));
				//BC 
				if (useBC) {
					//postConv_4G_grad
					log_gpuPtr(this->postConv_4G_grad, quadGBufferSize, localDir + "postConv_4G_grad_transition" + itos_cu(TIdx));
					//postBN_4G_grad
					log_gpuPtr(this->postBN_4G_grad, quadGBufferSize, localDir + "postBN_4G_grad_transition" + itos_cu(TIdx));
					//postReLU_4G_grad
					log_gpuPtr(this->postReLU_4G_grad, quadGBufferSize, localDir + "postReLU_4G_grad_transition" + itos_cu(TIdx));
				}
			}
			else {
				//postConv_data_gpu
				log_gpuPtr(this->postConv_data_gpu, postBufferSize, localDir + "postConv_data_gpu_transition" + itos_cu(TIdx));
				//postBN_data_gpu
				log_gpuPtr(this->postBN_data_gpu, postBufferSize, localDir + "postBN_data_gpu_transition" + itos_cu(TIdx));
				//postReLU_data_gpu
				log_gpuPtr(this->postReLU_data_gpu, postBufferSize, localDir + "postReLU_data_gpu_transition" + itos_cu(TIdx));
				if (useBC) {
					//postConv_4G
					if (BC_ultra_spaceEfficient) {
						log_gpuPtr(this->postConv_4G, quadGBufferSize, localDir + "postConv_4G_data_transition" + itos_cu(TIdx));
					}
					else {
						log_gpuPtr(this->postConv_4GVec[TIdx], quadGBufferSize, localDir + "postConv_4G_data_transition" + itos_cu(TIdx));
					}
					//postBN_4G
					log_gpuPtr(this->postBN_4G, quadGBufferSize, localDir + "postBN_4G_data_transition" + itos_cu(TIdx));
					//postReLU_4G
					log_gpuPtr(this->postReLU_4G, quadGBufferSize, localDir + "postReLU_4G_data_transition" + itos_cu(TIdx));
				}
			}
		}
		else {
			for (int transitionIdx = 0; transitionIdx < this->numTransition; ++transitionIdx) {
				int numChannel_moreWide = this->initChannel + this->growthRate * transitionIdx;
				int numChannel_quadG = 4 * growthRate;
				//global Mean/Variance
				log_gpuPtr(this->blobs_[3 * this->numTransition + transitionIdx]->mutable_gpu_data(), numChannel_moreWide, localDir + "globalMean_gpu_transition" + itos_cu(transitionIdx));
				log_gpuPtr(this->blobs_[4 * this->numTransition + transitionIdx]->mutable_gpu_data(), numChannel_moreWide, localDir + "globalVariance_gpu_transition" + itos_cu(transitionIdx));
				//ResultSaveMean/InvVariance
				log_gpuPtr(this->ResultSaveMean_gpu[transitionIdx], numChannel_moreWide, localDir + "ResultSaveMean_gpu_transition" + itos_cu(transitionIdx));
				log_gpuPtr(this->ResultSaveInvVariance_gpu[transitionIdx], numChannel_moreWide, localDir + "ResultSaveInvVariance_gpu_transition" + itos_cu(transitionIdx));
				if (useBC) {
					//global BC Mean/Variance
					log_gpuPtr(this->blobs_[8 * numTransition + transitionIdx]->mutable_gpu_data(), numChannel_quadG, localDir + "globalMean_BC_transition" + itos_cu(transitionIdx));
					log_gpuPtr(this->blobs_[9 * numTransition + transitionIdx]->mutable_gpu_data(), numChannel_quadG, localDir + "globalVar_BC_transition" + itos_cu(transitionIdx));
					//ResultSave BC Mean/InvVariance
					log_gpuPtr(this->ResultSaveMean_BC[transitionIdx], numChannel_quadG, localDir + "ResultSaveMean_BC_transition" + itos_cu(transitionIdx));
					log_gpuPtr(this->ResultSaveInvVariance_BC[transitionIdx], numChannel_quadG, localDir + "ResultSaveInvVariance_BC_transition" + itos_cu(transitionIdx));
				}
				//Filter_data/grad_gpu
				int filterSize;
				if (useBC) {
					filterSize = 4 * growthRate*growthRate * 3 * 3;
				}
				else {
					filterSize = (this->initChannel + this->growthRate*transitionIdx) * this->growthRate * 3 * 3;
				}
				log_gpuPtr(this->blobs_[transitionIdx]->mutable_gpu_data(), filterSize, localDir + "Filter_data_gpu_" + itos_cu(transitionIdx));
				log_gpuPtr(this->blobs_[transitionIdx]->mutable_gpu_diff(), filterSize, localDir + "Filter_grad_gpu_" + itos_cu(transitionIdx));
				//Scaler_data/grad_gpu
				log_gpuPtr(this->blobs_[transitionIdx + this->numTransition]->mutable_gpu_diff(), numChannel_moreWide, localDir + "Scaler_grad_gpu_" + itos_cu(transitionIdx));
				log_gpuPtr(this->blobs_[transitionIdx + this->numTransition]->mutable_gpu_data(), numChannel_moreWide, localDir + "Scaler_data_gpu_" + itos_cu(transitionIdx));
				//Bias_data/grad_gpu
				log_gpuPtr(this->blobs_[transitionIdx + 2 * this->numTransition]->mutable_gpu_diff(), numChannel_moreWide, localDir + "Bias_grad_gpu_" + itos_cu(transitionIdx));
				log_gpuPtr(this->blobs_[transitionIdx + 2 * this->numTransition]->mutable_gpu_data(), numChannel_moreWide, localDir + "Bias_data_gpu_" + itos_cu(transitionIdx));
				if (useBC) {
					//BC Filter
					int filterBC_size = (initChannel + growthRate*transitionIdx) * 4 * growthRate * 1 * 1;
					log_gpuPtr(this->blobs_[5 * numTransition + transitionIdx]->mutable_gpu_data(), filterBC_size, localDir + "Filter_data_BC_" + itos_cu(transitionIdx));
					log_gpuPtr(this->blobs_[5 * numTransition + transitionIdx]->mutable_gpu_diff(), filterBC_size, localDir + "Filter_grad_BC_" + itos_cu(transitionIdx));
					//BC scaler
					log_gpuPtr(this->blobs_[6 * numTransition + transitionIdx]->mutable_gpu_diff(), numChannel_quadG, localDir + "Scaler_grad_BC_" + itos_cu(transitionIdx));
					log_gpuPtr(this->blobs_[6 * numTransition + transitionIdx]->mutable_gpu_data(), numChannel_quadG, localDir + "Scaler_data_BC_" + itos_cu(transitionIdx));
					//BC bias
					log_gpuPtr(this->blobs_[7 * numTransition + transitionIdx]->mutable_gpu_diff(), numChannel_quadG, localDir + "Bias_grad_BC_" + itos_cu(transitionIdx));
					log_gpuPtr(this->blobs_[7 * numTransition + transitionIdx]->mutable_gpu_data(), numChannel_quadG, localDir + "Bias_data_BC_" + itos_cu(transitionIdx));
				}
			}
		}
	}

	

	
	void DenseBlockLayer::GPU_Initialization() {
		//std::cout<<"Pre DeviceSet"<<std::endl;
		//CUDA_CHECK(cudaSetDevice(1));
		//std::cout<<"Post DeviceSet"<<std::endl;
		//GPU intermediate ptrs
#if 1
		int bufferSize_byte = this->N*(this->initChannel + this->growthRate*this->numTransition)*this->H*this->W * sizeof(real_t);
		CUDA_CHECK(cudaMalloc(&this->postConv_data_gpu, bufferSize_byte));
		if (useDropout) {
			CUDA_CHECK(cudaMalloc(&this->postDropout_data_gpu, bufferSize_byte));
		}
		CUDA_CHECK(cudaMalloc(&this->postBN_data_gpu, bufferSize_byte));
		CUDA_CHECK(cudaMalloc(&this->postReLU_data_gpu, bufferSize_byte));
		CUDA_CHECK(cudaMalloc(&this->postConv_grad_gpu, bufferSize_byte));
		if (useDropout) {
			CUDA_CHECK(cudaMalloc(&this->postDropout_grad_gpu, bufferSize_byte));
		}
		CUDA_CHECK(cudaMalloc(&this->postBN_grad_gpu, bufferSize_byte));
		CUDA_CHECK(cudaMalloc(&this->postReLU_grad_gpu, bufferSize_byte));

		cudaMemset(this->postConv_data_gpu, 0, bufferSize_byte);
		cudaMemset(this->postBN_data_gpu, 0, bufferSize_byte);
		cudaMemset(this->postReLU_data_gpu, 0, bufferSize_byte);
		cudaMemset(this->postConv_grad_gpu, 0, bufferSize_byte);
		cudaMemset(this->postBN_grad_gpu, 0, bufferSize_byte);
		cudaMemset(this->postReLU_grad_gpu, 0, bufferSize_byte);
#endif 

		//workspace
		CUDA_CHECK(cudaMalloc(&this->workspace, this->workspace_size_bytes));
		cudaMemset(this->workspace, 0, this->workspace_size_bytes);
		CUDA_CHECK(cudaMalloc(&this->workspace2, this->workspace_size_bytes));
		cudaMemset(this->workspace2, 0, this->workspace_size_bytes);
		//handles and descriptors
		//cudnn handle
		this->cudnnHandlePtr = new cudnnHandle_t;
		cudaPrimalStream = new cudaStream_t;
		CUDNN_CHECK(cudnnCreate(this->cudnnHandlePtr));
		CUDA_CHECK(cudaStreamCreate(cudaPrimalStream));
		//CUDNN_CHECK(cudnnSetStream(*cudnnHandlePtr,*cudaPrimalStream));
		int extraHandle_num = 3;
		for (int i = 0; i < extraHandle_num; ++i) {
			cudnnHandle_t* localHandle = new cudnnHandle_t;
			cudaStream_t* localStream = new cudaStream_t;
			CUDNN_CHECK(cudnnCreate(localHandle));
			CUDA_CHECK(cudaStreamCreate(localStream));
			CUDNN_CHECK(cudnnSetStream(*localHandle, *localStream));
			extraHandles.push_back(localHandle);
			extraStreams.push_back(localStream);
		}
		//ReLU Activation Descriptor
		this->ReLUDesc = new cudnnActivationDescriptor_t;
		cudnn::createActivationDescriptor(ReLUDesc, CUDNN_ACTIVATION_RELU);
		//conv_y global tensor descriptor
		this->tensorDescriptor_conv_y = new cudnnTensorDescriptor_t;
		cudnn::createTensor4dDesc(this->tensorDescriptor_conv_y);
#if 1
		cudnn::setTensor4dDesc(this->tensorDescriptor_conv_y, this->N, this->growthRate, this->H, this->W, (this->numTransition*this->growthRate + this->initChannel)*this->H*this->W, this->H*this->W, this->W, 1);
#endif
		//BC
		int quadG_numValues = 4 * N*growthRate*H*W;
		int quadG_numBytes = quadG_numValues * sizeof(real_t);
		if (useBC) {
#if 1
			CUDA_CHECK(cudaMalloc(&postBN_4G, quadG_numBytes));
			CUDA_CHECK(cudaMalloc(&postBN_4G_grad, quadG_numBytes));
			CUDA_CHECK(cudaMalloc(&postReLU_4G, quadG_numBytes));
			CUDA_CHECK(cudaMalloc(&postReLU_4G_grad, quadG_numBytes));
			CUDA_CHECK(cudaMalloc(&postConv_4G_grad, quadG_numBytes));
			cudaMemset(postBN_4G, 0, quadG_numBytes);
			cudaMemset(postBN_4G_grad, 0, quadG_numBytes);
			cudaMemset(postReLU_4G, 0, quadG_numBytes);
			cudaMemset(postReLU_4G_grad, 0, quadG_numBytes);
			cudaMemset(postConv_4G_grad, 0, quadG_numBytes);
			if (BC_ultra_spaceEfficient) {
				CUDA_CHECK(cudaMalloc(&postConv_4G, quadG_numBytes));
				cudaMemset(postConv_4G, 0, quadG_numBytes);
			}
#endif
			quadG_tensorDesc = new cudnnTensorDescriptor_t;
			cudnn::createTensor4dDesc(quadG_tensorDesc);
#if 1
			cudnn::setTensor4dDesc(quadG_tensorDesc, N, 4 * growthRate, H, W, 4 * growthRate*H*W, H*W, W, 1);
#endif
			quadG_paramDesc = new cudnnTensorDescriptor_t;
			cudnn::createTensor4dDesc(quadG_paramDesc);
			cudnn::setTensor4dDesc(quadG_paramDesc, 1, 4 * growthRate, 1, 1, 4 * growthRate, 1, 1, 1);
			convBC_Descriptor = new cudnnConvolutionDescriptor_t;
			CUDNN_CHECK(cudnnCreateConvolutionDescriptor(convBC_Descriptor));
			CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*convBC_Descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, cudnn::dataType::type));
		}
		//per transition variables
		for (int i = 0; i < this->numTransition; ++i) {
			//Result Running/Saving Mean/Variance/InvVariance
			int localChannel = this->initChannel + i * this->growthRate;
			real_t* local_SaveMean;
			real_t* local_SaveInvVar;

			CUDA_CHECK(cudaMalloc(&local_SaveMean, localChannel * sizeof(real_t)));
			CUDA_CHECK(cudaMalloc(&local_SaveInvVar, localChannel * sizeof(real_t)));

			cudaMemset(local_SaveMean, 0, localChannel * sizeof(real_t));
			cudaMemset(local_SaveInvVar, 0, localChannel * sizeof(real_t));

			this->ResultSaveMean_gpu.push_back(local_SaveMean);
			this->ResultSaveInvVariance_gpu.push_back(local_SaveInvVar);

			//conv_x descriptor
			int conv_x_channels = this->initChannel + this->growthRate * i;
			cudnnTensorDescriptor_t * wide_Desc_local_x = new cudnnTensorDescriptor_t;
			cudnn::createTensor4dDesc(wide_Desc_local_x);
#if 1
			cudnn::setTensor4dDesc(wide_Desc_local_x, this->N, conv_x_channels, this->H, this->W, (this->numTransition*this->growthRate + this->initChannel)*this->H*this->W, this->H*this->W, this->W, 1);
			this->tensorDescriptorVec_conv_x.push_back(wide_Desc_local_x);
#endif
			//filter Descriptor for Convolution
			if (!useBC) {
				cudnnFilterDescriptor_t * localFilterDesc = new cudnnFilterDescriptor_t;
				cudnn::createFilterDesc(localFilterDesc, growthRate, conv_x_channels, 3, 3);
				this->filterDescriptorVec.push_back(localFilterDesc);
			}
			else {
				//3*3 convolution filter desc
				cudnnFilterDescriptor_t * localFilterDesc = new cudnnFilterDescriptor_t;
				cudnn::createFilterDesc(localFilterDesc, growthRate, 4 * growthRate, 3, 3);
				this->filterDescriptorVec.push_back(localFilterDesc);
				//1*1 convolution filter desc
				cudnnFilterDescriptor_t * localBottleneckFilterDesc = new cudnnFilterDescriptor_t;
				cudnn::createFilterDesc(localBottleneckFilterDesc, 4 * growthRate, conv_x_channels, 1, 1);
				this->BC_filterDescriptorVec.push_back(localBottleneckFilterDesc);
			}
			//BN channel-wise Descriptor
			int channelsBefore_self = initChannel + growthRate*i;
			cudnnTensorDescriptor_t * BNparam = new cudnnTensorDescriptor_t;
			cudnn::createTensor4dDesc(BNparam);
			cudnn::setTensor4dDesc(BNparam, 1, channelsBefore_self, 1, 1);
			this->tensorDescriptor_BN.push_back(BNparam);
			//Dropout Ptr and Descriptor
			if (useDropout) {
				size_t * sizeState = new size_t[1];
				size_t * sizeReserve = new size_t[1];
				CUDNN_CHECK(cudnnDropoutGetStatesSize((*cudnnHandlePtr), sizeState));
				CUDNN_CHECK(cudnnDropoutGetReserveSpaceSize(*tensorDescriptor_conv_y, sizeReserve));
				dropout_reserveSize.push_back(sizeReserve[0]);
				dropout_stateSize.push_back(sizeState[0]);
				void* localStatePtr;
				void* localReservePtr;
				CUDA_CHECK(cudaMalloc(&localStatePtr, sizeState[0]));
				CUDA_CHECK(cudaMalloc(&localReservePtr, sizeReserve[0]));
				dropout_state_gpu.push_back(localStatePtr);
				dropout_reserve_gpu.push_back(localReservePtr);
				cudnnDropoutDescriptor_t* localDropoutDesc = new cudnnDropoutDescriptor_t;
				cudnnCreateDropoutDescriptor(localDropoutDesc);
				cudnnSetDropoutDescriptor(*localDropoutDesc, *cudnnHandlePtr, dropoutAmount, localStatePtr, sizeState[0], DB_randomSeed);
				dropoutDescriptorVec.push_back(localDropoutDesc);
				DB_randomSeed += 1;
			}
			//BC
			if (useBC && (!BC_ultra_spaceEfficient)) {
				real_t* local_BC4G;
				CUDA_CHECK(cudaMalloc(&local_BC4G, quadG_numValues * sizeof(real_t)));
				cudaMemset(local_BC4G, 0, quadG_numBytes);
				postConv_4GVec.push_back(local_BC4G);
			}
			if (useBC) {
				real_t* BC_tmpMeanLocal;
				real_t* BC_tmpVarLocal;
				int numChannel_BC = 4 * growthRate;
				int byteChannel_BC = numChannel_BC * sizeof(real_t);
				CUDA_CHECK(cudaMalloc(&BC_tmpMeanLocal, numChannel_BC * sizeof(real_t)));
				CUDA_CHECK(cudaMalloc(&BC_tmpVarLocal, numChannel_BC * sizeof(real_t)));
				cudaMemset(BC_tmpMeanLocal, 0, byteChannel_BC);
				cudaMemset(BC_tmpVarLocal, 0, byteChannel_BC);
				BC_MeanInfVec.push_back(BC_tmpMeanLocal);
				BC_VarInfVec.push_back(BC_tmpVarLocal);
				real_t* BC_localSaveMean;
				real_t* BC_localSaveInvVar;
				CUDA_CHECK(cudaMalloc(&BC_localSaveMean, numChannel_BC * sizeof(real_t)));
				CUDA_CHECK(cudaMalloc(&BC_localSaveInvVar, numChannel_BC * sizeof(real_t)));
				cudaMemset(BC_localSaveMean, 0, byteChannel_BC);
				cudaMemset(BC_localSaveInvVar, 0, byteChannel_BC);
				ResultSaveMean_BC.push_back(BC_localSaveMean);
				ResultSaveInvVariance_BC.push_back(BC_localSaveInvVar);
			}
		}
		//Conv Descriptor
		this->conv_Descriptor = new cudnnConvolutionDescriptor_t;
		CUDNN_CHECK(cudnnCreateConvolutionDescriptor(this->conv_Descriptor));
		CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*this->conv_Descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, cudnn::dataType::type));

		//Mean and Var tmp
		int totalNumChannel = this->initChannel + this->growthRate * this->numTransition;
		CUDA_CHECK(cudaMalloc(&this->Mean_tmp, totalNumChannel * sizeof(real_t)));
		CUDA_CHECK(cudaMalloc(&this->Var_tmp, totalNumChannel * sizeof(real_t)));

		//Convolution Algorithms
		for (int transitionIdx = 0; transitionIdx < numTransition; ++transitionIdx) {
			cudnnTensorDescriptor_t conv_x_desc;
			cudnnTensorDescriptor_t conv_y_desc;
			cudnnFilterDescriptor_t conv_w_desc;
			cudnnTensorDescriptor_t BC_x_desc;
			cudnnTensorDescriptor_t BC_y_desc;
			cudnnFilterDescriptor_t BC_w_desc;
			if (useBC) {
				conv_x_desc = *(quadG_tensorDesc);
				conv_y_desc = *(tensorDescriptor_conv_y);
				conv_w_desc = *(filterDescriptorVec[transitionIdx]);
				BC_x_desc = *(tensorDescriptorVec_conv_x[transitionIdx]);
				BC_y_desc = *(quadG_tensorDesc);
				BC_w_desc = *(BC_filterDescriptorVec[transitionIdx]);
			}
			else {
				conv_x_desc = *(tensorDescriptorVec_conv_x[transitionIdx]);
				conv_y_desc = *(tensorDescriptor_conv_y);
				conv_w_desc = *(filterDescriptorVec[transitionIdx]);
			}
			//Conv Fwd Algo
			cudnnConvolutionFwdAlgo_t* conv_FwdAlgo_local = new cudnnConvolutionFwdAlgo_t;
			CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
				*cudnnHandlePtr,
				conv_x_desc, conv_w_desc, *conv_Descriptor, conv_y_desc,
				CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
				workspace_size_bytes, conv_FwdAlgo_local
			));
			conv_FwdAlgoVec.push_back(conv_FwdAlgo_local);
			//Conv Bwd Filter Algo
			cudnnConvolutionBwdFilterAlgo_t* conv_BwdFilter_local = new cudnnConvolutionBwdFilterAlgo_t;
			CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
				*cudnnHandlePtr,
				conv_x_desc, conv_y_desc, *conv_Descriptor, conv_w_desc,
				CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
				workspace_size_bytes, conv_BwdFilter_local
			));
			conv_BwdFilterAlgoVec.push_back(conv_BwdFilter_local);
			//Conv Bwd Data Algo
			cudnnConvolutionBwdDataAlgo_t* conv_BwdData_local = new cudnnConvolutionBwdDataAlgo_t;
			CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
				*(this->extraHandles[0]),
				conv_w_desc, conv_y_desc, *conv_Descriptor, conv_x_desc,
				CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
				workspace_size_bytes, conv_BwdData_local
			));
			conv_BwdDataAlgoVec.push_back(conv_BwdData_local);
			//BC Convolution
			if (useBC) {
				//BC Fwd Algo
				cudnnConvolutionFwdAlgo_t* BC_FwdAlgo_local = new cudnnConvolutionFwdAlgo_t;
				CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
					*cudnnHandlePtr,
					BC_x_desc, BC_w_desc, *convBC_Descriptor, BC_y_desc,
					CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
					workspace_size_bytes, BC_FwdAlgo_local
				));
				BC_FwdAlgoVec.push_back(BC_FwdAlgo_local);
				//BC Bwd Filter Algo
				cudnnConvolutionBwdFilterAlgo_t* BC_BwdFilter_local = new cudnnConvolutionBwdFilterAlgo_t;
				CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
					*cudnnHandlePtr,
					BC_x_desc, BC_y_desc, *convBC_Descriptor, BC_w_desc,
					CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
					workspace_size_bytes, BC_BwdFilter_local
				));
				BC_BwdFilterAlgoVec.push_back(BC_BwdFilter_local);
				//BC Bwd Data Algo
				cudnnConvolutionBwdDataAlgo_t* BC_BwdData_local = new cudnnConvolutionBwdDataAlgo_t;
				CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
					*(this->extraHandles[0]),
					BC_w_desc, BC_y_desc, *convBC_Descriptor, BC_x_desc,
					CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
					workspace_size_bytes, BC_BwdData_local
				));
				BC_BwdDataAlgoVec.push_back(BC_BwdData_local);
			}
		}
	}

	
	void cleanupBuffer(real_t* ptr_gpu, int count) {
		cudaMemset(ptr_gpu, 0, count * sizeof(real_t));
	}

	
	void DenseBlockLayer::LoopEndCleanup_gpu() {
		int valsBuffer = this->N * (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W;
		cleanupBuffer(this->postConv_data_gpu, valsBuffer);
		cleanupBuffer(this->postConv_grad_gpu, valsBuffer);
		if (useDropout) {
			cleanupBuffer(this->postDropout_data_gpu, valsBuffer);
			cleanupBuffer(this->postDropout_grad_gpu, valsBuffer);
		}
		cleanupBuffer(this->postBN_data_gpu, valsBuffer);
		cleanupBuffer(this->postBN_grad_gpu, valsBuffer);
		cleanupBuffer(this->postReLU_data_gpu, valsBuffer);
		cleanupBuffer(this->postReLU_grad_gpu, valsBuffer);
		int vals4G = N * 4 * growthRate*H*W;
		if (useBC) {
			cleanupBuffer(postConv_4G_grad, vals4G);
			cleanupBuffer(postBN_4G_grad, vals4G);
			cleanupBuffer(postReLU_4G_grad, vals4G);
		}
	}

	
	void DenseBlockLayer::resetDropoutDesc() {
		for (int transitionIdx = 0; transitionIdx < numTransition; ++transitionIdx) {
			std::cout << &(dropout_state_gpu[transitionIdx]) << "," << dropout_stateSize[transitionIdx] << std::endl;
			CUDNN_CHECK(cudnnSetDropoutDescriptor(
				*(dropoutDescriptorVec[transitionIdx]),
				*(this->cudnnHandlePtr),
				dropoutAmount,
				dropout_state_gpu[transitionIdx],
				dropout_stateSize[transitionIdx],
				DB_randomSeed
			));
			DB_randomSeed++;
		}
	}

	__global__ void sync_streams() {}

	
	void DenseBlockLayer::Forward_gpu(const vector<Blob*>& bottom,
		const vector<Blob*>& top) {
#if 0
		if (!this->gpuInited) {
			//std::cout<<"Initializing GPU local"<<std::endl;
			this->GPU_Initialization();
			this->gpuInited = true;
			//std::cout<< "GPUInited"<< std::endl;
		}
#endif
		clock_t begin_fwd = std::clock();//timer
		const real_t* bottom_data = bottom[0]->gpu_data();
		real_t* top_data = top[0]->mutable_gpu_data();
		const int count = bottom[0]->count();
		//copy to bottom_data to buffer with stride
		int chunkSize_copy_init = this->initChannel * this->H * this->W;
		int chunkStride_copy = (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W;
		if ((this->phase_ == TRAIN) && useDropout) {
			gpu_copy_one_to_many(bottom_data, this->postDropout_data_gpu, this->N, chunkSize_copy_init, chunkStride_copy);
		}
		else {
			gpu_copy_one_to_many(bottom_data, this->postConv_data_gpu, this->N, chunkSize_copy_init, chunkStride_copy);
		}
		int work_n = this->N * (this->initChannel + this->numTransition * this->growthRate) * this->H * this->W;
		//work in the buffer, transition by transition
		for (int transitionIdx = 0; transitionIdx < this->numTransition; ++transitionIdx) {
			//BN Fwd 
			real_t* BN_x_ptr;
			if (this->phase_ == TRAIN && useDropout) {
				BN_x_ptr = this->postDropout_data_gpu;
			}
			else {
				BN_x_ptr = this->postConv_data_gpu;
			}
			real_t* BN_y_ptr = this->postBN_data_gpu;
			real_t* BN_globalMean = this->blobs_[3 * this->numTransition + transitionIdx]->mutable_gpu_data();
			real_t* BN_globalVar = this->blobs_[4 * this->numTransition + transitionIdx]->mutable_gpu_data();
			cudnnTensorDescriptor_t * BN_paramDesc = tensorDescriptor_BN[transitionIdx];
			int numChannels = initChannel + growthRate*transitionIdx;
			real_t* local_MeanInf = this->Mean_tmp;
			real_t* local_VarInf = this->Var_tmp;

			if (this->phase_ == TEST) {
				CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
					*(this->cudnnHandlePtr), CUDNN_BATCHNORM_SPATIAL,
					cudnn::dataType::one, cudnn::dataType::zero,
					*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_x_ptr,
					*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_y_ptr,
					*BN_paramDesc,
					this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
					this->blobs_[2 * this->numTransition + transitionIdx]->gpu_data(),
					BN_globalMean, BN_globalVar, CUDNN_BN_MIN_EPSILON)
				);
			}
			else {
				real_t* batchMean = this->ResultSaveMean_gpu[transitionIdx];
				real_t* batchInvVar = this->ResultSaveInvVariance_gpu[transitionIdx];
				CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
					*(this->cudnnHandlePtr), CUDNN_BATCHNORM_SPATIAL,
					cudnn::dataType::one, cudnn::dataType::zero,
					*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_x_ptr,
					*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_y_ptr,
					*BN_paramDesc,
					this->blobs_[this->numTransition + transitionIdx]->mutable_gpu_data(),
					this->blobs_[2 * this->numTransition + transitionIdx]->mutable_gpu_data(),
					real_t(1), local_MeanInf, local_VarInf, CUDNN_BN_MIN_EPSILON,
					batchMean, batchInvVar)
				);
				//update global Mean/Var manually
					//Mean:
				caffe_gpu_axpby(numChannels, EMA_decay, local_MeanInf, real_t(1.0 - EMA_decay), BN_globalMean);
				//Var:
				caffe_gpu_axpby(numChannels, EMA_decay, local_VarInf, real_t(1.0 - EMA_decay), BN_globalVar);
			}
			//ReLU
			real_t* ReLU_x_ptr = this->postBN_data_gpu;
			real_t* ReLU_y_ptr = this->postReLU_data_gpu;
			CUDNN_CHECK(cudnnActivationForward(*(this->cudnnHandlePtr), *ReLUDesc,
				cudnn::dataType::one,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), ReLU_x_ptr,
				cudnn::dataType::zero,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), ReLU_y_ptr)
			);
			if (useBC) {
				//Convolution 1*1 kernel 
				real_t* conv_x_4G = postReLU_data_gpu;
				real_t* conv_y_4G;
				if (BC_ultra_spaceEfficient) {
					conv_y_4G = postConv_4G;
				}
				else {
					conv_y_4G = postConv_4GVec[transitionIdx];
				}
				//CONV_ALGO
				CUDNN_CHECK(cudnnConvolutionForward(*(cudnnHandlePtr),
					cudnn::dataType::one,
					*this->tensorDescriptorVec_conv_x[transitionIdx], conv_x_4G,
					*(BC_filterDescriptorVec[transitionIdx]),
					this->blobs_[5 * numTransition + transitionIdx]->gpu_data(),
					*(convBC_Descriptor), *BC_FwdAlgoVec[transitionIdx],
					workspace, workspace_size_bytes, cudnn::dataType::zero,
					*quadG_tensorDesc, conv_y_4G
				));
				//std::cout<<"BC Fwd Conv Done"<<std::endl;
				//BN 4G Fwd
				real_t* BN_x_4G = BC_ultra_spaceEfficient ? postConv_4G : postConv_4GVec[transitionIdx];
				real_t* BN_y_4G = postBN_4G;
				real_t* BN_BC_globalMean = this->blobs_[8 * numTransition + transitionIdx]->mutable_gpu_data();
				real_t* BN_BC_globalVar = this->blobs_[9 * numTransition + transitionIdx]->mutable_gpu_data();
				real_t* localBC_MeanInf = BC_MeanInfVec[transitionIdx];
				real_t* localBC_VarInf = BC_VarInfVec[transitionIdx];
				//std::cout<<"BC Fwd BN Prepared"<<std::endl;
				if (this->phase_ == TEST) {
					CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
						*cudnnHandlePtr, CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType::one, cudnn::dataType::zero,
						*quadG_tensorDesc, BN_x_4G,
						*quadG_tensorDesc, BN_y_4G,
						*quadG_paramDesc,
						this->blobs_[6 * numTransition + transitionIdx]->gpu_data(),
						this->blobs_[7 * numTransition + transitionIdx]->gpu_data(),
						BN_BC_globalMean, BN_BC_globalVar, CUDNN_BN_MIN_EPSILON)
					);
				}
				else {
					real_t* BC_batchMean = ResultSaveMean_BC[transitionIdx];
					real_t* BC_batchInvVar = ResultSaveInvVariance_BC[transitionIdx];
					CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
						*cudnnHandlePtr, CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType::one, cudnn::dataType::zero,
						*quadG_tensorDesc, BN_x_4G,
						*quadG_tensorDesc, BN_y_4G,
						*quadG_paramDesc,
						this->blobs_[6 * numTransition + transitionIdx]->mutable_gpu_data(),
						this->blobs_[7 * numTransition + transitionIdx]->mutable_gpu_data(),
						real_t(1), localBC_MeanInf, localBC_VarInf, CUDNN_BN_MIN_EPSILON,
						BC_batchMean, BC_batchInvVar
					));
					caffe_gpu_axpby(4 * growthRate, EMA_decay, localBC_MeanInf, real_t(1.0 - EMA_decay), BN_BC_globalMean);
					caffe_gpu_axpby(4 * growthRate, EMA_decay, localBC_VarInf, real_t(1.0 - EMA_decay), BN_BC_globalVar);
				}
				//std::cout<<"BC Fwd BN Done"<<std::endl;
			//ReLU 4G Fwd
				real_t* ReLU_BC_x = postBN_4G;
				real_t* ReLU_BC_y = postReLU_4G;
				CUDNN_CHECK(cudnnActivationForward(*cudnnHandlePtr, *ReLUDesc,
					cudnn::dataType::one,
					*quadG_tensorDesc, ReLU_BC_x,
					cudnn::dataType::zero,
					*quadG_tensorDesc, ReLU_BC_y
				));
				//std::cout<<"BC Fwd ReLU Done"<<std::endl;
			}
			//Convolution
			int delayChannel = this->initChannel + this->growthRate * transitionIdx;
			real_t* conv_x_local;
			cudnnTensorDescriptor_t* conv_x_localDesc;
			if (useBC) {
				conv_x_local = postReLU_4G;
				conv_x_localDesc = quadG_tensorDesc;
			}
			else {
				conv_x_local = postReLU_data_gpu;
				conv_x_localDesc = tensorDescriptorVec_conv_x[transitionIdx];
			}
			real_t* conv_y_local = this->postConv_data_gpu + delayChannel * this->H * this->W;
			//CONV_ALGO
			CUDNN_CHECK(cudnnConvolutionForward(*(this->cudnnHandlePtr),
				cudnn::dataType::one,
				*conv_x_localDesc, conv_x_local,
				*(filterDescriptorVec[transitionIdx]),
				this->blobs_[transitionIdx]->gpu_data(),
				*conv_Descriptor, *conv_FwdAlgoVec[transitionIdx],
				workspace, workspace_size_bytes, cudnn::dataType::zero,
				*(tensorDescriptor_conv_y), conv_y_local
			)
			);
			//Dropout
			if ((this->phase_ == TRAIN) && useDropout) {
				real_t* dropout_x_local = postConv_data_gpu + delayChannel*H*W;
				real_t* dropout_y_local = postDropout_data_gpu + delayChannel*H*W;
				CUDNN_CHECK(cudnnDropoutForward(*(this->cudnnHandlePtr),
					*(dropoutDescriptorVec[transitionIdx]),
					*tensorDescriptor_conv_y, dropout_x_local,
					*tensorDescriptor_conv_y, dropout_y_local,
					dropout_reserve_gpu[transitionIdx], dropout_reserveSize[transitionIdx]
				));
			}
			//this->logInternal_gpu("TClogFwd",transitionIdx,true,false);
		}
		//deploy top data
		if ((this->phase_ == TRAIN) && useDropout) {
			cudaMemcpy(top[0]->mutable_gpu_data(), postDropout_data_gpu, work_n * sizeof(real_t), cudaMemcpyDeviceToDevice);
		}
		else {
			cudaMemcpy(top[0]->mutable_gpu_data(), postConv_data_gpu, work_n * sizeof(real_t), cudaMemcpyDeviceToDevice);
		}
		//clock_t end_fwd = std::clock();
		//double elapsed_fwd = double(end_fwd - begin_fwd) / CLOCKS_PER_SEC;
		//std::cout<<"elapsed fwd gpu:"<<elapsed_fwd<<std::endl;
		//this->logInternal_gpu("TClogFwd",-1,false,false);
	}


	
	void DenseBlockLayer::Backward_gpu(const vector<Blob*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob*>& bottom) {
#if 0
		if (!this->gpuInited) {
			this->GPU_Initialization();
			this->gpuInited = true;
		}
#endif
		//clock_t begin_bwd = std::clock();
		//assuming buffers store already computed value, always propagate down
		real_t* bottom_diff = bottom[0]->mutable_gpu_diff();
		int work_n = N * (initChannel + growthRate*numTransition) * H * W;
		//deploy top diff
		if (useDropout) {
			cudaMemcpy(postDropout_grad_gpu, top[0]->mutable_gpu_diff(), work_n * sizeof(real_t), cudaMemcpyDeviceToDevice);
		}
		else {
			cudaMemcpy(postConv_grad_gpu, top[0]->mutable_gpu_diff(), work_n * sizeof(real_t), cudaMemcpyDeviceToDevice);
		}
		//Backward, transition by transition
		for (int transitionIdx = this->numTransition - 1; transitionIdx >= 0; --transitionIdx) {
			int channelsBefore_self = this->initChannel + transitionIdx * this->growthRate;
			//Using BN & ReLU Fwd to generate corresponding postBN,postReLU data for this transition 
			//BN Fwd
			real_t* BN_x_ptr;
			if (useDropout) {
				BN_x_ptr = postDropout_data_gpu;
			}
			else {
				BN_x_ptr = postConv_data_gpu;
			}
			real_t* BN_y_ptr = postBN_data_gpu;
			real_t* BN_globalMean = this->blobs_[3 * this->numTransition + transitionIdx]->mutable_gpu_data();
			real_t* BN_globalVar = this->blobs_[4 * this->numTransition + transitionIdx]->mutable_gpu_data();
			cudnnTensorDescriptor_t* BN_paramDesc = tensorDescriptor_BN[transitionIdx];
			real_t* local_MeanInf = Mean_tmp;
			real_t* local_VarInf = Var_tmp;
			real_t* batchMean = this->ResultSaveMean_gpu[transitionIdx];
			real_t* batchInvVar = this->ResultSaveInvVariance_gpu[transitionIdx];
			CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
				*(this->cudnnHandlePtr), CUDNN_BATCHNORM_SPATIAL,
				cudnn::dataType::one, cudnn::dataType::zero,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_x_ptr,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_y_ptr,
				*BN_paramDesc,
				this->blobs_[this->numTransition + transitionIdx]->mutable_gpu_data(),
				this->blobs_[2 * this->numTransition + transitionIdx]->mutable_gpu_data(),
				real_t(1), local_MeanInf, local_VarInf, CUDNN_BN_MIN_EPSILON,
				batchMean, batchInvVar)
			);

			/*CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
			*(this->cudnnHandlePtr),CUDNN_BATCHNORM_SPATIAL,
			cudnn::dataType::one,cudnn::dataType::zero,
			*(this->tensorDescriptorVec_conv_x[transitionIdx]),BN_x_ptr,
			*(this->tensorDescriptorVec_conv_x[transitionIdx]),BN_y_ptr,
			*BN_paramDesc,
			this->blobs_[this->numTransition+transitionIdx]->gpu_data(),
				this->blobs_[2*this->numTransition+transitionIdx]->gpu_data(),
			local_MeanInf,local_VarInf,CUDNN_BN_MIN_EPSILON)
		);*/
		//ReLU Fwd
			real_t* ReLU_x_ptr = this->postBN_data_gpu;
			real_t* ReLU_y_ptr = this->postReLU_data_gpu;
			CUDNN_CHECK(cudnnActivationForward(*(this->cudnnHandlePtr), *ReLUDesc,
				cudnn::dataType::one,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), ReLU_x_ptr,
				cudnn::dataType::zero,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), ReLU_y_ptr)
			);
			if (useBC) {
				//Fwd phase 
				//If BC Ultra SpaceEfficient, then need convolution Fwd 1*1
					//CONV_ALGO
				if (BC_ultra_spaceEfficient) {
					real_t* conv_x_4G = postReLU_data_gpu;
					real_t* conv_y_4G = postConv_4G;
					CUDNN_CHECK(cudnnConvolutionForward(*cudnnHandlePtr,
						cudnn::dataType::one,
						*this->tensorDescriptorVec_conv_x[transitionIdx], conv_x_4G,
						*(BC_filterDescriptorVec[transitionIdx]),
						this->blobs_[5 * numTransition + transitionIdx]->gpu_data(),
						*(convBC_Descriptor), *BC_FwdAlgoVec[transitionIdx],
						workspace, workspace_size_bytes, cudnn::dataType::zero,
						*quadG_tensorDesc, conv_y_4G
					));
				}
				//cudnnHandle_t * localFwdHandle = BC_ultra_spaceEfficient?cudnnHandlePtr:extraHandles[0];//TODO
				cudnnHandle_t * localFwdHandle = cudnnHandlePtr;
				//BC BN Fwd reconstruction
				real_t* BN_x_4G = BC_ultra_spaceEfficient ? postConv_4G : postConv_4GVec[transitionIdx];
				real_t* BN_y_4G = postBN_4G;
				real_t* localBC_MeanInf = BC_MeanInfVec[transitionIdx];
				real_t* localBC_VarInf = BC_VarInfVec[transitionIdx];
				real_t* BC_batchMean = ResultSaveMean_BC[transitionIdx];
				real_t* BC_batchInvVar = ResultSaveInvVariance_BC[transitionIdx];
				CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
					*cudnnHandlePtr, CUDNN_BATCHNORM_SPATIAL,
					cudnn::dataType::one, cudnn::dataType::zero,
					*quadG_tensorDesc, BN_x_4G,
					*quadG_tensorDesc, BN_y_4G,
					*quadG_paramDesc,
					this->blobs_[6 * numTransition + transitionIdx]->mutable_gpu_data(),
					this->blobs_[7 * numTransition + transitionIdx]->mutable_gpu_data(),
					real_t(1), localBC_MeanInf, localBC_VarInf, CUDNN_BN_MIN_EPSILON,
					BC_batchMean, BC_batchInvVar
				));
				/*CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
				  *localFwdHandle,CUDNN_BATCHNORM_SPATIAL,
				  cudnn::dataType::one,cudnn::dataType::zero,
				  *quadG_tensorDesc,BN_x_4G,
				  *quadG_tensorDesc,BN_y_4G,
				  *quadG_paramDesc,
				  this->blobs_[6*numTransition+transitionIdx]->gpu_data(),
				  this->blobs_[7*numTransition+transitionIdx]->gpu_data(),
				  localBC_MeanInf,localBC_VarInf,CUDNN_BN_MIN_EPSILON
				));*/
				//BC ReLU Fwd reconstruction
				real_t* ReLU_BC_x = postBN_4G;
				real_t* ReLU_BC_y = postReLU_4G;
				CUDNN_CHECK(cudnnActivationForward(*localFwdHandle, *ReLUDesc,
					cudnn::dataType::one,
					*quadG_tensorDesc, ReLU_BC_x,
					cudnn::dataType::zero,
					*quadG_tensorDesc, ReLU_BC_y
				));
			}
			//CUDA_CHECK(cudaStreamSynchronize(*(extraStreams[0])));
			//sync_streams<<<1, 1>>>();	
				//Now do Bwd
				//Dropout
			if (useDropout) {
				real_t* dropout_dy_ptr = postDropout_grad_gpu + channelsBefore_self*H*W;
				real_t* dropout_dx_ptr = postConv_grad_gpu + channelsBefore_self*H*W;
				CUDNN_CHECK(cudnnDropoutBackward(*(this->cudnnHandlePtr),
					*(dropoutDescriptorVec[transitionIdx]),
					*tensorDescriptor_conv_y, dropout_dy_ptr,
					*tensorDescriptor_conv_y, dropout_dx_ptr,
					dropout_reserve_gpu[transitionIdx], dropout_reserveSize[transitionIdx]
				));
			}
			//Conv
			real_t* filterGrad_local = this->blobs_[transitionIdx]->mutable_gpu_diff();
			real_t* filterData_local = this->blobs_[transitionIdx]->mutable_gpu_data();
			real_t* conv_x_local = useBC ? postReLU_4G : postReLU_data_gpu;
			real_t* conv_dy_local = postConv_grad_gpu + channelsBefore_self * this->H * this->W;
			real_t* conv_dx_local = useBC ? postReLU_4G_grad : postReLU_grad_gpu;
			cudnnTensorDescriptor_t * conv_x_localDesc = useBC ? quadG_tensorDesc : tensorDescriptorVec_conv_x[transitionIdx];
			//Conv w.r.t. filter
			//CONV_ALGO
			CUDNN_CHECK(cudnnConvolutionBackwardFilter(*(this->cudnnHandlePtr),
				cudnn::dataType::one,
				*conv_x_localDesc, conv_x_local,
				*(this->tensorDescriptor_conv_y), conv_dy_local,
				*(this->conv_Descriptor), *conv_BwdFilterAlgoVec[transitionIdx],
				this->workspace, this->workspace_size_bytes,
				cudnn::dataType::one,
				*(this->filterDescriptorVec[transitionIdx]), filterGrad_local
			)
			);
			//Conv w.r.t. x
				//CONV_ALGO
			CUDNN_CHECK(cudnnConvolutionBackwardData(*(this->extraHandles[0]),
				cudnn::dataType::one,
				*(this->filterDescriptorVec[transitionIdx]), filterData_local,
				*(this->tensorDescriptor_conv_y), conv_dy_local,
				*(this->conv_Descriptor), *conv_BwdDataAlgoVec[transitionIdx],
				this->workspace2, this->workspace_size_bytes,
				cudnn::dataType::zero,
				*conv_x_localDesc, conv_dx_local
			)
			);
			sync_streams << <1, 1 >> > ();
			if (useBC) {
				//BC ReLU Bwd
				real_t* BC_ReLU_y_local = postReLU_4G;
				real_t* BC_ReLU_dy_local = postReLU_4G_grad;
				real_t* BC_ReLU_x_local = postBN_4G;
				real_t* BC_ReLU_dx_local = postBN_4G_grad;
				CUDNN_CHECK(cudnnActivationBackward(*cudnnHandlePtr, *ReLUDesc,
					cudnn::dataType::one,
					*quadG_tensorDesc, BC_ReLU_y_local,
					*quadG_tensorDesc, BC_ReLU_dy_local,
					*quadG_tensorDesc, BC_ReLU_x_local,
					cudnn::dataType::zero,
					*quadG_tensorDesc, BC_ReLU_dx_local
				));
				//BC BN Bwd
				real_t* BC_BN_x_local = BC_ultra_spaceEfficient ? postConv_4G : postConv_4GVec[transitionIdx];
				real_t* BC_BN_dx_local = postConv_4G_grad;
				real_t* BC_BN_dy_local = postBN_4G_grad;
				real_t* BC_saveMean_local = ResultSaveMean_BC[transitionIdx];
				real_t* BC_saveInvVar_local = ResultSaveInvVariance_BC[transitionIdx];
				// CUDNN_CHECK(
				cudnnStatus_t sta =
					cudnnBatchNormalizationBackward(
						*cudnnHandlePtr,
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType::one,
						cudnn::dataType::zero,
#if CUDNN_VERSION >= 4005
						cudnn::dataType::one,
						cudnn::dataType::one,
#endif
						*quadG_tensorDesc,
						BC_BN_x_local,
						*quadG_tensorDesc,
						BC_BN_dy_local,
						*quadG_tensorDesc,
						BC_BN_dx_local,
						*quadG_paramDesc,
						this->blobs_[6 * numTransition + transitionIdx]->gpu_data(),
						this->blobs_[6 * numTransition + transitionIdx]->mutable_gpu_diff(),
						this->blobs_[7 * numTransition + transitionIdx]->mutable_gpu_diff(),
						CUDNN_BN_MIN_EPSILON,
						BC_saveMean_local,
						BC_saveInvVar_local
					);
				//);	

			//BC Conv 1*1 Bwd
				real_t* BC_filterGrad = this->blobs_[5 * numTransition + transitionIdx]->mutable_gpu_diff();
				real_t* BC_filterData = this->blobs_[5 * numTransition + transitionIdx]->mutable_gpu_data();
				real_t* BC_conv_x_local = postReLU_data_gpu;
				real_t* BC_conv_dy_local = postConv_4G_grad;
				real_t* BC_conv_dx_local = postReLU_grad_gpu;
				//Conv Bwd w.r.t. filter
				//CONV_ALGO
				CUDNN_CHECK(cudnnConvolutionBackwardFilter(*cudnnHandlePtr,
					cudnn::dataType::one,
					*tensorDescriptorVec_conv_x[transitionIdx], BC_conv_x_local,
					*quadG_tensorDesc, BC_conv_dy_local,
					*convBC_Descriptor, *BC_BwdFilterAlgoVec[transitionIdx],
					workspace, workspace_size_bytes,
					cudnn::dataType::one,
					*BC_filterDescriptorVec[transitionIdx], BC_filterGrad
				));
				//Conv Bwd w.r.t. data
				//CONV_ALGO
				CUDNN_CHECK(cudnnConvolutionBackwardData(*(extraHandles[0]),
					cudnn::dataType::one,
					*BC_filterDescriptorVec[transitionIdx], BC_filterData,
					*quadG_tensorDesc, BC_conv_dy_local,
					*convBC_Descriptor, *BC_BwdDataAlgoVec[transitionIdx],
					workspace2, workspace_size_bytes,
					cudnn::dataType::zero,
					*tensorDescriptorVec_conv_x[transitionIdx], BC_conv_dx_local
				));
				sync_streams << <1, 1 >> > ();
			}
			//ReLU Bwd
			real_t* ReLU_y_local = postReLU_data_gpu;
			real_t* ReLU_x_local = postBN_data_gpu;
			real_t* ReLU_dy_local = postReLU_grad_gpu;
			real_t* ReLU_dx_local = postBN_grad_gpu;
			CUDNN_CHECK(cudnnActivationBackward(*(this->cudnnHandlePtr), *ReLUDesc,
				cudnn::dataType::one,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), ReLU_y_local,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), ReLU_dy_local,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), ReLU_x_local,
				cudnn::dataType::zero,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), ReLU_dx_local)
			);
			//BN Bwd
			real_t* BN_x_local;
			real_t* BN_dx_local;
			if (useDropout) {
				BN_x_local = this->postDropout_data_gpu;
				BN_dx_local = this->postDropout_grad_gpu;
			}
			else {
				BN_x_local = this->postConv_data_gpu;
				BN_dx_local = this->postConv_grad_gpu;
			}
			real_t* BN_dy_local = this->postBN_grad_gpu;
			real_t* saveMean_local = this->ResultSaveMean_gpu[transitionIdx];
			real_t* saveInvVar_local = this->ResultSaveInvVariance_gpu[transitionIdx];
			//CUDNN_CHECK(
			cudnnBatchNormalizationBackward(*(this->cudnnHandlePtr),
				CUDNN_BATCHNORM_SPATIAL,
				cudnn::dataType::one, cudnn::dataType::one,
#if CUDNN_VERSION >= 4005
				cudnn::dataType::one, cudnn::dataType::one,
#endif	  
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_x_local,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_dy_local,
				*(this->tensorDescriptorVec_conv_x[transitionIdx]), BN_dx_local,
				*BN_paramDesc,
				this->blobs_[this->numTransition + transitionIdx]->gpu_data(),
				this->blobs_[this->numTransition + transitionIdx]->mutable_gpu_diff(),
				this->blobs_[2 * this->numTransition + transitionIdx]->mutable_gpu_diff(),
				CUDNN_BN_MIN_EPSILON, saveMean_local, saveInvVar_local
			);
			//);
				//this->logInternal_gpu("TClogBwd",transitionIdx,true,false);
				//this->logInternal_gpu("TClogBwd",transitionIdx,true,true);
		}
		//deploy buffer to bottom diff
		//this->logInternal_gpu("TClogBwd",-1,false,false);
		int chunkSize_copy_init = this->initChannel * this->H * this->W;
		int chunkStride_copy = (this->initChannel + this->numTransition * this->growthRate) * this->H * this->W;
		if (useDropout) {
			gpu_copy_many_to_one(postDropout_grad_gpu, bottom_diff, this->N, chunkSize_copy_init, chunkStride_copy);
			//this->resetDropoutDesc();
		}
		else {
			gpu_copy_many_to_one(postConv_grad_gpu, bottom_diff, this->N, chunkSize_copy_init, chunkStride_copy);
		}
		int numTotalChannels = initChannel + growthRate*numTransition;
		cleanupBuffer(this->Mean_tmp, numTotalChannels);
		cleanupBuffer(this->Var_tmp, numTotalChannels);
		this->LoopEndCleanup_gpu();
		//clock_t end_bwd = std::clock();
		//double elapsed_bwd = double(end_bwd - begin_bwd) / CLOCKS_PER_SEC;
		//std::cout<<"elapsed bwd time:"<<elapsed_bwd<<std::endl;
	}

	
	void DenseBlockLayer::Forward_gpu_public(const vector<Blob*>& bottom, const vector<Blob*>& top) {
		this->Forward_gpu(bottom, top);
	}

	
	void DenseBlockLayer::Backward_gpu_public(const vector<Blob*>& top, const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
		this->Backward_gpu(top, propagate_down, bottom);
	}

	
	void ReallocCudaMem(real_t** p, int size)
	{
		cudaFree(*p); *p = 0;
		CUDA_CHECK(cudaMalloc(p, size));
		cudaMemset(*p, 0, size);
	}

	
	void DenseBlockLayer::reshape_gpu_data(int oldh, int oldw,int oldn, int h, int w,int newn)
	{
		int bufferSize_byte_old = oldn*(this->initChannel + this->growthRate*this->numTransition)*oldh*oldw * sizeof(real_t);
		int bufferSize_byte_new = newn*(this->initChannel + this->growthRate*this->numTransition)*h*w * sizeof(real_t);
		if (bufferSize_byte_new > bufferSize_byte_old)
		{
			int bufferSize_byte = bufferSize_byte_new;

			ReallocCudaMem(&this->postConv_data_gpu, bufferSize_byte);

			if (useDropout) {
				ReallocCudaMem(&this->postDropout_data_gpu, bufferSize_byte);
			}
			ReallocCudaMem(&this->postBN_data_gpu, bufferSize_byte);
			ReallocCudaMem(&this->postReLU_data_gpu, bufferSize_byte);
			ReallocCudaMem(&this->postConv_grad_gpu, bufferSize_byte);

			if (useDropout)
			{
				ReallocCudaMem(&this->postDropout_grad_gpu, bufferSize_byte);
			}
			ReallocCudaMem(&this->postBN_grad_gpu, bufferSize_byte);
			ReallocCudaMem(&this->postReLU_grad_gpu, bufferSize_byte);
		}

		cudnn::setTensor4dDesc(this->tensorDescriptor_conv_y, newn, this->growthRate, h, w, (this->numTransition*this->growthRate + this->initChannel)*h*w, h*w, w, 1);

		int quadG_numValues_old = 4 * newn*growthRate*oldh*oldw;
		int quadG_numValues = 4 * newn*growthRate*h*w;
		int quadG_numBytes = quadG_numValues * sizeof(real_t);
		if (quadG_numValues > quadG_numValues_old)
		{
			if (useBC)
			{
				ReallocCudaMem(&postBN_4G, quadG_numBytes);
				ReallocCudaMem(&postBN_4G_grad, quadG_numBytes);
				ReallocCudaMem(&postReLU_4G, quadG_numBytes);
				ReallocCudaMem(&postReLU_4G_grad, quadG_numBytes);
				ReallocCudaMem(&postConv_4G_grad, quadG_numBytes);

				if (BC_ultra_spaceEfficient) {
					ReallocCudaMem(&postConv_4G, quadG_numBytes);
				}
			}
		}
		if (useBC)
		{
			cudnn::setTensor4dDesc(quadG_tensorDesc, newn, 4 * growthRate, h, w, 4 * growthRate*h*w, h*w, w, 1);
		}

		for (int i = 0; i < this->numTransition; ++i)
		{
			int conv_x_channels = this->initChannel + this->growthRate * i;
			cudnn::setTensor4dDesc(this->tensorDescriptorVec_conv_x[i], newn, conv_x_channels, h, w, (this->numTransition*this->growthRate + this->initChannel)*h*w, h*w, w, 1);
		}
	}

	template void DenseBlockLayer<float>::reshape_gpu_data(int oldh, int oldw, int oldn, int h, int w, int newn);
	template void DenseBlockLayer<double>::reshape_gpu_data(int oldh, int oldw, int oldn, int h, int w, int newn);

	template void DenseBlockLayer<float>::GPU_Initialization();
	template void DenseBlockLayer<double>::GPU_Initialization();

	INSTANTIATE_LAYER_GPU_FUNCS(DenseBlockLayer);
}  // namespace caffe
