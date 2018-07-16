#include <vector>
#include <algorithm>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace caffe;


int main(int argc, char* argv[]) {
  if (caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, 0);
  }

  Net net("../models/classification/deploy.prototxt");
  net.CopyTrainedLayersFrom("../models/classification/deploy.caffemodel");
  Mat img = imread("../classification/5.jpg",0);

  Profiler* profiler = Profiler::Get();
  profiler->TurnON();
  uint64_t tic = profiler->Now();
  
  // preprocess
  int height = img.rows;
  int width = img.cols;
  Mat imgResized;
  cv::resize(img, imgResized, Size(32, 32));

  vector<Mat> bgr;
  cv::split(imgResized, bgr);
  bgr[0].convertTo(bgr[0], CV_32F);
  bgr[0] = (bgr[0]-128.0)*0.00390625f;
  //bgr[1].convertTo(bgr[1], CV_32F, 0.00390625f, -128.f);
  //bgr[2].convertTo(bgr[2], CV_32F, 0.00390625f, -128.f);

  // fill network input
  shared_ptr<Blob> data = net.blob_by_name("data");
  data->Reshape(1, 1, imgResized.rows, imgResized.cols);
  const int bias = data->offset(0, 1, 0, 0);
  const int bytes = bias * sizeof(float);
  memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
 // memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
 // memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);

  // forward
  net.Forward();

  // 
  shared_ptr<Blob> result = net.blob_by_name("softmax");
  const float* result_data = result->cpu_data();
  std::vector<float> res_v(result_data, result_data + result->count());
  float max_value = 0.0;
  int index;
  for (int r_idx = 0; r_idx < result->count(); r_idx++) {
	  if (*(result_data + r_idx) >= max_value) {
		  max_value = *(result_data + r_idx);
		  index = r_idx;
	  }
		 
  }
  LOG(INFO) << "index " << index;

  uint64_t toc = profiler->Now();
  profiler->TurnOFF();
  profiler->DumpProfile("./classification-profile.json");

  LOG(INFO) << "Costs " << (toc - tic) / 1000.f << " ms";
  
  return 0;
}
