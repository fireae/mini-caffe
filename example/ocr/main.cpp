#include <algorithm>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;
using namespace caffe;


int main(int argc, char *argv[]) {
  if (caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, 0);
  }

  Net net("../models/ocr/inception-bn-res-blstm/deploy.prototxt");
  net.CopyTrainedLayersFrom("../models/ocr/inception-bn-res-blstm/deploy.caffemodel");
  Mat img = imread("../ocr/10.jpg");
  string label_file="../models/ocr/label.txt";
  std::vector<string> lines;
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    lines.push_back(string(line));

  Profiler *profiler = Profiler::Get();
  profiler->TurnON();
  uint64_t tic = profiler->Now();

  // preprocess
  int height = img.rows;
  int width = img.cols;
  Mat imgResized;
  float ratio = height*1.0 / 32.0;
  cv::resize(img, imgResized, cv::Size(), 1.0/ratio, 1.0/ratio);

  vector<Mat> bgr;
  cv::split(imgResized, bgr);
  bgr[0].convertTo(bgr[0], CV_32F, 1.f, -152.0f);
  bgr[1].convertTo(bgr[1], CV_32F, 1.f, -152.0f);
  bgr[2].convertTo(bgr[2], CV_32F, 1.f, -152.0f);

  // fill network input
  Blob* data = net.blob_by_name_mutable("data");
  data->Reshape(1, 3, imgResized.rows, imgResized.cols);
  const int bias = data->offset(0, 1, 0, 0);
  const int bytes = bias * sizeof(float);
  memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
  memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
  memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
  net.Reshape();
  // forward
  net.Forward(true);

  // get output, shape is N x 7
  shared_ptr<Blob> result = net.blob_by_name("result");
  const float *result_data = result->cpu_data();
  std::vector<float> res_v(result_data, result_data + result->count());
  for (int r_idx = 0; r_idx < result->count(); r_idx++) {
	  if (*(result_data +r_idx) >= 0) 
    LOG(INFO) << r_idx << " -- " << lines[*(result_data + r_idx)];
  }
  uint64_t toc = profiler->Now();
  profiler->TurnOFF();
  profiler->DumpProfile("./ocr-profile.json");

  LOG(INFO) << "Costs " << (toc - tic) / 1000.f << " ms";
  return 0;
}
