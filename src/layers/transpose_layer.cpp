#include <vector>
#include "./transpose_layer.hpp"
#include "../util/math_functions.hpp"

namespace caffe {

void transpose_cpu(const int count, const real_t* from_data, real_t* to_data, 
	const int* from_counts, const int* to_counts, const int* map, const int num_axes) {
  int from_inds[kMaxBlobAxes] = {0};
  for (int index = 0; index < count; index++) {
  	int from_index = index, to_index = 0;
  	for (int i = 0; i < num_axes; i++) {
			from_inds[i] = from_index / from_counts[i];
			from_index = from_index % from_counts[i];
		}
		for (int i = 0; i < num_axes; i++) {
			to_index += from_inds[map[i]] * to_counts[i];
		}

		*(to_data+to_index) = *(from_data+index);
  }
}


void TransposeLayer::LayerSetUp(const vector<Blob*>& bottom,
        const vector<Blob*>& top) {
	CHECK_NE(bottom[0], top[0]) << this->type() << " Layer does not support "
		"in-place computation.";
	transpose_param_ = this->layer_param_.transpose_param();
}


void TransposeLayer::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
	vector<int> shape = bottom[0]->shape();
	CHECK_GT(shape.size(), 0) << "the dimension of the transposed blob should "
		"be greater than 0.";
	CHECK_LE(shape.size(), kMaxBlobAxes) << "the dimension of the transposed blob should "
		"be less than kMaxBlobAxes (" << kMaxBlobAxes << ").";
	CHECK_EQ(shape.size(), transpose_param_.dim_size()) << "the dimensions of "
		"the top blob and bottom blob must be equal.";
	vector<int> top_shape = permute(shape);
	top[0]->Reshape(top_shape);

	const int num_axes = transpose_param_.dim_size();
	shape.clear();
	shape.push_back(num_axes);
	
	bottom_counts_.Reshape(shape);
	top_counts_.Reshape(shape);

	int* bottom_counts_data=bottom_counts_.mutable_cpu_data();
	int* top_counts_data = top_counts_.mutable_cpu_data();
	for (int i = 1; i < num_axes; i++) {
		*bottom_counts_data = bottom[0]->count(i);
		*top_counts_data = top[0]->count(i);
		bottom_counts_data++;
		top_counts_data++;
	}
	*bottom_counts_data = 1;
	*top_counts_data = 1;

	forward_map_.Reshape(shape);
	backward_map_.Reshape(shape);

	int* forward_map_data=forward_map_.mutable_cpu_data();
	int* backward_map_data=backward_map_.mutable_cpu_data();
	for (int i = 0; i < num_axes; i++) {
		*forward_map_data = transpose_param_.dim(i);
		backward_map_data[transpose_param_.dim(i)] = i;
		forward_map_data++;
	}

	shape.clear();
	shape.push_back(bottom[0]->count() * num_axes);
	buf_.Reshape(shape);

}


vector<int> TransposeLayer::permute(const vector<int>& vec) {
	vector<int> new_vec(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		new_vec[i] = vec[transpose_param_.dim(i)];
	}
	return new_vec;
}



void TransposeLayer::Forward_cpu(const vector<Blob*>& bottom, 
		const vector<Blob*>& top) {
	transpose_cpu(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data(), 
        bottom_counts_.cpu_data(), top_counts_.cpu_data(), forward_map_.cpu_data(), 
        bottom[0]->shape().size());
}

#ifndef USE_CUDA
STUB_GPU(TransposeLayer);
#endif

REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe
