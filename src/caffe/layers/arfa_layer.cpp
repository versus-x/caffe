#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/arfa_layer.hpp"

namespace caffe {
 // TODO: CORRECT CPU VERSION FOR DIFFERENT VARIANTS! Now works as Scaled/OnlyLeft:
template <typename Dtype>
void ARFALayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  ARFAParameter arfa_param = this->layer_param().arfa_param();
  int channels = bottom[0]->channels();
  channel_shared_ = arfa_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (arfa_param.scaled()==true) {
      this->blobs_.resize(2);
      if (channel_shared_) {
        this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
        this->blobs_[1].reset(new Blob<Dtype>(vector<int>(0)));
      } else {
        this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
        this->blobs_[1].reset(new Blob<Dtype>(vector<int>(1, channels)));
      }
    } else {
      this->blobs_.resize(1);
      if (channel_shared_) {
        this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
      } else {
        this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
      }
    }
    shared_ptr<Filler<Dtype> > filler;
    FillerParameter filler_param;
    filler_param.set_type("constant");

    filler_param.set_value(arfa_param.b());
    filler.reset(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());

    if (arfa_param.scaled()) {
      filler_param.set_value(arfa_param.c());
      filler.reset(GetFiller<Dtype>(filler_param));
      filler->Fill(this->blobs_[1].get());
    }
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1) << "Parameter b size is inconsistent with prototxt config";
    if (arfa_param.scaled())
      CHECK_EQ(this->blobs_[1]->count(), 1) << "Parameter c size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels) << "Parameter b size is inconsistent with prototxt config";
    if (arfa_param.scaled())
      CHECK_EQ(this->blobs_[1]->count(), channels) << "Parameter c size is inconsistent with prototxt config";
  }

  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_b_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_b_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_b_.count(), Dtype(1), multiplier_b_.mutable_cpu_data());
  if (arfa_param.scaled()) {
    multiplier_c_.Reshape(vector<int>(1, bottom[0]->count(1)));
    backward_buff_c_.Reshape(vector<int>(1, bottom[0]->count(1)));
    caffe_set(multiplier_c_.count(), Dtype(1), multiplier_c_.mutable_cpu_data());
  }
}

template <typename Dtype>
void ARFALayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2) << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void ARFALayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  Dtype* param_b_data_check = this->blobs_[0]->mutable_cpu_data();

  const Dtype* param_b_data = this->blobs_[0]->cpu_data();
  const Dtype* param_c_data = this->blobs_[1]->cpu_data();

  ARFAParameter arfa_param = this->layer_param().arfa_param();
  const Dtype param_a = arfa_param.a();
  const Dtype threshold_b_min = arfa_param.threshold_b_min();
  const Dtype threshold_b_max = arfa_param.threshold_b_max();

  for (int i = 0; i < this->blobs_[0]->count(); i++) {
    if (param_b_data[i]>threshold_b_max) {
      param_b_data_check[i] = threshold_b_max;
    }
    if (param_b_data[i]<threshold_b_min) {
      param_b_data_check[i] = threshold_b_min;
    }
  }

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    top_data[i] = bottom_data[i] <= Dtype(0) ? (param_a * bottom_data[i]) / (param_b_data[c] - bottom_data[i]) : bottom_data[i] * param_c_data[c];
  }
}

template <typename Dtype>
void ARFALayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* param_b_data = this->blobs_[0]->cpu_data();
  const Dtype* param_c_data = this->blobs_[1]->cpu_data();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  ARFAParameter arfa_param = this->layer_param().arfa_param();

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  if (this->param_propagate_down_[0]) {
    Dtype* param_b_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      if (bottom_data[i] < 0)
        param_b_diff[c] += top_diff[i] * (-top_data[i] / (param_b_data[c] - bottom_data[i]));
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      if (bottom_data[i] >= 0)
        bottom_diff[i] = top_diff[i] * param_c_data[c];
      else
        bottom_diff[i] = top_diff[i] * ((top_data[i] * param_b_data[c]) / (bottom_data[i] * (param_b_data[c] - bottom_data[i])));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ARFALayer);
#endif

INSTANTIATE_CLASS(ARFALayer);
REGISTER_LAYER_CLASS(ARFA);

}  // namespace caffe
