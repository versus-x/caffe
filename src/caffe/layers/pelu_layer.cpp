#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/pelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void PELULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  PELUParameter pelu_param = this->layer_param().pelu_param();
  int channels = bottom[0]->channels();
  channel_shared_ = pelu_param.channel_shared();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > filler;
    FillerParameter filler_param;
    filler_param.set_type("constant");

    filler_param.set_value(pelu_param.a());
    filler.reset(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());

    filler_param.set_value(pelu_param.b());
    filler.reset(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[1].get());
  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1) << "Parameter a size is inconsistent with prototxt config";
    CHECK_EQ(this->blobs_[1]->count(), 1) << "Parameter b size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels) << "Parameter a size is inconsistent with prototxt config";
    CHECK_EQ(this->blobs_[1]->count(), channels) << "Parameter b size is inconsistent with prototxt config";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_a_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_a_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_a_.count(), Dtype(1), multiplier_a_.mutable_cpu_data());
  multiplier_b_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_b_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_b_.count(), Dtype(1), multiplier_b_.mutable_cpu_data());
}

template <typename Dtype>
void PELULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2) << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void PELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  Dtype* param_a_data_check = this->blobs_[0]->mutable_cpu_data();
  Dtype* param_b_data_check = this->blobs_[1]->mutable_cpu_data();

  const Dtype* param_a_data = this->blobs_[0]->cpu_data();
  const Dtype* param_b_data = this->blobs_[1]->cpu_data();

  PELUParameter pelu_param = this->layer_param().pelu_param();
  const Dtype threshold_a_min = pelu_param.threshold_a_min();
  const Dtype threshold_a_max = pelu_param.threshold_a_max();
  const Dtype threshold_b_min = pelu_param.threshold_b_min();
  const Dtype threshold_b_max = pelu_param.threshold_b_max();

  for (int i = 0; i < this->blobs_[0]->count(); i++) {
    if (param_a_data[i]>threshold_a_max) {
      param_a_data_check[i]=threshold_a_max;
    }
    if (param_a_data[i]<threshold_a_min) {
      param_a_data_check[i]=threshold_a_min;
    }
  }
  for (int i = 0; i < this->blobs_[1]->count(); i++) {
    if (param_b_data[i]>threshold_b_max) {
      param_b_data_check[i]=threshold_b_max;
    }
    if (param_b_data[i]<threshold_b_min) {
      param_b_data_check[i]=threshold_b_min;
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
    top_data[i] = bottom_data[i] <= Dtype(0) ? (param_a_data[c] * bottom_data[i])/(param_b_data[c] - bottom_data[i]) : (param_a_data[c]/param_b_data[c]) * bottom_data[i];
  }
}

template <typename Dtype>
void PELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* param_a_data = this->blobs_[0]->cpu_data();
  const Dtype* param_b_data = this->blobs_[1]->cpu_data();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagate to params
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* param_a_diff = this->blobs_[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
        param_a_diff[c] += top_diff[i] * (top_data[i] / param_a_data[c]);
    }
  }
  if (this->param_propagate_down_[1]) {
    Dtype* param_b_diff = this->blobs_[1]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      if (bottom_data[i] > 0)
        param_b_diff[c] += top_diff[i] * (-top_data[i]/param_b_data[c]);
      else
        param_b_diff[c] += top_diff[i] * (-top_data[i]/(param_b_data[c]-bottom_data[i]));
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      if (bottom_data[i] >= 0)
        bottom_diff[i] = top_diff[i] * (param_a_data[c]/param_b_data[c]);
      else
        bottom_diff[i] = top_diff[i] * ((top_data[i]*param_b_data[c])/(bottom_data[i]*(param_b_data[c]-bottom_data[i])));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(PELULayer);
#endif

INSTANTIATE_CLASS(PELULayer);
REGISTER_LAYER_CLASS(PELU);

}  // namespace caffe

