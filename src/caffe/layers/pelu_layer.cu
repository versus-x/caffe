#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/pelu_layer.hpp"

namespace caffe {

// CUDA threshold
template <typename Dtype>
__global__ void PELUThreshold(const int n, Dtype* param_data, const Dtype threshold_min, const Dtype threshold_max) {
  CUDA_KERNEL_LOOP(index, n) {
    if (param_data[index] < threshold_min) {
      param_data[index] = threshold_min;
    }
    if (param_data[index] > threshold_max) {
      param_data[index] = threshold_max;
    }
  }
}

// CUDA kernel for forward
template <typename Dtype>
__global__ void PELUForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* param_a_data, const Dtype* param_b_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? (param_a_data[c] / param_b_data[c]) * in[index] : param_a_data[c]*(exp(in[index]/param_b_data[c])-1);
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void PELUBackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, const Dtype* out_data, Dtype* out_diff, 
    const Dtype* param_a_data, const Dtype* param_b_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in_data[index] >= 0)
      out_diff[index] = in_diff[index] * (param_a_data[c]/param_b_data[c]);
    else
      out_diff[index] = in_diff[index] * (param_a_data[c]/param_b_data[c])*(exp(in_data[index]/param_b_data[c]))  ;
  }
}

// CUDA kernel for element-wise parameter a backward
template <typename Dtype>
__global__ void PELUParamABackward(const int n, const int channels, const int dim,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff, const Dtype* param_a_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * (out_data[index] / param_a_data[c]);
    for ( int k = 1; k < rows; k++ ) {
        int index_k = index + k*rowPitch;
        int c = ((index_k) / dim) % channels / div_factor;
        out_diff[index] += in_diff[index_k] * (out_data[index_k] / param_a_data[c]);
    }
  }
}

// CUDA kernel for element-wise parameter b backward
template <typename Dtype>
__global__ void PELUParamBBackward(const int n, const int channels, const int dim,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, const Dtype* out_data, Dtype* out_diff, const Dtype* param_a_data, const Dtype* param_b_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in_data[index] > 0)
      out_diff[index] = in_diff[index] * (-in_data[index]*param_a_data[c]/(param_b_data[c]*param_b_data[c]));
    else
      out_diff[index] = in_diff[index] * (-exp(in_data[index]/param_b_data[c])*param_a_data[c]/(param_b_data[c]*param_b_data[c]));
    for ( int k = 1; k < rows; k++ ) {
      int index_k = index + k*rowPitch;
      int c = ((index_k) / dim) % channels / div_factor;
      if (in_data[index_k] > 0)
        out_diff[index] = in_diff[index_k] * (-in_data[index_k]*param_a_data[c]/(param_b_data[c]*param_b_data[c]));
      else
        out_diff[index] = in_diff[index_k] * (-exp(in_data[index_k]/param_b_data[c])*param_a_data[c]/(param_b_data[c]*param_b_data[c]));
    }
  }
}

template <typename Dtype>
void PELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  Dtype* param_a_data_check = this->blobs_[0]->mutable_gpu_data();
  const int param_a_count = this->blobs_[0]->count();
  Dtype* param_b_data_check = this->blobs_[1]->mutable_gpu_data();
  const int param_b_count = this->blobs_[1]->count();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  PELUParameter pelu_param = this->layer_param().pelu_param();
  const Dtype threshold_a_min = pelu_param.threshold_a_min();
  const Dtype threshold_a_max = pelu_param.threshold_a_max();
  const Dtype threshold_b_min = pelu_param.threshold_b_min();
  const Dtype threshold_b_max = pelu_param.threshold_b_max();

  PELUThreshold<Dtype><<<CAFFE_GET_BLOCKS(param_a_count), CAFFE_CUDA_NUM_THREADS>>>(param_a_count, param_a_data_check, threshold_a_min, threshold_a_max);
  CUDA_POST_KERNEL_CHECK;
  PELUThreshold<Dtype><<<CAFFE_GET_BLOCKS(param_b_count), CAFFE_CUDA_NUM_THREADS>>>(param_b_count, param_b_data_check, threshold_b_min, threshold_b_max);
  CUDA_POST_KERNEL_CHECK;

  const Dtype* param_a_data = this->blobs_[0]->gpu_data();
  const Dtype* param_b_data = this->blobs_[1]->gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  PELUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, channels, dim, bottom_data, top_data, param_a_data, param_b_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void PELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  const int div_factor = channel_shared_ ? channels : 1;

  const Dtype* param_a_data = this->blobs_[0]->gpu_data();
  const Dtype* param_b_data = this->blobs_[1]->gpu_data();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param a
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* param_a_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    PELUParamABackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, channels, dim, bottom[0]->num(), top[0]->offset(1), top_diff,
      top_data,
      backward_buff_a_.mutable_gpu_diff(), param_a_data, div_factor);
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_a_.gpu_diff(),
       multiplier_a_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), param_a_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_a_.gpu_diff(), multiplier_a_.gpu_data(), 1.,
        param_a_diff);
    }
  }
  // Propagate to param b
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[1]) {
    Dtype* param_b_diff = this->blobs_[1]->mutable_gpu_diff();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    PELUParamBBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, channels, dim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      bottom_data, top_data,
      backward_buff_b_.mutable_gpu_diff(), param_a_data, param_b_data, div_factor);
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_b_.gpu_diff(),
       multiplier_b_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[1]->count(), Dtype(dsum), param_b_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_b_.gpu_diff(), multiplier_b_.gpu_data(), 1.,
        param_b_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* param_a_data = this->blobs_[0]->gpu_data();
    const Dtype* param_b_data = this->blobs_[1]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    // NOLINT_NEXT_LINE(whitespace/operators)
    PELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
        count, channels, dim, top_diff, bottom_data, top_data, bottom_diff, param_a_data, param_b_data,
        div_factor);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PELULayer);

}  // namespace caffe

