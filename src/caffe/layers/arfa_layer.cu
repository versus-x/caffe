#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/arfa_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ARFAThreshold(const int n, Dtype* param_data, const Dtype threshold_min, const Dtype threshold_max) {
  CUDA_KERNEL_LOOP(index, n) {
    if (param_data[index] < threshold_min) {
      param_data[index] = threshold_min;
    }
    if (param_data[index] > threshold_max) {
      param_data[index] = threshold_max;
    }
  }
}

template <typename Dtype>
__global__ void ARFAForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype param_a, const Dtype* param_b_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] / param_b_data[c] : (in[index] * param_a) / (param_b_data[c] - in[index]);
  }
}

template <typename Dtype>
__global__ void ARFAForwardScaled(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype param_a, const Dtype* param_b_data, const Dtype* param_c_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? (param_c_data[c] / param_b_data[c]) * in[index] : (in[index] * param_a) / (param_b_data[c] - in[index]);
  }
}

template <typename Dtype>
__global__ void ARFAForwardOL(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype param_a, const Dtype* param_b_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : (in[index] * param_a) / (param_b_data[c] - in[index]);
  }
}

template <typename Dtype>
__global__ void ARFAForwardScaledOL(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype param_a, const Dtype* param_b_data, const Dtype* param_c_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? param_c_data[c] * in[index] : (in[index] * param_a) / (param_b_data[c] - in[index]);
  }
}

template <typename Dtype>
__global__ void ARFABackward(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, const Dtype* out_data, Dtype* out_diff, 
    const Dtype* param_b_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in_data[index] >= 0)
      out_diff[index] = in_diff[index] / param_b_data[c];
    else
      out_diff[index] = in_diff[index] * ((out_data[index] * param_b_data[c]) / (in_data[index] * (param_b_data[c] - in_data[index])));
  }
}

template <typename Dtype>
__global__ void ARFABackwardOL(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, const Dtype* out_data, Dtype* out_diff, 
    const Dtype* param_b_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in_data[index] >= 0)
      out_diff[index] = in_diff[index];
    else
      out_diff[index] = in_diff[index] * ((out_data[index] * param_b_data[c]) / (in_data[index] * (param_b_data[c] - in_data[index])));
  }
}

template <typename Dtype>
__global__ void ARFABackwardScaled(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, const Dtype* out_data, Dtype* out_diff, 
    const Dtype* param_b_data, const Dtype* param_c_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in_data[index] >= 0)
      out_diff[index] = in_diff[index] * param_c_data[c] / param_b_data[c];
    else
      out_diff[index] = in_diff[index] * ((out_data[index] * param_b_data[c]) / (in_data[index] * (param_b_data[c] - in_data[index])));
  }
}

template <typename Dtype>
__global__ void ARFABackwardScaledOL(const int n, const int channels, const int dim,
    const Dtype* in_diff, const Dtype* in_data, const Dtype* out_data, Dtype* out_diff, 
    const Dtype* param_b_data, const Dtype* param_c_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in_data[index] >= 0)
      out_diff[index] = in_diff[index] * param_c_data[c];
    else
      out_diff[index] = in_diff[index] * ((out_data[index] * param_b_data[c]) / (in_data[index] * (param_b_data[c] - in_data[index])));
  }
}

template <typename Dtype>
__global__ void ARFAParamBBackward(const int n, const int channels, const int dim,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, const Dtype* out_data, Dtype* out_diff, const Dtype* param_b_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in_data[index] > 0)
      out_diff[index] = in_diff[index] * (-out_data[index] / param_b_data[c]);
    else
      out_diff[index] = in_diff[index] * (-out_data[index] / (param_b_data[c] - in_data[index]));
    for ( int k = 1; k < rows; k++ ) {
      int index_k = index + k * rowPitch;
      int c = ((index_k) / dim) % channels / div_factor;
      if (in_data[index_k] > 0)
        out_diff[index] = in_diff[index_k] * (-out_data[index_k] / param_b_data[c]);
      else
        out_diff[index] = in_diff[index_k] * (-out_data[index_k] / (param_b_data[c] - in_data[index_k]));
    }
  }
}

template <typename Dtype>
__global__ void ARFAParamBBackwardOL(const int n, const int channels, const int dim,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* in_data, const Dtype* out_data, Dtype* out_diff, const Dtype* param_b_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    if (in_data[index] < 0)
      out_diff[index] = in_diff[index] * (-out_data[index] / param_b_data[c]);
    else
      out_diff[index] = 0;
    for ( int k = 1; k < rows; k++ ) {
      int index_k = index + k * rowPitch;
      if (in_data[index_k] < 0) {
        int c = ((index_k) / dim) % channels / div_factor;
        out_diff[index] = in_diff[index_k] * (-out_data[index_k] / (param_b_data[c] - in_data[index_k]));
      }
    }
  }
}

template <typename Dtype>
__global__ void ARFAParamCBackward(const int n, const int channels, const int dim,
    const int rows, const int rowPitch, const Dtype* in_data, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff, const Dtype* param_c_data, const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = (in_data[index] <= 0) ? 0 : in_diff[index] * (out_data[index] / param_c_data[c]);
    for ( int k = 1; k < rows; k++ ) {
      int index_k = index + k * rowPitch;
      if (in_data[index_k] > 0) {
        int c = ((index_k) / dim) % channels / div_factor;
        out_diff[index] += in_diff[index_k] * (out_data[index_k] / param_c_data[c]);
      }
    }
  }
}

template <typename Dtype>
void ARFALayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  Dtype* param_b_data_check = this->blobs_[0]->mutable_gpu_data();
  const int param_b_count = this->blobs_[0]->count();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  if (top[0] == bottom[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  }

  ARFAParameter arfa_param = this->layer_param().arfa_param();
  const Dtype param_a = arfa_param.a();
  const Dtype threshold_b_min = arfa_param.threshold_b_min();
  const Dtype threshold_b_max = arfa_param.threshold_b_max();

  const bool scaled = arfa_param.scaled();
  const bool only_left = arfa_param.only_left();

  ARFAThreshold<Dtype><<<CAFFE_GET_BLOCKS(param_b_count), CAFFE_CUDA_NUM_THREADS>>>(param_b_count, param_b_data_check, threshold_b_min, threshold_b_max);
  CUDA_POST_KERNEL_CHECK;

  const Dtype* param_b_data = this->blobs_[0]->gpu_data();

  if (scaled) {
    const Dtype* param_c_data = this->blobs_[1]->gpu_data();
    if (only_left) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ARFAForwardScaledOL<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, channels, dim, bottom_data, top_data, param_a, param_b_data, param_c_data, div_factor);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ARFAForwardScaled<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, channels, dim, bottom_data, top_data, param_a, param_b_data, param_c_data, div_factor);
    }
  } else {
    if (only_left) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ARFAForwardOL<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, channels, dim, bottom_data, top_data, param_a, param_b_data, div_factor);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ARFAForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, channels, dim, bottom_data, top_data, param_a, param_b_data, div_factor);
    }
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ARFALayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  const int div_factor = channel_shared_ ? channels : 1;

  const Dtype* param_b_data = this->blobs_[0]->gpu_data();

  ARFAParameter arfa_param = this->layer_param().arfa_param();
  const bool scaled = arfa_param.scaled();
  const bool only_left = arfa_param.only_left();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.gpu_data();
  }

  // Propagate to param b
  if (this->param_propagate_down_[0]) {
    Dtype* param_b_diff = this->blobs_[0]->mutable_gpu_diff();
    int cdim = channels * dim;
    if (only_left) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ARFAParamBBackwardOL<Dtype><<<CAFFE_GET_BLOCKS(cdim),
        CAFFE_CUDA_NUM_THREADS>>>(
        cdim, channels, dim, bottom[0]->num(), top[0]->offset(1), top_diff ,
        bottom_data, top_data,
        backward_buff_b_.mutable_gpu_diff(), param_b_data, div_factor);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      ARFAParamBBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
        CAFFE_CUDA_NUM_THREADS>>>(
        cdim, channels, dim, bottom[0]->num(), top[0]->offset(1), top_diff ,
        bottom_data, top_data,
        backward_buff_b_.mutable_gpu_diff(), param_b_data, div_factor);
    }
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_b_.gpu_diff(),
       multiplier_b_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), param_b_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_b_.gpu_diff(), multiplier_b_.gpu_data(), 1.,
        param_b_diff);
    }
  }
  // Propagate to param c
  if (this->param_propagate_down_[1]) {
    const Dtype* param_c_data = this->blobs_[1]->gpu_data();
    Dtype* param_c_diff = this->blobs_[1]->mutable_gpu_diff();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    ARFAParamCBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, channels, dim, bottom[0]->num(), top[0]->offset(1), bottom_data, top_diff,
      top_data,
      backward_buff_c_.mutable_gpu_diff(), param_c_data, div_factor);
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_c_.gpu_diff(),
       multiplier_c_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[1]->count(), Dtype(dsum), param_c_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_c_.gpu_diff(), multiplier_c_.gpu_data(), 1.,
        param_c_diff);
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* param_b_data = this->blobs_[0]->gpu_data();
    int div_factor = channel_shared_ ? channels : 1;
    if (scaled) {
      const Dtype* param_c_data = this->blobs_[1]->gpu_data();
      if (only_left) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        ARFABackwardScaledOL<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
            count, channels, dim, top_diff, bottom_data, top_data, bottom_diff, param_b_data, param_c_data,
            div_factor);
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        ARFABackwardScaled<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
            count, channels, dim, top_diff, bottom_data, top_data, bottom_diff, param_b_data, param_c_data,
            div_factor);
      }
    } else {
      if (only_left) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        ARFABackwardOL<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
            count, channels, dim, top_diff, bottom_data, top_data, bottom_diff, param_b_data,
            div_factor);
      } else {
        // NOLINT_NEXT_LINE(whitespace/operators)
        ARFABackward<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
            count, channels, dim, top_diff, bottom_data, top_data, bottom_diff, param_b_data,
            div_factor);
      }
    }
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ARFALayer);

}  // namespace caffe
