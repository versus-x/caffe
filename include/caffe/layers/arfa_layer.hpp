#ifndef CAFFE_ARFA_LAYER_HPP_
#define CAFFE_ARFA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/* Adaptive Rational Fraction Activation function */
template <typename Dtype>
class ARFALayer : public NeuronLayer<Dtype> {
 public:
  /**
   */
  explicit ARFALayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ARFA"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool channel_shared_;
  Blob<Dtype> multiplier_b_;  // dot multiplier for backward computation of params
  Blob<Dtype> backward_buff_b_;  // temporary buffer for backward computation
  Blob<Dtype> multiplier_c_;  // dot multiplier for backward computation of params
  Blob<Dtype> backward_buff_c_;  // temporary buffer for backward computation
  Blob<Dtype> bottom_memory_;  // memory for in-place computation
};


}  // namespace caffe

#endif  // CAFFE_ARFA_LAYER_HPP_
