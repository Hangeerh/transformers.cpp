#include "layer.hpp"
#include "compute_graph.hpp"

tr::Dense::Dense(size_t in_size, size_t out_size, bool bias) {
  m_in_size = in_size;
  m_out_size = out_size;
  m_bias = bias;
}

tr::CGTensorNode *tr::Dense::compile(CGTensorNode *in) {
  tr::CGTensorNode *weights = new tr::CGTensorNode;
  weights->data = new tr::Tensor<float>({m_in_size, m_out_size});
  weights->gradient = new tr::Tensor<float>();

  tr::CGMatmulNode *mul_op = new tr::CGMatmulNode;
  mul_op->lhs = in;
  mul_op->rhs = weights;

  tr::CGTensorNode *mul_out = new tr::CGTensorNode;
  mul_out->data = new tr::Tensor<float>({m_out_size, in->data->shape()[1]});
  mul_out->gradient = new tr::Tensor<float>();
  mul_out->creator = (void *)mul_op;

  if (!m_bias) {
    return mul_out;
  }

  tr::CGTensorNode *bias = new tr::CGTensorNode;
  bias->data = new tr::Tensor<float>(mul_out->data->shape());
  bias->gradient = new tr::Tensor<float>();

  tr::CGTensorNode *sum_out = new tr::CGTensorNode;
  sum_out->data = new tr::Tensor<float>(mul_out->data->shape());
  sum_out->gradient = new tr::Tensor<float>();

  tr::CGSumNode *sum_op = new tr::CGSumNode;
  sum_op->lhs = mul_out;
  sum_op->rhs = bias;

  sum_op->out = sum_out;
  sum_out->creator = (void *)sum_op;

  return sum_out;
}
