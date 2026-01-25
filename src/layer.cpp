#include "layer.hpp"
#include "compute_graph.hpp"

tr::Dense::Dense(size_t in_size, size_t out_size, bool bias) {
  m_in_size = in_size;
  m_out_size = out_size;
  m_bias = bias;
}

tr::CGTensorNode *tr::Dense::compile(CGTensorNode *in) {
  tr::CGTensorNode *weights = new tr::CGTensorNode;
  tr::CGMatmulNode *multiply_op = new tr::CGMatmulNode;

  tr::CGTensorNode *out = new tr::CGTensorNode;

  multiply_op->set_lhs(in);
  multiply_op->set_rhs(weights);

  if (m_bias) {
    tr::CGTensorNode *bias = new tr::CGTensorNode;
    tr::CGTensorNode *multiply_out = new tr::CGTensorNode;
    tr::CGMatsumNode *sum_op = new tr::CGMatsumNode;

    multiply_op->set_res(multiply_out);

    sum_op->set_rhs(multiply_out);
    sum_op->set_lhs(bias);

    sum_op->set_res(out);

    return out;
  } else {
    multiply_op->set_res(out);
    return out;
  }
}
