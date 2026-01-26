#include "compute_graph.hpp"
#include "tensor.hpp"

void tr::CGMatmulNode::forward() {
  tr::matmul_in_place(lhs->data, rhs->data, out->data);
}

void tr::CGSumNode::forward() {
  tr::matsum_in_place(lhs->data, rhs->data, out->data);
}
