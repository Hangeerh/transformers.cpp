#include "compute_graph.hpp"
#include "tensor.hpp"

void tr::CGMatsumNode::forward() {
  tr::matsum_in_place(lhs->data, lhs->data, result->data);
}

void tr::CGMatsumNode::set_rhs(CGTensorNode *rhs) { this->rhs = rhs; }
void tr::CGMatsumNode::set_lhs(CGTensorNode *lhs) { this->lhs = lhs; }
void tr::CGMatsumNode::set_res(CGTensorNode *res) { this->result = res; }

void tr::CGMatmulNode::forward() {
  tr::matmul_in_place(lhs->data, rhs->data, result->data);
}

void tr::CGMatmulNode::set_rhs(CGTensorNode *rhs) { this->rhs = rhs; }
void tr::CGMatmulNode::set_lhs(CGTensorNode *lhs) { this->lhs = lhs; }
void tr::CGMatmulNode::set_res(CGTensorNode *res) { this->result = res; }
