#pragma once
#include "tensor.hpp"
#pragma once

namespace tr {

struct CGTensorNode {
  Tensor<float> *data;
  Tensor<float> *gradient;
  void *creator;
};

struct CGMatmulNode {
  CGTensorNode *lhs;
  CGTensorNode *rhs;

  CGTensorNode *out;
};

struct CGSumNode {
  CGTensorNode *lhs;
  CGTensorNode *rhs;

  CGTensorNode *out;
};

} // namespace tr
