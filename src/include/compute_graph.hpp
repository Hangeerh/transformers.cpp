#pragma once
#include "tensor.hpp"

namespace tr {

struct CGTensorNode {
  Tensor<float> *data;
  Tensor<float> *gradient;
  void *creator;
};

class CGMatmulNode {
public:
  CGTensorNode *lhs;
  CGTensorNode *rhs;

  CGTensorNode *out;

  void forward();
};

class CGSumNode {
public:
  CGTensorNode *lhs;
  CGTensorNode *rhs;

  CGTensorNode *out;

  void forward();
};

} // namespace tr
