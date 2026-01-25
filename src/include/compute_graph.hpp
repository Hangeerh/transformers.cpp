#pragma once
#include "tensor.hpp"

namespace tr {

struct CGTensorNode {
  Tensor<float> *data;
  Tensor<float> *gradient;
};

class CGMatsumNode {
public:
  void forward();

  void set_rhs(CGTensorNode *rhs);
  void set_lhs(CGTensorNode *lhs);
  void set_res(CGTensorNode *res);

private:
  CGTensorNode *rhs;
  CGTensorNode *lhs;

  CGTensorNode *result;
};

class CGMatmulNode {
public:
  void forward();

  void set_rhs(CGTensorNode *rhs);
  void set_lhs(CGTensorNode *lhs);
  void set_res(CGTensorNode *res);

private:
  CGTensorNode *rhs;
  CGTensorNode *lhs;

  CGTensorNode *result;
};
}; // namespace tr
