#include "compute_graph.hpp"
#include "layer.hpp"
#include <iostream>

int main() {
  auto dense_layer = tr::Dense(4, 16, false);
  tr::CGTensorNode *in = new tr::CGTensorNode;

  tr::CGTensorNode *out = dense_layer.compile(in);

  auto mul_op = static_cast<tr::CGMatmulNode *>(out->creator);

  mul_op->forward();
}
