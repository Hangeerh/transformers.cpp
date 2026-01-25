#include "layer.hpp"

int main() {
  auto dense_layer = tr::Dense(4, 16, false);
  tr::CGTensorNode *in = new tr::CGTensorNode;

  tr::CGTensorNode *out = dense_layer.compile(in);
}
