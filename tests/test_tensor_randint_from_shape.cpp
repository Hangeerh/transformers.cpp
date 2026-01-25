#include "tensor.hpp"

int main() {
  tr::Tensor<int> a = tr::tensor_randint_in_shape({3, 4}, 1);
  assert(a.shape().at(0) == 3);
  assert(a.shape().at(1) == 4);
}
