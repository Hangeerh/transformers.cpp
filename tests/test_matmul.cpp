#include "transformers/tensor.hpp"
#include <cassert>

int main() {
  tr::Tensor<int> a = tr::tensor_randint_in_shape({4, 5}, 1);
  tr::Tensor<int> b = tr::tensor_randint_in_shape({5, 6}, 1);
  tr::Tensor<int> c = tr::matmul(a, b);
  assert(c.shape().at(0) == a.shape().at(0));
  assert(c.shape().at(1) == b.shape().at(1));
}
