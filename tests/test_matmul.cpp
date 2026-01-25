#include "tensor.hpp"
#include <cassert>
#include <vector>

void test_2x2_matmul() {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  tr::Tensor<float> a = tr::tensor_from_data(data, {2, 2});

  data = {1.0f, 2.0f, 3.0f, 4.0f};
  tr::Tensor<float> b = tr::tensor_from_data(data, {2, 2});

  tr::Tensor<float> c = tr::matmul(a, b);

  assert(c.shape()[0] == 2);
  assert(c.shape()[1] == 2);

  assert(c.at({0, 0}) == 7);
  assert(c.at({0, 1}) == 10);
  assert(c.at({1, 0}) == 15);
  assert(c.at({1, 1}) == 22);
}

int main() { test_2x2_matmul(); }
