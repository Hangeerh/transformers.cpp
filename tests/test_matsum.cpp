#include "transformers/tensor.hpp"

int main() {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  tr::Tensor<float> a = tr::tensor_from_data(data, {2, 2});

  data = {1.0f, 2.0f, 3.0f, 4.0f};
  tr::Tensor<float> b = tr::tensor_from_data(data, {2, 2});

  tr::Tensor<float> c = tr::matsum(a, b);

  assert(c.at({0, 0}) == 2);
  assert(c.at({0, 1}) == 4);
  assert(c.at({1, 0}) == 6);
  assert(c.at({1, 1}) == 8);
}
