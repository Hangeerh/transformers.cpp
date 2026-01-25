#include "tensor.hpp"
#include <cassert>
#include <vector>

int main() {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<size_t> shape = {2, 2};
  tr::Tensor<float> t = tr::tensor_from_data(data, shape);
  assert(t.at({0, 0}) == 1.0f);
  assert(t.at({0, 1}) == 2.0f);
  assert(t.at({1, 0}) == 3.0f);
  assert(t.at({1, 1}) == 4.0f);
}
