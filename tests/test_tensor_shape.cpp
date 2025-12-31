#include "../src/include/tensor.hpp"
#include <cassert>

int main() {
  std::vector<size_t> shape = {2, 2};
  tr::Tensor<int> tensor(shape);
  assert(tensor.shape() == shape);
}
