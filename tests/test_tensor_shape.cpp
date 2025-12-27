#include "../src/include/tensor.hpp"
#include <cassert>

int main() {
  std::vector<int> shape = {2, 2};
  transformer::Tensor<int> tensor(shape);
  assert(tensor.shape() == shape);
}
