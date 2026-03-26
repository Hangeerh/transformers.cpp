#include "tensor.hpp"

int main() {
  tr::Tensor<float> result({1, 2, 3, 4}, {2, 2});

  std::string text = result.display();

  const char *expected = "Tensor of shape (2,2)\n{\n  {1, 2}\n  {3, 4}\n}";
  assert(strcmp(text.c_str(), expected));
}
