#pragma once
#include <vector>

namespace transformer {

template <typename T> class Tensor {
private:
  std::vector<T> m_data;
  std::vector<int> m_shape;

public:
  Tensor(const std::vector<int> &shape) {
    m_shape = shape;
    size_t total_elements = 1;
    for (size_t i : m_shape) {
      total_elements *= i;
    }
    m_data.reserve(total_elements);
  }

  Tensor();

  std::vector<int> shape() { return m_shape; }
};

} // namespace transformer
