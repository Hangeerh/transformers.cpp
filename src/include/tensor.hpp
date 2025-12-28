#pragma once
#include <vector>

namespace transformer {

template <typename T> class Tensor {
private:
  std::vector<T> m_data;
  std::vector<size_t> m_shape;
  std::vector<size_t> m_strides;

  void set_strides() {
    size_t last = m_shape.size() - 1;
    size_t stride = 1;

    for (size_t i = last; i > 0; --i) {
      stride *= m_shape.at(i);
      m_strides.at(i - 1) = stride;
    }

    m_strides.back() = 1;
  }

  size_t calculate_index(const std::vector<size_t> &indices) {
    size_t index = 1;
    size_t count = indices.size();

    for (size_t i = 0; i < count; i++) {
      index += indices.at(i) * m_strides.at(i);
    }

    return index;
  }

public:
  Tensor(const std::vector<size_t> &shape) {
    m_shape = shape;
    m_strides = std::vector<size_t>(m_shape.size());

    set_strides();

    size_t total_elements = 1;
    for (size_t i : m_shape) {
      total_elements *= i;
    }

    m_data = std::vector<T>(total_elements);
  }

  std::vector<size_t> shape() { return m_shape; }

  T get(const std::vector<size_t> &indices) {
    return m_data.at(calculate_index(indices));
  }
};

} // namespace transformer
