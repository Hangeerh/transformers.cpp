#pragma once
#include <cassert>
#include <vector>

namespace tr {

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
    size_t index = 0;
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

  Tensor() {
    m_data = {(T)0};
    m_shape = {1};
    m_strides = {1};
  }

  const std::vector<size_t> &shape() {
    return (const std::vector<size_t> &)m_shape;
  }

  T &at(const std::vector<size_t> &indices) {

    size_t count = indices.size();

    assert(count == m_shape.size() &&
           "tr::Tensor::at dimension mismatch between tensor and indices");

    for (size_t i = 0; i < count; i++) {
      assert(indices.at(i) < m_shape.at(i) &&
             "tr::Tensor::at index out of bounds");
    }

    return m_data.at(calculate_index(indices));
  }
};

template <typename T> Tensor<T> mat_mul(Tensor<T> t1, Tensor<T> t2) {

  assert(t1.shape().size() == 2 && t2.shape().size() == 2 &&
         "tr::mat_mul() only accepts rank 2 tensors");

  assert(t1.shape().at(1) == t2.shape().at(0) &&
         "tr::mat_mul() dimension mismatch between multiplied matrices");

  size_t out_height = t1.shape().at(0);
  size_t out_width = t2.shape().at(1);
  size_t iters = t1.shape().at(1);

  Tensor<T> out({out_height, out_width});

  for (size_t i = 0; i < out_height; i++) {
    for (size_t j = 0; j < out_width; j++) {
      T accumulator = 0;
      for (size_t k = 0; k < iters; k++) {
        accumulator += t1.at({i, k}) * t2.at({k, j});
      }
      out.at({i, j}) = accumulator;
    }
  }

  return out;
}
} // namespace tr
